"""openbot-wake-listen: wake-word detection -> STT cascade -> one reactive
multi-turn session that lasts until "stop session" or "go to sleep".

Keeps the framework's core reliability properties: Ollama's grammar-
constrained {reply, tone_action} schema (common/reply_schema.py) makes the
model structurally unable to emit anything else, and movement is decided
by deterministic keyword-match on the transcript (movement/keywords.py),
never by the LLM -- a small local model reliably hears a move command but
doesn't reliably request the matching action.

Gestures/movement are requested from openbot-alive over a socket
(common/motor_client.py), not dispatched locally -- this process never
constructs its own Picarx(); see motor_client's docstring for why a second
Picarx() doesn't work on this hardware.

No competing background wake-word thread here (unlike the framework this
replaces) -- this process is single-threaded through the wake/session loop,
so the PortAudio-device-race class of bug that motivated one in the
original doesn't apply: there's nothing else trying to open the mic at the
same time.
"""
from __future__ import annotations

import json
import os
import queue
import re
import struct
import threading
import time
from collections import deque
from typing import Iterable, Iterator

import common.system  # noqa: F401 -- side effect, harmless even though this process owns no hardware

import config as cfg  # noqa: E402
from common import cognition, events as dash_events, health, mic_stream, persona as persona_mod  # noqa: E402
from common import commands, faces, journal, memory, reply_schema, sensors, speak_client as speak_mod, state, vision  # noqa: E402
from common.motor_client import dispatch as motor_dispatch  # noqa: E402
from common.speak_client import speak as speak_client  # noqa: E402
from common import stt  # noqa: E402
from common.stt import MIC_RATE, VOSK_RATE, resample_pcm  # noqa: E402
from movement.keywords import detect_movement_keyword, is_affirmative  # noqa: E402

import vosk  # noqa: E402

COMPONENT = "openbot-wake-listen"
# Matches the original design: no timeout on listening for the actual
# request after waking -- library behavior this framework carries forward
# deliberately, not a bug.
MAX_RECORD_S = 20.0
# After Vosk closes a phrase (its own endpointing, ~0.5s of silence), wait
# this long for more before replying -- a short thinking pause mid-sentence
# shouldn't end your turn, but Rocky shouldn't sit through 3s of silence either.
PAUSE_S = 0.8
# Audio kept from BEFORE speech was confirmed. Confirmation needs Vosk to hear two
# words (~1s in), so 0.5s of pre-roll clipped the start ("There is a delay
# between..." came out "Between..."). ~2s; leading silence costs Whisper nothing.
PREROLL_CHUNKS = 45
# Chat history kept for the length of one session -- what makes it a
# conversation instead of 20 unrelated one-liners. Bounded so a long
# session can't grow the prompt without limit.
MAX_HISTORY_MESSAGES = 16
# Said WHILE Rocky is talking, these cut it off (barge-in). Kept to a few
# distinct words: a grammar recognizer listening to a speaker playing
# Rocky's own voice will match anything it's allowed to match.
# No lone "quiet": Rocky's own voice through its speaker was recognized as
# "quiet" and switched quiet mode on by itself (2026-10-02).
INTERRUPT_PHRASES = ["stop", "wait", "hold on", "be quiet", "stop session", "go to sleep"]
# In the wake grammar so Vosk hears the name inside them. Whether a sleeping
# Rocky actually wakes is decided on Whisper's transcript ("wake up" in it) --
# see _on_wake -- since the grammar forces any speech onto these phrases.
def _wake_up_phrases(name: str) -> list[str]:
    return [f"{name} wake up", f"wake up {name}", f"hey {name} wake up"]

_offered_fist_bump = False
_whisper: "stt.Whisper | stt.RemoteWhisper | None" = None  # set in the background at startup; None until then
_wake_model = None  # set in main(); also used for the barge-in recognizer


def _vosk_model_path() -> str:
    # Matches the filenames the original README documents as already
    # downloaded on the robot -- see its "STT model experiments" section.
    name = "vosk-model-small-en-in-0.4" if cfg.STT_LANGUAGE.startswith("en-in") \
        else "vosk-model-small-en-us-0.15"
    return os.path.expanduser(f"~/.vosk_models/{name}")


def _rms(chunk: bytes) -> float:
    count = len(chunk) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"{count}h", chunk)
    return (sum(s * s for s in shorts) / count) ** 0.5


def _started(heard: str) -> bool:
    """Real speech, not noise: Vosk heard 2+ words (a lone "the"/"a" is what
    it hallucinates from a door click), or a phrase that survives the
    phantom-phrase filter."""
    return len(heard.split()) >= 2 or bool(stt.reject_hallucination(heard))


def _listen(mic: mic_stream.ArecordStream, onset_timeout_s: float, seed: bytes = b"",
            speculate=None, spec_out: dict | None = None) -> tuple[bytes, str]:
    """Waits (patiently, up to onset_timeout_s) for someone to start talking,
    then records until they stop -- both decided by a STREAMING Vosk
    recognizer (are there words?), not by loudness: a fixed loudness
    threshold broke whenever the mic gain or the room changed. Returns
    (audio, transcript) -- the transcript is Whisper's, Vosk's as fallback.
    `seed`: audio already heard (the wake word) to put in front, if speech follows.
    `speculate(audio, heard_from)` -> _Speculation, started at each pause; the
    one still valid when you stop is returned in spec_out["spec"]."""
    rec = vosk.KaldiRecognizer(_wake_model, VOSK_RATE)
    preroll: deque[bytes] = deque(maxlen=PREROLL_CHUNKS)
    captured: list[bytes] = []
    results: list[dict] = []
    started_at = pause_until = None
    spec = None
    deadline = time.time() + onset_timeout_s
    chunk_frames = mic.chunk_bytes // 2
    while True:
        # A patient wait inside a multi-turn session is normal, not a hang --
        # keep the watchdog fed or systemd kills a live conversation.
        health.record_success(COMPONENT, min_interval_s=5.0)
        chunk = mic.read(chunk_frames)
        if not chunk:
            break
        now = time.time()
        (captured if started_at else preroll).append(chunk)
        if rec.AcceptWaveform(resample_pcm(chunk, MIC_RATE, VOSK_RATE)):
            res = json.loads(rec.Result())
            if _started(res.get("text", "")):
                if not started_at:
                    started_at, captured = now, ([seed] if seed else []) + list(preroll)
                results.append(res)
                pause_until = now + PAUSE_S
                if speculate is not None and _whisper is not None:
                    if spec is not None:
                        spec.cancel()
                    spec = speculate(b"".join(captured), started_at - PREROLL_CHUNKS * chunk_frames / mic.rate - 1.0)
            elif started_at and not results:
                started_at, captured = None, []  # that "speech" was noise -- keep waiting
        else:
            heard = json.loads(rec.PartialResult()).get("partial", "")
            if _started(heard):
                if not started_at:
                    started_at, captured = now, ([seed] if seed else []) + list(preroll)
                pause_until = None  # still talking
                if spec is not None:
                    spec.cancel()  # not the end after all
                    spec = None
        if not started_at:
            if now > deadline:
                return b"", ""
            continue
        if (pause_until and now >= pause_until) or now - started_at > MAX_RECORD_S:
            break
    if not pause_until:  # cut off by MAX_RECORD_S mid-phrase
        results.append(json.loads(rec.FinalResult()))
    audio = b"".join(captured)
    if started_at:
        speak_mod.warm()  # amp on now, while we transcribe and think -- not after
    heard_from = (started_at or time.time()) - PREROLL_CHUNKS * chunk_frames / mic.rate - 1.0
    if spec is not None and pause_until and spec_out is not None:
        text = spec.transcript()
        if text:
            print(f"stt: used the transcript made during the pause: {text!r}")
            spec_out["spec"] = spec
            return audio, text
    if spec is not None:
        spec.cancel()
    text = stt.vosk_text(results)
    # One Whisper run on the whole utterance, after you stop. Tried and measured
    # worse: transcribing phrase-by-phrase (each call costs ~1.8s however short,
    # and split audio lost accuracy) and speculating at each pause (a run can't
    # be cancelled once started, so the real one queued behind a stale one).
    if _whisper is not None:
        t0 = time.monotonic()
        better = _whisper.transcribe(audio)
        print(f"stt: vosk {text!r} -> whisper {better!r} in {time.monotonic() - t0:.1f}s")
        text = better or text
    # Rocky may have been talking while the mic was open (a reflex, the mind, its
    # own reply's tail): drop its exact sentences, keep what the person said.
    said = speak_mod.said_between(heard_from, time.time())
    if said and text:
        cleaned = stt.remove_echo(text, said)
        if cleaned != text:
            print(f"stt: removed Rocky's own words: {text!r} -> {cleaned!r}")
            if not cleaned:
                dash_events.log_event("wake", f"ignored my own voice: {text!r}")
        text = cleaned
    return audio, text


def _reply_sentences(persona, Reply, user_text: str, history: list[dict], out: dict,
                     on_tone=lambda tone: motor_dispatch([tone])) -> Iterator[str]:
    """Streams the reply, yielding each finished sentence (persona-transformed)
    the moment it's complete -- speech starts while the rest is still being
    generated. on_tone(gesture) is called as soon as the gesture is named (it
    streams first). Never ends in silence: an unreachable LLM yields one plain
    "can't reach my brain" line. Side-effect free otherwise (a speculative
    reply may be thrown away): fills out["text"] / out["raw"]; `history` is
    only read -- the caller commits the exchange (_commit_history)."""
    prompt = persona.system_prompt_template(cfg.ALLOWED_ACTIONS, cfg.STATIONARY_ACTIONS)
    # A fresh camera frame with EVERY turn, not the mind's last look (up to 90s
    # old): "what am I holding?" got guesses ("a pen? a stylus?") without it.
    # Measured: +0.1-0.2s to the first word; frame grab ~0.04s.
    photo = vision.capture()
    conversation_prompt = journal.inject(
        ("The attached photo is what your camera sees right now -- use it when they ask "
         "about something they're showing you or what you see.\n" if photo else "") +
        "You can't switch your own modes by agreeing to -- only these exact spoken commands do: "
        f"\"be quiet\" (ends this conversation), \"go to sleep\" (everything off until \"{persona.name}, wake up\"). "
        "If they seem to want one of those, tell them the words to say instead of promising it. " +
        ("You drive only on these exact spoken commands: \"go to the <thing>\", \"come here\", "
         "\"explore\", \"stop\" (and simple forward/back/turn). Never claim you're driving otherwise -- "
         "tell them the words. " if cfg.CAN_DRIVE else
         "You have no wheels or arms: you can't move at all, so never claim you're moving. ") +
        "You can't press buttons or pick things up.\n\n"
        f"User transcript: {user_text}", query=user_text)
    rs, raw, gestured = reply_schema.ReplyStream(), "", False
    for delta in cognition.stream(cfg.LLM_BASE_URL, cfg.LLM_MODEL, conversation_prompt, system=prompt,
                                  json_schema=Reply.model_json_schema(), history=history, image_jpeg=photo):
        raw += delta
        ready = rs.feed(delta)  # parse first: tone_action may arrive in this same chunk as a sentence
        if not gestured and rs.tone_action is not None:
            gestured = True
            if rs.tone_action in cfg.STATIONARY_ACTIONS:
                on_tone(rs.tone_action)
        for sentence in ready:
            yield persona.transform(sentence)
    for sentence in rs.finish():
        yield persona.transform(sentence)
    if not raw:
        print("cognition unavailable: the reply stream was empty")
        out["text"] = "I can't reach my brain right now -- the LLM server's unreachable."
        yield out["text"]
        return
    if not rs.text:  # not the JSON shape we asked for -- say what came back rather than nothing
        print(f"reply parse failed; raw text: {raw!r}")
        yield persona.transform(raw.strip())
    out["text"], out["raw"] = rs.text or raw.strip(), raw


class _Speculation:
    """A reply started BEFORE we're sure you've finished: at each pause in your
    speech (Vosk closing a phrase), transcribe what's been said and start
    writing the reply in the background, holding the sentences. If the pause
    turns out to be the end (PAUSE_S later), the reply is already partly
    written -- that pause used to be dead time. If you keep talking it's
    cancelled. Nothing it does is visible until commit(): no gesture, no
    speech, no history (the Mac's Whisper + LLM make a wasted guess cheap)."""

    def __init__(self, persona, Reply, audio: bytes, heard_from: float, history: list[dict]):
        self.text = ""
        self.out: dict = {}
        self._cancelled = threading.Event()
        self._transcribed = threading.Event()
        self._cond = threading.Condition()
        self._sentences: list[str] = []
        self._done = self._committed = False
        self._tone: str | None = None
        threading.Thread(target=self._run, args=(persona, Reply, audio, heard_from, list(history)),
                         daemon=True).start()

    def _run(self, persona, Reply, audio: bytes, heard_from: float, history: list[dict]) -> None:
        try:
            text = _whisper.transcribe(audio) if _whisper is not None else ""
            said = speak_mod.said_between(heard_from, time.time())
            self.text = stt.remove_echo(text, said) if said and text else text
        finally:
            self._transcribed.set()
        reply = None
        try:
            if not self.text or self._cancelled.is_set() or commands.parse(self.text):
                return  # nothing to answer, or a command -- the turn handles those itself
            reply = _reply_sentences(persona, Reply, self.text, history, self.out, on_tone=self._on_tone)
            for sentence in reply:
                if self._cancelled.is_set():
                    break
                with self._cond:
                    self._sentences.append(sentence)
                    self._cond.notify_all()
        finally:
            if reply is not None:
                reply.close()  # cancelled mid-stream: stop the LLM generating unheard text
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _on_tone(self, tone: str) -> None:
        with self._cond:
            self._tone, committed = tone, self._committed
        if committed:
            motor_dispatch([tone])

    def transcript(self, timeout_s: float = 6.0) -> str:
        self._transcribed.wait(timeout_s)
        return self.text

    def cancel(self) -> None:
        self._cancelled.set()

    def ready_for(self, text: str) -> bool:
        return bool(text) and text == self.text and not self._cancelled.is_set()

    def commit(self, out: dict) -> Iterator[str]:
        """The held reply as a live sentence stream; fires the gesture now."""
        with self._cond:
            self._committed, tone = True, self._tone
        if tone:
            motor_dispatch([tone])
        i = 0
        while True:
            with self._cond:
                while i >= len(self._sentences) and not self._done:
                    self._cond.wait(0.5)
                if i >= len(self._sentences):
                    break
                sentence = self._sentences[i]
            i += 1
            yield sentence
        out.update(self.out)


def _commit_history(history: list[dict], user_text: str, out: dict) -> None:
    if out.get("raw"):
        history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": out["raw"]}]
        del history[:-MAX_HISTORY_MESSAGES]


def _speak_interruptible(persona, sentences: Iterable[str], mic: mic_stream.ArecordStream) -> tuple[str, str | None]:
    """Speaks sentences as they arrive (may be a generator still being fed by
    the LLM) while listening for an interrupt phrase. Returns (what was
    actually spoken, the interrupt phrase heard or None). An interrupt phrase Rocky has itself said
    in this reply is ignored -- otherwise Rocky saying "wait" through its own
    speaker would cut itself off."""
    pending: queue.Queue = queue.Queue()
    spoken: list[str] = []
    done, cut = threading.Event(), threading.Event()

    def _produce() -> None:
        try:
            for sentence in sentences:
                if cut.is_set():
                    break
                pending.put(sentence)
        finally:
            getattr(sentences, "close", lambda: None)()  # interrupted: stop the LLM generating unheard text
            pending.put(None)

    def _say() -> None:
        while (sentence := pending.get()) is not None and not cut.is_set():
            spoken.append(sentence)
            speak_client(sentence)
        done.set()

    threading.Thread(target=_produce, daemon=True).start()
    threading.Thread(target=_say, daemon=True).start()
    if not cfg.BARGE_IN_ENABLED or _wake_model is None:
        done.wait()
        return " ".join(spoken), None

    rec = vosk.KaldiRecognizer(_wake_model, VOSK_RATE, json.dumps(INTERRUPT_PHRASES + ["[unk]"]))
    mic.flush()
    chunk_frames = mic.chunk_bytes // 2
    while not done.is_set():
        chunk = mic.read(chunk_frames)
        if not chunk:
            break
        # FINISHED phrases only: partial guesses are where Rocky's own voice
        # (heard through its speaker) turned into false "quiet"/"wait" interrupts.
        if not rec.AcceptWaveform(resample_pcm(chunk, MIC_RATE, VOSK_RATE)):
            continue
        heard = " ".join(w for w in json.loads(rec.Result()).get("text", "").split() if w != "[unk]")
        said = set(re.findall(r"[a-z']+", " ".join(spoken).lower()))
        if any(p in heard and len(heard.split()) <= len(p.split()) + 1 and not set(p.split()) & said
               for p in INTERRUPT_PHRASES):
            cut.set()
            speak_mod.stop()
            dash_events.log_event("wake", f"interrupted by: {heard!r}")
            done.wait(2.0)
            return " ".join(spoken), heard
    return " ".join(spoken), None


def _say_plain(persona, text: str, mic: mic_stream.ArecordStream) -> None:
    spoken = persona.transform(text)
    journal.log("said", spoken)
    dash_events.log_event("reply", f"llm response: {spoken}")
    _speak_interruptible(persona, [spoken], mic)


def _do_navigation(persona, nav: tuple[str, str | None], mic: mic_stream.ArecordStream) -> bool:
    """Starts or stops a drive (it runs in openbot-alive -- movement/navigate.py,
    proven in sim/ first -- and Rocky says how it went). The conversation goes on."""
    from common.motor_client import cancel_navigation, navigate
    kind, target = nav
    if not cfg.CAN_DRIVE:
        _say_plain(persona, "I can't move -- I don't have wheels.", mic)
        return True
    if kind == "stop":
        _say_plain(persona, "Stopping." if cancel_navigation() else "Okay.", mic)
        return True
    ok, err = navigate({"approach": target} if kind == "approach" else {"explore": 60})
    if not ok:
        _say_plain(persona, "I'm already on my way -- say stop first." if "already" in err
                   else "My wheels aren't answering right now.", mic)
        return True
    journal.log("did", f"started driving {'to the ' + target if target else 'around to explore'}")
    _say_plain(persona, "Coming!" if target == "person" else f"Okay, heading to the {target}." if target
               else "Okay, exploring!", mic)
    return True


def _do_command(persona, cmd: str, mic: mic_stream.ArecordStream) -> bool:
    """Carries out a mode command. Returns whether the session continues."""
    if cmd in (commands.STOP_SESSION, commands.SLEEP):
        from common.motor_client import cancel_navigation
        cancel_navigation()  # never keep driving after "be quiet" / "go to sleep"
    if cmd == commands.STOP_SESSION:  # "be quiet" / "stop session"
        journal.log("did", "ended the conversation (asked to be quiet) -- still around in the background")
        _say_plain(persona, "Okay. Say my name if you need me.", mic)
        return False
    if cmd == commands.SLEEP:
        _say_plain(persona, persona.sleep_ack, mic)
        state.update_session({"asleep": True})
        motor_dispatch(["look down"], wait=True)  # head down: visibly asleep
        journal.log("did", f"went to sleep -- everything off, listening only for \"{persona.name}, wake up\"")
        dash_events.log_event("wake", "went to sleep")
        return False
    if cmd in (commands.LOUDER, commands.SOFTER):
        from common.system import get_volume, set_volume
        current = get_volume() or 100
        target = current + (15 if cmd == commands.LOUDER else -15)
        if cmd == commands.LOUDER and current >= 100:
            _say_plain(persona, "I'm already as loud as I go!", mic)
            return True
        new = set_volume(target)
        state.update_session({"volume": new})  # openbot-speak re-applies it after a reboot
        journal.log("did", f"volume {current}% -> {new}%")
        _say_plain(persona, "Louder now!" if cmd == commands.LOUDER else "Softer now.", mic)
        return True
    return True


def _run_turn(persona, Reply, user_text: str, history: list[dict],
              mic: mic_stream.ArecordStream, spec: "_Speculation | None" = None) -> tuple[bool, bool]:
    """One exchange. Returns (continue_session, interrupted) -- continue is
    False after "stop session" or "go to sleep"; interrupted is True if the
    person talked over the reply."""
    global _offered_fist_bump
    state.update_session({"last_heard": user_text, "last_heard_ts": time.time(), "last_activity_ts": time.time()})
    dash_events.log_event("wake", f"user: {user_text}")
    journal.log("heard", user_text)

    _learn_face(user_text)

    cmd = commands.parse(user_text, extra_sleep=persona.sleep_words)
    if cmd:
        if spec is not None:
            spec.cancel()
        dash_events.log_event("wake", f'"{user_text}" -> {cmd}')
        return _do_command(persona, cmd, mic), False

    nav = commands.navigation(user_text)  # after mode commands: "go to sleep" isn't a destination
    if nav and (cfg.CAN_DRIVE or nav[0] != "stop"):  # no wheels: a bare "stop"/"wait" is just talk
        if spec is not None:
            spec.cancel()
        dash_events.log_event("wake", f'"{user_text}" -> drive {nav}')
        return _do_navigation(persona, nav, mic), False

    move_action = detect_movement_keyword(user_text)
    if not move_action and _offered_fist_bump and is_affirmative(user_text):
        move_action = "fist bump"
    if move_action in cfg.ALLOWED_ACTIONS:
        motor_dispatch([move_action], wait=True)  # speech never starts mid-drive

    out: dict = {}
    if spec is not None and spec.ready_for(user_text):
        sentences = spec.commit(out)  # already written while we waited out the pause
    else:
        if spec is not None:
            spec.cancel()
        sentences = _reply_sentences(persona, Reply, user_text, history, out)
    spoken, cut_by = _speak_interruptible(persona, sentences, mic)
    _commit_history(history, user_text, out)
    _offered_fist_bump = "fist bump" in out.get("text", "").lower()
    dash_events.log_event("reply", f"llm response: {spoken}")
    journal.log("said", spoken)
    state.update_session({"last_activity_ts": time.time()})
    if cut_by:
        journal.log("interrupted", f"they cut me off: {cut_by!r}")
        cmd = commands.parse(cut_by)  # "be quiet" / "stop session" / "go to sleep" said over me
        if cmd:
            return _do_command(persona, cmd, mic), False
    return True, bool(cut_by)


def _learn_face(user_text: str) -> None:
    """"My name is Atul" with a face in view -> remember that face as Atul's.
    Also tops up the samples of someone recognized only weakly (new light,
    new angle), so recognition improves the more you talk to Rocky."""
    data = faces.read()
    embedding, seen = data.get("embedding"), data.get("faces") or []
    if not embedding or not seen:
        return
    name = faces.introduced_name(user_text)
    if name:
        n = faces.enroll(name, embedding)
        journal.log("learned", f"what {name} looks like ({n} face sample{'s' if n > 1 else ''})")
        if n == 1:
            memory.remember("person", name, "face", f"I can recognize {name}'s face -- they introduced themselves")
    elif seen[0].get("name") and seen[0].get("similarity", 1) < 0.55:
        faces.enroll(seen[0]["name"], embedding)


def _wake_greeting(persona, Reply, woke_from_sleep: bool = False) -> str:
    """A fresh, in-character greeting from the LLM every time -- no
    fallback content: if the LLM server's unreachable, say so plainly."""
    prompt = persona.system_prompt_template(cfg.ALLOWED_ACTIONS, cfg.STATIONARY_ACTIONS)
    greeting_prompt = journal.inject(
        "You were ASLEEP and someone just woke you up by saying your name. Greet them with one "
        "short, sleepy, in-character line." if woke_from_sleep else
        "Someone just said your wake word. Greet them with one short, in-character line."
    )
    result = cognition.ask(
        cfg.LLM_BASE_URL, cfg.LLM_MODEL, greeting_prompt,
        system=prompt, json_schema=Reply.model_json_schema(), timeout_s=15.0,
    )
    if result.status != cognition.AVAILABLE:
        print(f"wake greeting: cognition unavailable ({result.status}): {result.error}")
        dash_events.log_event("safety", f"wake greeting fallback ({result.status})")
        return persona.transform("Unable to think.")
    try:
        data = Reply.model_validate_json(result.text)
        return persona.transform(data.reply)
    except Exception as e:
        print(f"wake greeting parse failed ({e}); raw text: {result.text!r}")
        return persona.transform(result.text.strip())


def _run_session(persona, Reply, mic: mic_stream.ArecordStream, first_text: str | None = None) -> None:
    """Open until "stop session" or "go to sleep" -- silence, long pauses and
    unintelligible audio never end it. (After CONVERSATION_IDLE_S of nobody
    speaking, the mind starts thinking again while the session stays open --
    see state.conversation_active.)"""
    woke_from_sleep = bool(state.load_session().get("asleep"))
    try:
        state.update_session({"in_session": True, "listening": True, "asleep": False,
                              "last_activity_ts": time.time()})
        if woke_from_sleep:
            journal.log("woke", "someone woke me up")
            dash_events.log_event("wake", "woke up")
        motor_dispatch(["wave hands"])  # stationary wake acknowledgment; ends with the head centred
        history: list[dict] = []
        journal.log("woke", "someone said my name")
        if first_text:  # "Rocky, <instruction>" in one breath: answer it, no greeting
            keep_going, interrupted = _run_turn(persona, Reply, first_text, history, mic)
            if not keep_going:
                return
        else:
            greeting = _wake_greeting(persona, Reply, woke_from_sleep)
            dash_events.log_event("reply", f"wake greeting: {greeting}")
            journal.log("said", greeting)
            _, cut_by = _speak_interruptible(persona, [greeting], mic)
            interrupted = bool(cut_by)
            if cut_by and (cmd := commands.parse(cut_by)) and not _do_command(persona, cmd, mic):
                return

        while True:
            health.record_success(COMPONENT)
            if not interrupted:
                mic.flush()  # drop Rocky's own voice; after a barge-in, keep what the person is already saying
            spec_out: dict = {}
            _, text = _listen(mic, 60.0, speculate=lambda audio, heard_from: _Speculation(
                persona, Reply, audio, heard_from, history), spec_out=spec_out)
            if not text:
                interrupted = False  # nothing heard in a minute -> just keep listening
                continue
            keep_going, interrupted = _run_turn(persona, Reply, text, history, mic, spec=spec_out.get("spec"))
            if not keep_going:
                break
    finally:
        state.update_session({"in_session": False, "listening": False})


CONTINUE_WAIT_S = 1.2  # after "Rocky", how long to wait for "...go to sleep" before greeting


def _on_wake(persona, Reply, mic: mic_stream.ArecordStream, seed: bytes) -> None:
    """Heard "Rocky". Vosk's wake grammar closes the phrase at the pause after
    the name, so "Rocky, go to sleep" arrives as just "rocky" -- greeting right
    then talked over the rest. Wait CONTINUE_WAIT_S for more speech instead:
    if it comes, Whisper transcribes the whole thing (from "Rocky" on, via
    `seed`) and that's the first turn, no greeting. While asleep, wake only if
    Whisper actually heard "wake up" (the grammar invented it from "be quiet")."""
    _, text = _listen(mic, CONTINUE_WAIT_S, seed=seed)
    if not text and seed and _whisper is not None:
        # Nothing MORE was said -- but said in one breath ("Rocky wake up", "Rocky
        # what time is it"), the whole phrase is already in `seed`: Vosk only
        # reports the name once the phrase ends. Missing this refused "Rocky,
        # wake up" four times in a row (2026-10-02).
        text = _whisper.transcribe(seed)
    words = stt._words(text)
    if state.load_session().get("asleep"):
        if "wake up" in " ".join(words):
            _run_session(persona, Reply, mic)  # sleepy greeting
        else:
            dash_events.log_event("wake", f"heard {text or 'my name'!r} while asleep -- only \"{persona.name}, wake up\" wakes me")
        return
    instruction = [w for w in words if w not in (_wake_name(persona), "hey", "hi", "ok", "okay")]
    dash_events.log_event("wake", f"user: {text or _wake_name(persona)}")
    _run_session(persona, Reply, mic, first_text=text if instruction else None)


def _wake_name(persona) -> str:
    """The word that wakes it: the persona's name, last word if it has several.
    Must be a word the Vosk model knows -- pick a plain, common one."""
    return persona.name.lower().split()[-1]


def _load_whisper() -> None:
    global _whisper
    remote = stt.RemoteWhisper(cfg.LLM_BASE_URL, cfg.MAC_WHISPER_PORT, None) if cfg.MAC_WHISPER_PORT else None
    _whisper = remote  # usable at once; the Pi's own model takes ~26s to load as its fallback
    try:
        local = stt.Whisper(cfg.WHISPER_MODEL)
        print(f"stt: local whisper {cfg.WHISPER_MODEL} ready")
    except Exception as e:
        print(f"stt: local whisper unavailable: {e}")
        return
    if remote:
        remote.fallback = local
    else:
        _whisper = local


def main() -> None:
    global _wake_model
    persona = persona_mod.load()
    Reply = reply_schema.build_reply_model(cfg.STATIONARY_ACTIONS)
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)
    # A kill mid-session (watchdog, crash) skips _run_session's finally --
    # without this reset, in_session stays True and mind/alive stay frozen.
    state.update_session({"in_session": False, "listening": False})

    device = mic_stream.resolve_device(cfg.STT_DEVICE)
    mic = mic_stream.ArecordStream(device)
    mic.start_stream()

    wake_model = _wake_model = vosk.Model(_vosk_model_path())
    name = _wake_name(persona)
    wake_phrases = persona.wake_words + _wake_up_phrases(name)
    grammar = json.dumps(wake_phrases + ["[unk]"])
    wake_rec = vosk.KaldiRecognizer(wake_model, VOSK_RATE, grammar)
    wake_rec.SetWords(True)  # word timings: where "Rocky" started in the stream

    threading.Thread(target=_load_whisper, daemon=True).start()

    print(f"{COMPONENT}: listening for {wake_phrases}")
    health.record_success(COMPONENT)

    chunk_frames = mic.chunk_bytes // 2
    # Audio the wake recognizer has heard, by stream position -- so when it hears
    # "Rocky" we can hand Whisper everything from that word on.
    recent: deque[tuple[float, bytes]] = deque(maxlen=int(8 * mic.rate / chunk_frames))
    fed_s = 0.0
    levels: deque[float] = deque(maxlen=30)  # per-second peak loudness, for openbot-mind's surprise detection
    second_peak, next_publish = 0.0, time.time() + 1.0
    while True:
        health.record_success(COMPONENT, min_interval_s=5.0)
        chunk = mic.read(chunk_frames)
        if not chunk:
            if not mic.is_running():
                # arecord died (device busy/unplugged, e.g. grabbed at boot) -- exit
                # so systemd restarts us, instead of spinning deaf forever.
                raise SystemExit("arecord exited -- mic unavailable")
            time.sleep(0.05)
            continue
        second_peak = max(second_peak, _rms(chunk))
        if time.time() >= next_publish:
            levels.append(round(second_peak))
            sensors.publish_hearing(list(levels))
            second_peak, next_publish = 0.0, time.time() + 1.0
        if speak_mod.playing():
            continue  # its own voice ("Rocky ready!") must not count as its name
        resampled = resample_pcm(chunk, MIC_RATE, VOSK_RATE)
        recent.append((fed_s, chunk))
        fed_s += chunk_frames / mic.rate
        if wake_rec.AcceptWaveform(resampled):
            res = json.loads(wake_rec.Result())
            starts = [w["start"] for w in res.get("result", []) if w.get("word") in (name, "hey", "hi", "wake")]
            if name in res.get("text", "").split() and starts:
                seed = b"".join(c for t, c in recent if t >= starts[0] - 0.3)
                _on_wake(persona, Reply, mic, seed)
                wake_rec = vosk.KaldiRecognizer(wake_model, VOSK_RATE, grammar)
                wake_rec.SetWords(True)
                recent.clear()
                fed_s = 0.0


if __name__ == "__main__":
    main()
