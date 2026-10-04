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

import datetime
import json
import os
import queue
import re
import threading
import time
from collections import deque
from typing import Iterable, Iterator

import common.system  # noqa: F401 -- side effect, harmless even though this process owns no hardware

import config as cfg  # noqa: E402
from common import cognition, events as dash_events, health, mic_stream, persona as persona_mod  # noqa: E402
from common import commands, faces, journal, memory, react, reply_schema, sensors, speak_client as speak_mod, state, vision  # noqa: E402
from common import hearing, voices  # noqa: E402
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
# ...plus this once, when Whisper's transcript so far stops mid-sentence (no final
# . ? ! or a trailing "..."/"--"): "Also her name is not", "This is going to be a--"
# and "I can suggest how we..." were all answered mid-sentence (2026-10-02), and no
# finished sentence that day lacked the punctuation.
UNFINISHED_EXTRA_S = 0.8
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

_offered_move: str | None = None  # a wheel move the last reply offered ("want a fist bump?") -- "yes" does it
_whisper: "stt.Whisper | stt.RemoteWhisper | None" = None  # set in the background at startup; None until then
_wake_model = None  # set in main(); also used for the barge-in recognizer
_voices: "voices.VoiceEngine | None" = None
_voice_speaker: str | None = None  # this turn's speaker, known only by voice (no face said who)


def _vosk_model_path() -> str:
    # Matches the filenames the original README documents as already
    # downloaded on the robot -- see its "STT model experiments" section.
    name = "vosk-model-small-en-in-0.4" if cfg.STT_LANGUAGE.startswith("en-in") \
        else "vosk-model-small-en-us-0.15"
    return os.path.expanduser(f"~/.vosk_models/{name}")



def _started(heard: str) -> bool:
    """Real speech, not noise: Vosk heard 2+ words (a lone "the"/"a" is what
    it hallucinates from a door click), or a phrase that survives the
    phantom-phrase filter."""
    return len(heard.split()) >= 2 or bool(stt.reject_hallucination(heard))


def _listen(mic: mic_stream.ArecordStream, onset_timeout_s: float, seed: bytes = b"",
            speculate=None, spec_out: dict | None = None, stop=None) -> tuple[bytes, str]:
    """Waits (patiently, up to onset_timeout_s) for someone to start talking,
    then records until they stop -- both decided by a STREAMING Vosk
    recognizer (are there words?), not by loudness: a fixed loudness
    threshold broke whenever the mic gain or the room changed. Returns
    (audio, transcript) -- the transcript is Whisper's, Vosk's as fallback.
    `seed`: audio already heard (the wake word) to put in front, if speech follows.
    `speculate(audio, heard_from)` -> _Speculation, started at each pause; the
    one still valid when you stop is returned in spec_out["spec"]. `stop()`:
    asked about once a second while nobody is talking yet -- True ends the wait."""
    rec = vosk.KaldiRecognizer(_wake_model, VOSK_RATE)
    preroll: deque[bytes] = deque(maxlen=PREROLL_CHUNKS)
    captured: list[bytes] = []
    results: list[dict] = []
    started_at = pause_until = None
    extended = False  # the unfinished-sentence extra wait, at most once per pause
    spec = None
    deadline = time.time() + onset_timeout_s
    next_stop_check = 0.0
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
                pause_until, extended = now + PAUSE_S, False
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
            if stop is not None and now >= next_stop_check:
                next_stop_check = now + 1.0
                if stop():
                    return b"", ""
            continue
        if pause_until and now >= pause_until and not extended and spec is not None and spec.unfinished():
            pause_until, extended = now + UNFINISHED_EXTRA_S, True  # probably looking for the next word
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
            print(f"stt: used the transcript made during the pause: vosk "
                  f"{' '.join(r.get('text', '') for r in results)!r} -> {text!r} ({len(audio) / 2 / mic.rate:.1f}s audio)")
            spec_out["spec"] = spec
            return audio, text
        if spec.whisper_answered:  # Whisper already judged it noise (or all Rocky's own words) --
            spec.cancel()          # a second run on the same audio re-rolled "You can see the next one."
            return audio, ""
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
        text = better if better is not None else text  # "": Whisper's verdict was noise -- not Vosk's guess
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


def _planned_move(text: str) -> str | None:
    """The body move this transcript triggers -- keywords decide, never the LLM."""
    move = detect_movement_keyword(text)
    if not move and _offered_move and is_affirmative(text):
        move = _offered_move
    return move if move in cfg.ALLOWED_ACTIONS else None


def _reply_sentences(persona, Reply, user_text: str, history: list[dict], out: dict,
                     on_tone=lambda tone: motor_dispatch([tone]), moved: str | None = None) -> Iterator[str]:
    """Streams the reply, yielding each finished sentence (persona-transformed)
    the moment it's complete -- speech starts while the rest is still being
    generated. on_tone(gesture) is called as soon as the gesture is named (it
    streams first). Never ends in silence: an unreachable LLM yields one plain
    "can't reach my brain" line. Side-effect free otherwise (a speculative
    reply may be thrown away): fills out["text"] / out["raw"]; `history` is
    only read -- the caller commits the exchange (_commit_history). `moved`:
    the body move this turn triggered (it's told, so it never claims one that
    didn't happen -- "moving forward" got said with the wheels still)."""
    prompt = persona.system_prompt_template(cfg.ALLOWED_ACTIONS, cfg.STATIONARY_ACTIONS,
                                            cfg.describe_actions(cfg.STATIONARY_ACTIONS))
    called_only = not [w for w in stt._words(user_text) if w not in (_wake_name(persona), "hey", "hi", "ok", "okay")]
    # A fresh camera frame with every turn, not the mind's last look (up to 90s
    # old): "what am I holding?" got guesses ("a pen? a stylus?") without it.
    # Measured: +0.1-0.2s to the first word; frame grab ~0.04s. Not for a bare
    # "Rocky?": with a photo and nothing asked, it described the room every time.
    photo = None if called_only else vision.capture()
    if called_only:
        turn = ("They only said your name -- they want your attention. Answer in a few words, "
                "like \"Yes? What's up?\" -- don't describe anything.\n")
    else:
        turn = ("Talk naturally, like a friend in a conversation. Don't describe your surroundings "
                "unless they ask what you see, what they're showing or holding, or who they are.\n")
        if photo:
            turn += "The attached photo is what your camera sees right now -- use it for those questions.\n"
    who, by = faces.speaker(faces.read()), "face"
    if not who and _voice_speaker:
        who, by = _voice_speaker, "voice"
    if who:
        turn += f"The person talking to you is {who} (you recognize their {by}). Use their name when it fits.\n"
    if cfg.CAN_DRIVE:
        turn += (f"Because of what they just said, your body is doing \"{moved}\" right now -- you may say so.\n"
                 if moved else "Nothing they just said made you move -- don't say you're moving or about to.\n")
    near = sensors.read()
    if cfg.PLAYFUL_ACTIONS and near.get("distance") is not None and near["distance"] <= cfg.SAFE_DISTANCE:
        turn += (f"Right now something is almost touching your front ({near['distance']:.0f}cm) -- probably their "
                 f"hand or fist. If it fits the moment, your tone_action may also be: "
                 f"{cfg.describe_actions(cfg.PLAYFUL_ACTIONS)}.\n")
    conversation_prompt = journal.inject(
        turn +
        "You can't switch your own modes by agreeing to -- only these exact spoken commands do: "
        f"\"be quiet\" (ends this conversation), \"go to sleep\" (everything off until \"{persona.name}, wake up\"). "
        "If they seem to want one of those, tell them the words to say instead of promising it. " +
        ("You drive only on these exact spoken commands: \"go to the <thing>\", \"come here\", "
         "\"follow me\", \"explore\", \"stop\" (and simple forward/back/turn). Never claim you're driving otherwise -- "
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
            if rs.tone_action in cfg.TONE_ACTIONS:
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
        self.whisper_answered = False  # True: an empty text is Whisper's verdict, not "couldn't ask"
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
            heard = _whisper.transcribe(audio) if _whisper is not None else None
            text = heard or ""
            said = speak_mod.said_between(heard_from, time.time())
            self.text = stt.remove_echo(text, said) if said and text else text
            self.whisper_answered = heard is not None
        finally:
            self._transcribed.set()
        reply = None
        try:
            if not self.text or self._cancelled.is_set() or commands.parse(self.text):
                return  # nothing to answer, or a command -- the turn handles those itself
            reply = _reply_sentences(persona, Reply, self.text, history, self.out, on_tone=self._on_tone,
                                     moved=_planned_move(self.text))
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

    def unfinished(self) -> bool:
        """Whisper's text, if already in, stops mid-sentence."""
        t = self.text.rstrip() if self._transcribed.is_set() else ""
        return bool(t) and (t.endswith(("...", "-")) or t[-1] not in ".?!")

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


def _say_line(persona, situation: str, fallback: str, mic: mic_stream.ArecordStream) -> None:
    """Says what the LLM wants to say about `situation` (in its current mood),
    with the gesture it picks; `fallback` only if the LLM is unreachable."""
    spoken, tone = react.line(persona, situation, fallback)
    if tone:
        motor_dispatch([tone])
    journal.log("said", spoken)
    dash_events.log_event("reply", f"llm response: {spoken}")
    _speak_interruptible(persona, [spoken], mic)


def _do_navigation(persona, nav: tuple[str, str | None], mic: mic_stream.ArecordStream) -> bool:
    """Starts or stops a drive (it runs in openbot-alive -- movement/navigate.py,
    proven in sim/ first -- and Rocky says how it went). The conversation goes on."""
    from common.motor_client import cancel_navigation, navigate
    kind, target = nav
    if not cfg.CAN_DRIVE:
        _say_line(persona, "They asked you to drive somewhere, but you have no wheels -- you can't move at all.",
                  "I can't move -- I don't have wheels.", mic)
        return True
    if kind == "stop":
        _say_line(persona, "They said stop -- you've just stopped driving." if cancel_navigation()
                  else "They said stop, but you weren't moving anyway.", "Okay.", mic)
        return True
    ok, err = navigate({"follow": True} if kind == "follow" else {"approach": target} if kind == "approach"
                       else {"explore": 60})
    if not ok:
        _say_line(persona, "They asked you to drive somewhere, but you're already on your way somewhere -- "
                  "they'd have to say stop first." if "already" in err
                  else "They asked you to drive somewhere, but your wheels aren't answering right now.",
                  "I can't drive right now.", mic)
        return True
    journal.log("did", "started following them" if kind == "follow"
                else f"started driving {'to the ' + target if target else 'around to explore'}")
    _say_line(persona, "They said follow me, and you've just started following them." if kind == "follow"
              else "They said come here, and you've just set off toward them." if target == "person"
              else f"They asked you to go to the {target}, and you've just set off toward it." if target
              else "They told you to go explore, and you've just set off.", "Okay!", mic)
    return True


def _wake_up(persona, mic: mic_stream.ArecordStream) -> None:
    """"Rocky, wake up" with nothing after it: awake in the background -- camera,
    mind and reflexes resume on the cleared flag -- and one sleepy word, no session."""
    state.update_session({"asleep": False})
    motor_dispatch(["look ahead"], wait=True)  # head back up
    journal.log("woke", "someone woke me up")
    dash_events.log_event("wake", "woke up")
    _say_line(persona, "You were asleep and they just said \"wake up\". You're awake now, back to your usual self "
              "in the background -- one short, sleepy word. (Saying your name starts a conversation.)",
              "Mm. I'm up.", mic)


REMOTE_FRESH_S = 60.0  # a texted command older than this (wake-listen was down) is dropped, not done late


def _do_remote(persona, mic: mic_stream.ArecordStream, in_session: bool) -> bool:
    """A mode skill used by text on WhatsApp (skills sleep, wake_up,
    end_conversation, volume -- session["remote_command"]), carried out as if
    it had just been said. Returns whether the conversation (if any) goes on."""
    s = state.load_session()
    cmd = s.get("remote_command")
    if not cmd:
        return True
    state.update_session({"remote_command": None})
    if time.time() - s.get("remote_command_ts", 0) > REMOTE_FRESH_S:
        return True
    dash_events.log_event("wake", f"texted command: {cmd}")
    if cmd == commands.WAKE and s.get("asleep"):
        _wake_up(persona, mic)
    elif (cmd == commands.SLEEP and not s.get("asleep")) or (cmd == commands.STOP_SESSION and in_session) \
            or cmd in (commands.LOUDER, commands.SOFTER):
        return _do_command(persona, cmd, mic)
    return True


def _do_command(persona, cmd: str, mic: mic_stream.ArecordStream) -> bool:
    """Carries out a mode command. Returns whether the session continues."""
    if cmd in (commands.STOP_SESSION, commands.SLEEP):
        from common.motor_client import cancel_navigation
        cancel_navigation()  # never keep driving after "be quiet" / "go to sleep"
    if cmd == commands.STOP_SESSION:  # "be quiet" / "stop session"
        journal.log("did", "ended the conversation (asked to be quiet) -- still around in the background")
        _say_line(persona, f"They asked you to be quiet: this conversation is over, but you'll still be around, and "
                  f"saying \"{persona.name}\" starts a new one. Acknowledge that briefly.",
                  "Okay. Say my name if you need me.", mic)
        return False
    if cmd == commands.SLEEP:
        _say_line(persona, f"They told you to go to sleep -- everything switches off until they say "
                  f"\"{persona.name}, wake up\". Say a short, sleepy goodnight.", persona.sleep_ack, mic)
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
            _say_line(persona, "They asked you to speak louder, but you're already at full volume.",
                      "I'm already as loud as I go!", mic)
            return True
        new = set_volume(target)
        state.update_session({"volume": new})  # openbot-speak re-applies it after a reboot
        journal.log("did", f"volume {current}% -> {new}%")
        _say_line(persona, f"They asked you to speak {'louder' if cmd == commands.LOUDER else 'softer'}, and you "
                  f"just turned your volume {'up' if cmd == commands.LOUDER else 'down'} to {new}%. Acknowledge it.",
                  "Louder now!" if cmd == commands.LOUDER else "Softer now.", mic)
        return True
    return True


def _learn_voice(who: str, pcm16k: bytes) -> None:
    """The face says who's talking: what they said adds to their voice samples."""
    emb = _voices.embed(pcm16k)
    if emb is not None and voices.enroll(who, emb) == 1:
        memory.remember("person", who, "voice", f"I can recognize {who}'s voice -- I heard them while I saw them")
        journal.log("learned", f"what {who} sounds like")


def _whose_voice(pcm16k: bytes) -> str | None:
    emb = _voices.embed(pcm16k)
    if emb is None:
        return None
    name, score = voices.identify(emb)
    print(f"voices: best match {name or '-'} ({score:.2f})")
    return name


def _run_turn(persona, Reply, user_text: str, history: list[dict],
              mic: mic_stream.ArecordStream, spec: "_Speculation | None" = None,
              audio: bytes = b"") -> tuple[bool, bool]:
    """One exchange. Returns (continue_session, interrupted) -- continue is
    False after "stop session" or "go to sleep"; interrupted is True if the
    person talked over the reply. `audio`: what they said, for voices."""
    global _offered_move, _voice_speaker
    state.update_session({"last_heard": user_text, "last_heard_ts": time.time(), "last_activity_ts": time.time()})
    dash_events.log_event("wake", f"user: {user_text}")
    who = faces.speaker(faces.read())
    _voice_speaker = None
    if audio and _voices is not None:
        pcm16k = resample_pcm(audio, MIC_RATE, VOSK_RATE)
        if who:  # learned in the background: no reply waits for it
            threading.Thread(target=_learn_voice, args=(who, pcm16k), daemon=True).start()
        else:
            who = _voice_speaker = _whose_voice(pcm16k)
            if who and spec is not None:  # the reply written ahead didn't know who was talking
                spec.cancel()
                spec = None
    journal.log("heard", f"{who}: {user_text}" if who else user_text)

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

    move_action = _planned_move(user_text)
    if move_action:
        # Queued, not awaited: the words come WITH the move ("watch me dance" while dancing), not
        # after it. dispatch() still reports whether the body accepted it (down, or mid-drive).
        if motor_dispatch([move_action]):
            journal.log("did", f"moved: {move_action}")
            dash_events.log_event("wake", f'"{user_text}" -> move {move_action}')
        else:  # alive down or mid-drive: a reply written ahead must not claim the move
            move_action = None
            if spec is not None:
                spec.cancel()

    out: dict = {}
    if spec is not None and spec.ready_for(user_text):
        sentences = spec.commit(out)  # already written while we waited out the pause
    else:
        if spec is not None:
            spec.cancel()
        sentences = _reply_sentences(persona, Reply, user_text, history, out, moved=move_action)
    spoken, cut_by = _speak_interruptible(persona, sentences, mic)
    _commit_history(history, user_text, out)
    said = out.get("text", "").lower()
    _offered_move = next((a for a in cfg.PLAYFUL_ACTIONS if a in said), None)
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
    """"My name is Atul" with ONE face in view -> remember that face as Atul's.
    Also tops up the samples of someone recognized only weakly (new light,
    new angle), so recognition improves the more you talk to Rocky. "Not Ana,
    Anna." renames a name only just learned (a misheard introduction).
    One face only, and a real match only: with two in view, or a name merely
    carried by tracking, whose face it is would be a guess -- and a wrong guess
    becomes training data."""
    fixed = faces.corrected_name(user_text)
    if fixed:
        old, new = fixed
        (memory.MIND_DIR / "people" / f"{memory.slug(old)}.md").unlink(missing_ok=True)
        memory.remember("person", new, "name", f"I first misheard {new}'s name as {old}")
        journal.log("learned", f"{old} is really called {new} -- I misheard the name")
        return
    data = faces.read()
    embedding, seen = data.get("embedding"), data.get("faces") or []
    if not embedding or not seen:
        return
    name = faces.introduced_name(user_text)
    if name and len(seen) > 1:
        journal.log("noticed", f"{name} introduced themselves, but I see {len(seen)} faces -- not sure which is them")
    elif name:
        n = faces.enroll(name, embedding)
        journal.log("learned", f"what {name} looks like ({n} face sample{'s' if n > 1 else ''})")
        if n == 1:
            memory.remember("person", name, "face", f"I can recognize {name}'s face -- they introduced themselves")
    elif len(seen) == 1 and seen[0].get("name") and faces.MATCH_COSINE <= seen[0].get("similarity", 0) < 0.55:
        faces.enroll(seen[0]["name"], embedding)


_greetings: list[tuple[str, str | None]] = []  # the next one, written ahead -- see _prepare_greeting


def _prepare_greeting(persona) -> None:
    """Writes the NEXT greeting now, in the background, so a bare "Rocky" is
    answered at once instead of after an LLM round (~1-2s of silence that read
    as not having heard). Still fresh each time: one line, used once."""
    def _run() -> None:
        _greetings[:] = [_wake_greeting(persona, prepared=False)]
    threading.Thread(target=_run, daemon=True).start()


def _wake_greeting(persona, prepared: bool = True) -> tuple[str, str | None]:
    """(greeting, gesture): fresh from the LLM every time -- no fallback
    content: if the LLM server's unreachable, say so plainly.
    `prepared`: take the one written ahead if there is one."""
    if prepared and _greetings:
        return _greetings.pop()
    return react.line(persona, "Someone just said your name to get your attention. Greet them.",
                      "Unable to think.", timeout_s=15.0)


def _seed_history(opener: str | None) -> list[dict]:
    """Where the last conversation left off (journal), and the line the mind
    just said if it opened this one -- so a new session continues a
    relationship instead of starting from nothing."""
    history: list[dict] = []
    last = journal.last_conversation(datetime.datetime.now() - datetime.timedelta(minutes=10))
    if last:
        when, lines = last
        history += [{"role": "user", "content": f"(Your last conversation, {when} -- for continuity, don't recite it:)\n"
                                                + "\n".join(lines)},
                    {"role": "assistant", "content": json.dumps({"tone_action": "none", "reply": "(remembered)"})}]
    if opener:
        history += [{"role": "user", "content": "(nobody said anything yet)"},
                    {"role": "assistant", "content": json.dumps({"tone_action": "none", "reply": opener})}]
    return history


def _run_session(persona, Reply, mic: mic_stream.ArecordStream, first_text: str | None = None,
                 opener: str | None = None) -> None:
    """Open until "stop session" or "go to sleep" -- silence, long pauses and
    unintelligible audio never end it. (After CONVERSATION_IDLE_S of nobody
    speaking, the mind starts thinking again while the session stays open --
    see state.conversation_active.) `opener`: the mind spoke first and they answered."""
    woke_from_sleep = bool(state.load_session().get("asleep"))
    try:
        state.update_session({"in_session": True, "listening": True, "asleep": False,
                              "last_activity_ts": time.time()})
        if woke_from_sleep:  # "Rocky, wake up, <instruction>" in one breath
            journal.log("woke", "someone woke me up")
            dash_events.log_event("wake", "woke up")
        history = _seed_history(opener)
        journal.log("woke", "they answered me" if opener else "someone said my name")
        if first_text:  # "Rocky, <instruction>" in one breath: answer it, no greeting
            keep_going, interrupted = _run_turn(persona, Reply, first_text, history, mic)
            if not keep_going:
                return
        else:
            greeting, tone = _wake_greeting(persona)
            if tone:
                motor_dispatch([tone])  # the greeting's own gesture -- the LLM's pick, not a fixed wave
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
            audio, text = _listen(mic, 60.0, speculate=lambda audio, heard_from: _Speculation(
                persona, Reply, audio, heard_from, history), spec_out=spec_out,
                stop=lambda: bool(state.load_session().get("remote_command")))
            if not _do_remote(persona, mic, in_session=True):  # texted /be-quiet or /go-to-sleep
                break
            if not text:
                interrupted = False  # nothing heard in a minute -> just keep listening
                continue
            keep_going, interrupted = _run_turn(persona, Reply, text, history, mic, spec=spec_out.get("spec"),
                                                audio=audio)
            if not keep_going:
                break
    finally:
        state.update_session({"in_session": False, "listening": False})
        _prepare_greeting(persona)  # for the next "Rocky"


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
        text = _whisper.transcribe(seed) or ""
        print(f"stt: wake seed ({len(seed) / (2 * MIC_RATE):.1f}s) -> whisper {text!r}")
    words = stt._words(text)
    fillers = (_wake_name(persona), "hey", "hi", "ok", "okay")
    if state.load_session().get("asleep"):
        joined = " ".join(words)
        if "wake up" not in joined:
            dash_events.log_event("wake", f"heard {text or 'my name'!r} while asleep -- only \"{persona.name}, wake up\" wakes me")
            return
        # "Rocky, wake up" alone: back to awake-in-the-background, like "be quiet"
        # leaves it -- no conversation, no greeting. With more after it ("wake up,
        # what time is it"), that's the first turn.
        rest = [w for w in joined.split("wake up", 1)[1].split() if w not in fillers]
        if rest:
            _run_session(persona, Reply, mic, first_text=" ".join(rest))
        else:
            _wake_up(persona, mic)
        return
    instruction = [w for w in words if w not in fillers]
    if not instruction:  # with one, _run_turn logs it
        dash_events.log_event("wake", f"user: {text or _wake_name(persona)}")
    _run_session(persona, Reply, mic, first_text=text if instruction else None)


def _wake_name(persona) -> str:
    """The word that wakes it: the persona's name, last word if it has several.
    Must be a word the Vosk model knows -- pick a plain, common one."""
    return persona.name.lower().split()[-1]


def _load_senses() -> None:
    """Voice prints, in the background -- optional: no model, no names. (Sound
    names are openbot-ears' job.)"""
    global _voices
    try:
        _voices = voices.VoiceEngine()
    except Exception as e:
        print(f"voices: off ({e})")


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
    Reply = reply_schema.build_reply_model(cfg.TONE_ACTIONS)
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)
    # A kill mid-session (watchdog, crash) skips _run_session's finally --
    # without this reset, in_session stays True and mind/alive stay frozen.
    state.update_session({"in_session": False, "listening": False})

    # openbot-ears owns the mic; this listens to its live audio (same format, whole chunks)
    mic = mic_stream.ArecordStream("openbot-ears", rate=MIC_RATE, chunk_frames=hearing.CHUNK_FRAMES,
                                   cmd=hearing.pcm_command())
    mic.start_stream()

    wake_model = _wake_model = vosk.Model(_vosk_model_path())
    name = _wake_name(persona)
    wake_phrases = persona.wake_words + _wake_up_phrases(name)
    grammar = json.dumps(wake_phrases + ["[unk]"])
    wake_rec = vosk.KaldiRecognizer(wake_model, VOSK_RATE, grammar)
    wake_rec.SetWords(True)  # word timings: where "Rocky" started in the stream

    threading.Thread(target=_load_whisper, daemon=True).start()
    threading.Thread(target=_load_senses, daemon=True).start()
    _prepare_greeting(persona)

    print(f"{COMPONENT}: listening for {wake_phrases}")
    health.record_success(COMPONENT)

    chunk_frames = mic.chunk_bytes // 2
    # Audio the wake recognizer has heard, by stream position -- so when it hears
    # "Rocky" we can hand Whisper everything from that word on.
    recent: deque[tuple[float, bytes]] = deque(maxlen=int(8 * mic.rate / chunk_frames))
    fed_s = 0.0
    next_invite_check = 0.0
    while True:
        health.record_success(COMPONENT, min_interval_s=5.0)
        chunk = mic.read(chunk_frames)
        if not chunk:
            if not mic.is_running():
                # openbot-ears' audio stopped (it restarted, or the mic is gone) -- exit
                # so systemd restarts us and we reconnect, instead of sitting deaf forever.
                raise SystemExit("the audio from openbot-ears stopped")
            time.sleep(0.05)
            continue
        if speak_mod.playing():
            continue  # its own voice ("Rocky ready!") must not count as its name
        if time.time() >= next_invite_check:
            next_invite_check = time.time() + 1.0
            invite = state.load_session()
            if invite.get("remote_command"):  # texted on WhatsApp: done as if just said
                _do_remote(persona, mic, in_session=False)
                continue
            remaining = invite.get("invite_until", 0) - time.time()
            if remaining > 0:
                # The mind just said something on its own: listen for an answer without the wake
                # word, for the rest of the window. An answer opens a normal conversation.
                state.update_session({"invite_until": 0, "in_session": True, "listening": True,
                                      "last_activity_ts": time.time()})
                mic.flush()
                _, text = _listen(mic, remaining)
                if text:
                    dash_events.log_event("wake", f"answered without the wake word: {text}")
                    _run_session(persona, Reply, mic, first_text=text, opener=invite.get("invite_text"))
                else:
                    state.update_session({"in_session": False, "listening": False})
                wake_rec = vosk.KaldiRecognizer(wake_model, VOSK_RATE, grammar)
                wake_rec.SetWords(True)
                recent.clear()
                fed_s = 0.0
                continue
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
