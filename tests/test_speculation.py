"""Replying before we're sure you've finished (services.wake_listen._Speculation):
  - end of speech -> first sentence ready, with vs without speculation
  - a pause MID-sentence doesn't produce a reply to half the sentence: the guess
    is cancelled and the full sentence is what gets answered
Fake mic + the Mac's Whisper + the live LLM; nothing spoken or moved (speech and
motors stubbed); isolated state.

    cd ~/openbot && python3 -m tests.test_speculation
"""
from tests._audio import FakeMic, isolate_state, noise, say, seconds

isolate_state()

import time  # noqa: E402

import vosk  # noqa: E402

import config as cfg  # noqa: E402
import services.wake_listen as wl  # noqa: E402
from common import persona as persona_mod, reply_schema, stt  # noqa: E402

vosk.SetLogLevel(-1)
wl._wake_model = vosk.Model(wl._vosk_model_path())
wl._whisper = stt.RemoteWhisper(cfg.LLM_BASE_URL, cfg.MAC_WHISPER_PORT, None)
wl.motor_dispatch = lambda actions, wait=False: True
persona, Reply = persona_mod.load(), reply_schema.build_reply_model(cfg.STATIONARY_ACTIONS)
first_ready: list[float] = []


def fake_speak(persona, sentences, mic):
    got = []
    for s in sentences:
        if not got:
            first_ready.append(time.time())
        got.append(s)
    return " ".join(got), None


wl._speak_interruptible = fake_speak


def turn(audio: bytes, speech_end: float, use_spec: bool) -> tuple[str, float]:
    history, spec_out = [], {}
    first_ready.clear()
    mic = FakeMic(audio)
    t0 = mic.t0
    speculate = (lambda a, hf: wl._Speculation(persona, Reply, a, hf, history)) if use_spec else None
    _, text = wl._listen(mic, 20, speculate=speculate, spec_out=spec_out)
    wl._run_turn(persona, Reply, text, history, mic, spec=spec_out.get("spec"))
    return text, first_ready[0] - (t0 + speech_end)


q = say("What can you see in front of you right now?")
audio = noise(1) + q + noise(4)
for use_spec in (False, True, False, True):
    text, lag = turn(audio, 1 + seconds(q), use_spec)
    print(f"{'speculating' if use_spec else 'waiting    '}: first sentence ready {lag:.2f}s after you stopped ({text!r})")

# a short pause mid-sentence (shorter than PAUSE_S -- a longer one ends the turn,
# speculation or not) must not get half the question answered
a, b = say("Tell me about the color"), say("of the wall behind me.")
audio = noise(1) + a + noise(0.5) + b + noise(4)
text, lag = turn(audio, 1 + seconds(a) + 0.5 + seconds(b), True)
print(f"mid-sentence pause -> answered {text!r}")
assert "wall" in text.lower() and "color" in text.lower(), text
print("test_speculation: ok")
