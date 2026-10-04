"""Waking: "Rocky" alone vs "Rocky, <instruction>" in one breath, awake and asleep
(services.wake_listen._on_wake). Fake mic with synthetic speech; the Mac's
Whisper; _run_session stubbed (records what it was asked to do); isolated state.

    cd ~/openbot && python3 -m tests.test_wake
"""
from tests._audio import FakeMic, isolate_state, noise, say

isolate_state()

import vosk  # noqa: E402

import config as cfg  # noqa: E402
import services.wake_listen as wl  # noqa: E402
from common import persona as persona_mod, state, stt  # noqa: E402

vosk.SetLogLevel(-1)
wl._wake_model = vosk.Model(wl._vosk_model_path())
wl._whisper = stt.RemoteWhisper(cfg.LLM_BASE_URL, cfg.MAC_WHISPER_PORT, None)
sessions = []
wl._run_session = lambda persona, Reply, mic, first_text=None: sessions.append(first_text or "<greeting>")
wl._say_line = lambda persona, situation, fallback, mic: sessions.append("<wake-ack>")
wl.motor_dispatch = lambda actions, wait=False: True
persona = persona_mod.load()
rocky = say(persona.name)


def wake(continuation: str | None, asleep: bool = False, seed: bytes = rocky) -> str | None:
    state.update_session({"asleep": asleep})
    sessions.clear()
    audio = noise(0.4) + (say(continuation) if continuation else b"") + noise(2.5)
    wl._on_wake(persona, None, FakeMic(audio), seed)
    got = sessions[0] if sessions else None
    print(f"{'asleep' if asleep else 'awake':6} {persona.name!r} + {continuation!r:22} -> {got!r}")
    return got


assert wake(None) == "<greeting>"                                  # just the name: greet
assert "sleep" in wake("go to sleep").lower()                       # one breath: the instruction, no greeting
assert "time" in wake("what time is it").lower()
assert wake("wake up", asleep=True) == "<wake-ack>" and not state.load_session()["asleep"]  # awake, no session
assert "time" in wake("wake up what time is it", asleep=True).lower()  # wake up + instruction: that's the first turn
assert wake(None, asleep=True) is None                              # asleep: name alone does nothing
assert wake("is a funny name", asleep=True) is None                 # asleep: name in passing does nothing
# One breath, no pause: Vosk only reports "rocky" after the WHOLE phrase, so the
# instruction is already in the seed and nothing more follows (the real failure:
# "Rocky, wake up" refused four times).
assert wake(None, asleep=True, seed=say(f"{persona.name} wake up")) == "<wake-ack>"
assert "time" in wake(None, seed=say(f"{persona.name} what time is it")).lower()
print("test_wake: ok")
