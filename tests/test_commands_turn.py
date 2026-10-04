"""Voice commands through the real conversation turn (services.wake_listen._run_turn):
"be quiet" / "stop session" end the conversation (Rocky stays awake); "go to
sleep" ends it and sleeps, head down. Then the same commands texted on WhatsApp
(/be-quiet, /go-to-sleep, /wake-up -> wake_listen._do_remote). Speech, motors
and the LLM are stubbed -- the robot stays silent and still. Isolated state.

    cd ~/openbot && python3 -m tests.test_commands_turn
"""
from tests._audio import isolate_state

isolate_state()

import config as cfg  # noqa: E402
import services.wake_listen as wl  # noqa: E402
import time  # noqa: E402

from common import commands, persona as persona_mod, reply_schema, state  # noqa: E402

said, moved = [], []
wl._speak_interruptible = lambda persona, sentences, mic: (said.append(" ".join(sentences)) or (" ".join(sentences), None))
wl.motor_dispatch = lambda actions, wait=False: moved.append(actions) or True
wl.speak_client = lambda text, persona=None: said.append(text)
persona, Reply = persona_mod.load(), reply_schema.build_reply_model(cfg.TONE_ACTIONS)


def turn(text: str) -> bool:
    keep, _ = wl._run_turn(persona, Reply, text, [], mic=None)
    s = state.load_session()
    print(f"{text!r:22} -> continue={keep!s:5} asleep={s.get('asleep')!s:5} said={said[-1]!r}")
    return keep


assert turn("Be quiet.") is False and not state.load_session()["asleep"]   # conversation over, still awake
assert turn(f"{persona.name}, quiet") is False and not state.load_session()["asleep"]
assert turn("Stop session") is False and not state.load_session()["asleep"]
assert turn("Go to sleep") is False and state.load_session()["asleep"] and ["look down"] in moved


def texted(cmd: str, in_session: bool, age: float = 0.0) -> bool:
    state.update_session({"remote_command": cmd, "remote_command_ts": time.time() - age})
    keep = wl._do_remote(persona, None, in_session)
    s = state.load_session()
    assert not s["remote_command"]  # taken once
    print(f"texted {cmd!r:14} in_session={in_session!s:5} -> continue={keep!s:5} asleep={s.get('asleep')!s:5}")
    return keep


state.update_session({"asleep": False})
assert texted(commands.STOP_SESSION, in_session=False)  # no conversation: nothing to end
assert texted(commands.STOP_SESSION, in_session=True) is False and not state.load_session()["asleep"]
assert texted(commands.SLEEP, in_session=False, age=300) and not state.load_session()["asleep"]  # stale: dropped
assert texted(commands.SLEEP, in_session=False) is False and state.load_session()["asleep"]
assert texted(commands.WAKE, in_session=False) and not state.load_session()["asleep"] and ["look ahead"] in moved
print("test_commands_turn: ok")
