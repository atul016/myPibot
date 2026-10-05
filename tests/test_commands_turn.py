"""Ending a conversation and sleeping, through the real conversation turn
(services.wake_listen._run_turn) and the real LLM deciding (skills): "be quiet" /
"stop session" end the conversation (Rocky stays awake); "go to sleep" ends it
and sleeps, head down. Then the same, used by text on WhatsApp (skills
end_conversation, sleep, wake_up, volume -> wake_listen._do_remote). Speech and
motors are stubbed -- the robot stays silent and still. Isolated state.

    cd ~/openbot && set -a && . ./openbot.env && set +a && python3 -m tests.test_commands_turn
"""
from tests._audio import isolate_state

isolate_state()

import config as cfg  # noqa: E402
import common.motor_client as motor  # noqa: E402
import services.wake_listen as wl  # noqa: E402
import time  # noqa: E402

from common import persona as persona_mod, reply_schema, state  # noqa: E402

said, moved = [], []
wl._speak_interruptible = lambda persona, sentences, mic: (said.append(" ".join(sentences)) or (" ".join(sentences), None))
# the skills' muscles reach the body through common.motor_client: stubbed at the source
motor.dispatch = wl.motor_dispatch = lambda actions, wait=False, timeout_s=15.0: moved.append(actions) or True
motor.cancel_navigation = lambda timeout_s=2.0: False
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


def texted(cmd: dict, in_session: bool, age: float = 0.0) -> bool:
    state.update_session({"remote_command": cmd, "remote_command_ts": time.time() - age})
    keep = wl._do_remote(persona, None, in_session)
    s = state.load_session()
    assert not s["remote_command"]  # taken once
    print(f"texted {cmd!r:14} in_session={in_session!s:5} -> continue={keep!s:5} asleep={s.get('asleep')!s:5}")
    return keep


state.update_session({"asleep": False})
END, SLEEP = {"skill": "end_conversation"}, {"skill": "sleep"}
assert texted(END, in_session=False)  # no conversation: nothing to end
assert texted(END, in_session=True) is False and not state.load_session()["asleep"]
assert texted(SLEEP, in_session=False, age=300) and not state.load_session()["asleep"]  # stale: dropped
assert texted(SLEEP, in_session=False) is False and state.load_session()["asleep"]
assert texted({"skill": "wake_up"}, in_session=False) and not state.load_session()["asleep"] and ["look ahead"] in moved
import common.system as system  # noqa: E402 -- volume by text (skills/volume): turned down at home, said out loud
volume = [60]
system.get_volume, system.set_volume = (lambda: volume[0]), (lambda v: volume.__setitem__(0, v) or v)
assert texted({"skill": "volume", "change": "softer"}, in_session=False) and volume[0] == 45
print("test_commands_turn: ok")
