"""Voice commands through the real conversation turn (services.wake_listen._run_turn):
"be quiet" / "stop session" end the conversation (Rocky stays awake); "go to
sleep" ends it and sleeps, head down. Speech, motors and the LLM are stubbed --
the robot stays silent and still. Isolated state.

    cd ~/openbot && python3 -m tests.test_commands_turn
"""
from tests._audio import isolate_state

isolate_state()

import config as cfg  # noqa: E402
import services.wake_listen as wl  # noqa: E402
from common import persona as persona_mod, reply_schema, state  # noqa: E402

said, moved = [], []
wl._speak_interruptible = lambda persona, sentences, mic: (said.append(" ".join(sentences)) or (" ".join(sentences), None))
wl.motor_dispatch = lambda actions, wait=False: moved.append(actions) or True
wl.speak_client = lambda text, persona=None: said.append(text)
persona, Reply = persona_mod.load(), reply_schema.build_reply_model(cfg.STATIONARY_ACTIONS)


def turn(text: str) -> bool:
    keep, _ = wl._run_turn(persona, Reply, text, [], mic=None)
    s = state.load_session()
    print(f"{text!r:22} -> continue={keep!s:5} asleep={s.get('asleep')!s:5} said={said[-1]!r}")
    return keep


assert turn("Be quiet.") is False and not state.load_session()["asleep"]   # conversation over, still awake
assert turn(f"{persona.name}, quiet") is False and not state.load_session()["asleep"]
assert turn("Stop session") is False and not state.load_session()["asleep"]
assert turn("Go to sleep") is False and state.load_session()["asleep"] and ["look down"] in moved
print("test_commands_turn: ok")
