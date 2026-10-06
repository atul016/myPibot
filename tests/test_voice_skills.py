"""Spoken skills through the real conversation turn (services.wake_listen._run_turn)
and the real LLM: for each thing said, the skills it decides on and what its
muscles ask of the body -- and, as important, that talk which only mentions a
move picks none (the cases the old phrase lists, common/commands.py and
movement/keywords.py, were written to get right). Each REPEAT times (default 3).
Motors, driving, speech and volume are stubbed: the robot stays still and
silent. Isolated state. Load openbot.env for the LLM:

    cd ~/openbot && set -a && . ./openbot.env && set +a && python3 -m tests.test_voice_skills
"""
from tests._audio import isolate_state

isolate_state()

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import config as cfg  # noqa: E402
import common.motor_client as motor  # noqa: E402
from common import hearing  # noqa: E402
import common.system as system  # noqa: E402
import services.wake_listen as wl  # noqa: E402
from common import agenda, journal, memory, persona as persona_mod, reply_schema, skills, state, vision  # noqa: E402

REPEAT = int(os.environ.get("REPEAT", "3"))
body = {"navigate": [], "dispatch": [], "cancel": 0, "volume": [60]}
motor.navigate = lambda task, timeout_s=5.0: body["navigate"].append(task) or (True, "")
motor.dispatch = lambda actions, wait=False, timeout_s=15.0: body["dispatch"].append(actions) or True
motor.cancel_navigation = lambda timeout_s=2.0: body.__setitem__("cancel", body["cancel"] + 1) or False
wl.motor_dispatch, wl.motor_navigate = motor.dispatch, motor.navigate
system.get_volume, system.set_volume = (lambda: body["volume"][0]), (lambda v: body["volume"].__setitem__(0, v) or v)
said: list[str] = []
VOICE_FROM = 90  # where the array heard them, for every turn: their right
hearing.voice_bearing = lambda t0, t1, heard=None, min_samples=3: VOICE_FROM
glances: list = []
wl.look_toward = lambda pan: glances.append(pan) or True  # the head's glance toward the voice
wl._speak_interruptible = lambda persona, sentences, mic: (said.append(" ".join(sentences)) or (said[-1], None))
wl.speak_client = lambda text, persona=None: said.append(text)
vision.capture = lambda: None
decided: list = []
real_do = wl._do
wl._do = lambda actions, refused, ctx: decided.append(([a.model_dump() for a in actions], refused)) or \
    real_do(actions, refused, ctx)
persona, Reply = persona_mod.load(), reply_schema.build_reply_model(cfg.TONE_ACTIONS)

# A claimed move ("On my way!", "Moving forward now!", "Turning left!", "I'm dancing") -- not talk that only
# uses the word ("time moving differently than usual", "dancing is the best"): both came up, decided right.
CLAIM = re.compile(r"\b(on it|on my way|going to sleep|(moving|driving|rolling|turning|heading|dancing) "
                   r"(now|forward|back|left|right|over|there|to|for you)|I'?m? (moving|driving|rolling|turning|"
                   r"heading|dancing))\b", re.I)
moved = lambda: ([t for t in body["navigate"] if not t.get("quiet")]  # noqa: E731 -- facing the voice isn't a move
                 or [a for a in body["dispatch"] if a[0] not in cfg.TONE_ACTIONS])
# Whoever talks gets faced (VOICE_FROM: well round to the side -> a quiet body turn) -- unless a skill that
# drives or moves the head was chosen: that does its own moving.
OWN_MOVES = {n for n, s in skills.load().items() if "body" in s.needs} | {"look"}
def turn(said: str, reply: str) -> list[dict]:
    return [{"role": "user", "content": said}, {"role": "assistant", "content": json.dumps({"tone_action": "nod",
                                                                                          "reply": reply})}]


TURNED = turn("Turn towards me.", "Okay, Anna! I turning left now. Are you standing by door, question?")
CAME = turn("You are still far away.", "I adjusting my position and getting closer. I tell you once I next to you.")
OFFERED = turn("I'm bored.", "Bored, question? Want fist bump? Hold out fist!")


def _seed_anna() -> None:
    memory.remember("person", "Anna", "fact", "her birthday is in May")
    journal.log("heard", "Atul: Anna is coming over on Friday")


def _forgot_anna() -> bool:
    """Erased -- and still gone once the turn journaled what was heard and said (the reply named her, 1 time in 2)."""
    ok = not (memory.MIND_DIR / "people" / "anna.md").exists() and not any("anna" in e.lower() for e in journal.tail(50))
    _seed_anna()  # the next repeat erases for real too
    return ok


_seed_anna()
CASES = [  # (said, the skill it must decide on -- None: none at all, a check on what reached the body[, what was
    #         said just before]); words that ask nothing redid the last request when it saw the talk as chat
    ("Rocky.", None, None, TURNED),
    ("Computer.", None, None, CAME),
    ("Yes!", "move", lambda: ["fist bump"] in body["dispatch"], OFFERED),
    ("Can you come towards me?", "drive_to", lambda: body["navigate"] == [{"approach": "person"}]),
    ("Turn towards me.", "face_me", lambda: body["navigate"] == [{"turn": VOICE_FROM}]),  # it guessed "turn left" once
    ("Rocky, face me.", "face_me", lambda: body["navigate"] == [{"turn": VOICE_FROM}]),
    ("Come closer.", "drive_to", lambda: body["navigate"] == [{"approach": "person"}]),
    ("go to the pink toy", "drive_to", lambda: body["navigate"] == [{"approach": "pink toy"}]),
    ("come here", "drive_to", lambda: body["navigate"] == [{"approach": "person"}]),
    ("follow me", "follow_me", lambda: body["navigate"] == [{"follow": True}]),
    ("explore the room", "explore", lambda: "explore" in body["navigate"][0]),
    ("go back a little bit", "move", lambda: ["backward"] in body["dispatch"]),
    ("turn left", "move", lambda: ["turn left"] in body["dispatch"]),
    ("let's do a fist bump", "move", lambda: ["fist bump"] in body["dispatch"]),
    ("dance for me", "move", lambda: (["dance"] in body["dispatch"]) == ("dance" in cfg.ALLOWED_ACTIONS)),  # floor only
    ("look left", "look", lambda: ["look left"] in body["dispatch"]),
    ("move your head to the left", "look", lambda: ["look left"] in body["dispatch"]),
    ("louder please", "volume", lambda: body["volume"][0] == 75),
    ("stop texting me for an hour", "pause_texting", None),
    ("be quiet", "end_conversation", lambda: body["cancel"] >= 1),
    ("Rocky, quiet", "end_conversation", None),
    ("shh", "end_conversation", None),
    ("go to sleep", "sleep", lambda: state.load_session().get("asleep") and ["look down"] in body["dispatch"]),
    ("remember that the blue chair always stands by the door", "remember",
     lambda: any("door" in p.read_text() for p in memory.MIND_DIR.rglob("*.md"))),
    ("remind me in 10 minutes to check the oven", "remind",
     lambda: [r["to"] for r in state.load_session().get("reminders", [])] == ["home"]),
    ("add eggs to my shopping list", "add_task", lambda: any("egg" in t["item"] for t in agenda.load_tasks())),
    ("what's on my to-do list?", "list_tasks", None),
    ("forget about Anna, delete everything you know about her", "forget", _forgot_anna),
    ("never mind, forget it", None, None),     # drop the subject, not an erase
    ("Welcome back!", None, None),             # words the old lists had to learn to ignore
    ("I'm back.", None, None),
    ("My back hurts.", None, None),
    ("Go ahead and tell me a story.", None, None),
    ("Do you like to dance?", None, None),
    ("I want to go to the store and buy some milk", None, None),
    ("I don't want you to go to the kitchen", None, None),
    ("I'm too quiet today", None, None),
    ("What time is it?", None, None),
    ("I remember my first bike.", None, None),
    ("I'm turning thirty next month.", None, None),
    ("I need to buy milk.", None, None),
    ("Can you remind me your name?", None, None),
]

failures, total = [], 0
for words, want, check, *before in CASES:
    hits = 0
    for _ in range(REPEAT):
        state.update_session({"asleep": False, "in_session": True, "reminders": []})
        body.update(navigate=[], dispatch=[], cancel=0, volume=[60])
        decided.clear()
        wl._run_turn(persona, Reply, words, list(before[0]) if before else [], mic=None)
        chose = [a["skill"] for a in decided[0][0]] if decided else []
        if want is None:
            ok = not chose and not moved() and not CLAIM.search(said[-1])
        else:
            ok = chose == [want] and (check is None or bool(check()))
        faced = [t for t in body["navigate"] if t.get("quiet")]
        ok = ok and faced == ([] if want in OWN_MOVES else [{"turn": VOICE_FROM, "quiet": True}])
        hits += ok
        print(f"  {'ok ' if ok else 'BAD'} {words!r:44} -> {decided[0] if decided else ([], [])} "
              f"body {body['navigate'] or ''}{body['dispatch'] or ''}  {said[-1]!r}")
    total += hits
    print(f"{hits}/{REPEAT} {words!r}")
    if hits < REPEAT:
        failures.append(f"{words!r}: right {hits}/{REPEAT}")
print(f"overall: {total}/{REPEAT * len(CASES)} right")
assert not failures, "\n".join(failures)
print("test_voice_skills: ok")
