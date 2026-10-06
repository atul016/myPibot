"""What the mind may do, pinned down. First what services.mind.validate_expression
lets through -- for every action its reflection can pick, a valid example and what
it becomes, and what gets refused: text too long, a number out of range, a choice
that doesn't exist, driving in the dark, sleeping when not tired or with someone in
view. Then that each one, carried out (_dispatch_expression; a tool through its
skill), reaches the body -- every action is a skill now (skills/*/, `where: mind`),
and a muscle that raises fails here instead of quietly doing nothing. Written
against the hand-made validator, kept passing when the tools moved to one spec each.
No LLM; the body, speech, camera, ears and memory search are stubbed. Isolated state.

    cd ~/openbot && set -a && . ./openbot.env && set +a && python3 -m tests.test_mind_actions
"""
from tests._audio import isolate_state

isolate_state()

import time  # noqa: E402

import config as cfg  # noqa: E402
import common.motor_client as motor  # noqa: E402
import services.mind as mind  # noqa: E402
from common import faces, hearing, memory, persona as persona_mod, skills, state, vision  # noqa: E402

dark, tired, face = [False], [False], [False]
vision.last_look = lambda: {"scene": vision.DARK_SCENE if dark[0] else "a desk with a laptop"}
mind._needs = lambda: ["tired: battery at 10%"] if tired[0] else []
faces.read = lambda: {"faces": [{"name": "Anna"}] if face[0] else []}


def ok(action: str, params: dict, want: dict) -> None:
    got = mind.validate_expression({"action": action, "params": params})
    assert got == (action, want), f"{action} {params}: got {got}, want {want}"


def bad(action: str, params: dict) -> None:
    try:
        got = mind.validate_expression({"action": action, "params": params})
    except mind.MindError:
        return
    raise AssertionError(f"{action} {params} was accepted: {got}")


ok("look", {"direction": "left"}, {"direction": "left"})
ok("look", {}, {"direction": "ahead"})
bad("look", {"direction": "behind"})
ok("recall", {"query": "Anna"}, {"query": "Anna"})
bad("recall", {})
bad("recall", {"query": "x" * 201})
ok("speak", {"text": "Hello there"}, {"text": "Hello there"})
bad("speak", {"text": ""})
bad("speak", {"text": "x" * 301})
ok("gesture", {"name": cfg.STATIONARY_ACTIONS[0]}, {"name": cfg.STATIONARY_ACTIONS[0]})
for playful in cfg.PLAYFUL_ACTIONS:
    ok("gesture", {"name": playful}, {"name": playful})
bad("gesture", {"name": "fly"})
ok("remember", {"kind": "person", "about": "Anna", "category": "Likes", "text": "likes tea"},
   {"kind": "person", "about": "Anna", "category": "Likes", "text": "likes tea"})  # memory.remember slugs it
ok("remember", {"kind": "place", "about": "desk", "text": "the lamp is on the left"},
   {"kind": "place", "about": "desk", "category": "note", "text": "the lamp is on the left"})
bad("remember", {"kind": "planet", "about": "Mars", "text": "red"})
bad("remember", {"kind": "person", "about": "x" * 61, "text": "hi"})
bad("remember", {"kind": "person", "about": "Anna", "text": "x" * 301})
if cfg.SOUNDS:
    ok("play_sound", {"name": cfg.SOUNDS[0]}, {"name": cfg.SOUNDS[0]})
    bad("play_sound", {"name": "moo"})
ok("remind_me", {"in_minutes": "5", "about": "check the door"}, {"in_minutes": 5, "about": "check the door"})
bad("remind_me", {"in_minutes": 0, "about": "x"})
bad("remind_me", {"in_minutes": 721, "about": "x"})
bad("remind_me", {"in_minutes": "soon", "about": "x"})
ok("watch", {"for": "sound", "about": "the knock"}, {"for": "sound", "about": "the knock"})
bad("watch", {"for": "dancing", "about": "x"})
ok("rest", {"minutes": 30}, {"minutes": 30})
bad("rest", {"minutes": 0})
bad("rest", {"minutes": 721})
ok("wish", {"text": "wheels that climb stairs"}, {"text": "wheels that climb stairs"})
bad("wish", {"text": "x" * 201})
if cfg.CAN_DRIVE:
    ok("drive_to", {"thing": "pink toy"}, {"thing": "pink toy"})
    bad("drive_to", {"thing": "x" * 41})
    ok("explore", {"seconds": 5}, {"seconds": 5})        # not refused: the muscle keeps it to 10-120 (below)
    ok("explore", {}, {"seconds": 30})
    bad("explore", {"seconds": "a while"})
    dark[0] = True
    bad("drive_to", {"thing": "pink toy"})               # never drive in the dark
    bad("explore", {"seconds": 60})
    dark[0] = False
bad("sleep", {})                                        # not tired
tired[0] = True
ok("sleep", {}, {})
face[0] = True
bad("sleep", {})                                        # someone's right in front of it
tired[0] = face[0] = False
for no_params in ("time_check", "introspect", "wait", "listen"):
    ok(no_params, {"anything": 1}, {})
bad("fly", {})
bad("speak", "hello")                                   # params that aren't an object
try:
    mind.validate_expression(["speak"])
    raise AssertionError("a list was accepted")
except mind.MindError:
    pass

# --- carried out: each reaches the body (stubbed), and a failing muscle fails here ----------------
said, body = [], {"dispatch": [], "navigate": []}
mind.speak_client = lambda text, persona=None: said.append(text)
motor.dispatch = lambda actions, wait=False, timeout_s=15.0: body["dispatch"].append(actions) or True
motor.navigate = lambda task, timeout_s=5.0: body["navigate"].append(task) or (True, "")
vision.look = lambda base_url, model, direction="ahead", head=None: {"scene": f"a desk, looking {direction}",
                                                                      "changed": False}
hearing.read = lambda: {"levels": [300] * 10, "sounds": ["Silence"] * 10}
memory.recall = lambda query, limit=6: [f"people/anna: [likes] {query} likes tea"]
time.sleep = lambda s: None  # listen's few seconds
skills.attempt = lambda action, ctx: skills.run(action, ctx)  # no swallowing: a muscle that raises fails the test
cfg.EXPRESSION_COOLDOWN_S = 0
persona, book = persona_mod.load(), mind._skills()


def do(action: str, params: dict) -> str | None:
    name, params = mind.validate_expression({"action": action, "params": params}, book)
    if book[name].tool:
        return skills.use(name, mind._ctx(persona), params, book)
    mind._dispatch_expression(persona, book[name], params, book)
    return None


state.update_session({"asleep": False})
do("speak", {"text": "Hello there"})
assert "Hello there" in said[-1] and state.load_session()["invite_until"] > time.time()
do("time_check", {})
assert "currently" in said[-1]  # Rocky drops the "is"
do("introspect", {})
assert "working" in said[-1] or "trouble" in said[-1]
do("gesture", {"name": cfg.STATIONARY_ACTIONS[0]})
assert body["dispatch"][-1] == [cfg.STATIONARY_ACTIONS[0]]
if cfg.SOUNDS:
    do("play_sound", {"name": cfg.SOUNDS[0]})
    assert body["dispatch"][-1] == [cfg.SOUNDS[0]]
do("remember", {"kind": "person", "about": "Anna", "category": "Likes", "text": "likes green tea"})
note = memory.read_note("person", "Anna")
assert len(note) == 1 and note[0].startswith("- [likes] likes green tea (") and note[0].endswith(" mind)"), note  # (when, how)
do("remind_me", {"in_minutes": 5, "about": "check the door"})
assert [r["about"] for r in state.load_session()["reminders"]] == ["check the door"]
do("watch", {"for": "sound", "about": "the knock"})
assert [w["for"] for w in state.load_session()["watches"]] == ["sound"]
do("rest", {"minutes": 30})
assert state.load_session()["rest_until"] > time.time() + 29 * 60
do("wish", {"text": "wheels that climb stairs"})
assert "wheels that climb stairs" in memory.read_note("self", "wishes")[-1]
do("wait", {})
if cfg.CAN_DRIVE:
    do("drive_to", {"thing": "pink toy"})
    do("explore", {"seconds": 5})
    assert body["navigate"] == [{"approach": "pink toy"}, {"explore": 10}]
assert do("look", {"direction": "left"}).startswith("look left: a desk, looking left")
assert body["dispatch"][-2:] == [["look left"], ["look ahead"]]  # and back
assert do("listen", {}).startswith("listen: ")
assert do("recall", {"query": "Anna"}).startswith("recall 'Anna': people/anna")
tired[0] = True
do("sleep", {})
assert state.load_session()["asleep"] and body["dispatch"][-1] == ["look down"]
n = len(said)
do("speak", {"text": "Anyone there?"})                  # asleep: the policy holds it back
assert len(said) == n
state.update_session({"asleep": False})
cfg.EXPRESSION_COOLDOWN_S = 120
do("speak", {"text": "Anyone there?"})                  # it just spoke: the cooldown holds it back
assert len(said) == n
print("test_mind_actions: ok")
