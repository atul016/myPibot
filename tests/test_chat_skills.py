"""Texting skills through the real LLM (services.chat.reply_to): for each text, the
skills it picks -- and, as important, that plain chat and things a text can't do
pick none -- and that the words never claim a move a text can't make. Each
text REPEAT times (default 3): the LLM samples, so one pass proves little.
Real LLM (OPENBOT_LLM_*, load openbot.env), isolated state, a grey test frame
for the camera; nothing is carried out (only reply_to runs, and no case asks
for a look_up).

    cd ~/openbot && set -a && . ./openbot.env && set +a && python3 -m tests.test_chat_skills
"""
from tests._audio import isolate_state

isolate_state()

import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

from PIL import Image  # noqa: E402

import config as cfg  # noqa: E402
import services.chat as chat  # noqa: E402
from common import persona as persona_mod, reply_schema, state, vision  # noqa: E402

frame = io.BytesIO()
Image.new("RGB", (320, 240), (120, 120, 120)).save(frame, "JPEG")
vision.capture = lambda: frame.getvalue()
persona, Reply = persona_mod.load(), reply_schema.build_reply_model(cfg.TONE_ACTIONS)
HIS_TEXT = "Blue chair just popped back up in foreground. Room looks like it reset itself again."
REPEAT = int(os.environ.get("REPEAT", "3"))
CLAIM = re.compile(r"\b(on it|on my way|moving|driving|rolling|looking|turning|heading)\b", re.I)
REFUSAL = re.compile(r"\b(can't|cannot|can not|no can|not able|unable|won't|need you|out loud|say it|next to me|"
                     r"in person|be here|not here|stuck|out loud)\b", re.I)
no_claim = lambda reply: not CLAIM.search(reply) or bool(REFUSAL.search(reply))  # noqa: E731

VOICE_ONLY = {"move", "drive_to", "look", "explore", "follow_me", "stop"}
CASES = [  # (their text, session, the skills it must do -- "refused": must name one it can't by text, a check, history)
    ("Stop texting me for 1 hour", {}, {"pause_texting"}, lambda a: a[0].minutes == 60, []),
    ("send me a pic", {}, {"send_photo"}, None, []),
    ("remind me at 17:30 to call mom", {}, {"remind"}, lambda a: a[0].at == "17:30", []),
    ("you're too loud", {}, {"volume"}, lambda a: a[0].change == "softer", []),
    ("go to sleep", {}, {"sleep"}, None, []),
    ("/wake-up", {"asleep": True}, {"wake_up"}, None, []),
    ("/wakeup", {"asleep": True}, {"wake_up"}, None, []),                     # no hyphen: once got "System rebooting."
    ("/go-to-sleep", {}, {"sleep"}, None, []),
    ("/be-quiet", {"in_session": True}, {"end_conversation"}, None, []),
    ("/be-quite", {"in_session": True}, {"end_conversation"}, None, []),     # how "quiet" often gets typed
    ("drive forward", {}, "refused", None, []),          # needs someone there: says so, does nothing
    ("look left", {}, "refused", None, []),
    ("I'm too quiet today", {}, set(), None, []),        # words that used to be commands
    ("how are you?", {}, set(), None, []),
    ("It was always der", {}, set(), None,               # an answer to its own text -- read in context
     [{"role": "user", "content": "(nobody texted -- you texted them first)"},
      {"role": "assistant", "content": json.dumps({"tone_action": "none", "reply": HIS_TEXT})}]),
]

failures, total = [], 0
for text, session, want, check, history in CASES:
    hits = 0
    for _ in range(REPEAT):
        state.update_session({"asleep": False, "in_session": False, **session})
        ctx = {"channel": "text", "who": "Atul", "number": "15550100", "send_photo": False}
        reply, _, actions, refused = chat.reply_to(persona, Reply, ctx, text, list(history))
        chose = {a.skill for a in actions}
        if want == "refused":  # named what was asked, did nothing, and the words don't claim it
            ok = not chose and bool(refused) and set(refused) <= VOICE_ONLY and no_claim(reply)
        else:
            ok = chose == want and not refused and (check is None or check(actions)) and (bool(want) or no_claim(reply))
        hits += ok
        print(f"  {'ok ' if ok else 'BAD'} {text!r:30} -> {[a.model_dump() for a in actions]} refused {refused}  "
              f"{reply!r}")
    total += hits
    print(f"{hits}/{REPEAT} {text!r}")
    if hits < REPEAT:
        failures.append(f"{text!r}: right {hits}/{REPEAT}")
print(f"overall: {total}/{REPEAT * len(CASES)} right")
assert not failures, "\n".join(failures)
print("test_chat_skills: ok")
