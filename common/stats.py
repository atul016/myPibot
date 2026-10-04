"""The dashboard's Stats tab: is Rocky getting better, in numbers -- per day,
from what it already writes (the journal, state/outcomes/, state/jev/).
Talking: texts it started, and the replies / reactions / silence they got.
Its mind: Jev's calls and how often it said "text"; confusion -- thoughts
that the room changed, "something is moving" notices, and WhatsApp replies
that claim a move a text can't make (a phrase match: "possible", not proof).
"""
from __future__ import annotations

import datetime as dt
import re

from . import journal
from .events import read_jsonl
from .state import STATE_DIR

CONFUSED = re.compile(r"layout|shifted|room (?:has )?changed|scene has (?:completely )?changed|without me moving", re.I)
MOVE_CLAIM = re.compile(r"\b(?:I(?:'m| am)? moving|I move\b|moving forward|on my way|rolling|I(?:'m)? driving|"
                        r"looking away|head turned|turning (?:left|right))", re.I)


def day(d: dt.date) -> dict:
    lines = journal.entries(d)
    kinds: dict[str, int] = {}
    for ln in lines:
        k = ln[1:ln.index("]")]
        kinds[k] = kinds.get(k, 0) + 1
    outs = read_jsonl(STATE_DIR / "outcomes" / f"{d.isoformat()}.jsonl", 10**6)
    texts = [o for o in outs if o.get("kind") != "unprompted"]
    jev = [c for c in read_jsonl(STATE_DIR / "jev" / f"{d.isoformat()}.jsonl", 10**6)]
    picks = [((c.get("response") or {}).get("answers") or {}).get("pick") or {} for c in jev]
    texting = [p for c, p in zip(jev, picks) if "stay_quiet" in (((c.get("request") or {}).get("questions") or {})
                                                                   .get("pick", {}).get("criteria") or {})]
    replied = sum(1 for o in texts if o.get("replied"))
    silent = sum(1 for o in texts if o.get("replied") is False)
    return {
        "day": d.isoformat(),
        "thoughts": kinds.get("thought", 0), "said": kinds.get("said", 0), "heard": kinds.get("heard", 0),
        "texts_started": kinds.get("texted", 0), "replied": replied, "no_reply": silent,
        "reactions": sum(1 for o in texts if o.get("reaction")),
        "reply_rate": round(replied / (replied + silent), 2) if replied + silent else None,
        "jev_calls": len(jev), "jev_errors": sum(1 for c in jev if c.get("status") != 200),
        "jev_said_text": sum(1 for p in texting if p.get("choice") == "text"), "jev_texting_calls": len(texting),
        "room_changed_thoughts": sum(1 for ln in lines if ln.startswith("[thought]") and CONFUSED.search(ln)),
        "moving_notices": sum(1 for ln in lines if ln.startswith("[noticed]") and "something is moving" in ln),
        "possible_move_claims": sum(1 for ln in lines if ln.startswith("[said]") and "(WhatsApp to" in ln
                                    and MOVE_CLAIM.search(ln)),
        "held_back": kinds.get("held_back", 0),
    }


def week(days: int = 7, today: dt.date | None = None) -> list[dict]:
    """Newest day first."""
    today = today or dt.date.today()
    return [day(today - dt.timedelta(days=i)) for i in range(days)]


def demo() -> None:
    import json, shutil, tempfile
    from pathlib import Path
    from . import memory
    global STATE_DIR
    orig, test_dir = (STATE_DIR, memory.MIND_DIR), Path(tempfile.mkdtemp())
    STATE_DIR, memory.MIND_DIR = test_dir, test_dir / "mind"
    try:
        d = dt.date(2026, 10, 3)
        (test_dir / "mind" / "journal").mkdir(parents=True)
        (test_dir / "mind" / "journal" / "2026-10-03.md").write_text(
            "- [thought] 10:00 (confused) The room layout shifted again without me moving\n"
            "- [thought] 10:01 (bored) nothing new\n"
            "- [noticed] 10:02 something is moving in front of you (change 40)\n"
            "- [texted] 10:03 Is that your jacket?\n"
            "- [said] 10:04 (WhatsApp to Atul) Okay, I moving forward!\n"
            "- [said] 10:05 (WhatsApp to Atul) Hi there!\n")
        (test_dir / "outcomes").mkdir()
        (test_dir / "outcomes" / "2026-10-03.jsonl").write_text(
            json.dumps({"kind": "jev", "replied": True, "reaction": "👍"}) + "\n"
            + json.dumps({"kind": "summary", "replied": False}) + "\n" + json.dumps({"kind": "unprompted"}) + "\n")
        s = day(d)
        assert (s["thoughts"], s["texts_started"], s["replied"], s["no_reply"], s["reactions"], s["reply_rate"]) == \
               (2, 1, 1, 1, 1, 0.5), s
        assert (s["room_changed_thoughts"], s["moving_notices"], s["possible_move_claims"]) == (1, 1, 1), s
        assert week(2, d)[0]["day"] == "2026-10-03" and week(2, d)[1]["thoughts"] == 0
    finally:
        STATE_DIR, memory.MIND_DIR = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("stats: ok")
