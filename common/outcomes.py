"""What happened after Rocky did something on its own -- the results it learns
from. Each outcome is kept twice:
  - state/outcomes/<day>.jsonl, never trimmed: training data, with the time the
    choice was made (decided_ts) to join with that moment's Jev call (state/jev/);
  - a line in its lesson note "what gets a reaction", which its next choices
    (how_people_reacted_before), Jev's texting question, and the nightly
    "what works" rules all read.
Spoken lines and gestures: did people react in the next few seconds (mind.py).
Texts it started: did they reply, and how fast (services/chat.py).
"""
from __future__ import annotations

import datetime as dt
import time

from . import memory
from .events import append_jsonl, read_jsonl
from .state import STATE_DIR

OUTCOME_DIR = STATE_DIR / "outcomes"
REACTION_NOTE = "what gets a reaction"
REPLY_WINDOW_S = 3 * 3600  # a reply later than this isn't an answer to the text


def record(what: str, outcome: str, **fields) -> None:
    """`what`: "said ..." / "texted ..."; `outcome`: "Atul: replied after 2 min". `fields` go
    into the training record only (kind, replied, after_s, decided_ts...)."""
    now = time.time()
    append_jsonl(OUTCOME_DIR / f"{dt.date.fromtimestamp(now).isoformat()}.jsonl",
                 {"ts": now, "what": what, "outcome": outcome, **fields})
    memory.remember("lesson", REACTION_NOTE, "reaction", f"{dt.datetime.now():%a %I:%M %p} I {what} -> {outcome}")


def recent(n: int = 6) -> list[str]:
    """The last n outcomes, oldest first, as "texted "..." -> Atul: no reply" (today and yesterday)."""
    today = dt.date.today()
    rows = []
    for day in (today - dt.timedelta(days=1), today):
        rows += read_jsonl(OUTCOME_DIR / f"{day.isoformat()}.jsonl", n)
    return [f"{r['what']} -> {r['outcome']}" for r in rows[-n:]]


def reply_words(after_s: float | None) -> str:
    if after_s is None:
        return "no reply"
    return "replied right away" if after_s < 60 else f"replied after {after_s / 60:.0f} min"


def demo() -> None:
    import shutil, tempfile
    from pathlib import Path
    global OUTCOME_DIR
    orig, mind_dir = (OUTCOME_DIR, memory.MIND_DIR), Path(tempfile.mkdtemp())
    OUTCOME_DIR, memory.MIND_DIR = mind_dir / "outcomes", mind_dir / "mind"
    try:
        record('texted "is that your jacket?"', "Atul: " + reply_words(130), kind="text", replied=True, after_s=130)
        record('said "hello"', "Atul: they talked back to me", kind="speech")
        assert recent() == ['texted "is that your jacket?" -> Atul: replied after 2 min',
                            'said "hello" -> Atul: they talked back to me']
        assert memory.read_note("lesson", REACTION_NOTE)[-1].endswith('I said "hello" -> Atul: they talked back to me')
        assert reply_words(None) == "no reply" and reply_words(5) == "replied right away"
    finally:
        OUTCOME_DIR, memory.MIND_DIR = orig
        shutil.rmtree(mind_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("outcomes: ok")
