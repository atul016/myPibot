"""One continuous record of Rocky's life: every process appends what it
heard, said, noticed, thought, did and found to ONE daily markdown note,
state/mind/journal/YYYY-MM-DD.md, one line each:

    - [heard] 14:02 what's the weather like
    - [thought] 14:02 (curious) Atul sounds tired today
    - [found] 14:03 look left: a coffee mug next to the keyboard

Replaces the old split (thoughts.jsonl for the mind, nothing at all for
conversations): the mind now sees what was said to it, and conversations
see what the mind was doing. Being a Basic Memory note, every line is also
searchable via memory.recall() ("what happened last time it got dark?").

Append-only, never rewritten in place -- three processes append to it and
the memory server watches it. The rolling "today so far" summary lives in
its own note (summaries/YYYY-MM-DD.md), replaced atomically.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from filelock import FileLock, Timeout as FileLockTimeout

from . import memory
from .state import STATE_DIR, atomic_write

LOCK_PATH = STATE_DIR / "journal.lock"
_LINE = re.compile(r"^- \[(\w+)\] (\d\d:\d\d) (.*)$")


def _day_path(day: dt.date) -> Path:
    return memory.MIND_DIR / "journal" / f"{day.isoformat()}.md"


def log(kind: str, text: str, now: dt.datetime | None = None) -> None:
    """Best-effort: a failed journal write must never break the caller."""
    now = now or dt.datetime.now()
    line = f"- [{memory.slug(kind)[:20]}] {now:%H:%M} {memory.one_line(text, 400)}\n"
    try:
        with FileLock(str(LOCK_PATH), timeout=5):
            path = memory.ensure_note("journal", now.date().isoformat(),
                                      f"Journal {now.date().isoformat()}", "journal")
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    except (FileLockTimeout, OSError):
        pass


def _entries(day: dt.date) -> list[str]:
    try:
        lines = _day_path(day).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [f"[{m[1]}] {m[2]} {m[3]}" for m in map(_LINE.match, lines) if m]


def tail(n: int = 15, now: dt.datetime | None = None) -> list[str]:
    """Last n entries, oldest first, reaching back into yesterday after midnight."""
    today = (now or dt.datetime.now()).date()
    out = _entries(today)[-n:]
    if len(out) < n:
        out = [f"(yesterday) {e}" for e in _entries(today - dt.timedelta(days=1))[-(n - len(out)):]] + out
    return out


def entries(day: dt.date) -> list[str]:
    return _entries(day)


# --- rolling summary ---------------------------------------------------------

def _summary_path(day: dt.date) -> Path:
    return memory.MIND_DIR / "summaries" / f"{day.isoformat()}.md"


def read_summary(day: dt.date) -> str:
    try:
        text = _summary_path(day).read_text(encoding="utf-8")
    except OSError:
        return ""
    return text.split("---\n", 2)[-1].strip()


def write_summary(day: dt.date, summary: str) -> None:
    path = _summary_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, f"---\ntitle: Summary {day.isoformat()}\ntype: summary\n"
                       f"permalink: {memory.PROJECT}/summaries/{day.isoformat()}\n---\n\n"
                       f"{memory.one_line(summary, 2000)}\n")


# --- prompt context shared by every LLM call that speaks as Rocky -------------

def inject(prompt_text: str, query: str = "") -> str:
    """Prepends what Rocky sees, today's summary, its recent life, and
    memories relevant to `query` -- the one formatter for the reactive
    turn, the wake greeting and proximity reactions, so they can't drift
    apart. (The mind loop builds its own richer awareness dict.)"""
    import time
    from . import faces, vision  # deferred: vision imports cognition; keep this module light

    sections = []
    look = vision.last_look()
    if look.get("scene") and time.time() - look.get("ts", 0) < 600:
        sections.append(f"What you can see through your camera right now: {look['scene']}")
    people = faces.describe(faces.read())
    if people:
        sections.append(f"Who's in front of you right now (from your camera): {people}")
    summary = read_summary(dt.date.today())
    if summary:
        sections.append(f"Your day so far: {summary}")
    recent = tail(10)
    if recent:
        sections.append("Your recent life (heard/said/noticed/thought/did):\n" + "\n".join(recent))
    related = memory.recall(query) if query else None
    if related:
        sections.append("Things you remember that might be relevant:\n" + "\n".join(related))
    return "\n\n".join(sections + [prompt_text])


def demo() -> None:
    import shutil, tempfile
    global LOCK_PATH
    orig_dir, orig_lock = memory.MIND_DIR, LOCK_PATH
    test_dir = Path(tempfile.mkdtemp())
    memory.MIND_DIR, LOCK_PATH = test_dir, test_dir / "journal.lock"
    try:
        y = dt.datetime(2026, 10, 1, 23, 58)
        log("heard", "rocky what time is it", now=y)
        log("said", "late!\nsecond line", now=y)
        t = dt.datetime(2026, 10, 2, 0, 1)
        log("noticed", "the lights went out", now=t)
        assert _day_path(t.date()).read_text().startswith(
            f"---\ntitle: Journal 2026-10-02\ntype: journal\npermalink: {memory.PROJECT}/journal/2026-10-02\n---\n")
        assert tail(3, now=t) == ["(yesterday) [heard] 23:58 rocky what time is it",
                                  "(yesterday) [said] 23:58 late! second line",
                                  "[noticed] 00:01 the lights went out"]
        assert tail(1, now=t) == ["[noticed] 00:01 the lights went out"]
        write_summary(t.date(), "Quiet night.")
        write_summary(t.date(), "Quiet night, then the lights went out.")
        assert read_summary(t.date()) == "Quiet night, then the lights went out."
        assert read_summary(dt.date(2000, 1, 1)) == ""
    finally:
        memory.MIND_DIR, LOCK_PATH = orig_dir, orig_lock
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("journal: ok")
