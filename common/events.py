"""Cross-process event log for the dashboard: state/events.jsonl,
append-only, trimmed periodically. Replaces the old single-process
in-memory deque now that the dashboard is a separate process reading
shared state instead of a live object reference.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .state import STATE_DIR, atomic_write

EVENTS_PATH = STATE_DIR / "events.jsonl"
MAX_EVENTS = 200


def log_event(kind: str, text: str) -> None:
    """kind: "heard"/"reply"/"safety"/"wake"/"mic" -- descriptive only,
    used by the dashboard to color-code log rows."""
    append_jsonl(EVENTS_PATH, {"ts": time.time(), "kind": kind, "text": text}, MAX_EVENTS)


def recent_events(limit: int = MAX_EVENTS) -> list[dict]:
    return read_jsonl(EVENTS_PATH, limit)


def append_jsonl(path: Path, record: dict, keep: int | None = None) -> None:
    """One JSON line onto `path`; with `keep`, trimmed to the last `keep` lines
    now and then. Also the Jev call log (common/jev.py), which keeps everything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    if keep is None:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) > keep * 2:  # trim in batches, not every write
        atomic_write(path, "\n".join(lines[-keep:]) + "\n")


def read_jsonl(path: Path, limit: int) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def demo() -> None:
    import shutil, tempfile
    global EVENTS_PATH
    orig = EVENTS_PATH
    test_dir = Path(tempfile.mkdtemp())
    EVENTS_PATH = test_dir / "events.jsonl"
    try:
        log_event("safety", "test event")
        events = recent_events()
        assert len(events) == 1
        assert events[0]["kind"] == "safety"
    finally:
        EVENTS_PATH = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("events: ok")
