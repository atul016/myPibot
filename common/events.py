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
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.time(), "kind": kind, "text": text})
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    _trim_if_needed()


def _trim_if_needed() -> None:
    try:
        lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) > MAX_EVENTS * 2:  # trim in batches, not every write
        atomic_write(EVENTS_PATH, "\n".join(lines[-MAX_EVENTS:]) + "\n")


def recent_events(limit: int = MAX_EVENTS) -> list[dict]:
    try:
        lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
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
