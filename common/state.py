"""Shared, cross-process robot state: FileLock-guarded JSON, atomic writes.

Multiple services (alive, wake-listen, mind, speak, dashboard) read/write
the same state/session.json instead of one process's in-memory attributes --
this module is the coordination point.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

from .persona import CURRENT

STATE_DIR = Path(os.environ.get("OPENBOT_STATE_DIR", Path(__file__).resolve().parent.parent / "state"))
SESSION_PATH = STATE_DIR / "session.json"
LOCK_PATH = STATE_DIR / "session.json.lock"
LOCK_TIMEOUT_S = 10.0

DEFAULT_SESSION: dict[str, Any] = {
    "persona": CURRENT,
    "listening": False,
    "in_session": False,
    "last_heard": "",
    "mood": "neutral",
    "confirm_motion_allowed": True,
    "asleep": False,  # "go to sleep": camera off, no thinking, no moving -- the wake word wakes it
}

# A session stays open until "stop session" -- but the mind shouldn't stay
# frozen behind a session nobody has spoken in for a while.
CONVERSATION_IDLE_S = 120.0


def _ensure_dir_writable(directory: Path) -> None:
    """openbot-speak (root) and every other service (the normal deploying
    user) share state/ -- whichever creates a directory here first must
    leave it world-writable (sticky, like /tmp), or the other user's
    writes fail outright with a plain PermissionError. Re-asserted on
    every call rather than trusted from a prior run, so it self-heals if
    some process's mkdir raced ahead with default permissions. A
    PermissionError here means a *different* user already owns the
    directory with the wrong mode -- nothing this process can do about
    that case; fix once by hand (chmod 1777) if it happens."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o1777)
    except PermissionError:
        pass


def atomic_write(path: Path, content: str) -> None:
    _ensure_dir_writable(STATE_DIR)      # in case path.parent is a subdirectory (e.g. state/health/)
    _ensure_dir_writable(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the temp file mode 0600 (owner-only) -- fine for
        # this process, but openbot-speak (root) and every other service
        # (the normal user) read each other's files under state/ (e.g. the
        # dashboard reading openbot-speak's own health record). Without
        # this, health.status() silently reads a PermissionError as
        # "missing" and a perfectly healthy root-owned service shows as
        # offline to every non-root reader.
        os.chmod(tmp, 0o644)
        os.replace(tmp, str(path))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def ensure_session() -> None:
    if not SESSION_PATH.exists():
        atomic_write(SESSION_PATH, json.dumps(DEFAULT_SESSION, indent=2))


def load_session() -> dict[str, Any]:
    """Best-effort read -- returns {} on any failure (missing/corrupt file,
    lock contention). policy.py treats {} as fail-closed for audio/motion,
    not as "no flags set"."""
    try:
        with FileLock(str(LOCK_PATH), timeout=LOCK_TIMEOUT_S):
            ensure_session()
            return json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (FileLockTimeout, OSError, json.JSONDecodeError):
        return {}


def update_session(fields: dict[str, Any]) -> None:
    with FileLock(str(LOCK_PATH), timeout=LOCK_TIMEOUT_S):
        ensure_session()
        try:
            data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = dict(DEFAULT_SESSION)
        data.update(fields)
        atomic_write(SESSION_PATH, json.dumps(data, indent=2))


def conversation_active(session: dict | None = None) -> bool:
    """Someone is actually talking with Rocky: a session is open AND there was
    speech in it within CONVERSATION_IDLE_S. An open-but-idle session lets the
    mind think (and speak up) again."""
    s = session if session is not None else load_session()
    return bool(s.get("in_session")) and time.time() - s.get("last_activity_ts", 0) < CONVERSATION_IDLE_S


def mark_self_noise() -> None:
    """Stamps "the robot itself just made sound/motion" -- openbot-mind
    ignores loudness/vision surprises for a few seconds after this, or
    Rocky hears its own voice, gets surprised, speaks again, forever.
    Best-effort: never raises (called from best-effort clients)."""
    try:
        update_session({"self_noise_ts": time.time()})
    except Exception:
        pass


def demo() -> None:
    import shutil
    global STATE_DIR, SESSION_PATH, LOCK_PATH
    test_dir = Path(tempfile.mkdtemp())
    STATE_DIR, SESSION_PATH, LOCK_PATH = test_dir, test_dir / "session.json", test_dir / "session.json.lock"
    try:
        assert load_session() == DEFAULT_SESSION
        update_session({"persona": "test"})
        assert load_session()["persona"] == "test"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("state: ok")
