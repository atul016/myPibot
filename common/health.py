"""Per-service health: state/health/<name>.json, status derived at read
time (never stored), so a dead service can't leave a lying "ok" behind.
Replaces a single process-wide watchdog heartbeat with per-service
visibility -- "which service died," not just "something did."

Also owns the systemd watchdog ping (sd_notify's WATCHDOG=1): each service
calls start_watchdog() once at startup, and record_success()/
record_failure() from its own work loop -- the ping only flows while that
service's own health record is fresh, so a hung main thread (which never
exits, so Restart=always never fires) gets force-restarted by systemd
instead.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

from .state import STATE_DIR, atomic_write

HEALTH_DIR = STATE_DIR / "health"

DEFAULT_STALE_AFTER_S = 120.0
FAILING_AFTER_N = 3


def _path(component: str) -> Path:
    return HEALTH_DIR / f"{component}.json"


def record_success(component: str, *, min_interval_s: float = 0.0) -> None:
    """min_interval_s throttles the write (a 20Hz sensor loop shouldn't
    fsync every tick) -- never throttles the status itself."""
    path = _path(component)
    now = time.time()
    if min_interval_s > 0 and path.exists():
        try:
            last = json.loads(path.read_text())
            if now - last.get("last_success_ts", 0) < min_interval_s:
                return
        except (OSError, json.JSONDecodeError):
            pass
    atomic_write(path, json.dumps({"last_success_ts": now, "consecutive_failures": 0}, indent=2))


def record_failure(component: str, error: str) -> None:
    """Failures never throttle, and clear the instant a success lands --
    a flapping component must not read as "failing" while it's working."""
    path = _path(component)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data["consecutive_failures"] = data.get("consecutive_failures", 0) + 1
    data["last_failure_ts"] = time.time()
    data["last_error"] = error
    atomic_write(path, json.dumps(data, indent=2))


def status(component: str, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> str:
    """ok -> degraded -> stale -> failing/missing."""
    path = _path(component)
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "missing"
    if data.get("consecutive_failures", 0) >= FAILING_AFTER_N:
        return "failing"
    if time.time() - data.get("last_success_ts", 0) > stale_after_s:
        return "stale"
    if data.get("consecutive_failures", 0) > 0:
        return "degraded"
    return "ok"


def all_status(components: list[str]) -> dict[str, str]:
    return {c: status(c) for c in components}


def _sd_notify(message: str) -> bool:
    address = os.environ.get("NOTIFY_SOCKET", "")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())
        return True
    except OSError:
        return False


def start_watchdog(component: str, stale_after_s: float, ping_interval_s: float) -> None:
    def _loop() -> None:
        while True:
            if status(component, stale_after_s) in ("ok", "degraded"):
                _sd_notify("WATCHDOG=1")
            time.sleep(ping_interval_s)

    threading.Thread(target=_loop, name=f"watchdog-{component}", daemon=True).start()


def demo() -> None:
    import shutil, tempfile
    global HEALTH_DIR
    orig = HEALTH_DIR
    test_dir = Path(tempfile.mkdtemp())
    HEALTH_DIR = test_dir
    try:
        assert status("x") == "missing"
        record_success("x")
        assert status("x") == "ok"
        for _ in range(3):
            record_failure("x", "boom")
        assert status("x") == "failing"
        record_success("x")
        assert status("x") == "ok"
    finally:
        HEALTH_DIR = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("health: ok")
