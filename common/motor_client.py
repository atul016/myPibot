"""Client used by wake_listen.py and mind.py to request a gesture/movement
dispatch from openbot-alive -- the one process that owns the persistent
Picarx()/ActionFlow instance.

Originally this project tried a GPIO *lease*: openbot-alive would yield its
Picarx() handle on request, and the requesting process would construct its
own. That does not work on this hardware -- confirmed live: a second
Picarx() raised `lgpio.error: 'GPIO busy'` while openbot-alive's instance
was still running, even after signaling it to stop. lgpio's GPIO line
claims are exclusive per open chip handle, and stopping motors/ActionFlow
does not release them; only closing the chip handle does, and robot_hat's
own Pin.close() only frees a pin that registered an interrupt callback, not
a plain output. Rather than reach into robot_hat's internals to force a
release, one process owns the hardware permanently and everyone else asks
it to act over a socket -- the same privilege-boundary shape already used
for audio (common.speak_client / services.speak).
"""
from __future__ import annotations

import json
import os
import socket

from .state import mark_self_noise

SOCK_PATH = os.environ.get("OPENBOT_ALIVE_SOCK", "/tmp/openbot-alive.sock")


def dispatch(actions: list[str], wait: bool = False, timeout_s: float = 15.0) -> bool:
    """Best-effort: returns False (never raises) on failure -- a caller
    losing the ability to gesture must not crash the caller."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"actions": actions, "wait": wait}).encode() + b"\n")
            resp = sock.recv(4096)
        mark_self_noise()  # servo whine reaches the mic too
        return bool(json.loads(resp).get("ok", False))
    except Exception:
        return False


def navigate(task: dict, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Start a drive in openbot-alive: {"approach": "pink toy"} or {"explore": steps}.
    Returns at once (the drive runs there, and Rocky says how it went)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"navigate": task}).encode() + b"\n")
            resp = json.loads(sock.recv(4096))
        return bool(resp.get("ok")), str(resp.get("error", ""))
    except Exception as e:
        return False, str(e)


def look_toward(pan_deg: float, timeout_s: float = 2.0) -> bool:
    """Glance the head toward a voice (degrees, + right) -- openbot-alive's face tracker holds it
    there a few seconds, then a face it finds takes over. False if alive didn't answer."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"look_toward": float(pan_deg)}).encode() + b"\n")
            ok = bool(json.loads(sock.recv(4096)).get("ok"))
        mark_self_noise()  # the head's servos whine too
        return ok
    except Exception:
        return False


def cancel_navigation(timeout_s: float = 2.0) -> bool:
    """Stop any drive now. True if one was running."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"navigate_cancel": True}).encode() + b"\n")
            return bool(json.loads(sock.recv(4096)).get("was_driving"))
    except Exception:
        return False


def demo() -> None:
    # Confirms a connection failure returns False rather than raising --
    # against a socket that can't exist, so running this check on the robot
    # never sends a real gesture to a live openbot-alive.
    global SOCK_PATH
    orig, SOCK_PATH = SOCK_PATH, "/nonexistent/openbot-alive.sock"
    try:
        assert dispatch(["nod"], timeout_s=1.0) is False
    finally:
        SOCK_PATH = orig


if __name__ == "__main__":
    demo()
    print("motor_client: ok")
