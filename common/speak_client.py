"""Client used by every non-root service to request speech from the
openbot-speak daemon (services/speak.py) -- the one root-owned process that
touches enable_speaker()/disable_speaker()/actual playback, and keeps
Piper's voice model loaded once instead of reloading per utterance. Talks
over a Unix domain socket rather than spawning a subprocess per call: one
persistent daemon is the whole privilege boundary, and no service other
than speak.py itself needs root.
"""
from __future__ import annotations

import json
import os
import socket

from .state import mark_self_noise
from .persona import CURRENT

SOCK_PATH = os.environ.get("OPENBOT_SPEAK_SOCK", "/tmp/openbot-speak.sock")
# openbot-speak writes "1" here while ANY audio plays (speech or sound effect,
# from any process) and "0" when it stops. The listener ignores the mic while
# it's "1" and for ECHO_TAIL_S after -- otherwise Rocky's own voice from a
# reflex or the mind ("Back away!") came back as something the user said.
PLAYING_FLAG = "/tmp/openbot-playing"
ECHO_TAIL_S = 0.7
# Everything Rocky said, with when: {"text", "start", "end"} per line (written by
# openbot-speak). The conversation listener removes these exact sentences from
# what Whisper heard (stt.remove_echo) instead of going deaf while Rocky talks.
SAID_LOG = "/tmp/openbot-said.jsonl"


def speak(text: str, persona: str = CURRENT, timeout_s: float = 30.0) -> bool:
    """Best-effort: returns False (never raises) on failure -- losing
    speech output must not crash the caller."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"persona": persona, "text": text}).encode() + b"\n")
            resp = sock.recv(4096)
        mark_self_noise()
        return bool(json.loads(resp).get("ok", False))
    except Exception:
        return False


def play_sound(path: str, timeout_s: float = 20.0) -> bool:
    """Play a sound effect (wav under picar-x/sounds) through openbot-speak --
    the one process allowed to open the audio device."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"cmd": "sound", "path": path}).encode() + b"\n")
            ok = bool(json.loads(sock.recv(4096)).get("ok", False))
        mark_self_noise()
        return ok
    except Exception:
        return False


def playing(tail_s: float = ECHO_TAIL_S) -> bool:
    """True while Rocky is making sound, or just finished (echo/reverb)."""
    import time
    try:
        with open(PLAYING_FLAG) as f:
            on = f.read(1) == "1"
        return on or time.time() - os.path.getmtime(PLAYING_FLAG) < tail_s
    except OSError:
        return False


def said_between(start: float, end: float) -> list[str]:
    """Texts Rocky was speaking at any point between start and end (unix time)."""
    try:
        with open(SAID_LOG) as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []
    return [x["text"] for x in lines if x.get("end", 0) >= start and x.get("start", 0) <= end]


def warm() -> None:
    """Fire-and-forget: get the amp ready for an imminent reply."""
    import threading

    def _send() -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect(SOCK_PATH)
                sock.sendall(json.dumps({"cmd": "warm"}).encode() + b"\n")
                sock.recv(64)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def stop(timeout_s: float = 2.0) -> bool:
    """Cut off whatever is being spoken right now (barge-in)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(SOCK_PATH)
            sock.sendall(json.dumps({"cmd": "stop"}).encode() + b"\n")
            return bool(json.loads(sock.recv(4096)).get("ok", False))
    except Exception:
        return False


def demo() -> None:
    # Confirms a connection failure returns False rather than raising --
    # against a socket that can't exist, so running this check on the robot
    # never speaks out loud through a live openbot-speak.
    global SOCK_PATH
    orig, SOCK_PATH = SOCK_PATH, "/nonexistent/openbot-speak.sock"
    try:
        assert speak("test", timeout_s=1.0) is False
        assert stop(timeout_s=1.0) is False
    finally:
        SOCK_PATH = orig


if __name__ == "__main__":
    demo()
    print("speak_client: ok")
