"""Hearing, as the rest of OpenBot sees it: openbot-ears (services/ears.py) is the
microphone's only owner; everyone else asks it, the way openbot-camera serves
the camera. This module is the asking side, plus the audio format both sides
must agree on.

  read()          the mind's question: each of the last HISTORY_S seconds' peak
                  loudness and what it sounded like -- None if ears isn't answering
  voice_bearing() which way a voice came from in a stretch of time (a reSpeaker
                  XVF3800's direction finder): degrees from Rocky's front, + right
  pcm_command()   what openbot-wake-listen runs instead of arecord: the live audio
"""
from __future__ import annotations

import json
import math
import os
import struct
import time
import urllib.request

MIC_DEVICE = os.environ.get("OPENBOT_STT_DEVICE", "USB PnP Sound Device")  # part of the mic's name in `arecord -l`
EARS_PORT = int(os.environ.get("OPENBOT_EARS_PORT", "9001"))
EARS_URL = f"http://127.0.0.1:{EARS_PORT}"  # localhost only: this is the house's live audio
CHUNK_FRAMES = 2048     # one mic chunk, 46ms at stt.MIC_RATE -- ears sends whole ones, so samples never split
HISTORY_S = 30          # seconds of loudness + sound names ears keeps
OWN_VOICE = "own voice"  # a second's name while Rocky himself was making sound -- not news
# The array's angle that points at Rocky's front -- talk from in front of him while `python3 -m
# services.ears --doa` runs and read it (Walle: 178). Unset: no voice direction. The XVF3800, LEDs up,
# counts clockwise (talk from his right: the number goes up); OPENBOT_DOA_CCW=1 for one that goes down.
DOA_FRONT = os.environ.get("OPENBOT_DOA_FRONT")
DOA_CCW = os.environ.get("OPENBOT_DOA_CCW", "0") == "1"


def rms(chunk: bytes) -> float:
    count = len(chunk) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"{count}h", chunk[:count * 2])
    return (sum(s * s for s in shorts) / count) ** 0.5


def pcm_command() -> list[str]:
    """Raw 16-bit mono PCM at stt.MIC_RATE on stdout, from openbot-ears. Retries
    while ears starts (systemd doesn't wait for it to be ready); if ears goes
    away mid-stream, curl exits and wake-listen restarts itself, as it does
    when arecord dies."""
    return ["curl", "-sSN", "--retry", "30", "--retry-connrefused", "--retry-delay", "1", f"{EARS_URL}/pcm"]


def read(timeout_s: float = 1.0) -> dict | None:
    """{"levels": [peak loudness per second], "sounds": [name or None per second],
    "ts"} -- oldest first, the two lists aligned. None if openbot-ears isn't
    answering (down, or the mic is gone) -- not the same as a quiet room."""
    try:
        with urllib.request.urlopen(f"{EARS_URL}/hearing", timeout=timeout_s) as r:
            data = json.loads(r.read())
    except (OSError, ValueError):
        return None
    return data if time.time() - data.get("ts", 0) < 5 else None


def bearing(raw: float, front: float, ccw: bool = False) -> float:
    """The array's raw angle as degrees from Rocky's front, + to his right, -180..180."""
    return ((front - raw if ccw else raw - front) + 180) % 360 - 180


def circular_median(angles: list[float]) -> float:
    """The median of directions in degrees, taken around their mean direction -- so 179 and
    -179 are neighbours, not opposites."""
    rad = [math.radians(a) for a in angles]
    mean = math.degrees(math.atan2(sum(map(math.sin, rad)), sum(map(math.cos, rad))))
    rel = sorted((a - mean + 180) % 360 - 180 for a in angles)
    return (mean + rel[len(rel) // 2] + 180) % 360 - 180


def voice_bearing(t0: float, t1: float, heard: dict | None = None, min_samples: int = 3) -> float | None:
    """Which way the voice came from between t0 and t1 (wall clock -- what someone just said):
    degrees from Rocky's front, + right. None without the array, or too little speech then."""
    heard = heard if heard is not None else read()
    found = [b for t, b in (heard or {}).get("voices") or [] if t0 <= t <= t1]
    return round(circular_median(found)) if len(found) >= min_samples else None


def describe(names: list) -> str:
    """"it sounded like: Speech, Door" -- in order, each once; Rocky's own voice left out."""
    named = list(dict.fromkeys(n for n in names if n and n != OWN_VOICE))
    return "it sounded like: " + ", ".join(named) if named else "nothing you could put a name to"


def demo() -> None:
    assert rms(struct.pack("4h", 3, -3, 3, -3)) == 3.0 and rms(b"") == 0.0
    assert describe(["Speech", None, "Speech", OWN_VOICE, "Door"]) == "it sounded like: Speech, Door"
    assert describe([None, OWN_VOICE]) == "nothing you could put a name to"
    assert pcm_command()[-1].endswith("/pcm") and EARS_URL.startswith("http://127.0.0.1:")
    assert bearing(178, 178) == 0 and bearing(269, 178) == 91 and bearing(89, 178) == -89  # Walle: front 178, clockwise
    assert bearing(358, 178) == -180 and bearing(10, 0, ccw=True) == -10
    assert round(circular_median([170, 179, -178, -170, 175])) == 179  # behind him: across the wrap
    assert round(circular_median([-5, 0, 3, 90])) == 3                # one stray reading doesn't drag it
    heard = {"voices": [[10.0, 80], [10.5, 85], [11.0, 82], [20.0, -60], [20.1, -62], [20.2, -61]]}
    assert voice_bearing(9, 12, heard) == 82 and voice_bearing(19.9, 21, heard) == -61
    assert voice_bearing(12, 19, heard) is None and voice_bearing(0, 30, {}) is None


if __name__ == "__main__":
    demo()
    print("hearing: ok")
