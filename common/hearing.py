"""Hearing, as the rest of OpenBot sees it: openbot-ears (services/ears.py) is the
microphone's only owner; everyone else asks it, the way openbot-camera serves
the camera. This module is the asking side, plus the audio format both sides
must agree on.

  read()          the mind's question: each of the last HISTORY_S seconds' peak
                  loudness and what it sounded like -- None if ears isn't answering
  pcm_command()   what openbot-wake-listen runs instead of arecord: the live audio
"""
from __future__ import annotations

import json
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


def describe(names: list) -> str:
    """"it sounded like: Speech, Door" -- in order, each once; Rocky's own voice left out."""
    named = list(dict.fromkeys(n for n in names if n and n != OWN_VOICE))
    return "it sounded like: " + ", ".join(named) if named else "nothing you could put a name to"


def demo() -> None:
    assert rms(struct.pack("4h", 3, -3, 3, -3)) == 3.0 and rms(b"") == 0.0
    assert describe(["Speech", None, "Speech", OWN_VOICE, "Door"]) == "it sounded like: Speech, Door"
    assert describe([None, OWN_VOICE]) == "nothing you could put a name to"
    assert pcm_command()[-1].endswith("/pcm") and EARS_URL.startswith("http://127.0.0.1:")


if __name__ == "__main__":
    demo()
    print("hearing: ok")
