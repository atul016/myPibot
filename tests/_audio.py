"""Shared helpers for tests/: synthetic speech, room noise, and a fake mic
that plays audio at real-time pace (so PAUSE_S and other timers mean what
they do live). Needs espeak + sox (installed by setup.sh).
"""
from __future__ import annotations

import os
import random
import struct
import subprocess
import tempfile
import time

RATE = 44100


def isolate_state() -> str:
    """Point OpenBot's state/ at a throwaway folder. Call BEFORE importing any
    OpenBot module (state.STATE_DIR is read at import time) -- a test must
    never write into Rocky's real memory, journal or session."""
    path = tempfile.mkdtemp(prefix="openbot-test-")
    os.environ["OPENBOT_STATE_DIR"] = path
    return path


def say(text: str, rate: int = RATE) -> bytes:
    """`text` spoken by espeak, as raw 16-bit mono PCM at `rate`."""
    wav = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(["espeak", "-w", wav, text], check=True)
        return subprocess.run(["sox", wav, "-r", str(rate), "-c", "1", "-b", "16", "-e", "signed-integer",
                               "-t", "raw", "-"], capture_output=True, check=True).stdout
    finally:
        if os.path.exists(wav):
            os.remove(wav)


def noise(seconds: float, amplitude: int = 50) -> bytes:
    """Quiet-room hiss (RMS ~30, like this robot's mic in a quiet room)."""
    n = int(RATE * seconds)
    return struct.pack(f"<{n}h", *(random.randint(-amplitude, amplitude) for _ in range(n)))


def seconds(pcm: bytes) -> float:
    return len(pcm) / (2 * RATE)


class FakeMic:
    """Stands in for mic_stream.ArecordStream: plays `audio`, then endless hiss."""
    chunk_bytes, rate = 4096, RATE

    def __init__(self, audio: bytes):
        self.audio, self.pos, self.t0 = audio, 0, time.time()

    def read(self, n_frames: int) -> bytes:
        due = self.t0 + self.pos / (2 * RATE)
        if due > time.time():
            time.sleep(due - time.time())
        chunk = self.audio[self.pos:self.pos + self.chunk_bytes]
        self.pos += self.chunk_bytes
        return chunk or noise(0.05)

    def flush(self) -> int:
        return 0
