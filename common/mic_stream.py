"""Microphone capture via an `arecord` subprocess, not PortAudio (PyAudio/
sounddevice).

PortAudio's ALSA backend has been observed silently dropping a large
fraction of samples on this class of C-Media USB mic -- invisible on every
offline metric (the recorded WAV looks clean, because what's missing is
missing, not corrupted or clipped) and only caught by a live loopback
check. `arecord` on the same device does not exhibit this. Ported from
SPARK's src/pxh/mic_stream.py, which carries the original diagnosis.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque

# ~9s of backlog at the default 2048-frame/44100Hz chunk size -- generous
# enough to survive an STT+LLM round without dropping, small enough that a
# genuinely stuck consumer doesn't grow unbounded.
BACKLOG_CHUNKS = 200


def resolve_device(name_substring: str) -> str:
    """Find the ALSA `plughw:X,Y` device whose `arecord -l` listing contains
    `name_substring`. Raises if not found -- a silently-wrong device is
    worse than a startup crash."""
    out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("card") and name_substring.lower() in line.lower():
            card = line.split("card", 1)[1].split(":", 1)[0].strip()
            dev = line.split("device", 2)[1].split(":", 1)[0].strip()
            return f"plughw:{card},{dev}"  # plug: converts rate/channels for mics that only do 48kHz or stereo
    raise RuntimeError(f"no arecord device matching {name_substring!r}")


class ArecordStream:
    """Mirrors the small subset of a PyAudio/sounddevice stream this
    project's callers use: start_stream()/read(n)/stop_stream(). A reader
    thread drains arecord's stdout pipe into a bounded deque so a slow
    consumer (STT, an LLM call) doesn't block arecord and force an ALSA
    overrun -- the pipe itself only buffers ~64KB (~0.37s at 44100Hz/
    16-bit/mono), far less than one STT+LLM round takes.
    """

    def __init__(self, device: str, rate: int = 44100, channels: int = 1, chunk_frames: int = 2048):
        self.device = device
        self.rate = rate
        self.channels = channels
        self.chunk_bytes = chunk_frames * 2 * channels
        self._proc: subprocess.Popen | None = None
        self._buf: deque[bytes] = deque()
        self.dropped_chunks = 0

    def start_stream(self) -> None:
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
             "-c", str(self.channels), "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        def _drain() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            while True:
                chunk = self._proc.stdout.read(self.chunk_bytes)
                if not chunk:
                    return
                if len(self._buf) > BACKLOG_CHUNKS:
                    self._buf.popleft()
                    self.dropped_chunks += 1
                self._buf.append(chunk)

        threading.Thread(target=_drain, daemon=True).start()

    def read(self, n_frames: int, exception_on_overflow: bool = False) -> bytes:
        while not self._buf:
            if self._proc is not None and self._proc.poll() is not None:
                return b""
            time.sleep(0.005)
        return self._buf.popleft()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def flush(self) -> int:
        n = len(self._buf)
        self._buf.clear()
        return n

    def stop_stream(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None


def demo() -> None:
    # No live audio hardware in this check -- confirms resolve_device()
    # raises cleanly rather than silently returning a wrong device.
    try:
        resolve_device("definitely-not-a-real-device-xyz")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    except FileNotFoundError:
        pass  # arecord not installed on this (dev) machine -- acceptable


if __name__ == "__main__":
    demo()
    print("mic_stream: ok")
