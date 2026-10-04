"""openbot-ears: the ONLY process that opens the microphone -- like
openbot-camera for the camera. Reads it with `arecord` (common/mic_stream.py)
and serves, on 127.0.0.1 only (this is the house's live audio):

    /pcm      the live audio: raw 16-bit mono PCM at stt.MIC_RATE, in whole mic
              chunks, from the moment you connect. openbot-wake-listen listens
              to this (common/hearing.pcm_command) instead of opening the mic.
    /hearing  the last HISTORY_S seconds, oldest first: each second's peak
              loudness and what it sounded like -- YAMNet's best guess
              (common/sounds.py), None if unsure, OWN_VOICE while Rocky was
              making sound. openbot-mind asks this (common/hearing.read).

Before this, wake-listen owned the mic and wrote loudness to a file only while
idle -- for as long as a conversation stayed open the mind got nothing and
decided its ears were dead (2026-10-04). Measured on the Pi 5 that day, ears ->
HTTP -> curl -> wake-listen's buffer: 2.7ms median, 6ms worst, per 46ms chunk
(most of it mic_stream.read's 5ms poll, which reading arecord directly had too).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common import health, mic_stream, sounds, speak_client
from common.hearing import CHUNK_FRAMES, EARS_PORT, HISTORY_S, MIC_DEVICE, OWN_VOICE, rms
from common.stt import MIC_RATE, VOSK_RATE, resample_pcm

COMPONENT = "openbot-ears"
RING_CHUNKS = 256      # ~12s of audio kept for /pcm clients; one further behind than this skips ahead
NAME_MIN_SCORE = 0.3   # YAMNet less sure than this: the second goes unnamed


class Ring:
    """The mic's chunks, numbered. Each /pcm client keeps its own place, and
    sockets are written outside the lock -- a stalled client never holds up
    the mic or anyone else; it just skips ahead, by whole chunks."""

    def __init__(self, size: int = RING_CHUNKS) -> None:
        self.size, self.chunks, self.seq = size, [b""] * size, 0  # seq: chunks written so far
        self.cond = threading.Condition()

    def put(self, chunk: bytes) -> None:
        with self.cond:
            self.chunks[self.seq % self.size] = chunk
            self.seq += 1
            self.cond.notify_all()

    def get(self, pos: int, timeout_s: float = 1.0) -> tuple[list[bytes], int]:
        """(the chunks from `pos` on, the next pos) -- waits up to timeout_s for one."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > pos, timeout_s)
            pos = max(pos, self.seq - self.size)
            return [self.chunks[i % self.size] for i in range(pos, self.seq)], self.seq


class Meter:
    """Each second: its peak loudness and, from a worker thread (so naming never
    slows the audio down), what it sounded like."""

    def __init__(self, classifier=None, playing=speak_client.playing) -> None:
        self.levels: list[int] = []
        self.names: list[str | None] = []
        self.count, self.ts = 0, 0.0  # seconds measured so far, and when the last one closed
        self.lock = threading.Lock()
        self.classifier, self.playing = classifier, playing
        self.todo: queue.Queue = queue.Queue(maxsize=2)
        self._peak, self._audio, self._next = 0.0, [], time.time() + 1.0
        if classifier is not None:
            threading.Thread(target=self._namer, daemon=True).start()

    def add(self, chunk: bytes) -> None:
        self._peak = max(self._peak, rms(chunk))
        self._audio.append(chunk)
        if time.time() < self._next:
            return
        own = self.playing()  # Rocky's own voice isn't news -- and "Speech" would claim someone talked
        with self.lock:
            self.levels = (self.levels + [round(self._peak)])[-HISTORY_S:]
            self.names = (self.names + [OWN_VOICE if own else None])[-HISTORY_S:]
            self.count, self.ts = self.count + 1, time.time()
        if self.classifier is not None and not own:
            try:
                self.todo.put_nowait((self.count, b"".join(self._audio)))
            except queue.Full:
                pass  # naming fell behind: this second stays unnamed
        self._peak, self._audio, self._next = 0.0, [], time.time() + 1.0

    def _namer(self) -> None:
        while True:
            count, audio = self.todo.get()
            try:
                name, score = self.classifier.classify(resample_pcm(audio, MIC_RATE, VOSK_RATE))[0]
            except Exception as e:  # one bad second must not stop the naming
                print(f"{COMPONENT}: naming failed: {e}")
                continue
            with self.lock:
                i = len(self.names) - 1 - (self.count - count)
                if 0 <= i < len(self.names) and score >= NAME_MIN_SCORE:
                    self.names[i] = name

    def snapshot(self) -> dict:
        with self.lock:
            return {"levels": list(self.levels), "sounds": list(self.names), "ts": self.ts}


def make_handler(ring: Ring, meter: Meter) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet: the mind asks every few seconds
            pass

        def do_GET(self) -> None:
            if self.path == "/hearing":
                body = json.dumps(meter.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/pcm":
                self.send_response(200)
                self.send_header("Content-Type", f"audio/L16; rate={MIC_RATE}; channels=1")
                self.end_headers()
                pos = ring.seq  # from now on
                try:
                    while True:
                        chunks, pos = ring.get(pos)
                        for chunk in chunks:
                            self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the listener went away
            else:
                self.send_error(404)
    return Handler


def main() -> None:
    health.start_watchdog(COMPONENT, 30.0, 10.0)
    mic = mic_stream.ArecordStream(mic_stream.resolve_device(MIC_DEVICE), rate=MIC_RATE, chunk_frames=CHUNK_FRAMES)
    mic.start_stream()
    try:
        classifier = sounds.Classifier()
    except Exception as e:  # optional, like the camera's detectors: no model, no names
        print(f"{COMPONENT}: sound names off ({e})")
        classifier = None
    ring, meter = Ring(), Meter(classifier)
    server = ThreadingHTTPServer(("127.0.0.1", EARS_PORT), make_handler(ring, meter))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"{COMPONENT}: listening to the mic, serving 127.0.0.1:{EARS_PORT} (/pcm, /hearing)")
    while True:
        chunk = mic.read(CHUNK_FRAMES)
        if not chunk:
            if not mic.is_running():
                raise SystemExit("arecord exited -- mic unavailable")  # systemd restarts us
            time.sleep(0.05)
            continue
        ring.put(chunk)
        meter.add(chunk)
        health.record_success(COMPONENT, min_interval_s=5.0)


def demo() -> None:
    import urllib.request
    # Ring: two clients get the same whole chunks; a stalled one skips ahead without holding anyone up
    ring = Ring(size=4)
    a = b = ring.seq
    for i in range(3):
        ring.put(bytes([i]) * 4)
    got_a, a = ring.get(a)
    assert got_a == [b"\x00" * 4, b"\x01" * 4, b"\x02" * 4] and a == 3
    for i in range(3, 9):
        ring.put(bytes([i]) * 4)
    got_b, b = ring.get(b)  # b stalled for 9 chunks: only the newest 4 are kept
    assert got_b == [bytes([i]) * 4 for i in range(5, 9)] and b == 9
    assert ring.get(9, timeout_s=0.05) == ([], 9)  # nothing new: returns after the timeout

    # Meter: per-second peaks; Rocky's own voice marked; names land on the right second
    class Fake:
        def classify(self, pcm16k):
            return [("Knock", 0.8)]
    speaking = [False]
    meter = Meter(Fake(), playing=lambda: speaking[0])
    loud = (b"\x10\x27" * 64)  # 10000
    for chunk, own in ((b"\x00\x00" * 64, False), (loud, False), (loud, True)):
        speaking[0] = own
        meter._next = 0.0  # close a second on every chunk
        meter.add(chunk)
    deadline = time.time() + 2
    while meter.snapshot()["sounds"][:2] != ["Knock", "Knock"] and time.time() < deadline:
        time.sleep(0.01)
    snap = meter.snapshot()
    assert snap["levels"] == [0, 10000, 10000] and snap["sounds"] == ["Knock", "Knock", OWN_VOICE], snap

    # HTTP: /hearing's shape, and two /pcm clients receiving identical, chunk-aligned audio
    ring = Ring()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ring, meter))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    assert json.loads(urllib.request.urlopen(f"{url}/hearing", timeout=2).read())["sounds"][-1] == OWN_VOICE
    clients = [urllib.request.urlopen(f"{url}/pcm", timeout=2) for _ in range(2)]
    time.sleep(0.2)  # both connected before the audio starts
    for i in range(5):
        ring.put(bytes([i]) * 8)
    heard = [c.read(40) for c in clients]
    assert heard[0] == heard[1] == b"".join(bytes([i]) * 8 for i in range(5)), heard
    server.shutdown()
    print("ears: ok")


if __name__ == "__main__":
    import sys
    demo() if "--check" in sys.argv else main()
