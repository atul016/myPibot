"""openbot-camera: the ONLY process that opens the camera. Runs
`rpicam-vid` as a continuous MJPEG stream (not vilib -- its camera thread
has a documented live deadlock) and serves it over HTTP:

    /mjpg          live video (multipart MJPEG -- an <img> tag plays it)
    /snapshot.jpg  latest frame; ?after=<unix ts> waits (<=2s) for a newer one

Everything else that wants an image -- openbot-mind's looks, the
dashboard's Camera tab -- asks this service instead of opening the camera:
the Pi's camera can only be open in one process at a time.

Same shape as openbot-speak for audio and openbot-alive for motors: one
owner per piece of hardware, everyone else asks. Pure stdlib.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common import faces, health, state

COMPONENT = "openbot-camera"
PORT = int(os.environ.get("OPENBOT_CAMERA_PORT", "9000"))
FPS = int(os.environ.get("OPENBOT_CAMERA_FPS", "10"))
FACE_INTERVAL_S = 0.2  # ~5 face checks a second (detect ~40ms + recognize ~25ms per face)
# This robot's ov5647 is the NoIR variant -- see common/vision.py.
TUNING_FILE = os.environ.get("OPENBOT_CAMERA_TUNING", "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json")
# Any command that writes an MJPEG stream to stdout. Default: the Pi camera.
# A USB webcam: "ffmpeg -loglevel error -f v4l2 -input_format mjpeg -video_size 640x480
# -framerate 10 -i /dev/video0 -c:v copy -f mjpeg -"
CAMERA_CMD = os.environ.get("OPENBOT_CAMERA_CMD", "")

_frame: bytes = b""
_frame_ts = 0.0
_new_frame = threading.Condition()


def split_frames(buf: bytes) -> tuple[list[bytes], bytes]:
    """Complete JPEGs (SOI ff d8 .. EOI ff d9) found in buf, plus the
    unconsumed tail. Safe on MJPEG: inside entropy-coded data every 0xff
    is byte-stuffed, so ff d9 only ever appears as a real end-of-image."""
    frames = []
    while True:
        start = buf.find(b"\xff\xd8")
        if start < 0:
            return frames, b""
        end = buf.find(b"\xff\xd9", start + 2)
        if end < 0:
            return frames, buf[start:]
        frames.append(buf[start:end + 2])
        buf = buf[end + 2:]


def _asleep() -> bool:
    return bool(state.load_session().get("asleep"))


def _capture_loop() -> None:
    """Streams while awake. "Go to sleep" turns the camera OFF (rpicam-vid
    stopped, camera released) until Rocky is woken."""
    global _frame, _frame_ts
    tuning = ["--tuning-file", TUNING_FILE] if os.path.exists(TUNING_FILE) else []
    while True:
        if _asleep():
            with _new_frame:
                _frame = b""  # no stale picture of the room while "asleep"
            health.record_success(COMPONENT, min_interval_s=5.0)  # asleep on purpose, not hung
            time.sleep(2)
            continue
        cmd = shlex.split(CAMERA_CMD) if CAMERA_CMD else \
            ["rpicam-vid", "-n", "-t", "0", "--codec", "mjpeg", "--width", "640", "--height", "480",
             "--framerate", str(FPS), "-q", "70", *tuning, "-o", "-"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as e:  # no such program: no camera here -- keep the service up, just eyeless
            print(f"{COMPONENT}: can't run {cmd[0]!r} ({e}) -- no camera; set OPENBOT_CAMERA_CMD for a USB webcam")
            health.record_success(COMPONENT, min_interval_s=5.0)  # missing on purpose, not hung
            time.sleep(60)
            continue
        buf = b""
        next_sleep_check = time.time() + 2
        assert proc.stdout is not None
        while chunk := proc.stdout.read1(65536):
            if time.time() >= next_sleep_check:
                next_sleep_check = time.time() + 2
                if _asleep():
                    proc.terminate()
                    break
            frames, buf = split_frames(buf + chunk)
            if frames:
                with _new_frame:
                    _frame, _frame_ts = frames[-1], time.time()
                    _new_frame.notify_all()
                health.record_success(COMPONENT, min_interval_s=5.0)
        proc.wait()
        if _asleep():
            print(f"{COMPONENT}: asleep -- camera off")
            continue
        health.record_failure(COMPONENT, f"camera command exited ({proc.returncode})")
        time.sleep(2)  # camera unplugged/busy -- retry, don't spin


def _face_loop() -> None:
    """Who's in frame, ~5x a second -> state/faces.json (common/faces.py).
    Optional: if the models aren't downloaded, the stream works without it."""
    try:
        engine = faces.FaceEngine()
    except Exception as e:
        print(f"{COMPONENT}: face detection off ({e})")
        return
    last = 0.0
    while True:
        with _new_frame:
            _new_frame.wait_for(lambda: _frame_ts > last, timeout=5.0)
            frame, last = _frame, _frame_ts
        if frame:
            try:
                faces.publish(engine.analyze(frame))
            except Exception as e:  # one bad frame must not stop face tracking for good
                print(f"{COMPONENT}: face analysis failed: {e}")
        time.sleep(FACE_INTERVAL_S)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet: the dashboard polls
        pass

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/snapshot.jpg":
            after = float(parse_qs(url.query).get("after", ["0"])[0])
            with _new_frame:
                _new_frame.wait_for(lambda: _frame_ts > after, timeout=2.0)
                frame, ts = _frame, _frame_ts
            if not frame or ts <= after:
                self.send_error(503, "no fresh frame")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("X-Frame-Ts", str(ts))
            self.end_headers()
            self.wfile.write(frame)
        elif url.path == "/mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = 0.0
            try:
                while True:
                    with _new_frame:
                        _new_frame.wait_for(lambda: _frame_ts > last, timeout=5.0)
                        frame, last = _frame, _frame_ts
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab
        else:
            self.send_error(404)


def main() -> None:
    health.start_watchdog(COMPONENT, 60.0, 10.0)
    threading.Thread(target=_capture_loop, daemon=True).start()
    threading.Thread(target=_face_loop, daemon=True).start()
    print(f"{COMPONENT}: streaming on :{PORT} (/mjpg, /snapshot.jpg)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def demo() -> None:
    a, b = b"\xff\xd8AAA\xff\xd9", b"\xff\xd8B\xff\x00B\xff\xd9"
    frames, rest = split_frames(b"junk" + a + b + b"\xff\xd8partial")
    assert frames == [a, b] and rest == b"\xff\xd8partial"
    assert split_frames(b"") == ([], b"") and split_frames(b"no jpeg here") == ([], b"")


if __name__ == "__main__":
    import sys
    demo() if "--check" in sys.argv else main()
