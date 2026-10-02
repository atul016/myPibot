"""Eyes: grab the latest frame from openbot-camera (the camera's sole
owner, streaming MJPEG -- falls back to a one-shot rpicam-still if that
service is down; never vilib, whose camera thread has a documented live
deadlock) -> the vision-capable LLM describes the
scene and says whether it changed since the last look. Latest look is
kept in state/vision.json so the conversation prompt and the dashboard
can read what the robot currently sees.

The Pi's camera opens in one process at a time, so openbot-camera holds
it and everyone else asks it for frames.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from . import cognition
from .state import STATE_DIR, atomic_write

VISION_PATH = STATE_DIR / "vision.json"
SNAPSHOT_PATH = "/tmp/openbot-snap.jpg"  # latest frame -- also served by the dashboard's Camera tab
CAPTURE_TIMEOUT_S = 10.0
CAMERA_URL = os.environ.get("OPENBOT_CAMERA_URL", "http://127.0.0.1:9000")
# This robot's ov5647 is the NoIR variant (no IR filter): the default tuning
# renders the room green-cyan and dark things purple -- confirmed side by
# side 2026-10-02. The NoIR tuning gives neutral whites. Pi 5 (pisp) path;
# missing file -> libcamera's default.
TUNING_FILE = os.environ.get("OPENBOT_CAMERA_TUNING", "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json")
# Mean pixel brightness (0-255) below which the frame is "too dark to see"
# and never goes to the LLM -- a vision model shown near-black noise
# invents colors ("reddish-brown"... then "purple"), which read as fake
# scene changes all night. Calibration knob for this camera/room.
DARK_BRIGHTNESS = 40  # measured: pitch-dark room reads ~25 at max auto-gain
DARK_SCENE = "It's too dark to see anything."

_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "changed": {"type": "boolean"},
        "what_changed": {"type": "string"},
    },
    "required": ["scene", "changed", "what_changed"],
}


def capture(path: str = SNAPSHOT_PATH) -> bytes | None:
    """640x480 JPEG taken AFTER this call started (so a head turn that just
    finished is in it), or None if the camera is missing/busy/hung. Also
    saved to `path`."""
    import requests
    try:
        resp = requests.get(f"{CAMERA_URL}/snapshot.jpg", params={"after": time.time()}, timeout=4)
        if resp.ok and resp.content:
            with open(path, "wb") as f:
                f.write(resp.content)
            return resp.content
    except requests.RequestException:
        pass  # camera service down -> open the camera ourselves
    tuning = ["--tuning-file", TUNING_FILE] if TUNING_FILE and os.path.exists(TUNING_FILE) else []
    try:
        subprocess.run(["rpicam-still", "-n", "-t", "300", "--width", "640", "--height", "480",
                        "-q", "70", *tuning, "-o", path],
                       capture_output=True, timeout=CAPTURE_TIMEOUT_S, check=True)
        with open(path, "rb") as f:
            return f.read()
    except (OSError, subprocess.SubprocessError):
        return None


def brightness(jpeg: bytes) -> float:
    import io
    from PIL import Image, ImageStat
    return ImageStat.Stat(Image.open(io.BytesIO(jpeg)).convert("L")).mean[0]


def last_look() -> dict[str, Any]:
    try:
        return json.loads(VISION_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def look(base_url: str, model: str, direction: str = "ahead") -> dict[str, Any] | None:
    """Capture + describe. Returns {scene, changed, kind, what_changed, direction, ts}
    or None if the camera or the LLM failed. `changed` compares against the
    last look in the SAME direction -- a head turn isn't a scene change."""
    jpeg = capture()
    if jpeg is None:
        return None
    prev = last_look().get("by_direction", {}).get(direction, "")
    if brightness(jpeg) < DARK_BRIGHTNESS:
        changed = bool(prev) and prev != DARK_SCENE
        return _save({"scene": DARK_SCENE, "changed": changed, "kind": "lights_off",
                      "what_changed": "the lights went out" if changed else "", "direction": direction})
    prompt = ("You are a small desk robot looking through your camera "
              f"(head turned {direction}). Describe what you see in one short, concrete sentence. "
              "Your camera has no infrared filter, so black or dark things (clothes, hair, shadows) "
              "show up purple or magenta -- treat that tint as dark, don't call things purple because of it. "
              + (f"Last time you looked this way you saw: \"{prev}\". Say whether anything meaningful "
                 "changed (people, objects, lighting) -- ignore tiny camera noise."
                 if prev else "This is your first look this way, so changed is false."))
    result = cognition.ask(base_url, model, prompt, json_schema=_SCHEMA, image_jpeg=jpeg,
                           timeout_s=45.0, num_predict=200)
    if result.status != cognition.AVAILABLE:
        return None
    try:
        data = json.loads(result.text)
    except json.JSONDecodeError:
        return None
    if prev == DARK_SCENE:  # the model can't compare against a frame it never saw
        return _save({"scene": str(data.get("scene", "")), "changed": True, "kind": "lights_on",
                      "what_changed": "the lights came on", "direction": direction})
    return _save({"scene": str(data.get("scene", "")), "changed": bool(data.get("changed")) and bool(prev),
                  "kind": "scene", "what_changed": str(data.get("what_changed", "")), "direction": direction})


_EXTENT = {"type": "object", "properties": {"visible": {"type": "boolean"}, "left": {"type": "number"},
                                            "right": {"type": "number"}, "top": {"type": "number"},
                                            "bottom": {"type": "number"}},
           "required": ["visible", "left", "right", "top", "bottom"]}


def locate(base_url: str, model: str, target: str, jpeg: bytes) -> tuple[float, float, float, float] | None:
    """Where `target` is in the frame: its box (left, top, right, bottom) as
    fractions of the image (0,0 = top left) -- or None if it isn't visible. With the head
    tilted down, an object's bottom edge low in the frame means it's close,
    whatever its size. For driving to it (movement/navigate.py)."""
    result = cognition.ask(base_url, model,
                           f"Is there a {target} in this camera image? If so, give its bounding box as fractions of "
                           "the image: left and right (0.0 = left side, 1.0 = right side), top and bottom (0.0 = top, "
                           "1.0 = bottom). If not, visible=false. Dark things may look purple (no IR filter).",
                           json_schema=_EXTENT, image_jpeg=jpeg, timeout_s=10.0, num_predict=80)
    if result.status != cognition.AVAILABLE:
        return None
    try:
        d = json.loads(result.text)
        left, right = sorted((float(d["left"]), float(d["right"])))
        top, bottom = sorted((float(d["top"]), float(d["bottom"])))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not d.get("visible") or not (0.0 <= left < right <= 1.0) or not 0.0 <= top < bottom <= 1.0:
        return None
    return left, top, right, bottom


def list_objects(base_url: str, model: str, jpeg: bytes) -> list[str]:
    """Short names of the distinct things in view (for exploring)."""
    result = cognition.ask(base_url, model, "List the distinct objects you can see (short names, max 8).",
                           image_jpeg=jpeg, timeout_s=10.0, num_predict=100,
                           json_schema={"type": "object", "properties": {"objects": {"type": "array", "items": {"type": "string"}}},
                                        "required": ["objects"]})
    try:
        return [str(x)[:40] for x in json.loads(result.text).get("objects", [])][:8] if result.status == cognition.AVAILABLE else []
    except (json.JSONDecodeError, AttributeError):
        return []


def _save(out: dict[str, Any]) -> dict[str, Any]:
    out["ts"] = time.time()
    direction = out["direction"]
    record = last_look()
    by_direction = record.get("by_direction", {})
    by_direction[direction] = out["scene"]
    atomic_write(VISION_PATH, json.dumps({**out, "by_direction": by_direction}))
    return out


def demo() -> None:
    import shutil, tempfile
    from pathlib import Path
    global VISION_PATH
    orig = VISION_PATH
    test_dir = Path(tempfile.mkdtemp())
    VISION_PATH = test_dir / "vision.json"
    try:
        assert last_look() == {}
        if capture() is None:  # no camera on a dev machine -- look() must fail soft, not raise
            assert look("http://192.0.2.1:1/v1", "m") is None
        first = _save({"scene": DARK_SCENE, "changed": False, "what_changed": "", "direction": "ahead"})
        assert last_look()["by_direction"] == {"ahead": DARK_SCENE} and first["ts"]
        try:
            import io
            from PIL import Image
            buf = io.BytesIO(); Image.new("L", (8, 8), 5).save(buf, "JPEG")
            assert brightness(buf.getvalue()) < DARK_BRIGHTNESS
        except ImportError:
            pass  # dev machine without PIL
    finally:
        VISION_PATH = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("vision: ok")
