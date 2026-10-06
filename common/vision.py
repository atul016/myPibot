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

from . import cognition, sensors
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
# Mean brightness differing this much from the last look the same way: the light changed (a lamp, the
# sun), so what the model calls new objects is reported as light, not as things appearing.
LIGHT_CHANGE = 40
DARK_SCENE = "It's too dark to see anything."
# A look is compared only with the last one from about the same ROOM direction (where the body faces
# + where the head points), in steps this wide: the head follows faces (up to 60 deg) and glances
# about (+-8), so "ahead" was often a different view -- "the room layout shifted again without me
# moving" came up 55 times in one day.
VIEW_STEP_DEG = 20
# Moved (picked up, driven): it doesn't know which way its body faces until a look matches a view in
# its map; after this many looks that match nothing, it's somewhere new -- a fresh map.
BEARING_TRIES = 3

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


def view_key(pan: float, tilt: float) -> str:
    return f"{round(pan / VIEW_STEP_DEG) * VIEW_STEP_DEG:+d},{round(tilt / VIEW_STEP_DEG) * VIEW_STEP_DEG:+d}"


def head_words(pan: float, tilt: float) -> str:
    """"head turned 30 deg to your right" -- pan + = right, tilt + = up."""
    parts = ([f"{abs(pan):.0f} degrees to your {'right' if pan > 0 else 'left'}"] if abs(pan) >= 5 else []) + \
            ([f"{abs(tilt):.0f} degrees {'up' if tilt > 0 else 'down'}"] if abs(tilt) >= 5 else [])
    return "head turned " + " and ".join(parts) if parts else "head pointing straight ahead"


def lost_bearings() -> None:
    """Picked up, set down, or it drove: it no longer knows which way its body
    faces. The map of views stays -- the next looks try to recognize one."""
    record = last_look()
    if not record.get("lost"):
        atomic_write(VISION_PATH, json.dumps({**record, "lost": {"since": time.time(), "tries": 0}}))


def turn_by(degrees: float) -> None:
    """The body turned in place this far (the IMU felt it; + = right): which way it faces is
    still known -- no need to recognize a view again."""
    record = last_look()
    if record and not record.get("lost"):
        heading = (record.get("heading", 0.0) + degrees + 180) % 360 - 180
        atomic_write(VISION_PATH, json.dumps({**record, "heading": round(heading, 1)}))


def facing_words() -> str:
    """Which way the body faces, for the mind's prompt."""
    record = last_look()
    if record.get("lost"):
        return "not sure -- you were moved and haven't recognized anything around you yet"
    h = record.get("heading", 0.0)
    return ("the way you faced when you first mapped this room" if abs(h) < 10 else
            f"turned {abs(h):.0f} degrees to the {'right' if h > 0 else 'left'} of how you faced when you first "
            "mapped this room")


def _find_bearings(base_url: str, model: str, scene: str, head: tuple[float, float],
                   views: dict[str, str]) -> float | None:
    """The body's heading, if `scene` (seen with the head at `head`) is one of the
    views in the map -- the LLM matches them by what's in them. None: no match."""
    options = {k: v for k, v in views.items() if v and v != DARK_SCENE}
    if not options:
        return None
    listing = "\n".join(f"- {k}: {v}" for k, v in options.items())
    result = cognition.ask(
        base_url, model,
        "You are a small robot. Someone just moved you, so you don't know which way you're facing. Earlier you "
        f"saw these views (label: what you saw):\n{listing}\n\nNow you see: \"{scene}\". Is this the same view as "
        "one of them -- the same things, the same part of the room? Answer with its label, or \"none\" if it "
        "doesn't clearly match one.",
        json_schema={"type": "object", "properties": {"same_as": {"type": "string", "enum": [*options, "none"]}},
                     "required": ["same_as"]},
        timeout_s=30.0, num_predict=40)
    if result.status != cognition.AVAILABLE:
        return None
    try:
        same = json.loads(result.text).get("same_as")
    except (json.JSONDecodeError, AttributeError):
        return None
    if same not in options:
        return None
    return (float(same.split(",")[0]) - head[0] + 180) % 360 - 180

def look(base_url: str, model: str, direction: str = "ahead",
         head: tuple[float, float] | None = None) -> dict[str, Any] | None:
    """Capture + describe. Returns {scene, changed, kind, what_changed, direction, head, ts}
    or None if the camera or the LLM failed. `changed` compares against the last
    look from about the same head angle (`head`: (pan, tilt); default: where
    openbot-alive says the head is) -- a head turn isn't a scene change."""
    jpeg = capture()
    if jpeg is None:
        return None
    if head is None:
        h = sensors.read().get("head") or {}
        head = (h.get("pan", 0.0), h.get("tilt", 0.0))
    record = last_look()
    heading, lost = record.get("heading", 0.0), record.get("lost")
    key = view_key(heading + head[0], head[1])
    prev = "" if lost else record.get("views", {}).get(key, "")
    light = brightness(jpeg)
    if light < DARK_BRIGHTNESS:
        changed = bool(prev) and prev != DARK_SCENE
        return _save({"scene": DARK_SCENE, "changed": changed, "kind": "lights_off",
                      "what_changed": "the lights went out" if changed else "", "direction": direction, "head": head},
                     heading, lost, light)
    from . import objects  # deferred: objects -> faces is heavier than this module needs at import
    detected = objects.names(objects.read())
    prompt = ("You are a small home robot looking through your camera "
              f"({head_words(*head)}). Describe what you see in one short, concrete sentence. "
              + (f"Your object detector sees {detected} -- trust it over guesses. " if detected else "") +
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
                      "what_changed": "the lights came on", "direction": direction, "head": head}, heading, lost, light)
    out = {"scene": str(data.get("scene", "")), "changed": bool(data.get("changed")) and bool(prev),
           "kind": "scene", "what_changed": str(data.get("what_changed", "")), "direction": direction, "head": head}
    lit = light_change(record.get("view_light", {}).get(key) if prev else None, light)
    if lit:
        out.update(changed=True, kind=lit, what_changed=f"the light {'came up' if lit == 'lights_on' else 'went down'} "
                   "-- things may only look different, not be different")
    if lost:
        found = _find_bearings(base_url, model, out["scene"], head, record.get("views", {}))
        if found is not None:
            heading, lost = found, None
            out["bearings"] = (f"you recognize this view: you're now facing {facing_words_for(heading)} -- "
                               "you were turned, the room didn't change")
        elif lost["tries"] + 1 >= BEARING_TRIES:
            heading, lost = 0.0, None
            record["views"] = {}
            atomic_write(VISION_PATH, json.dumps(record))
            out["bearings"] = "you don't recognize any of this -- somewhere new, so you start a fresh map of it"
        else:
            lost = {**lost, "tries": lost["tries"] + 1}
    return _save(out, heading, lost, light)


def light_change(before: float | None, now: float) -> str | None:
    """"lights_on"/"lights_off" when the frame is much brighter/darker than the last look the same way."""
    if before is None or abs(now - before) <= LIGHT_CHANGE:
        return None
    return "lights_on" if now > before else "lights_off"


def facing_words_for(heading: float) -> str:
    return ("the way you faced when you first mapped this room" if abs(heading) < 10 else
            f"{abs(heading):.0f} degrees to the {'right' if heading > 0 else 'left'} of how you faced at first")


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


def _save(out: dict[str, Any], heading: float = 0.0, lost: dict | None = None,
          light: float | None = None) -> dict[str, Any]:
    """by_direction: the last scene per named look (the dashboard's Camera tab);
    views: the map -- per room direction (heading + head pan), what the next look
    that way is compared with (view_light: how bright it was); not added to while
    lost (which way is unknown)."""
    out["ts"] = time.time()
    record = last_look()
    by_direction = {**record.get("by_direction", {}), out["direction"]: out["scene"]}
    views, view_light = record.get("views", {}), record.get("view_light", {})
    if not lost:
        pan, tilt = out.get("head", (0, 0))
        views = {**views, view_key(heading + pan, tilt): out["scene"]}
        if light is not None:
            view_light = {**view_light, view_key(heading + pan, tilt): round(light, 1)}
    atomic_write(VISION_PATH, json.dumps({**out, "by_direction": by_direction, "views": views,
                                          "view_light": view_light, "heading": heading, "lost": lost}))
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
        # a glance (+-8 deg) is the same view; following a face 30 deg right, or "look left", is not
        assert view_key(0, 0) == view_key(8, -5) == "+0,+0" and view_key(30, 0) != view_key(0, 0) != view_key(-45, 0)
        _save({"scene": "a desk", "changed": False, "what_changed": "", "direction": "ahead", "head": (30, 0)})
        assert last_look()["views"][view_key(30, 0)] == "a desk" and facing_words().startswith("the way you faced")
        turn_by(100)  # turned in place, felt by the IMU: still known
        assert last_look()["heading"] == 100.0 and facing_words().startswith("turned 100 degrees to the right")
        turn_by(-100)
        lost_bearings()  # picked up and turned: which way it faces is unknown -- but the map stays
        assert last_look()["lost"] and last_look()["views"]["+40,+0"] == "a desk" and facing_words().startswith("not sure")
        real_ask = cognition.ask
        try:  # the bed it used to see 40 deg to its left is now straight ahead: turned 40 deg left
            cognition.ask = lambda *a, **k: cognition.CognitionResult(cognition.AVAILABLE, '{"same_as": "-40,+0"}')
            assert _find_bearings("u", "m", "a messy bed", (0, 0), {"-40,+0": "a messy bed", "+40,+0": "a desk"}) == -40
            cognition.ask = lambda *a, **k: cognition.CognitionResult(cognition.AVAILABLE, '{"same_as": "none"}')
            assert _find_bearings("u", "m", "a kitchen", (0, 0), {"-40,+0": "a messy bed"}) is None
        finally:
            cognition.ask = real_ask
        assert facing_words_for(-40) == "40 degrees to the left of how you faced at first"
        assert head_words(0, 3) == "head pointing straight ahead"
        assert light_change(None, 120) is None and light_change(100, 120) is None  # first look / same light
        assert light_change(60, 150) == "lights_on" and light_change(150, 60) == "lights_off"
        _save({"scene": "a lamp", "changed": False, "what_changed": "", "direction": "ahead", "head": (0, 0)}, light=150.0)
        assert last_look()["view_light"][view_key(0, 0)] == 150.0
        assert head_words(-45, 0) == "head turned 45 degrees to your left"
        assert head_words(30, 10) == "head turned 30 degrees to your right and 10 degrees up"
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
