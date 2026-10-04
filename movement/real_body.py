"""navigate.Body on the real PiCar-X -- the same interface sim/world.SimBody
implements, so the driving proven in the simulator runs unchanged here.

Lives inside openbot-alive (the only process that owns Picarx). Angles in
navigate are + = LEFT; picarx's steering and head pan are + = RIGHT, so
they're flipped here. Targets are found by the Mac's vision model in a
fresh frame from openbot-camera (common/vision.locate); while driving, an
OpenCV tracker started from that box follows the target at ~10Hz.

Calibration knobs (measure on the real car, then mirror them in sim/world.py):
CAMERA_FOV, FLOOR_ORDER, SERVO_SETTLE_S.
"""
from __future__ import annotations

import threading
import time

import requests

from common import objects, vision
from movement.navigate import Sighting
from movement.triggers import POISONED_ADC

CAMERA_FOV = 54.0               # degrees across the 640px frame (ov5647, as in the sim)
FLOOR_ORDER = (0, 1, 2)         # grayscale indices for (left, centre, right) -- verify by lifting one side
SERVO_SETTLE_S = 0.35
# Head tilt while driving: looking DOWN at the surface ahead. Level (0) looks over
# small things on a desk -- found live: the camera saw the ceiling light, not the
# mouse 25cm ahead. picarx tilt: + = up.
DRIVE_TILT = -20
TICK_S = 0.05                   # same 20Hz as the sim and the safety loop
PERSON_FRESH_S = 0.6            # follow: a detection older than this isn't where they are now
DETECT_S = 0.15                 # a frame from after the head settled is published about this much later


def _new_tracker():
    """Best OpenCV tracker this build has: CSRT follows size changes (the box
    grows as we close in), KCF/MIL are the fallbacks (no opencv-contrib)."""
    import cv2
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        for mod in (cv2, getattr(cv2, "legacy", None)):
            make = getattr(mod, name, None)
            if make:
                return make()
    raise RuntimeError("no OpenCV tracker in this cv2 build")


def _decode(jpeg: bytes):
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)  # half size: faster tracking


class RealBody:
    def __init__(self, car, monitor, cfg, cancel: threading.Event):
        self.car, self.monitor, self.cfg, self.cancel = car, monitor, cfg, cancel
        self.pan = 0.0
        self.tilt = DRIVE_TILT          # navigate.follow looks up instead (alive sets FOLLOW_TILT)
        self.settled = 0.0              # when the head last stopped moving
        self.tracker = None

    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def distance(self) -> float | None:
        return self.monitor.current_distance()  # median of the last 5 reads (see triggers.smoothed_distance)

    def floor(self) -> tuple[bool, bool, bool]:
        for _ in range(3):
            g = list(self.car.get_grayscale_data())
            if g[:2] != list(POISONED_ADC[:2]):
                break
        ref = self.cfg.CLIFF_REFERENCE
        over = [v <= r for v, r in zip(g, ref)]
        return tuple(over[i] for i in FLOOR_ORDER)

    def look(self, pan_deg: float) -> None:
        self.pan = max(-60.0, min(60.0, pan_deg))
        self.tracker = None  # it was following the old view
        self.car.set_cam_pan_angle(int(-self.pan))
        self.car.set_cam_tilt_angle(self.tilt)
        time.sleep(SERVO_SETTLE_S)
        self.settled = time.time()

    def _frame(self) -> bytes | None:
        return vision.capture()

    def _sighting(self, left: float, top: float, right: float, bottom: float) -> Sighting:
        return Sighting(bearing=(0.5 - (left + right) / 2) * CAMERA_FOV + self.pan, width=right - left, bottom=bottom)

    def locate(self, target: str, jpeg: bytes | None = None) -> Sighting | None:
        jpeg = jpeg or self._frame()
        box = vision.locate(self.cfg.LLM_BASE_URL, self.cfg.LLM_MODEL, target, jpeg) if jpeg else None
        if box is None:
            print(f"drive: look (pan {self.pan:+.0f}): no {target}")
            return None
        s = self._sighting(*box)
        print(f"drive: look (pan {self.pan:+.0f}): {target} at {s.bearing:+.0f} deg, width {s.width:.2f}, "
              f"bottom {s.bottom:.2f}, distance {self.distance()}")
        return s

    def acquire(self, target: str) -> Sighting | None:
        """Find it with the vision model, then start the tracker on that same frame."""
        self.tracker = None
        jpeg = self._frame()
        if not jpeg:
            return None
        box = vision.locate(self.cfg.LLM_BASE_URL, self.cfg.LLM_MODEL, target, jpeg)
        if box is None:
            print(f"drive: look (pan {self.pan:+.0f}): no {target}")
            return None
        img = _decode(jpeg)
        h, w = img.shape[:2]
        left, top, right, bottom = box
        rect = (int(left * w), int(top * h), max(4, int((right - left) * w)), max(4, int((bottom - top) * h)))
        self.tracker = _new_tracker()
        self.tracker.init(img, rect)
        s = self._sighting(*box)
        print(f"drive: acquired {target} at {s.bearing:+.0f} deg, width {s.width:.2f}, bottom {s.bottom:.2f}, "
              f"distance {self.distance()}, box {rect} ({type(self.tracker).__name__})")
        return s

    def track(self, target: str) -> Sighting | None:
        """Where the tracker has it in the latest frame, or None (lost -> acquire again)."""
        # ponytail: no periodic vision-model recheck; a tracker drifting onto the
        # wrong thing goes unnoticed until it's lost. Add a ~2s background locate if seen live.
        if self.tracker is None:
            return None
        try:
            jpeg = requests.get(f"{vision.CAMERA_URL}/snapshot.jpg", timeout=1).content
            img = _decode(jpeg)
        except Exception:
            return None
        ok, (x, y, bw, bh) = self.tracker.update(img)
        if not ok:
            print(f"drive: lost {target}")
            self.tracker = None
            return None
        h, w = img.shape[:2]
        s = self._sighting(x / w, y / h, (x + bw) / w, (y + bh) / h)
        print(f"drive:   track {s.bearing:+.0f} deg, width {s.width:.2f}, bottom {s.bottom:.2f}, distance {self.distance()}")
        return s

    def person(self) -> Sighting | None:
        """The biggest (nearest) person openbot-camera's detector sees now -- it's
        asked to run on every frame while we follow. Right after a head move
        (standing still), waits (<=1s) for a detection of the new view; never
        while driving -- the drive loop checks the floor every 50ms."""
        # ponytail: biggest box, no memory of WHICH person -- two people can swap. Track by
        # bearing (or face) if that's seen live.
        while True:
            objects.want_fast()
            found, ts = objects.latest()
            if ts > self.settled + DETECT_S and time.time() - ts < PERSON_FRESH_S:
                break
            if time.time() - self.settled > 1.0 or getattr(self, "started", None):  # driving: never wait here
                return None
            time.sleep(0.03)
        people = [self._sighting(*o["box"]) for o in found if o["name"] == "person"]
        if not people:
            return None
        s = max(people, key=lambda p: p.width)
        s.bottom = None  # head tipped up: their feet are below the frame, so the bottom edge isn't distance
        return s

    def say(self, text: str) -> None:
        """Out loud, in the persona's voice -- without waiting: the search goes on meanwhile."""
        from common import persona
        from common.speak_client import speak
        threading.Thread(target=speak, args=(persona.load().transform(text),), daemon=True).start()

    def glance(self) -> list[str]:
        jpeg = self._frame()
        return vision.list_objects(self.cfg.LLM_BASE_URL, self.cfg.LLM_MODEL, jpeg) if jpeg else []

    def drive(self, speed: float, steer_deg: float, seconds: float, reverse: bool = False,
              guard_cm: float = 10.0) -> tuple[str, float]:
        """Like SimBody.drive: checks the floor and the ultrasonic every tick while
        going forward (there's no rear sensor), always stops the motors."""
        steer = max(-30.0, min(30.0, steer_deg))
        self.car.set_dir_servo_angle(int(-steer))
        time.sleep(0.05)
        t0 = time.monotonic()
        result = "ok"
        try:
            (self.car.backward if reverse else self.car.forward)(int(speed))
            while time.monotonic() - t0 < seconds:
                if self.cancel.is_set():
                    result = "cancelled"
                    break
                if not reverse:
                    if any(self.floor()):
                        result = "cliff"
                        break
                    d = self.distance()
                    if d is not None and d < guard_cm:
                        result = "blocked"
                        break
                time.sleep(TICK_S)
        finally:
            self.car.stop()
            self.car.set_dir_servo_angle(0)
        driven = time.monotonic() - t0
        print(f"drive: {'back' if reverse else 'fwd '} steer {steer:+.0f} {driven:.2f}/{seconds:.2f}s -> {result}")
        return result, driven

    # --- continuous driving (navigate.approach): it checks the floor/distance every tick ---
    def start_drive(self, speed: float, steer_deg: float, reverse: bool = False) -> None:
        self.set_steer(steer_deg)
        (self.car.backward if reverse else self.car.forward)(int(speed))
        self.started = time.monotonic()
        print(f"drive: cruise start, steer {steer_deg:+.0f}, distance {self.distance()} "
              f"(raw {[round(d) for d in self.monitor._recent if d is not None]}), floor {self.floor()}")

    def set_steer(self, steer_deg: float) -> None:
        self.car.set_dir_servo_angle(int(-max(-30.0, min(30.0, steer_deg))))

    def stop(self) -> None:
        self.car.stop()
        self.car.set_dir_servo_angle(0)
        if getattr(self, "started", None):
            print(f"drive: cruise stop after {time.monotonic() - self.started:.2f}s, distance {self.distance()}, "
                  f"floor {self.floor()}")
            self.started = None

    def wait(self, seconds: float) -> str:
        time.sleep(seconds)
        return "cancelled" if self.cancel.is_set() else "ok"
