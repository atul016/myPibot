"""Shared sensor snapshot: state/sensors.json, published by openbot-alive
(the only process with a live Picarx handle), read by openbot-mind and the
dashboard. Decouples "who has the hardware" from "who wants the last
reading" now that they're separate processes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .state import STATE_DIR, atomic_write

SENSORS_PATH = STATE_DIR / "sensors.json"
MOTION_PATH = STATE_DIR / "motion.json"


def publish(distance: float | None, grayscale: list | None, latches: dict[str, bool],
            battery_v: float | None = None, battery_pct: float | None = None, driving: bool = False,
            head: dict | None = None, imu: dict | None = None) -> None:
    """head: {"pan", "tilt", "moved_ts"} -- where the camera points (pan + = right, tilt + = up)
    and when it last moved: its own glances must not read as the room changing.
    imu: common/imu.py's Tracker.snapshot() -- None without an IMU."""
    atomic_write(SENSORS_PATH, json.dumps({
        "distance": distance, "grayscale": grayscale, "latches": latches,
        "battery_v": battery_v, "battery_pct": battery_pct, "driving": driving, "head": head, "imu": imu,
        "ts": time.time(),
    }))


def read() -> dict[str, Any]:
    """{} if missing or stale -- openbot-alive down, or no body at all (a
    reading left over from a PiCar-X run must not look live)."""
    try:
        data = json.loads(SENSORS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if time.time() - data.get("ts", 0) < 5 else {}


def publish_motion(levels: list[float]) -> None:
    """Rolling frame-to-frame change (0-255 mean abs diff, downscaled), twice a
    second, oldest first -- published by openbot-camera, read by openbot-mind:
    something moving in view is a surprise even before anyone speaks."""
    atomic_write(MOTION_PATH, json.dumps({"levels": levels, "ts": time.time()}))


def read_motion() -> list[float]:
    try:
        data = json.loads(MOTION_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("levels", []) if time.time() - data.get("ts", 0) < 5 else []


def demo() -> None:
    import shutil, tempfile
    global SENSORS_PATH, MOTION_PATH
    orig, orig_m = SENSORS_PATH, MOTION_PATH
    test_dir = Path(tempfile.mkdtemp())
    SENSORS_PATH, MOTION_PATH = test_dir / "sensors.json", test_dir / "motion.json"
    try:
        publish(12.5, [1400, 1420, 1390], {"cliff": False})
        data = read()
        assert data["distance"] == 12.5 and data["driving"] is False
        atomic_write(SENSORS_PATH, json.dumps({**data, "ts": time.time() - 60}))
        assert read() == {}  # stale
        assert read_motion() == []
        publish(50.0, None, {}, head={"pan": 20.0, "tilt": 0.0, "moved_ts": 1.0})
        assert read()["head"]["pan"] == 20.0
        publish_motion([1.0, 20.0])
        assert read_motion() == [1.0, 20.0]
    finally:
        SENSORS_PATH, MOTION_PATH = orig, orig_m
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("sensors: ok")
