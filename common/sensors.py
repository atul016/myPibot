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
HEARING_PATH = STATE_DIR / "hearing.json"


def publish(distance: float | None, grayscale: list | None, latches: dict[str, bool],
            battery_v: float | None = None, battery_pct: float | None = None) -> None:
    atomic_write(SENSORS_PATH, json.dumps({
        "distance": distance, "grayscale": grayscale, "latches": latches,
        "battery_v": battery_v, "battery_pct": battery_pct, "ts": time.time(),
    }))


def read() -> dict[str, Any]:
    """{} if missing or stale -- openbot-alive down, or no body at all (a
    reading left over from a PiCar-X run must not look live)."""
    try:
        data = json.loads(SENSORS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if time.time() - data.get("ts", 0) < 5 else {}


def publish_hearing(levels: list[float]) -> None:
    """Rolling per-second peak mic loudness (RMS), oldest first --
    published by openbot-wake-listen (the mic's only reader), read by
    openbot-mind for surprise detection. A list, not one value, so a
    reader sampling slower than once a second can't miss a spike."""
    atomic_write(HEARING_PATH, json.dumps({"levels": levels, "ts": time.time()}))


def read_hearing() -> list[float]:
    """[] if missing or stale (wake-listen down or mid-conversation)."""
    try:
        data = json.loads(HEARING_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("levels", []) if time.time() - data.get("ts", 0) < 5 else []


def demo() -> None:
    import shutil, tempfile
    global SENSORS_PATH, HEARING_PATH
    orig, orig_h = SENSORS_PATH, HEARING_PATH
    test_dir = Path(tempfile.mkdtemp())
    SENSORS_PATH, HEARING_PATH = test_dir / "sensors.json", test_dir / "hearing.json"
    try:
        publish(12.5, [1400, 1420, 1390], {"cliff": False})
        data = read()
        assert data["distance"] == 12.5
        atomic_write(SENSORS_PATH, json.dumps({**data, "ts": time.time() - 60}))
        assert read() == {}  # stale
        assert read_hearing() == []
        publish_hearing([300.0, 2500.0])
        assert read_hearing() == [300.0, 2500.0]
    finally:
        SENSORS_PATH, HEARING_PATH = orig, orig_h
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("sensors: ok")
