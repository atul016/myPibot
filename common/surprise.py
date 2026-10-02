"""Surprise detection: turns a short rolling history of sensor samples into
plain-English "something just changed" events. openbot-mind samples every
few seconds and reflects IMMEDIATELY on a surprise, instead of only on its
slow idle timer -- the difference between a robot that checks in every
five minutes and one that notices you walk up.

Each event is (kind, text) -- kind is what common/agenda.py's watches
match on (approach, leave, sound, picked_up, put_down, battery_low; vision
adds scene, lights_on, lights_off). Pure functions over plain lists, no hardware, no I/O. Thresholds are
calibration knobs (ultrasonic jitter, room noise), not truths.
"""
from __future__ import annotations

from statistics import median

NEAR_CM = 25.0      # "something is close"
FAR_CM = 45.0       # "nothing was close" -- the gap between the two is hysteresis
LOUD_RATIO = 3.0    # recent loudness vs. the room's recent baseline
LOUD_FLOOR = 1500.0  # RMS; below this nothing counts as loud however quiet the room was (ambient ~270, speech ~900+)


def _valid(xs):
    return [x for x in xs if x is not None and x > 1]  # ultrasonic -1/-2 glitch codes


Event = tuple[str, str]  # (kind, plain-English text)


def distance_events(distances: list) -> list[Event]:
    """distances: oldest-first, one per sample. Compares the median of the
    last 3 samples to the median of the ones before, so one glitchy read
    can't fire an event."""
    recent, before = _valid(distances[-3:]), _valid(distances[-10:-3])
    if len(recent) < 2 or len(before) < 2:
        return []
    r, b = median(recent), median(before)
    if b >= FAR_CM and r <= NEAR_CM:
        return [("approach", f"something came close in front of you ({b:.0f}cm -> {r:.0f}cm)")]
    if b <= NEAR_CM and r >= FAR_CM:
        return [("leave", f"whatever was in front of you moved away ({b:.0f}cm -> {r:.0f}cm)")]
    return []


def loudness_events(levels: list) -> list[Event]:
    """levels: oldest-first RMS loudness samples from the mic."""
    if len(levels) < 5:
        return []
    baseline = median(levels[:-2])
    peak = max(levels[-2:])
    if peak >= LOUD_FLOOR and peak >= LOUD_RATIO * max(baseline, 1.0):
        return [("sound", f"a sudden sound (loudness {peak:.0f}, room was ~{baseline:.0f})")]
    return []


def latch_events(prev: dict, cur: dict) -> list[Event]:
    """alive.py's own debounced safety latches, on their rising edge."""
    if cur.get("cliff") and not prev.get("cliff"):
        return [("picked_up", "the ground under you vanished -- picked up, or at an edge")]
    if not cur.get("cliff") and prev.get("cliff"):
        return [("put_down", "you're back on solid ground")]
    return []


BATTERY_WARN_PCT = (20.0, 10.0)


def battery_events(prev_pct: float | None, pct: float | None) -> list[Event]:
    """Crossing DOWN through 20% or 10% -- once per crossing, so a reading
    jittering around the line doesn't nag. None (unknown/charging) never fires."""
    if prev_pct is None or pct is None:
        return []
    for level in BATTERY_WARN_PCT:
        if prev_pct > level >= pct:
            return [("battery_low", f"your battery just dropped to {pct:.0f}% -- you're getting tired")]
    return []


def demo() -> None:
    assert distance_events([80, 82, 79, 81, 80, 20, 18, 19]) == \
        [("approach", "something came close in front of you (80cm -> 19cm)")]
    assert distance_events([80, 82, 79, 81, 80, 20, 80, 81]) == []      # one glitch isn't an approach
    assert distance_events([20, 19, 21, 20, 60, 61, 62])[0][0] == "leave"
    assert distance_events([80, -1, -2, 81, -1, 20]) == []              # not enough valid reads
    assert loudness_events([300, 280, 310, 290, 300, 2500])[0][0] == "sound"
    assert loudness_events([300, 280, 310, 290, 300, 600]) == []        # louder, but not loud
    assert loudness_events([2000, 2100, 2050, 1900, 2000, 2400]) == []  # already-loud room
    assert latch_events({"cliff": False}, {"cliff": True})[0][0] == "picked_up"
    assert latch_events({"cliff": True}, {"cliff": True}) == []
    assert battery_events(21.0, 19.5)[0][0] == "battery_low"
    assert battery_events(19.5, 19.0) == [] and battery_events(11.0, 9.0)[0][0] == "battery_low"
    assert battery_events(None, 5.0) == [] and battery_events(50.0, None) == []


if __name__ == "__main__":
    demo()
    print("surprise: ok")
