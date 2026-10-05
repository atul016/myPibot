"""Surprise detection: turns a short rolling history of sensor samples into
plain-English "something just changed" events. openbot-mind samples every
few seconds and reflects IMMEDIATELY on a surprise, instead of only on its
slow idle timer -- the difference between a robot that checks in every
five minutes and one that notices you walk up.

Each event is (kind, text) -- kind is what common/agenda.py's watches
match on (approach, leave, sound, picked_up, put_down, battery_low; vision
adds scene, lights_on, lights_off; the camera adds motion). Pure functions over plain lists, no hardware, no I/O. Thresholds are
calibration knobs (ultrasonic jitter, room noise), not truths.
"""
from __future__ import annotations

import os

from statistics import median

NEAR_CM = 25.0      # "something is close"
FAR_CM = 45.0       # "nothing was close" -- the gap between the two is hysteresis
LOUD_RATIO = 3.0    # recent loudness vs. the room's recent baseline
# RMS; below this nothing counts as loud however quiet the room was. Mic-specific: a USB dongle reads
# ambient ~270, speech ~900+; a reSpeaker XVF3800 (noise-suppressed) ambient ~70, speech up to ~3500,
# a clap 6000-11000 -- Walle sets 5000. Check yours in 127.0.0.1:9001/hearing while you talk and clap.
LOUD_FLOOR = float(os.environ.get("OPENBOT_LOUD_FLOOR", "1500"))


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


TURNED_DEG = 30.0  # turned in place by someone at least this far: noticed
TIPPED_DEG = 55.0  # tilted further: on its side or back


def imu_events(before: dict | None, now: dict | None) -> list[Event]:
    """What the body felt (common/imu.py) between two still moments: someone turned it, it's
    lying tipped, it's level again. Nothing across an openbot-alive restart (its count starts at 0)."""
    if not before or not now or before.get("epoch") != now.get("epoch"):
        return []
    events: list[Event] = []
    turned = now["turned_by_others"] - before["turned_by_others"]
    if abs(turned) >= TURNED_DEG:
        events.append(("turned", f"someone turned you about {5 * round(abs(turned) / 5)} degrees to your "
                                 f"{'right' if turned > 0 else 'left'}"))
    if now["tilt"] >= TIPPED_DEG > before["tilt"]:
        events.append(("tipped", f"you're lying tilted about {now['tilt']:.0f} degrees -- tipped over, or someone's "
                                 "holding you that way; you can't drive like this"))
    elif before["tilt"] >= TIPPED_DEG > now["tilt"]:
        events.append(("righted", "you're level on your wheels again"))
    return events


MOTION_LEVEL = 12.0  # mean abs pixel change (0-255) on a 32x24 grey frame; a person moving ~20-60, noise ~2-5


def motion_events(levels: list) -> list[Event]:
    """levels: oldest-first, twice a second. Two high samples in a row -- one
    jump is the head itself turning (a glance), sustained change is something
    moving in view."""
    if len(levels) >= 2 and min(levels[-2:]) >= MOTION_LEVEL:
        return [("motion", f"something is moving in front of you (change {max(levels[-2:]):.0f})")]
    return []


def latch_events(prev: dict, cur: dict, lifted: bool | None = None) -> list[Event]:
    """alive.py's own debounced safety latches, on their rising edge. lifted: whether the IMU
    felt someone move the body just now (None without an IMU) -- the ground vanishing while
    nobody touched it is an edge, not a pickup."""
    if cur.get("cliff") and not prev.get("cliff"):
        if lifted is False:
            return [("edge", "the ground in front of you vanished -- you're at an edge")]
        return [("picked_up", "the ground under you vanished -- " + ("someone picked you up" if lifted else
                                                                      "picked up, or at an edge"))]
    if not cur.get("cliff") and prev.get("cliff"):
        return [("put_down", "you're back on solid ground")]
    if cur.get("danger") and not prev.get("danger"):
        return [("approach", "something is right in front of your nose, almost touching you")]
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


# At or under this, in this many readings in a row (30s apart), the Pi shuts down cleanly -- a brownout
# mid-write can corrupt its SD card. A motor's sag or one bad read isn't an empty battery; under
# PLAUSIBLE_V the reading is a missing battery or an ADC glitch (a 2S pack cuts off near 6V).
SHUTDOWN_PCT, SHUTDOWN_READINGS, PLAUSIBLE_V = 5.0, 3, 5.0


def battery_critical(low_count: int, volts: float | None, pct: float | None) -> int:
    """The new count of low readings in a row; SHUTDOWN_READINGS means shut down."""
    low = pct is not None and volts is not None and volts >= PLAUSIBLE_V and pct <= SHUTDOWN_PCT
    return low_count + 1 if low else 0


def demo() -> None:
    n = 0
    for volts, pct in [(6.1, 4.0), (6.3, 12.0), (6.1, 4.0), (6.1, 4.0), (6.1, 4.0)]:  # one sag reading resets it
        n = battery_critical(n, volts, pct)
    assert n == SHUTDOWN_READINGS
    assert battery_critical(2, 0.4, 0.0) == 0 and battery_critical(2, None, None) == 0  # no battery / misread
    assert distance_events([80, 82, 79, 81, 80, 20, 18, 19]) == \
        [("approach", "something came close in front of you (80cm -> 19cm)")]
    assert distance_events([80, 82, 79, 81, 80, 20, 80, 81]) == []      # one glitch isn't an approach
    assert distance_events([20, 19, 21, 20, 60, 61, 62])[0][0] == "leave"
    assert distance_events([80, -1, -2, 81, -1, 20]) == []              # not enough valid reads
    loud = LOUD_FLOOR * 1.7  # relative: the floor is a per-mic setting (OPENBOT_LOUD_FLOOR)
    assert loudness_events([300, 280, 310, 290, 300, loud])[0][0] == "sound"
    assert loudness_events([300, 280, 310, 290, 300, 600]) == []        # louder, but not loud
    assert loudness_events([loud] * 5 + [loud * 1.2]) == []  # already-loud room
    assert motion_events([2, 3, 30, 28])[0][0] == "motion"
    assert motion_events([2, 3, 30, 2]) == [] and motion_events([30]) == []  # one jump: the head turned
    assert latch_events({"cliff": False}, {"cliff": True})[0][0] == "picked_up"
    assert latch_events({"cliff": True}, {"cliff": True}) == []
    assert latch_events({}, {"cliff": True}, lifted=False)[0][0] == "edge"           # nobody touched it
    assert "someone picked you up" in latch_events({}, {"cliff": True}, lifted=True)[0][1]
    still = {"epoch": 1.0, "turned_by_others": 0.0, "tilt": 2.0}
    turned = imu_events(still, {**still, "turned_by_others": -88.0})
    assert turned == [("turned", "someone turned you about 90 degrees to your left")], turned
    assert imu_events(still, {**still, "turned_by_others": 12.0}) == []              # a nudge
    assert imu_events(still, {**still, "epoch": 2.0, "turned_by_others": 90.0}) == []  # alive restarted
    assert imu_events(still, {**still, "tilt": 85.0})[0][0] == "tipped"
    assert imu_events({**still, "tilt": 85.0}, still) == [("righted", "you're level on your wheels again")]
    assert latch_events({"danger": False}, {"danger": True})[0][0] == "approach"
    assert battery_events(21.0, 19.5)[0][0] == "battery_low"
    assert battery_events(19.5, 19.0) == [] and battery_events(11.0, 9.0)[0][0] == "battery_low"
    assert battery_events(None, 5.0) == [] and battery_events(50.0, None) == []


if __name__ == "__main__":
    demo()
    print("surprise: ok")
