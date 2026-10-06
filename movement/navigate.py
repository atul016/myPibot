"""Driving with a purpose: "go to the pink toy", "explore the table".

Written against a small Body interface, so the same code drives the
simulator (sim/world.py -- tested across scenarios x seeds by
tests/test_navigation_sim.py) and the real robot. Angles: degrees, + = LEFT
for steering, head pan and bearings (the real Body flips them for picarx,
whose + is right).

Safety, in layers:
  1. Body.drive() stops by itself the instant a floor sensor sees an edge or
     the ultrasonic sees something within guard_cm -- forward only: there's
     no rear sensor.
  2. Reversing is blind, so Driver.back() only ever retraces ground just
     driven over forward ("reverse credit"). The sim's first version backed
     up freely and fell off the table in 15/25 runs of one scenario.
  3. Turns are forward arcs (guarded), with short credited reverses when an
     arc meets an edge -- and they turn AWAY from the side that saw it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Sighting:
    bearing: float   # degrees from the body's heading, + = left
    width: float     # angular width as a fraction of the camera's field of view (bigger = closer)
    # Real robot only: the object's bottom edge as a fraction of the image height,
    # head tilted down. Low in the frame = close, whatever the object's size --
    # width alone isn't distance (a 12cm mouse filled a third of the view from
    # 35cm away). None in the sim, which uses width.
    bottom: float | None = None


class Body(Protocol):
    def distance(self) -> float | None: ...                 # ultrasonic cm, None = no echo
    def floor(self) -> tuple[bool, bool, bool]: ...          # (left, centre, right): True = over an edge
    def look(self, pan_deg: float) -> None: ...              # head pan
    def locate(self, target: str) -> Sighting | None: ...
    def glance(self) -> list[str]: ...                       # labels of what the camera sees now
    def drive(self, speed: float, steer_deg: float, seconds: float, reverse: bool = False,
              guard_cm: float = 10.0) -> tuple[str, float]: ...  # ("ok"|"cliff"|"blocked"|"fell"|"hit", seconds driven)
    # Continuous driving (approach): keep the wheels turning and steer as the target moves.
    def start_drive(self, speed: float, steer_deg: float, reverse: bool = False) -> None: ...
    def set_steer(self, steer_deg: float) -> None: ...
    def stop(self) -> None: ...
    def wait(self, seconds: float) -> str: ...                # let time pass driving: "ok"|"fell"|"hit"|"cancelled"
    def acquire(self, target: str) -> Sighting | None: ...   # slow, smart (the vision model): find + start tracking
    def track(self, target: str) -> Sighting | None: ...     # fast (a video tracker, ~10Hz): where it is now, or lost
    def person(self) -> Sighting | None: ...                 # follow(): the object detector's newest person, or None
    def heading(self) -> float | None: ...                   # turn_by(): degrees turned so far, + right (an IMU)


# --- tuning (each one exercised by the sim's scenarios) ---------------------------
SPEED = 20               # slow: ~10 cm/s
STEP_S = 0.4             # one guarded step
ARRIVE_CM = 16           # ultrasonic says it's right there...
ARRIVE_BEARING = 15      # ...and it's roughly straight ahead
ARRIVE_WIDTH = 0.33      # or it fills a third of the view
# The ultrasonic reads whatever is ahead -- a box in front of the target too. Its
# reading only counts as "arrived" if the target itself also LOOKS close (the sim
# caught a "reached the pink toy" at 0 steps: the 15cm reading was the box).
NEAR_WIDTH = 0.2
ARRIVE_BOTTOM = 0.88     # real camera: bottom edge in the lowest 12% of the frame = right at the bumper
NEAR_BOTTOM = 0.7


def _arrived(s: Sighting, d: float | None) -> bool:
    if s.bottom is not None:
        return s.bottom >= ARRIVE_BOTTOM and abs(s.bearing) <= ARRIVE_BEARING
    return s.width >= ARRIVE_WIDTH or (d is not None and d <= ARRIVE_CM and abs(s.bearing) <= ARRIVE_BEARING
                                       and s.width >= NEAR_WIDTH)


def _near(s: Sighting) -> bool:
    """Close and centred enough that what the ultrasonic sees is probably the target."""
    return abs(s.bearing) <= ARRIVE_BEARING and (s.bottom >= NEAR_BOTTOM if s.bottom is not None
                                                  else s.width >= NEAR_WIDTH)
STEER_GAIN = 1.3
SEARCH_PANS = (0, 40, -40, 60, -60)
BACKUP_S = 0.5
MAX_EDGE_HITS = 2        # the floor ran out this many times heading for it -> it's past the edge
TURN_S = 1.6             # forward-arc time for a "turn" (~45-60 degrees at full lock)
# Stop distance while cruising. The ultrasonic is a ~+/-15 degree cone: something
# 6cm off-centre is inside it only while it's > ~22cm away, so a 14cm stop let it
# slide out of view and the bumper's corner hit it (the sim: most collisions).
# 30cm covers the body's width. Final approach to a close, centred target: CLOSE_GUARD_CM.
CRUISE_GUARD_CM = 30.0
CLOSE_GUARD_CM = 14.0
PASS_WIDE_DEG = 12       # after going around an obstacle, steer this much away from it...
PASS_WIDE_S = 1.6        # ...for this long
CTRL_TICK = 0.05         # continuous driving: floor + distance checked every tick (20Hz, as before)...
TRACK_EVERY = 2          # ...and the target's direction updated from the tracker every 2nd (10Hz)
MAX_DRIVE_S = 60.0
SIDE_DEG = 25            # target beyond the forward camera view (54 deg): stop and turn to it first


@dataclass
class Outcome:
    done: bool
    reason: str
    steps: int = 0
    seen: list[str] = field(default_factory=list)


class Stop(Exception):
    """The body fell or hit something -- end the behaviour (the sim counts it)."""


class Driver:
    """Body.drive with the reverse-credit rule: backing up is only allowed
    over ground just driven forward."""

    def __init__(self, body: Body, turn_guard_cm: float):
        self.body = body
        self.credit = 0.0  # seconds of forward driving available to retrace
        # How close something may get during a turn. Exploring: CRUISE (nothing is
        # meant to be approached). Going to a target: CLOSE -- the 30cm cruise
        # guard stopped turns for the TARGET itself, and the endless back-and-arc
        # walked Rocky into an edge at a shallow angle (the sim: 9/25 falls).
        self.turn_guard_cm = turn_guard_cm

    def forward(self, steer: float, seconds: float, guard_cm: float = 10.0) -> str:
        _check_cancel(self.body)
        result, driven = self.body.drive(SPEED, _clamp(steer), seconds, guard_cm=guard_cm)
        self.credit = min(self.credit + driven, 3.0)
        if result in ("fell", "hit", "cancelled"):
            raise Stop(result)
        return result

    def back(self, steer: float = 0.0, seconds: float = BACKUP_S) -> bool:
        seconds = min(seconds, self.credit)
        if seconds < 0.1:
            return False  # no known-clear ground behind: don't reverse blind
        result, driven = self.body.drive(SPEED, _clamp(steer), seconds, reverse=True)
        self.credit -= driven
        if result in ("fell", "hit", "cancelled"):
            raise Stop(result)
        return True

    def edge_side(self) -> int:
        """+1 if the edge is on the left (turn right), -1 on the right, 0 unknown."""
        left, _, right = self.body.floor()
        return 1 if left and not right else -1 if right and not left else 0

    def turn(self, left: bool, seconds: float = TURN_S, guard_cm: float | None = None) -> None:
        """A forward arc at full lock; where it meets an edge or obstacle, a short
        credited reverse with the wheels the other way, then carry on."""
        steer = 30.0 if left else -30.0
        done = 0.0
        tries = 0
        while done < seconds and tries < 6:
            tries += 1
            before = self.credit
            result = self.forward(steer, min(0.4, seconds - done), guard_cm=guard_cm or self.turn_guard_cm)
            done += max(0.0, self.credit - before) or 0.05
            if result != "ok":
                if not self.back(-steer, 0.6):
                    return  # pinned with nowhere safe to back into -- stop turning


def _check_cancel(body: Body) -> None:
    """The real Body can be told to stop ("Rocky, stop"); the sim's never is."""
    if getattr(body, "cancelled", lambda: False)():
        raise Stop("cancelled")


def _clamp(v: float, lim: float = 30.0) -> float:
    return max(-lim, min(lim, v))


def search(body: Body, target: str) -> Sighting | None:
    """Look around with the head (no driving); bearing returned in the body frame."""
    look = getattr(body, "acquire", body.locate)
    found = None
    for pan in SEARCH_PANS:
        body.look(pan)
        found = look(target)
        if found:
            break
    body.look(0)
    return found


def approach(body: Body, target: str, max_rounds: int = 40) -> Outcome:
    """Drive up to the thing called `target`, smoothly: find it once with the
    vision model (standing still), then keep the wheels turning and steer from
    a fast tracker, checking the floor and the ultrasonic every 50ms. Edges,
    obstacles and losing the target stop the car and fall back to the careful
    recovery moves the stop-and-go version proved in the sim. Gives up rather
    than follow the target over an edge."""
    drv = Driver(body, turn_guard_cm=CLOSE_GUARD_CM)
    misses = edge_hits = 0
    last = None
    driven = wide_until = wide_side = 0.0
    try:
        for rnd in range(max_rounds):
            _check_cancel(body)
            s = getattr(body, "acquire", body.locate)(target) or search(body, target)
            if s is None:
                misses += 1
                # Never seen it: one turn to check behind, then say so -- no blind
                # wandering (the sim: wandering to find a hidden target ended in
                # tight arcs near edges, and a wheel over one). Seen before and lost
                # (it went out of view while turning): a few arcs toward where it was.
                if misses > (3 if last else 2):
                    return Outcome(False, f"I can't see the {target} from here", rnd)
                drv.turn(left=last is None or last.bearing >= 0, seconds=TURN_S if last else 2 * TURN_S)
                continue
            misses = 0

            # --- cruise: wheels turning, steering from the tracker, guards every tick ---
            event, moving, ticks = None, False, 0
            while driven < MAX_DRIVE_S:
                _check_cancel(body)
                if s is None:
                    event = "lost"
                    break
                last = s
                d = body.distance()
                if _arrived(s, d):
                    body.stop()
                    return Outcome(True, f"reached the {target}", rnd)
                if abs(s.bearing) > SIDE_DEG:
                    event = "side"
                    break
                if any(body.floor()):
                    event = "cliff"
                    break
                if d is not None and d < (CLOSE_GUARD_CM if _near(s) else CRUISE_GUARD_CM):
                    event = "blocked"
                    break
                bias = wide_side * PASS_WIDE_DEG if driven < wide_until else 0.0
                steer = _clamp(s.bearing * STEER_GAIN + bias)
                if moving:
                    body.set_steer(steer)
                else:
                    body.start_drive(SPEED, steer)
                    moving = True
                result = body.wait(CTRL_TICK)
                driven += CTRL_TICK
                drv.credit = min(drv.credit + CTRL_TICK, 3.0)
                if result != "ok":
                    raise Stop(result)
                ticks += 1
                if ticks % TRACK_EVERY == 0:
                    s = body.track(target)
            body.stop()
            if event is None:
                break  # out of driving time
            if event == "side":  # well off to the side: arc toward it
                drv.turn(left=last.bearing > 0, seconds=0.8)
            elif event == "cliff":
                edge_hits += 1
                side = drv.edge_side()
                drv.back()
                if edge_hits >= MAX_EDGE_HITS and abs(last.bearing) <= 25:
                    return Outcome(False, f"the {target} is past the edge -- I can't get there safely", rnd)
                if side:
                    drv.turn(left=side < 0, seconds=0.8)
            elif event == "blocked":
                # What's in front may BE the target. A tracker's box hardly grows as
                # we close in (live: bottom 0.49 with the mouse at 16cm), so when the
                # target is dead ahead, take one fresh look with the vision model.
                if not _near(last) and abs(last.bearing) <= ARRIVE_BEARING and hasattr(body, "acquire"):
                    last = body.acquire(target) or last
                if _near(last):
                    return Outcome(True, f"reached the {target}", rnd)
                # Go around: back off far enough that the arc's sideways swing clears
                # it, arc toward the target's side (measured 22/25 vs 6/25 the other
                # way), then pass WIDE -- the ultrasonic only sees straight ahead.
                drv.back(seconds=1.4)
                drv.turn(left=last.bearing >= 0, seconds=1.2)
                wide_until, wide_side = driven + PASS_WIDE_S, (1.0 if last.bearing >= 0 else -1.0)
            # "lost": just look again (next round)
        return Outcome(False, f"couldn't reach the {target} in time", max_rounds)
    except Stop as e:
        return Outcome(False, str(e))
    finally:
        if hasattr(body, "stop"):
            body.stop()


def look_around(body: Body) -> list[str]:
    seen: set[str] = set()
    for pan in (-60, -30, 0, 30, 60):
        body.look(pan)
        seen.update(body.glance())
    body.look(0)
    return sorted(seen)


def explore(body: Body, max_steps: int = 120, rng: random.Random | None = None) -> Outcome:
    """Wander, for the FLOOR: drive in straight legs (edges met head-on are the
    only ones the narrow floor sensor reliably catches), look around every few
    steps, turn away from obstacles. The first edge means "this is a table":
    the sim showed wandering on one can't be made safe with these sensors (the
    floor sensor spans ~5cm, the wheels ~14cm -- at a shallow angle a wheel goes
    over first), so it stops driving and just looks around."""
    rng = rng or random.Random()
    drv = Driver(body, turn_guard_cm=CRUISE_GUARD_CM)
    seen: set[str] = set(look_around(body))
    stuck = 0
    try:
        for step in range(max_steps):
            _check_cancel(body)
            if step and step % 8 == 0:
                seen.update(look_around(body))
            result = drv.forward(0.0, STEP_S, guard_cm=CRUISE_GUARD_CM)
            if result == "cliff":
                drv.back()
                seen.update(look_around(body))
                return Outcome(True, "this is a table -- I won't wander near its edges, so I looked around instead",
                               step, sorted(seen))
            if result == "blocked":
                stuck += 1
                drv.back()
                drv.turn(left=rng.random() < 0.5, seconds=rng.uniform(1.0, 2.4))
                if stuck >= 10:
                    return Outcome(True, "boxed in -- stopping", step, sorted(seen))
            else:
                stuck = max(0, stuck - 1)
        return Outcome(True, "explored", max_steps, sorted(seen))
    except Stop as e:
        return Outcome(False, str(e), 0, sorted(seen))


# --- turning to face someone (a voice's direction; needs a heading: the IMU) --------------------
FACE_DONE_DEG = 12.0      # this close to their direction: facing them
FACE_MOVE_S = 0.8         # one arc at full lock (~25-30 degrees at SPEED)
FACE_MAX_MOVES = 16
FACE_WRONG_WAY_DEG = 8.0  # a move that leaves it this much further off: a wrong sign, or a push -- stop
FACE_STALLED_MOVES = 3    # moves in a row that turn it under 2 degrees: the wheels aren't turning it (or the IMU stopped)


def turn_by(body: Body, degrees: float, max_moves: int = FACE_MAX_MOVES) -> Outcome:
    """Turns about `degrees` (+ right) where it stands: a three-point turn -- an arc forward
    at full lock, then back over that ground with the wheels the other way (both turn it the
    same way) -- each a guarded Driver move, watching its heading rather than counting
    seconds. Stops facing that way; when there's no room to move; when a move leaves it
    further off than before (it must never spin round on a wrong sign); or when moves stop
    turning it (motors off, or a heading that stopped changing -- never drive on blind)."""
    start = body.heading()
    if start is None:
        return Outcome(False, "I can't feel which way I'm turning")
    target, drv, left, stuck, idle = start + degrees, Driver(body, CLOSE_GUARD_CM), degrees, 0, 0
    try:
        for move in range(max_moves):
            if abs(left) <= FACE_DONE_DEG:
                return Outcome(True, "facing them now", move)
            steer = -30.0 if left > 0 else 30.0  # Driver's convention: + steers left
            if not (move % 2 and drv.back(-steer, FACE_MOVE_S)):
                result = drv.forward(steer, FACE_MOVE_S, guard_cm=CLOSE_GUARD_CM)
                stuck = stuck + 1 if result != "ok" and drv.credit < 0.1 else 0
                if stuck >= 2:
                    return Outcome(False, "there's no room to turn here", move + 1)
            now_left = target - body.heading()
            if abs(now_left) > abs(left) + FACE_WRONG_WAY_DEG:
                return Outcome(False, "I was turning the wrong way, so I stopped", move + 1)
            idle = idle + 1 if abs(now_left - left) < 2.0 else 0
            if idle >= FACE_STALLED_MOVES:
                return Outcome(False, "my wheels aren't turning me", move + 1)
            left = now_left
    except Stop as e:
        return Outcome(False, str(e))
    finally:
        body.stop()
    return Outcome(abs(left) <= 2 * FACE_DONE_DEG, "facing them now" if abs(left) <= 2 * FACE_DONE_DEG
                   else f"I turned, but I'm still about {abs(left):.0f} degrees off", max_moves)


# --- following a person ("follow me") ----------------------------------------------------
# Calibration knobs. Measured 2026-10-04, one person standing in front of the real
# camera (~13cm off the floor): detected ~9x/s at every tilt and distance tried. At
# tilt 30 their box was 0.72 of the view wide at 0.5m, 0.43 at 1m, 0.34 at 2m; level
# (tilt 0) the widths didn't track distance at all (more of the body shows farther away).
FOLLOW_SPEED = 30          # the house rule's test speed
FOLLOW_TILT = 30           # real robot: head tipped up -- body width then tracks distance
FOLLOW_STOP_CM = 50.0      # moving, something this close ahead (them, usually) -> stop...
FOLLOW_GO_CM = 70.0        # ...and wait until it's this far again (no stop-go jitter)
FOLLOW_CLOSE_WIDTH = 0.6   # this wide in the view = close, ~0.65m (the ultrasonic misses people off-centre)
FOLLOW_GO_WIDTH = 0.5      # ...and narrower than this = they've walked on, ~0.8m
FOLLOW_SMALL_WIDTH = 0.2   # something inside FOLLOW_STOP_CM while they look this small isn't them
FOLLOW_GRACE_S = 0.5       # the detector missed a frame or two: keep going on the last sighting
FOLLOW_MAX_TRIES = 6        # searches and head turns in a row without driving -> give up
FOLLOW_SEEK_SWEEPS = 3      # lost: call out, then this many more looks around (~10s) before giving up
FOLLOW_MAX_S = 300.0


def _find_person(body: Body) -> tuple[Sighting | None, float]:
    """Standing still, look for them with the head -- it stays pointed at them.
    (the sighting, the head's pan)."""
    for pan in SEARCH_PANS:
        body.look(pan)
        for _ in range(3):  # the detector misses a frame now and then
            s = body.person()
            if s is not None:
                return s, pan
            body.wait(0.15)
    body.look(0)
    return None, 0.0


def _head_for(bearing: float, pan: float) -> float:
    """Where the head should point to keep them in view while driving toward
    them: straight ahead when they're roughly ahead, else turned their way in
    20-degree steps -- with slack, so it doesn't flip-flop at a boundary."""
    if pan == 0 and abs(bearing) <= SIDE_DEG or pan != 0 and abs(bearing) < SIDE_DEG - 10:
        return 0.0
    if pan != 0 and abs(bearing - pan) <= 20:
        return pan
    return max(-60.0, min(60.0, 20.0 * round(bearing / 20)))


def follow(body: Body, max_s: float = FOLLOW_MAX_S) -> Outcome:
    """"Follow me": stay near the person in view -- steer toward them, stop
    when they're close, go again when they walk on. The head only turns while
    the car stands still (the detector reports a frame ~0.1s old: a moving
    head would put them in the wrong place), and points where _head_for says,
    so a person off to the side stays in view while Rocky turns toward them.
    Lost: stop and look around for them -- not there: call out and keep
    looking a little while. The first floor edge ends it -- they
    can step off a table, Rocky can't."""
    moving, pan, last, seen, t, looks, stuck = False, 0.0, None, 0.0, 0.0, 0, 0
    try:
        while t < max_s:
            _check_cancel(body)
            if any(body.floor()):
                return Outcome(False, "the floor ends here, so I stopped at the edge", looks)
            if stuck > FOLLOW_MAX_TRIES:
                return Outcome(False, "I can't get to you from here", looks)
            s = body.person()
            if s is not None:
                last, seen = s, t
            elif t - seen > FOLLOW_GRACE_S:  # gone: look for them, standing still
                if moving:
                    body.stop()
                    moving = False
                s, pan = _find_person(body)
                if s is None:  # call out, and give them a moment to come back into view
                    getattr(body, "say", lambda text: None)("Where did you go?")
                    for _ in range(FOLLOW_SEEK_SWEEPS):
                        _check_cancel(body)
                        s, pan = _find_person(body)
                        if s is not None:
                            break
                    else:
                        return Outcome(False, "I lost sight of you", looks)
                last, seen, looks, stuck = s, t, looks + 1, stuck + 1
                continue
            else:
                s = last  # a missed frame or two: carry on as before (None at the very start: wait)
            d = body.distance()
            if s is None:
                pass
            elif d is not None and d < FOLLOW_STOP_CM and s.width < FOLLOW_SMALL_WIDTH:
                return Outcome(False, "something is in my way", looks)
            elif moving:
                if s.width >= FOLLOW_CLOSE_WIDTH or (d is not None and d < FOLLOW_STOP_CM) \
                        or _head_for(s.bearing, pan) != pan:
                    body.stop()  # close enough -- or the head has to turn, which it only does standing still
                    moving = False
                else:
                    body.set_steer(_clamp(s.bearing * STEER_GAIN))
            elif s.width < FOLLOW_GO_WIDTH and (d is None or d > FOLLOW_GO_CM):  # they walked on
                head = _head_for(s.bearing, pan)
                if head != pan:
                    body.look(head)
                    pan, last, looks, stuck = head, None, looks + 1, stuck + 1
                    seen = t - FOLLOW_GRACE_S  # that sighting was through the old view
                    continue
                body.start_drive(FOLLOW_SPEED, _clamp(s.bearing * STEER_GAIN))
                moving, stuck = True, 0
            result = body.wait(CTRL_TICK)
            t += CTRL_TICK
            if result != "ok":
                raise Stop(result)
        return Outcome(True, "followed you as long as I'm allowed to in one go", looks)
    except Stop as e:
        return Outcome(False, str(e), looks)
    finally:
        body.stop()
