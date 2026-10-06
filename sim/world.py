"""A small 2D tabletop simulator for Rocky's driving (movement/navigate.py).

Ideas from lucascosolo/picarx-training (a sim that stands in for sensors and
motors so the decision code can practice), written for this robot's own
interfaces. Pure stdlib -- runs on the Mac in seconds.

Units: cm, seconds, degrees at the API (radians inside). x right, y up.

  World    the table (the only floor -- off it is a fall), obstacles, labelled
           objects (also obstacles: you drive up to them, not through them)
  SimBody  the navigate.Body interface over a World: ultrasonic, the three
           floor sensors, the camera's view, and guarded driving

Every physical number below is a calibration knob, not a measurement of the
real car -- tune them against the robot, the physical world doesn't match a
model until it's been measured.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

# --- PiCar-X geometry (calibration knobs) ------------------------------------
WHEELBASE = 12.0          # rear axle -> front axle
REAR_OVERHANG = 4.0       # rear bumper behind the rear axle
FRONT_REACH = 18.0        # front bumper ahead of the rear axle
HALF_TRACK = 7.0          # wheel contact points, either side of the centre line
# The body for collisions: (ahead of rear axle, radius). Must end at the bumper
# (FRONT_REACH), where the ultrasonic is -- a first version poked 4cm past it, so
# the sim's body hit things the sensor still read as 4cm away.
BODY_CIRCLES = ((0.0, 8.0), (6.0, 8.0), (11.0, 7.0))
FLOOR_SENSORS = ((19.0, 2.5), (19.0, 0.0), (19.0, -2.5))  # (ahead, left) of the rear axle: grayscale left, centre, right
MAX_STEER = 30.0
CM_PER_S_PER_SPEED = 0.5  # drive(speed=20) -> 10 cm/s

# --- sensors ---------------------------------------------------------------------
ULTRASONIC_RAYS = tuple(float(d) for d in range(-15, 16, 3))  # the beam: a ~30-degree cone, sampled every 3 degrees
ULTRASONIC_MAX = 250.0
ULTRASONIC_NOISE = 1.0                # cm
ULTRASONIC_GLITCH = 0.03              # chance of a -2 "no echo" read when there IS something
CAMERA_FOV = 54.0
CAMERA_RANGE = 300.0
TICK = 0.05                           # the real safety loop polls at ~20Hz


def _rot(x: float, y: float, h: float) -> tuple[float, float]:
    return x * math.cos(h) - y * math.sin(h), x * math.sin(h) + y * math.cos(h)


@dataclass
class Obstacle:
    shape: str                 # "circle" or "rect"
    x: float
    y: float
    r: float = 0.0             # circle
    w: float = 0.0             # rect (x, y = lower-left corner)
    h: float = 0.0
    label: str | None = None   # what the camera calls it; None = an unremarkable thing
    # A walking person (circle): from (x, y) to each waypoint in turn at `speed` cm/s,
    # starting after `wait` s; a waypoint [x, y, s] stands there s seconds. A
    # collision counts only when Rocky moves into it.
    walk: list | None = None
    speed: float = 0.0
    wait: float = 0.0
    start: tuple | None = None

    def __post_init__(self) -> None:
        if self.walk and self.start is None:
            self.start = (self.x, self.y)

    def center(self) -> tuple[float, float]:
        return (self.x, self.y) if self.shape == "circle" else (self.x + self.w / 2, self.y + self.h / 2)

    def hits_circle(self, cx: float, cy: float, r: float) -> bool:
        if self.shape == "circle":
            return math.hypot(cx - self.x, cy - self.y) < r + self.r
        nx, ny = min(max(cx, self.x), self.x + self.w), min(max(cy, self.y), self.y + self.h)
        return math.hypot(cx - nx, cy - ny) < r

    def ray(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """Distance along the unit ray (dx, dy) from (ox, oy) to this obstacle, or None."""
        if self.shape == "circle":
            fx, fy = ox - self.x, oy - self.y
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - self.r * self.r
            disc = b * b - c
            if disc < 0:
                return None
            t = -b - math.sqrt(disc)
            return t if t >= 0 else None
        tmin, tmax = -math.inf, math.inf
        for o, d, lo, hi in ((ox, dx, self.x, self.x + self.w), (oy, dy, self.y, self.y + self.h)):
            if abs(d) < 1e-9:
                if not lo <= o <= hi:
                    return None
                continue
            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
        return tmin if tmax >= tmin and tmin >= 0 else None

    def half_width(self) -> float:
        return self.r if self.shape == "circle" else max(self.w, self.h) / 2


@dataclass
class Pose:
    x: float      # rear-axle centre
    y: float
    h: float      # heading, radians

    def point(self, ahead: float, left: float = 0.0) -> tuple[float, float]:
        dx, dy = _rot(ahead, left, self.h)
        return self.x + dx, self.y + dy


@dataclass
class World:
    table: tuple[float, float, float, float]          # x, y, w, h -- the floor Rocky stands on
    obstacles: list[Obstacle]
    pose: Pose
    pan: float = 0.0                                   # head pan, degrees (+ = left, as in the sim's frame)
    rng: random.Random = field(default_factory=random.Random)
    path: list[tuple[float, float]] = field(default_factory=list)
    events: list[tuple[str, float, float]] = field(default_factory=list)  # ("cliff"|"blocked"|"fell"|"hit", x, y)
    fell: bool = False
    collided: bool = False
    time: float = 0.0

    # --- scenario files ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> tuple["World", dict]:
        spec = json.loads(Path(path).read_text())
        rng = random.Random(seed)
        jitter = spec.get("start_jitter", {"xy": 3.0, "deg": 8.0})
        sx, sy, sdeg = spec["robot"]
        pose = Pose(sx + rng.uniform(-1, 1) * jitter["xy"], sy + rng.uniform(-1, 1) * jitter["xy"],
                    math.radians(sdeg + rng.uniform(-1, 1) * jitter["deg"]))
        obstacles = [Obstacle(**o) for o in spec.get("obstacles", [])]
        world = cls(tuple(spec["table"]), obstacles, pose, rng=rng)
        world.path.append(pose.point(WHEELBASE / 2))
        return world, spec

    # --- physics ------------------------------------------------------------------
    def on_table(self, x: float, y: float) -> bool:
        tx, ty, tw, th = self.table
        return tx <= x <= tx + tw and ty <= y <= ty + th

    def wheels(self, pose: Pose | None = None) -> list[tuple[float, float]]:
        p = pose or self.pose
        return [p.point(a, s * HALF_TRACK) for a in (0.0, WHEELBASE) for s in (-1, 1)]

    def walk(self) -> None:
        """Move the walking people to where they are at self.time."""
        for o in self.obstacles:
            if not o.walk:
                continue
            t, (ax, ay) = self.time - o.wait, o.start
            o.x, o.y = ax, ay
            for bx, by, *pause in o.walk:
                need = math.hypot(bx - ax, by - ay) / o.speed
                if t < need:
                    f = max(0.0, t) / need if need else 0.0
                    o.x, o.y = ax + (bx - ax) * f, ay + (by - ay) * f
                    break
                o.x, o.y = ax, ay = bx, by
                t -= need + (pause[0] if pause else 0.0)
                if t < 0:
                    break

    def collides(self, pose: Pose) -> Obstacle | None:
        self.walk()
        for ahead, r in BODY_CIRCLES:
            cx, cy = pose.point(ahead)
            for o in self.obstacles:
                if o.hits_circle(cx, cy, r):
                    return o
        return None

    def step(self, v: float, steer_deg: float, dt: float) -> None:
        """Kinematic bicycle model about the rear axle. A move into an obstacle
        is refused (records a collision); a wheel off the table is a fall."""
        p = self.pose
        h = p.h + v / WHEELBASE * math.tan(math.radians(steer_deg)) * dt
        nxt = Pose(p.x + v * math.cos(h) * dt, p.y + v * math.sin(h) * dt, h)
        hit = self.collides(nxt)
        if hit is not None:
            if not self.collided:
                self.events.append(("hit", *nxt.point(WHEELBASE)))
            self.collided = True
            return
        self.pose = nxt
        self.time += dt
        self.path.append(nxt.point(WHEELBASE / 2))
        if not all(self.on_table(*w) for w in self.wheels()):
            if not self.fell:
                self.events.append(("fell", *nxt.point(WHEELBASE / 2)))
            self.fell = True

    # --- sensors --------------------------------------------------------------------
    def ultrasonic(self) -> float:
        """Like the real sensor: cm, or -2 for no echo (nothing in range, or a glitch)."""
        self.walk()
        ox, oy = self.pose.point(FRONT_REACH)
        best = None
        for deg in ULTRASONIC_RAYS:
            dx, dy = _rot(1.0, 0.0, self.pose.h + math.radians(deg))
            for o in self.obstacles:
                t = o.ray(ox, oy, dx, dy)
                if t is not None and (best is None or t < best):
                    best = t
        if best is None or best > ULTRASONIC_MAX or self.rng.random() < ULTRASONIC_GLITCH:
            return -2.0
        return max(2.0, best + self.rng.gauss(0, ULTRASONIC_NOISE))

    def floor(self) -> list[bool]:
        """The three floor sensors (left, centre, right): True = over the edge (sees no table)."""
        return [not self.on_table(*self.pose.point(a, l)) for a, l in FLOOR_SENSORS]

    def visible(self) -> list[tuple[Obstacle, float, float]]:
        """Labelled objects in the camera's view: (object, bearing in degrees from
        the BODY's heading, + = left; angular width as a fraction of the FOV)."""
        self.walk()
        cx, cy = self.pose.point(FRONT_REACH - 4)
        axis = self.pose.h + math.radians(self.pan)
        out = []
        for o in self.obstacles:
            if not o.label:
                continue
            ox, oy = o.center()
            dist = math.hypot(ox - cx, oy - cy)
            if dist > CAMERA_RANGE or dist < 1:
                continue
            ang = math.degrees(math.atan2(oy - cy, ox - cx) - axis)
            ang = (ang + 180) % 360 - 180
            if abs(ang) > CAMERA_FOV / 2:
                continue
            if self._occluded(cx, cy, ox, oy, o):
                continue
            width = math.degrees(2 * math.atan2(o.half_width(), dist)) / CAMERA_FOV
            out.append((o, ang + self.pan, min(width, 1.0)))
        return out

    def _occluded(self, ax: float, ay: float, bx: float, by: float, target: Obstacle) -> bool:
        dist = math.hypot(bx - ax, by - ay)
        dx, dy = (bx - ax) / dist, (by - ay) / dist
        for o in self.obstacles:
            if o is target:
                continue
            t = o.ray(ax, ay, dx, dy)
            if t is not None and t < dist - target.half_width():
                return True
        return False

    # --- picture -----------------------------------------------------------------------
    def svg(self, title: str = "") -> str:
        tx, ty, tw, th = self.table
        pad = 30
        x0, y0 = min(tx, min(o.center()[0] - o.half_width() for o in self.obstacles) if self.obstacles else tx) - pad, ty - pad
        x1 = max(tx + tw, max((o.center()[0] + o.half_width() for o in self.obstacles), default=tx + tw)) + pad
        y1 = max(ty + th, max((o.center()[1] + o.half_width() for o in self.obstacles), default=ty + th)) + pad
        W, H = x1 - x0, y1 - y0
        fy = lambda y: y1 - y + y0  # flip: SVG's y grows downward  # noqa: E731
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {W} {H + 18}" '
                 f'width="{int(W * 3)}" height="{int((H + 18) * 3)}" font-family="sans-serif" font-size="6">',
                 f'<rect x="{x0}" y="{y0}" width="{W}" height="{H + 18}" fill="#f4f1ea"/>',
                 f'<rect x="{tx}" y="{fy(ty + th)}" width="{tw}" height="{th}" fill="#d9c7a3" stroke="#8a7350"/>']
        for o in self.obstacles:
            if o.walk:
                pts = " ".join(f"{x:.1f},{fy(y):.1f}" for x, y, *_ in [o.start, *o.walk])
                parts.append(f'<polyline points="{pts}" fill="none" stroke="#e05a8a" stroke-dasharray="2,2" '
                             'stroke-width="0.8"/>')
        for o in self.obstacles:
            fill = "#e05a8a" if o.label else "#888"
            cx, cy = o.center()
            if o.shape == "circle":
                parts.append(f'<circle cx="{o.x}" cy="{fy(o.y)}" r="{o.r}" fill="{fill}"/>')
            else:
                parts.append(f'<rect x="{o.x}" y="{fy(o.y + o.h)}" width="{o.w}" height="{o.h}" fill="{fill}"/>')
            if o.label:
                parts.append(f'<text x="{cx}" y="{fy(cy) - o.half_width() - 2}" text-anchor="middle">{o.label}</text>')
        if len(self.path) > 1:
            pts = " ".join(f"{x:.1f},{fy(y):.1f}" for x, y in self.path)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="#2b6cb0" stroke-width="1.2"/>')
        sx, sy = self.path[0]
        parts.append(f'<circle cx="{sx}" cy="{fy(sy)}" r="2.5" fill="#2b6cb0"/>')
        corners = [self.pose.point(a, l) for a, l in ((-REAR_OVERHANG, -8), (FRONT_REACH, -8), (FRONT_REACH, 8), (-REAR_OVERHANG, 8))]
        parts.append('<polygon points="' + " ".join(f"{x:.1f},{fy(y):.1f}" for x, y in corners)
                     + f'" fill="{"#c53030" if self.fell or self.collided else "#2f855a"}" fill-opacity="0.8"/>')
        marks = {"cliff": "#dd6b20", "blocked": "#805ad5", "fell": "#c53030", "hit": "#c53030"}
        for kind, x, y in self.events:
            parts.append(f'<circle cx="{x:.1f}" cy="{fy(y):.1f}" r="1.8" fill="{marks[kind]}"/>')
        parts.append(f'<text x="{x0 + 3}" y="{y0 + H + 12}" font-size="7">{title}</text>')
        return "\n".join(parts + ["</svg>"])


VLM_SECONDS = 0.5      # the Mac's vision model, per look -- the robot stands still meanwhile
TRACKER_NOISE = 1.0    # degrees: the fast video tracker's jitter
PERSON_MISS = 0.15     # the object detector misses a person in this share of frames


class SimBody:
    """navigate.Body over a World."""

    def __init__(self, world: World):
        self.w = world
        self.cmd = (0.0, 0.0)  # continuous driving: (cm/s, steer degrees)

    # --- continuous driving (navigate.approach's smooth mode) ---------------------
    def start_drive(self, speed: float, steer_deg: float, reverse: bool = False) -> None:
        self.cmd = (speed * CM_PER_S_PER_SPEED * (-1 if reverse else 1), max(-MAX_STEER, min(MAX_STEER, steer_deg)))

    def set_steer(self, steer_deg: float) -> None:
        self.cmd = (self.cmd[0], max(-MAX_STEER, min(MAX_STEER, steer_deg)))

    def stop(self) -> None:
        self.cmd = (0.0, self.cmd[1])

    def wait(self, seconds: float) -> str:
        """Let time pass with the current motor command: "ok", or "fell" / "hit"."""
        t = 0.0
        while t < seconds - 1e-9:
            if self.cmd[0]:
                self.w.step(self.cmd[0], self.cmd[1], TICK)
                if self.w.fell or self.w.collided:
                    return "fell" if self.w.fell else "hit"
            else:
                self.w.time += TICK
            t += TICK
        return "ok"

    def acquire(self, target: str):
        """The slow, smart look (the vision model): finds the target and (re)starts the tracker."""
        self.w.time += VLM_SECONDS
        return self.locate(target)

    def track(self, target: str):
        """The fast tracker: where the target is now, from the video -- or None if lost."""
        s = self.locate(target)
        if s is not None:
            s.bearing += self.w.rng.gauss(0, TRACKER_NOISE)
        return s

    def person(self):
        """The object detector's newest person (~8/s on the real Pi): noisy, sometimes missed."""
        return None if self.w.rng.random() < PERSON_MISS else self.track("person")

    def distance(self) -> float | None:
        d = self.w.ultrasonic()
        return d if d > 1 else None

    def floor(self) -> tuple[bool, bool, bool]:
        return tuple(self.w.floor())

    def look(self, pan_deg: float) -> None:
        self.w.pan = max(-60.0, min(60.0, pan_deg))
        self.w.time += 0.3  # servo travel + settle

    def locate(self, target: str):
        from movement.navigate import Sighting
        for o, bearing, width in self.w.visible():
            if o.label and target.lower() in o.label.lower():
                return Sighting(bearing, width)
        return None

    def glance(self) -> list[str]:
        return sorted({o.label for o, _, _ in self.w.visible() if o.label})

    def heading(self) -> float:
        """Degrees turned, + right -- what the real robot's IMU reports (the pose's h is + left)."""
        return -math.degrees(self.w.pose.h)

    def drive(self, speed: float, steer_deg: float, seconds: float, reverse: bool = False,
              guard_cm: float = 10.0) -> tuple[str, float]:
        """Drive, checking the floor and the ultrasonic every TICK like the real
        loop: stops and reports "cliff" / "blocked" (forward only -- there's no
        rear sensor), or "ok". Returns (result, seconds actually driven)."""
        steer = max(-MAX_STEER, min(MAX_STEER, steer_deg))
        v = speed * CM_PER_S_PER_SPEED * (-1 if reverse else 1)
        t = 0.0
        while t < seconds - 1e-9:
            if not reverse:
                if any(self.floor()):
                    self.w.events.append(("cliff", *self.w.pose.point(FRONT_REACH)))
                    return "cliff", t
                d = self.distance()
                if d is not None and d < guard_cm:
                    self.w.events.append(("blocked", *self.w.pose.point(FRONT_REACH)))
                    return "blocked", t
            self.w.step(v, steer, TICK)
            if self.w.fell or self.w.collided:
                return ("fell" if self.w.fell else "hit"), t
            t += TICK
        return "ok", t
