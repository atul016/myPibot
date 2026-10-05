"""The body's inertial sense: a DFRobot Gravity 6 DOF IMU (SEN0692) on I2C --
which way the body has turned, how it's tilted, whether it's moving. openbot-alive
reads it 25 times a second and publishes Tracker.snapshot() in sensors.json["imu"].

  axes      config.IMU_AXES: the sensor axes that point forward, right and down, as
            mounted ("+x,+y,+z": the board flat, its X arrow forward; Walle: "-y,+x,+z").
  heading   degrees turned, right-positive like the head's pan: the turn rate about
            the vertical -- the gyro projected on gravity, so it stays right while he's
            held at an angle -- added up while he moves. Still, nothing is added (the
            rate is noise then) and the gyro's bias is learned instead.
  turned_by_others  the part of the heading that changed while he wasn't moving
            himself (a gesture, a move, a drive, the cliff reflex): someone turned him.
  epoch     when this count started -- openbot-alive restarting zeroes the heading.
"""
from __future__ import annotations

import contextlib
import math
import time

ADDR = 0x4A
PID, MODE, ACC_RANGE, GYR_RANGE, DATA = 0x0001, 0x0005, 0x0006, 0x0007, 0x0010  # 16-bit registers
ACC_G, GYR_DPS = 8.0, 2000.0  # the ranges Sensor() sets: +-8 g (code 2), +-2000 deg/s (code 4)


class Sensor:
    """The board's own protocol (DFRobot_Multi_DOF_IMU): write the register's address,
    low byte first, wait 10 ms while its firmware fetches it, then read. `lock`:
    held around each transfer (not the wait) -- openbot-alive's movement.bus.BUS_LOCK."""

    def __init__(self, bus: int = 1, addr: int = ADDR, lock=None) -> None:
        from smbus2 import SMBus, i2c_msg  # deferred: the Tracker below needs no hardware
        self._msg, self.addr, self.bus = i2c_msg, addr, SMBus(bus)
        self._lock = lock or contextlib.nullcontext()
        pid = self._read(PID, 2)
        if pid[0] | pid[1] << 8 < 0x0006:
            raise OSError(f"no 6 DOF IMU at {addr:#x} (product id {pid})")
        for reg, value in ((MODE, 0x02), (ACC_RANGE, 2), (GYR_RANGE, 4)):  # normal mode, +-8 g, +-2000 deg/s
            self._transfer(self._msg.write(addr, [reg & 0xFF, reg >> 8, value, 0]))
            time.sleep(0.02)

    def _transfer(self, msg) -> None:
        with self._lock:
            self.bus.i2c_rdwr(msg)

    def _read(self, reg: int, n: int) -> list[int]:
        self._transfer(self._msg.write(self.addr, [reg & 0xFF, reg >> 8]))
        time.sleep(0.01)
        msg = self._msg.read(self.addr, n)
        self._transfer(msg)
        return list(msg)

    def sample(self) -> tuple[list[float], list[float]]:
        """(acceleration in g, turn rate in deg/s), sensor axes x y z."""
        d = self._read(DATA, 12)
        v = [(d[i] | d[i + 1] << 8) - (0x10000 if d[i + 1] & 0x80 else 0) for i in range(0, 12, 2)]
        return [x * ACC_G / 32768 for x in v[:3]], [x * GYR_DPS / 32768 for x in v[3:]]


def axes(spec: str) -> list[tuple[int, int]]:
    """"-y,+x,+z" -> (sign, sensor axis) for forward, right, down."""
    out = [(-1 if p.strip().startswith("-") else 1, "xyz".index(p.strip()[-1].lower())) for p in spec.split(",")]
    if len(out) != 3 or sorted(i for _, i in out) != [0, 1, 2]:
        raise ValueError(f"IMU axes {spec!r}: name x, y and z once each, e.g. \"-y,+x,+z\"")
    return out


def to_body(v: list[float], ax: list[tuple[int, int]]) -> list[float]:
    return [sign * v[i] for sign, i in ax]


class Tracker:
    STILL_DPS = 2.0     # every axis turning slower than this (after its bias), and
    STILL_G = 0.04      # the acceleration within this of 1 g: still
    BIAS_RATE = 0.02    # per still sample, how far the learned bias moves toward the reading
    DOWN_RATE = 0.2     # per sample, how far "which way is down" moves toward the reading
    HOLD_S = 2.0        # "moving" stays true this long after the last motion

    def __init__(self, now: float) -> None:
        self.epoch, self.last = now, None
        self.heading = self.external = 0.0
        self.bias = [0.0, 0.0, 0.0]
        self.down = [0.0, 0.0, 1.0]  # body axes (forward, right, down) -- from the accelerometer, smoothed
        self.moved_ts = self.moved_by_others_ts = 0.0

    def update(self, acc: list[float], gyr: list[float], now: float, self_moving: bool) -> None:
        """One sample in body axes (forward, right, down): acc in g, gyr in deg/s."""
        dt = now - self.last if self.last is not None and now - self.last < 0.5 else 0.0  # never across a stall
        self.last = now
        g = math.sqrt(sum(a * a for a in acc)) or 1.0
        self.down = [d + self.DOWN_RATE * (-a / g - d) for d, a in zip(self.down, acc)]  # at rest it reads "up"
        n = math.sqrt(sum(d * d for d in self.down)) or 1.0
        self.down = [d / n for d in self.down]
        rate = [w - b for w, b in zip(gyr, self.bias)]
        if all(abs(r) < self.STILL_DPS for r in rate) and abs(g - 1.0) < self.STILL_G:
            if not self_moving:
                self.bias = [b + self.BIAS_RATE * (w - b) for b, w in zip(self.bias, gyr)]
            return
        self.moved_ts = now
        turn = sum(r * d for r, d in zip(rate, self.down)) * dt  # about the vertical: + is to the right
        self.heading += turn
        if not self_moving:
            self.external += turn
            self.moved_by_others_ts = now

    def snapshot(self, now: float) -> dict:
        fwd, right, down = self.down
        return {"heading": round(self.heading, 1), "turned_by_others": round(self.external, 1),
                "tilt": round(math.degrees(math.acos(max(-1.0, min(1.0, down)))), 1),
                "pitch": round(math.degrees(math.asin(max(-1.0, min(1.0, -fwd)))), 1),   # + nose up
                "roll": round(math.degrees(math.asin(max(-1.0, min(1.0, right)))), 1),   # + right side down
                "moving": now - self.moved_ts < self.HOLD_S,
                "moved_by_others": now - self.moved_by_others_ts < self.HOLD_S,
                "epoch": self.epoch, "ts": now}


def posture_words(snap: dict) -> str | None:
    """"tilted 30 degrees, nose up" -- None when he's about level (the usual)."""
    if snap["tilt"] < 15:
        return None
    if snap["tilt"] > 60:
        return f"lying on your {'back' if snap['pitch'] > 45 else 'nose' if snap['pitch'] < -45 else 'side'}"
    which = max(("nose up" if snap["pitch"] > 0 else "nose down", abs(snap["pitch"])),
                (f"{'right' if snap['roll'] > 0 else 'left'} side down", abs(snap["roll"])), key=lambda w: w[1])[0]
    return f"tilted {snap['tilt']:.0f} degrees, {which}"


def demo() -> None:
    assert axes("-y,+x,+z") == [(-1, 1), (1, 0), (1, 2)]
    assert to_body([0.1, -0.2, 0.9], axes("-y,+x,+z")) == [0.2, 0.1, 0.9]  # forward is -y on Walle
    for bad in ("x,y", "x,x,z", "+x,+y,+w"):
        try:
            axes(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    t, now = Tracker(0.0), 0.0
    flat = [0.0, 0.0, -1.0]  # at rest the accelerometer reads "up": -1 g along down
    for _ in range(300):     # still, with a gyro that reads 0.5 deg/s too much on the vertical
        now += 0.04
        t.update(flat, [0.0, 0.0, 0.5], now, False)
    assert abs(t.bias[2] - 0.5) < 0.01 and t.heading == 0.0  # learned, and nothing added while still
    for _ in range(25):      # turned 90 deg/s to the right for a second, by someone
        now += 0.04
        t.update(flat, [0.0, 0.0, 90.5], now, False)
    s = t.snapshot(now)
    assert abs(s["heading"] - 90) < 4 and abs(s["turned_by_others"] - 90) < 4 and s["moving"], s
    for _ in range(25):      # turns 90 to the left on its own: heading follows, "by others" doesn't
        now += 0.04
        t.update(flat, [0.0, 0.0, -89.5], now, True)
    s = t.snapshot(now)
    assert abs(s["heading"]) < 6 and abs(s["turned_by_others"] - 90) < 4, s
    assert not t.snapshot(now + 3)["moving"] and posture_words(s) is None
    nose_up = [math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))]  # held 30 deg nose up
    held, now = Tracker(0.0), 0.0
    for _ in range(60):      # turning about the room's vertical: the gyro sees it split across axes
        now += 0.04
        held.update(nose_up, [-45 * math.sin(math.radians(30)), 0.0, 45 * math.cos(math.radians(30))], now, False)
    s = held.snapshot(now)
    assert abs(s["tilt"] - 30) < 1 and abs(s["pitch"] - 30) < 1 and abs(s["heading"] - 45 * 2.36) < 6, s
    assert posture_words(s) == "tilted 30 degrees, nose up"
    side = Tracker(0.0)
    for i in range(30):
        side.update([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], i * 0.04, False)  # right side down, lying still
    assert posture_words(side.snapshot(1.2)) == "lying on your side"


if __name__ == "__main__":
    demo()
    print("imu: ok")
