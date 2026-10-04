"""SafetyMonitor: cliff/too-close detection, polled from services/alive.py's
own loop. The ONE reflex here is safety: back away from a cliff. Something
coming close is not a reflex any more -- the danger/caution latches are
published (common/sensors.py) and openbot-mind decides what, if anything,
to do about it (a fist bump, a bullfight charge, a word, nothing) in its
current mood. The fixed "fist bump on caution / bullfight on danger"
reactions that used to live here were canned behaviour.
"""
from __future__ import annotations

import threading
import time
from collections import deque


from common import events as dash_events
from common import health
from common import journal
from movement.actions import is_real_cliff


DISTANCE_WINDOW = 5    # ~0.25s at the 20Hz poll
# robot_hat's fake grayscale reading, forever, once Picarx() resets the HAT's MCU after ADC objects exist
POISONED_ADC = (2571, 3085, 3599)


def smoothed_distance(samples: list) -> float | None:
    from statistics import median
    valid = [d for d in samples if d is not None and d > 1]
    return median(valid) if len(valid) * 2 > len(samples) and len(valid) >= 3 else None


class SafetyMonitor:
    # Both cliff and too-close latches require the condition (or its
    # clearing) to hold continuously for this long before believing a
    # transition -- fixes an oscillation (a single "clear" reading drives
    # forward, re-detects, repeats forever) and duplicate-firing on one
    # glitchy reading.
    CLEAR_HOLD_SEC = 2.0

    DISTANCE_POLL_INTERVAL = 0.05
    DISTANCE_STALE_SEC = 1.0
    # A desk edge is a static hazard -- no need to re-read the grayscale
    # sensor faster than actions.py's own cliff-aware drive already polls
    # it *while actually driving*, the one time a fresh reading matters.
    GRAYSCALE_POLL_INTERVAL = 0.05

    def __init__(self, car, action_flow, cfg):
        self.car = car
        self.action_flow = action_flow
        self.cfg = cfg

        self._latches = {name: {"active": False, "clear_since": None}
                          for name in ("cliff", "danger", "caution")}
        self._distance: float | None = None
        self._distance_time: float = 0.0
        self._grayscale_error_active = False
        self._glitches, self._glitch_logged_at = 0, 0.0
        self.paused = False  # set by alive while Rocky is asleep
        self._recent: deque = deque(maxlen=DISTANCE_WINDOW)
        self._grayscale: list | None = None
        self._grayscale_time: float = 0.0
        self._grayscale_poll_time: float = 0.0

    def start_distance_watchdog(self) -> None:
        """Picarx.get_distance() bit-bangs the ultrasonic echo pin and can
        hang indefinitely -- isolating the read on its own daemon thread
        means the poll loop only ever touches a plain cached float."""
        def _poll() -> None:
            while True:
                if self.paused:  # asleep: the ultrasonic is off too
                    time.sleep(0.5)
                    continue
                try:
                    self._distance = self.car.get_distance()
                except Exception:
                    self._distance = None
                self._recent.append(self._distance)
                self._distance_time = time.time()
                time.sleep(self.DISTANCE_POLL_INTERVAL)

        threading.Thread(target=_poll, name="distance_watchdog", daemon=True).start()

    def current_distance(self) -> float | None:
        """Median of the last DISTANCE_WINDOW readings, ignoring the sensor's
        -1/-2 codes (-2 = no echo: nothing within range). None unless most of
        the window is valid. One stray short echo used to fire a reflex on its
        own -- the echo is timed in Python, so a busy moment in another thread
        stretches or shrinks it -- and with the motors on, a reflex drives."""
        return smoothed_distance(list(self._recent))

    def _log(self, msg: str) -> None:
        print(msg)
        dash_events.log_event("safety", msg)

    def _react(self, triggers_move: bool, active_msg: str, disabled_msg: str, *actions: str) -> None:
        if triggers_move:
            self.action_flow.add_action(*actions)
            msg = active_msg
            journal.log("reflex", msg)  # body reacted on its own -- part of the life record
        else:
            msg = disabled_msg
        self._log(msg)

    def latch_active(self, name: str) -> bool:
        return self._latches[name]["active"]

    def snapshot(self) -> dict:
        """Last-known sensor readings + latch state, for common.sensors to
        publish -- openbot-mind and the dashboard read that file instead of
        this instance directly, since they run in separate processes."""
        return {
            "distance": self.current_distance(),
            "grayscale": self._grayscale,
            "latches": {name: self.latch_active(name) for name in self._latches},
        }

    def _latch(self, name: str, active: bool) -> bool:
        """Edge-triggered: fires (returns True) once on the false->true
        transition, resets once `active` has stayed continuously False for
        CLEAR_HOLD_SEC."""
        latch = self._latches[name]
        if active:
            latch["clear_since"] = None
            if not latch["active"]:
                latch["active"] = True
                return True
            return False

        if not latch["active"]:
            return False
        now = time.time()
        if latch["clear_since"] is None:
            latch["clear_since"] = now
        elif now - latch["clear_since"] >= self.CLEAR_HOLD_SEC:
            latch["active"] = False
            latch["clear_since"] = None
        return False

    def trigger_cliff(self) -> None:
        health.record_success("openbot-alive", min_interval_s=5.0)
        if time.time() - self._grayscale_poll_time < self.GRAYSCALE_POLL_INTERVAL:
            return
        self._grayscale_poll_time = time.time()
        try:
            grayscale = self.car.get_grayscale_data()
            if list(grayscale[:2]) == list(POISONED_ADC[:2]):
                return  # robot_hat's known bogus ADC values (reset_mcu after ADC init) -- not a reading
            partial_low = self.car.get_cliff_status(grayscale)
            cliff = is_real_cliff(grayscale, self.cfg.CLIFF_REFERENCE)
            if partial_low and not cliff:
                # These fire about once a second on this module (one channel
                # reading 0) -- log a summary at most once a minute, or they
                # flood the dashboard and push real events out of it.
                self._glitches += 1
                if time.time() - self._glitch_logged_at >= 60:
                    self._log(f"grayscale glitches: {self._glitches} in the last minute "
                              f"(latest {grayscale}) -- ignored, cliff check unaffected")
                    self._glitches, self._glitch_logged_at = 0, time.time()
                cliff = self.latch_active("cliff")
        except Exception as e:
            if not self._grayscale_error_active:
                self._grayscale_error_active = True
                self._log(f"grayscale sensor read failed: {e}")
            return
        self._grayscale_error_active = False
        self._grayscale, self._grayscale_time = grayscale, time.time()

        if self._latch("cliff", cliff):
            self._react(self.cfg.CLIFF_TRIGGERS_MOVE,
                        f"cliff detected -- backing away (grayscale: {grayscale}, reference: {self.cfg.CLIFF_REFERENCE})",
                        f"cliff detected (movement disabled) (grayscale: {grayscale}, reference: {self.cfg.CLIFF_REFERENCE})",
                        "backward")

    def trigger_too_close(self) -> None:
        """Keeps the danger/caution latches current (debounced, see _latch).
        No reaction here: they're published for openbot-mind, which decides."""
        health.record_success("openbot-alive", min_interval_s=5.0)
        distance = self.current_distance()
        valid = distance is not None and (time.time() - self._distance_time) <= self.DISTANCE_STALE_SEC
        in_danger = valid and distance <= self.cfg.DANGER_DISTANCE
        in_caution = valid and self.cfg.DANGER_DISTANCE < distance <= self.cfg.SAFE_DISTANCE
        if self._latch("danger", in_danger):
            self._log(f"danger: {distance:.1f}cm -- something right in front (for the mind to think about)")
        if self._latch("caution", in_caution):
            self._log(f"caution: {distance:.1f}cm -- something near (for the mind to think about)")

    def poll(self) -> None:
        """Call once per loop tick from services/alive.py -- cliff first,
        falling off a desk is more urgent than a hand nearby."""
        self.trigger_cliff()
        self.trigger_too_close()
