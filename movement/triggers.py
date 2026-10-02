"""SafetyMonitor: cliff/too-close detection, polled from services/alive.py's
own loop (there's no VoiceAssistant trigger-polling framework anymore --
alive.py owns this loop directly). Hardware-generic; the persona is
consulted for its system prompt (to generate the spoken reaction line) and
its text transform.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque

from picarx.preset_actions import ActionStatus

from common import cognition
from common import events as dash_events
from common import health
from common import policy
from common import state
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

    def __init__(self, car, action_flow, speak_fn, cfg, persona):
        self.car = car
        self.action_flow = action_flow
        self.speak_fn = speak_fn  # callable(text) -> None, routed through policy
        self.cfg = cfg
        self.persona = persona

        self._latches = {name: {"active": False, "clear_since": None}
                          for name in ("cliff", "danger", "caution")}
        self._last_reaction_at = 0.0
        self._bullfighting = False
        self._reaction_in_flight = False
        self._distance: float | None = None
        self._distance_time: float = 0.0
        self._grayscale_error_active = False
        self._glitches, self._glitch_logged_at = 0, 0.0
        self._playful, self._playful_checked = True, 0.0
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

    def _safety_result_speaks(self) -> bool:
        return bool(self.cfg.SAFETY_TRIGGERS_SPEAK)

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

    def _playful_ok(self) -> bool:
        """Fist bump / bullfight are play, not safety: off while asleep, in
        quiet hours, or mid-conversation -- someone leaning in to talk isn't
        an intrusion (the cliff reflex is NOT gated by this). Cached -- polled at 20Hz."""
        if time.time() - self._playful_checked > 5.0:
            quiet = (self.cfg.QUIET_HOURS_START, self.cfg.QUIET_HOURS_END)
            self._playful = policy.evaluate("presence", quiet_hours=quiet).allowed and not state.conversation_active()
            self._playful_checked = time.time()
        return self._playful

    def _reaction_cooldown_ok(self) -> bool:
        return time.time() - self._last_reaction_at >= self.cfg.REACTION_COOLDOWN_SEC

    def _generate_reaction_line(self, action: str) -> str:
        """One short, in-character reaction line from the LLM -- no
        fallback content: if the LLM server's unreachable, say so plainly rather
        than pretending with canned flavor text. No JSON schema either
        (unlike a normal turn) -- this needs a short phrase, not a
        structured {reply, tone_action}; the physical gesture is already
        queued separately and doesn't wait on this."""
        is_bullfight = action == self.cfg.BULLFIGHT_SEQUENCE[-1]
        situation = "got too close and you're charging at it" if is_bullfight \
            else "came near for a friendly bump"
        prompt_text = self.persona.system_prompt_template(self.cfg.ALLOWED_ACTIONS, self.cfg.STATIONARY_ACTIONS)
        reaction_prompt = journal.inject(
            f"Something just {situation}. React out loud with one short, in-character line "
            "-- a few words, not a full sentence."
        )
        # Schema-constrained to just the words: free text under the persona's
        # {reply, tone_action} system prompt came back as the literal
        # 'Reply: "Hey there!" tone_action: ...' and got spoken out loud.
        result = cognition.ask(
            self.cfg.LLM_BASE_URL, self.cfg.LLM_MODEL, reaction_prompt,
            system=prompt_text, timeout_s=8.0,
            json_schema={"type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"]},
        )
        if result.status != cognition.AVAILABLE:
            return self.persona.transform("Unable to think.")
        try:
            line = str(json.loads(result.text)["line"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            return ""
        return self.persona.transform(line) if line else ""

    def _speak_reaction(self, action: str) -> None:
        """In-the-moment reaction line for a just-triggered proximity
        gesture, generated fresh by the LLM. Runs on its own thread: this
        poll loop must not block on an LLM call + TTS playback, or cliff
        detection stalls for that whole stretch -- the physical gesture
        (queued separately, see _react()) never waits on this."""
        if self._reaction_in_flight:
            return

        quiet = (self.cfg.QUIET_HOURS_START, self.cfg.QUIET_HOURS_END)
        if not policy.evaluate("audio", quiet_hours=quiet).allowed:
            return  # same quiet-hours gate as every other audible path -- a 3am hand wave stays silent
        if state.conversation_active():
            return  # don't blurt "Back away!" over a conversation (the gesture still happens)

        def _run() -> None:
            self._reaction_in_flight = True
            try:
                text = self._generate_reaction_line(action)
                if not text:
                    return
                dash_events.log_event("reply", f"{self.persona.name.lower()}: {text}")
                journal.log("said", text)
                self.speak_fn(text)
            finally:
                self._reaction_in_flight = False

        threading.Thread(target=_run, name="reaction_speech", daemon=True).start()

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
            if self._safety_result_speaks():
                self.speak_fn("<<<Cliff detected -- edge nearby>>>")

    def trigger_too_close(self) -> None:
        health.record_success("openbot-alive", min_interval_s=5.0)
        distance = self.current_distance()
        stale = (time.time() - self._distance_time) > self.DISTANCE_STALE_SEC
        valid = distance is not None and not stale
        dist_str = f"{distance:.1f}" if distance is not None else "?"

        in_danger = valid and distance <= self.cfg.DANGER_DISTANCE
        in_caution = valid and self.cfg.DANGER_DISTANCE < distance <= self.cfg.SAFE_DISTANCE

        danger_edge = self._latch("danger", in_danger)
        danger_now = self.latch_active("danger")

        if self._bullfighting:
            # Confirmed live: the ultrasonic sensor's own -1/-2 glitch
            # codes (filtered to invalid/"?" by current_distance()) fire
            # most often exactly when something is very close -- the one
            # moment this loop most needs a real reading. The latch alone
            # doesn't distinguish "still close" from "no idea right now,"
            # so a repeat charge needs a FRESH valid reading, not just an
            # unexpired latch -- otherwise a string of glitchy ticks with
            # nothing between them keeps it charging at an unconfirmed
            # target. Not the same as ending the episode: a momentarily
            # unreadable sensor pauses the charging, it doesn't declare
            # the encounter over (that's still the latch's own
            # CLEAR_HOLD_SEC, below).
            if not valid:
                return
            # A latched-but-stale danger reading (e.g. a cliff-triggered
            # backup moved the sensor, not the object) should never win
            # over an active cliff -- treat it the same as danger
            # actually clearing.
            if danger_now and not self.latch_active("cliff"):
                if self.action_flow.status == ActionStatus.STANDBY:
                    self.action_flow.add_action(self.cfg.BULLFIGHT_SEQUENCE[-1])
                    self._log(f"danger: {dist_str}cm -- still close, charging again")
                return
            self._bullfighting = False
            self._last_reaction_at = time.time()
            reason = "cliff nearby" if self.latch_active("cliff") else f"{dist_str}cm"
            self._log(f"danger clear ({reason}) -- bullfight episode over, cooldown started")
            # Fall through -- same tick can still start a fist bump below.

        elif danger_now and not self.latch_active("cliff"):
            if self._reaction_cooldown_ok():
                if self.cfg.DANGER_TRIGGERS_MOVE and self._playful_ok():
                    self._bullfighting = True
                    self._speak_reaction(self.cfg.BULLFIGHT_SEQUENCE[-1])
                    self._react(True, f"danger: {dist_str}cm -- {' then '.join(self.cfg.BULLFIGHT_SEQUENCE)}",
                                "", *self.cfg.BULLFIGHT_SEQUENCE)
                # (no log when skipped or on cooldown -- those lines were most of the dashboard)
            if danger_edge and self._safety_result_speaks():
                self.speak_fn(f"<<<Ultrasonic sense danger: {dist_str}cm>>>")
            return

        if self._latch("caution", in_caution):
            if self._reaction_cooldown_ok():
                self._last_reaction_at = time.time()
                playful = self.cfg.CAUTION_TRIGGERS_MOVE and self._playful_ok()
                if playful:
                    self._speak_reaction(self.cfg.FIST_BUMP_SEQUENCE[-1])
                if playful:
                    self._react(True, f"caution: {dist_str}cm -- {' then '.join(self.cfg.FIST_BUMP_SEQUENCE)}",
                                "", *self.cfg.FIST_BUMP_SEQUENCE)
            if self._safety_result_speaks():
                self.speak_fn(f"<<<Ultrasonic sense caution: {dist_str}cm>>>")

    def poll(self) -> None:
        """Call once per loop tick from services/alive.py -- cliff first,
        falling off a desk is more urgent than a hand nearby."""
        self.trigger_cliff()
        self.trigger_too_close()
