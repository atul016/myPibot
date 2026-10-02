"""Single chokepoint every spoken or expressive action passes through --
reactive (wake-listen) and autonomous (mind) alike -- so quiet hours and
the motion-confirm gate can't be bypassed by a caller that forgot to check.
Mirrors SPARK's tool-voice sink: one gate, not "every caller remembers to."
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .state import load_session

Effect = str  # "audio" | "motion" | "presence" | "other"


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


def is_quiet_hours(now: dt.datetime, start_h: int, end_h: int) -> bool:
    """start_h/end_h in local 24h time; wraps past midnight (e.g. 21->8).
    start_h == end_h means quiet hours are disabled."""
    if start_h == end_h:
        return False
    h = now.hour
    if start_h < end_h:
        return start_h <= h < end_h
    return h >= start_h or h < end_h


def evaluate(effect: Effect, *, quiet_hours: tuple[int, int], now: dt.datetime | None = None) -> Verdict:
    now = now or dt.datetime.now()

    # Checked before session load, and for "presence" too -- servo noise
    # from an idle gesture is exactly the kind of thing quiet hours exist
    # to suppress, even though it's not a drive command. Wall-clock only,
    # so an unreadable session can't accidentally skip this check.
    if effect in ("audio", "motion", "presence") and is_quiet_hours(now, *quiet_hours):
        return Verdict(False, "quiet hours")

    session = load_session()
    if not session:
        # Fails CLOSED for audio/motion: an unreadable session must never
        # be read as "no flags set, go ahead." A presence-only action (a
        # gaze shift, an LED) is harmless enough to fail open instead.
        if effect in ("audio", "motion"):
            return Verdict(False, "session state unreadable")
        return Verdict(True)

    if session.get("asleep") and effect in ("audio", "motion", "presence"):
        return Verdict(False, "asleep")

    if effect == "motion" and not session.get("confirm_motion_allowed", True):
        return Verdict(False, "motion not confirmed")

    return Verdict(True)


def demo() -> None:
    assert is_quiet_hours(dt.datetime(2026, 1, 1, 22, 0), 21, 8)
    assert is_quiet_hours(dt.datetime(2026, 1, 1, 3, 0), 21, 8)
    assert not is_quiet_hours(dt.datetime(2026, 1, 1, 12, 0), 21, 8)
    assert not is_quiet_hours(dt.datetime(2026, 1, 1, 12, 0), 9, 9)

    night, day = dt.datetime(2026, 1, 1, 22, 0), dt.datetime(2026, 1, 1, 12, 0)
    assert not evaluate("presence", quiet_hours=(21, 8), now=night).allowed
    assert evaluate("presence", quiet_hours=(21, 8), now=day).allowed
    assert not evaluate("audio", quiet_hours=(21, 8), now=night).allowed

    global load_session
    real = load_session
    try:
        load_session = lambda: {"asleep": True}
        assert evaluate("presence", quiet_hours=(21, 8), now=day).reason == "asleep"
        assert evaluate("other", quiet_hours=(21, 8), now=day).allowed  # bookkeeping still fine asleep
    finally:
        load_session = real


if __name__ == "__main__":
    demo()
    print("policy: ok")
