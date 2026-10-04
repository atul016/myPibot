"""Single chokepoint every spoken or expressive action passes through --
reactive (wake-listen) and autonomous (mind) alike -- so sleep and
the motion-confirm gate can't be bypassed by a caller that forgot to check.
Mirrors SPARK's tool-voice sink: one gate, not "every caller remembers to."
"""
from __future__ import annotations

from dataclasses import dataclass

from .state import load_session

Effect = str  # "audio" | "motion" | "presence" | "other"


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


def evaluate(effect: Effect) -> Verdict:
    """No clock here (quiet hours were removed 2026-10-03): the only ways to
    hush Rocky are "go to sleep" and the motion-confirm flag."""
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
    global load_session
    real = load_session
    try:
        load_session = lambda: {"asleep": True}
        assert evaluate("presence").reason == "asleep"
        assert evaluate("other").allowed  # bookkeeping still fine asleep
        load_session = lambda: {"asleep": False}
        assert evaluate("audio").allowed  # awake at any hour: no clock gate
        load_session = lambda: {"confirm_motion_allowed": False}
        assert evaluate("motion").reason == "motion not confirmed" and evaluate("audio").allowed
    finally:
        load_session = real


if __name__ == "__main__":
    demo()
    print("policy: ok")
