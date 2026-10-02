"""Deterministic movement-keyword fallback.

A small local LLM reliably hears a movement command but doesn't reliably
request the matching action (confirmed live on the original desk-bot: "go
back" got a reply with no action, or an unrelated one). A stationary reply
to an explicit move command is a much worse failure than an occasional
over-eager trigger, so movement is decided here, not by the LLM -- the
reactive voice-loop turn uses this function's result as the sole source of
truth for whether the robot moves, hardware-generic and persona-independent.
"""
from __future__ import annotations

import re

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bturn\s+left\b", re.I), "turn left"),
    (re.compile(r"\bturn\s+right\b", re.I), "turn right"),
    (re.compile(r"\b(go\s+)?back(ward)?s?\b|\breverse\b", re.I), "backward"),
    (re.compile(r"\b(go\s+)?forward\b|\bahead\b", re.I), "forward"),
    (re.compile(r"\bfist\s*bump\b|\bbump\s+fists?\b", re.I), "fist bump"),
    (re.compile(r"\bbull\s*fight\b", re.I), "bullfight"),
)


def detect_movement_keyword(heard_text: str | None) -> str | None:
    """Return a movement-action name if heard_text plainly asks to move.
    heard_text is the STT transcript of what the person said, not the
    LLM's reply. Returns None on no match (including empty/None input)."""
    if not heard_text:
        return None
    for pattern, action in _PATTERNS:
        if pattern.search(heard_text):
            return action
    return None


_AFFIRMATIVE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|okay|ok|alright|definitely|absolutely)\b", re.I
)


def is_affirmative(heard_text: str | None) -> bool:
    """True if heard_text is a short "yes"-shaped answer -- only meaningful
    in the specific context of the bot's own previous reply having just
    offered something (e.g. a fist bump)."""
    return bool(heard_text) and _AFFIRMATIVE.match(heard_text) is not None


def demo() -> None:
    assert detect_movement_keyword("go back please") == "backward"
    assert detect_movement_keyword("turn left now") == "turn left"
    assert detect_movement_keyword("what's the weather") is None
    assert detect_movement_keyword(None) is None
    assert is_affirmative("yes please")
    assert not is_affirmative("no thanks")


if __name__ == "__main__":
    demo()
    print("movement/keywords: ok")
