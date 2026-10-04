"""Deterministic movement-keyword fallback.

A small local LLM reliably hears a movement command but doesn't reliably
request the matching action (confirmed live on the original desk-bot: "go
back" got a reply with no action, or an unrelated one). A stationary reply
to an explicit move command is a much worse failure than an occasional
over-eager trigger, so movement is decided here, not by the LLM -- the
reactive voice-loop turn uses this function's result as the sole source of
truth for whether the robot moves, hardware-generic and persona-independent.

Driving needs the WHOLE utterance to be the command (after dropping the
bot's name and politeness words), like common/commands.py's navigation --
searching inside sentences drove the car on "welcome back", "I'm back" and
"go ahead and tell me". Fist bump and bullfight are unusual enough to be
found anywhere in a sentence.
"""
from __future__ import annotations

import re

from common.persona import CURRENT

# "Rocky, can you go back a little bit please?" -> "go back"
_FILLER = {CURRENT, "hey", "hi", "ok", "okay", "please", "now", "just", "can", "could", "would", "you",
           "a", "little", "bit", "again"}
_DRIVE = {phrase: action for action, phrases in {
    "forward": {"forward", "go forward", "move forward", "come forward", "drive forward",
                "ahead", "move ahead", "drive ahead", "go straight", "go straight ahead"},
    "backward": {"back", "go back", "move back", "back up", "backward", "backwards", "go backward",
                 "go backwards", "move backward", "move backwards", "drive back", "reverse", "back off"},
    "turn left": {"turn left", "go left"},
    "turn right": {"turn right", "go right"},
    "dance": {"dance", "dance for me", "do dance", "do your dance", "show me dance", "let's dance", "dance time"},
}.items() for phrase in phrases}
_ANYWHERE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfist\s*bump\b|\bbump\s+fists?\b", re.I), "fist bump"),
    (re.compile(r"\bbull\s*fight\b", re.I), "bullfight"),
)


def detect_movement_keyword(heard_text: str | None) -> str | None:
    """Return a movement-action name if heard_text plainly asks to move.
    heard_text is the STT transcript of what the person said, not the
    LLM's reply. Returns None on no match (including empty/None input)."""
    if not heard_text:
        return None
    words = [w for w in re.findall(r"[a-z']+", heard_text.lower()) if w not in _FILLER]
    if " ".join(words) in _DRIVE:
        return _DRIVE[" ".join(words)]
    for pattern, action in _ANYWHERE:
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
    assert detect_movement_keyword(f"{CURRENT.capitalize()}, move ahead.") == "forward"
    assert detect_movement_keyword("Come forward.") == "forward"
    assert detect_movement_keyword("Can you go back a little bit, please?") == "backward"
    for chat in ("Welcome back!", "I'm back.", "My back hurts.", "Go ahead and tell me a story.",
                 "Go ahead.", "Come back.", "Right.", "I went back home", "turn left and then right"):
        assert detect_movement_keyword(chat) is None, chat
    assert detect_movement_keyword("Let's do a fist bump!") == "fist bump"
    assert detect_movement_keyword(f"{CURRENT.capitalize()}, dance for me please!") == "dance"
    assert detect_movement_keyword("Do you like to dance?") is None
    assert detect_movement_keyword(None) is None
    assert is_affirmative("yes please")
    assert not is_affirmative("no thanks")


if __name__ == "__main__":
    demo()
    print("movement/keywords: ok")
