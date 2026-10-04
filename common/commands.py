"""Spoken mode commands -- matched in code, never left to the LLM:

    "be quiet" / "quiet" / "stop session" -> end the conversation. Rocky keeps
                         living in the background (watching, thinking, may speak up).
    "go to sleep"      -> everything off but the voice listener (and the nightly
                         memory review); ONLY "Rocky, wake up" wakes it
    "louder" / "softer" -> speaker volume
    "go to the <thing>" / "come here" / "explore" / "stop" -> driving (navigation())

Short, plain phrases on purpose (the main user prefers simple words). The
whole utterance must BE the command (after dropping "rocky"/"please"/
punctuation) -- "I'm too quiet today" must not switch modes.
"""
from __future__ import annotations

import re

from .persona import CURRENT

STOP_SESSION, SLEEP, LOUDER, SOFTER = "stop_session", "sleep", "louder", "softer"
WAKE = "wake"  # texted only: a spoken "wake up" goes through the wake word
# Texted on WhatsApp (services/chat.py), exactly; wake-listen carries them out as if
# they had been said out loud. "quite": how "quiet" often gets typed, too.
SLASH = {"/be-quiet": STOP_SESSION, "/be-quite": STOP_SESSION, "/go-to-sleep": SLEEP, "/wake-up": WAKE}

_PHRASES = {
    STOP_SESSION: {"stop session", "end session", "stop the session", "end the session", "session stop",
                   "session over", "stop conversation", "end conversation",
                   # "be quiet" ends the conversation too (2026-10-02: one command, not a separate quiet mode).
                   # "quite" etc: how Whisper actually transcribed a spoken "quiet".
                   "quiet", "be quiet", "quiet now", "shh", "shush", "silence", "stay quiet", "quiet please",
                   "quite", "be quite", "quite now", "quite please"},
    SLEEP: {"go to sleep", "sleep", "sleep now", "go sleep", "good night", "goodnight", "sleep mode"},
    LOUDER: {"louder", "volume up", "speak louder", "talk louder", "more volume", "turn it up", "louder please",
             "a bit louder", "little louder", "increase volume", "i can't hear you"},
    # "quieter" is volume, NOT the end of the conversation
    SOFTER: {"softer", "volume down", "speak softer", "talk softer", "quieter", "lower volume", "less volume",
             "turn it down", "not so loud", "too loud", "a bit softer", "little softer", "decrease volume",
             "speak quieter", "talk quieter", "a bit quieter"},
}


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def parse(text: str, extra_sleep: list[str] | None = None) -> str | None:
    """The command `text` is, or None for ordinary speech."""
    t = _normalize(text)
    sleep_words = {_normalize(w) for w in extra_sleep or []}
    candidates = {t, " ".join(w for w in t.split() if w not in {CURRENT, "hey", "hi", "please", "ok", "okay"})}
    for cand in candidates:
        for cmd, phrases in _PHRASES.items():
            if cand in phrases or (cmd == SLEEP and cand in sleep_words):
                return cmd
    # "stop session" can arrive glued to other words ("okay stop session thanks")
    if re.search(r"\b(stop|end) (the )?session\b", t) and len(t.split()) <= 5:
        return STOP_SESSION
    return None


# Repeated verbs are fine ("go, move, go to the pink toy"); still anchored, so
# "I don't want you to go to the kitchen" isn't a drive.
_GO_TO = re.compile(r"^(?:please )?(?:head|(?:(?:go|drive|move|roll) )*(?:go|drive|move|roll))(?: over)? "
                    r"to(?:wards?)? (?:the |that |my |a |your )?(.+?)(?: please)?$")
_COME = {"come here", "come to me", "come over here", "come over", "come", "come here please", "come closer",
         "come back", "come back here"}
_EXPLORE = {"explore", "explore the room", "explore around", "go explore", "explore the table", "go exploring",
            "explore a bit", "look around the room"}
_STOP = {"stop", "stop moving", "stop driving", "halt", "freeze", "stop it", "stop stop", "wait", "hold on",
         "don't move", "stay", "stay there", "stop there"}


def navigation(text: str) -> tuple[str, str | None] | None:
    """A driving request: ("approach", thing) / ("explore", None) / ("stop", None),
    or None. "come here" drives to the person ("person" is what the camera
    looks for)."""
    if parse(text):  # "go to sleep" is a mode command, not a place to drive to
        return None
    t = " ".join(w for w in _normalize(text).split() if w not in (CURRENT, "hey", "ok", "okay"))
    if t in _STOP:
        return "stop", None
    if t in _COME:
        return "approach", "person"
    if t in _EXPLORE:
        return "explore", None
    m = _GO_TO.match(t)
    if m and len(m.group(1).split()) <= 4:  # a thing, not a sentence ("go to the store and buy milk")
        return "approach", m.group(1)
    return None


def demo() -> None:
    n = CURRENT.capitalize()  # the wake name gets dropped, whichever persona is active
    assert parse("Stop session.") == STOP_SESSION and parse(f"{n}, stop the session please") == STOP_SESSION
    assert parse("okay stop session thanks") == STOP_SESSION
    assert parse("Be quiet.") == STOP_SESSION and parse(f"{n}, quiet") == STOP_SESSION and parse("quite") == STOP_SESSION
    assert parse("Go to sleep") == SLEEP and parse(f"{n} go to sleep.") == SLEEP and parse("Good night!") == SLEEP
    # ordinary speech is NOT a command
    assert parse("I'm too quiet today") is None and parse("I want to sleep early tonight") is None
    assert parse("Can you talk about the weather?") is None and parse("tell me about your session") is None
    assert parse("hey, nap time", extra_sleep=["nap time"]) == SLEEP
    assert parse("Louder!") == LOUDER and parse(f"{n}, volume up") == LOUDER
    assert parse("Quieter please") == SOFTER and parse("too loud") == SOFTER
    assert navigation(f"{n}, go to the pink toy.") == ("approach", "pink toy")
    assert navigation("Go, move, go to the pink toy.") == ("approach", "pink toy")
    assert navigation("I don't want you to go to the kitchen") is None
    assert navigation("move head to the left") is None  # a head move, not a drive
    assert navigation("Come back.") == ("approach", "person")
    assert navigation("Drive over to my mug please") == ("approach", "mug")
    assert navigation("Come here!") == ("approach", "person") and navigation("explore the room") == ("explore", None)
    assert navigation("Stop!") == ("stop", None) and navigation(f"{n}, stop moving") == ("stop", None)
    assert navigation("I want to go to the store and buy some milk") is None
    assert navigation("What is this?") is None and parse("stop") is None  # bare "stop" isn't "stop session"
    assert navigation("Go to sleep.") is None and navigation(f"{n}, go to sleep") is None  # mode command wins


if __name__ == "__main__":
    demo()
    print("commands: ok")
