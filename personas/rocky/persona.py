"""Assembles Rocky as a Persona bundle -- the framework's default example
persona. Everything here is identity/flavor text; no framework logic.
"""
from __future__ import annotations

from common.persona import Persona

from . import prompt as _prompt
from . import transform as _transform
from . import voice as _voice

NAME = "Rocky"

WAKE_WORDS = ["rocky", "hey rocky", "hi rocky"]
SLEEP_WORDS = ["go to sleep", "rocky go to sleep"]
SLEEP_ACK = "Good. Me sleep. you observe"

PERSONA = Persona(
    name=NAME,
    wake_words=WAKE_WORDS,
    sleep_words=SLEEP_WORDS,
    sleep_ack=SLEEP_ACK,
    system_prompt_template=_prompt.build_system_prompt,
    piper_voice="en_US-lessac-low",
    transform=_transform.rocky_transform,
    speak_overlay=_voice.speak_rocky_overlay,
)
