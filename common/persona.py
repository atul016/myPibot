"""Loads the active persona bundle (name, wake words, prompt, voice, text
transform) by name -- OPENBOT_PERSONA env var, default "rocky". The
framework itself carries no bot identity; everything bot-flavored lives in
personas/<name>/.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Persona:
    name: str
    wake_words: list[str]
    sleep_words: list[str]
    sleep_ack: str
    # (allowed_actions, stationary_actions, gesture_guide) -> prompt. gesture_guide: "name (what it's for); ..."
    # for the stationary ones (config.describe_actions) -- the body's meanings, so the persona never names actions.
    system_prompt_template: Callable[[list[str], list[str], str], str]
    piper_voice: str
    transform: Callable[[str], str] = field(default=lambda text: text)
    speak_overlay: Callable[..., None] | None = None  # optional TTS overlay hook: (text, piper_voice, should_stop=callable)


# The active persona's bundle name = its name, lowercased (personas/<name>/).
CURRENT = os.environ.get("OPENBOT_PERSONA", "rocky")


def load(name: str | None = None) -> Persona:
    name = name or CURRENT
    module = importlib.import_module(f"personas.{name}.persona")
    return module.PERSONA


def demo() -> None:
    persona = load("rocky")
    assert persona.name
    assert persona.wake_words
    assert persona.transform("test") is not None
    assert persona.system_prompt_template(["nod"], ["nod"], "nod (agreement)")


if __name__ == "__main__":
    demo()
    print("persona: ok")
