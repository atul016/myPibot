"""One in-character line for a situation -- and the gesture that goes with
it -- from the LLM, in the robot's current mood. The one path for every
"something happened, say something" moment (wake greeting, command
acknowledgments, how a drive went), so nothing it says is a fixed string
and nothing it does with its body is a canned sequence: the model sees the
situation, its mood, its day and the people around it (journal.inject) and
decides. Falls back to `fallback` only when the LLM can't be reached.
"""
from __future__ import annotations

import json

import config as cfg

from . import cognition, journal, reply_schema

_Reply = reply_schema.build_reply_model(cfg.TONE_ACTIONS)


def line(persona, situation: str, fallback: str, timeout_s: float = 12.0) -> tuple[str, str | None]:
    """(what to say, persona-transformed; tone gesture or None). `situation`
    is one or two plain sentences: what just happened and what's wanted of
    the line ("Greet them with one short line")."""
    prompt = journal.inject(situation + " One short line, in character -- don't describe what you see. Call "
                            "people by name only if your camera lists them in front of you right now -- not from "
                            "your recent life (that may be someone who's gone).")
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt,
                           system=persona.system_prompt_template(cfg.ALLOWED_ACTIONS, cfg.STATIONARY_ACTIONS,
                                                                 cfg.describe_actions(cfg.STATIONARY_ACTIONS)),
                           json_schema=_Reply.model_json_schema(), timeout_s=timeout_s, num_predict=120)
    if result.status != cognition.AVAILABLE:
        print(f"react: cognition unavailable ({result.status}): {result.error}")
        return persona.transform(fallback), None
    try:
        data = json.loads(result.text)
        text = str(data.get("reply", "")).strip()
        tone = data.get("tone_action")
    except (json.JSONDecodeError, AttributeError):
        text, tone = result.text.strip(), None
    return persona.transform(text or fallback), tone if tone in cfg.TONE_ACTIONS else None


def demo() -> None:
    class P:
        transform = staticmethod(lambda t: t.upper())
        system_prompt_template = staticmethod(lambda a, s, g="": "")
    # offline LLM -> the fallback, transformed, no gesture
    global cognition
    real, Result = cognition, cognition.CognitionResult
    try:
        class _Off:
            AVAILABLE = cognition.AVAILABLE
            @staticmethod
            def ask(*a, **k):
                return Result(status=real.OFFLINE, error="x")
        cognition = _Off
        assert line(P(), "hi", "okay") == ("OKAY", None)
    finally:
        cognition = real


if __name__ == "__main__":
    demo()
    print("react: ok")
