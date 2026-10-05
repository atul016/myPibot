"""Structured LLM response schema: {reply, tone_action}, enforced via
Ollama's grammar-constrained `format` field so the model is structurally
unable to emit a response that doesn't match -- a stronger guarantee than a
prompt instruction alone. tone_action is a Literal built from the hardware's
own stationary-action list (config.STATIONARY_ACTIONS) so the schema and
the real action vocabulary can't independently drift apart.

No movement field: what it does is its skills (common/skills.py),
decided by a separate call before the reply -- which is told what was done,
so the words can't claim a move that didn't happen.
"""
from __future__ import annotations

import re
from typing import Literal, Type

from pydantic import BaseModel, Field


def build_reply_model(stationary_actions: list[str]) -> Type[BaseModel]:
    ToneAction = Literal[tuple(stationary_actions) or ("none",)]  # no body -> nothing to gesture

    class Reply(BaseModel):
        # tone_action FIRST: the model generates fields in schema order, so a
        # streamed reply names its gesture before its words -- the gesture can
        # start while the first sentence is still being written.
        tone_action: ToneAction = Field(
            description="Which stationary gesture matches the EMOTIONAL TONE of "
                        "`reply` -- every reply has one, so this is never empty. "
                        "Doesn't move the wheels."
        )
        reply: str = Field(
            description="What the robot says out loud. Never mention, name, or "
                        "hint at tone_action in any form -- the action is "
                        "performed separately and is visible on its own."
        )
    return Reply


class ReplyStream:
    """Incremental reader for a streamed {"tone_action": ..., "reply": "..."}
    JSON object: feed() it text deltas as they arrive and get back each
    COMPLETE sentence of `reply` as soon as it's finished (plus tone_action
    once it's seen) -- so speech can start long before the JSON closes."""

    _ESC = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "b": "", "f": "", "r": ""}

    def __init__(self) -> None:
        self.buf = ""
        self.text = ""          # decoded reply so far
        self.tone_action: str | None = None
        self._pos: int | None = None  # scan position inside the reply string
        self._done = False
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        if self.tone_action is None:
            m = re.search(r'"tone_action"\s*:\s*"([^"]*)"', self.buf)
            if m:
                self.tone_action = m.group(1)
        if self._pos is None and not self._done:
            m = re.search(r'"reply"\s*:\s*"', self.buf)
            if m:
                self._pos = m.end()
        while self._pos is not None and self._pos < len(self.buf):
            c = self.buf[self._pos]
            if c == "\\":
                if self._pos + 1 >= len(self.buf):
                    break  # escape split across deltas -- wait for more
                e = self.buf[self._pos + 1]
                if e == "u":
                    if self._pos + 6 > len(self.buf):
                        break
                    self.text += chr(int(self.buf[self._pos + 2:self._pos + 6], 16))
                    self._pos += 6
                else:
                    self.text += self._ESC.get(e, e)
                    self._pos += 2
                continue
            if c == '"':
                self._pos, self._done = None, True
                break
            self.text += c
            self._pos += 1
        return self._sentences(final=self._done)

    def finish(self) -> list[str]:
        """Whatever's left (the stream ended, maybe mid-sentence)."""
        return self._sentences(final=True)

    def _sentences(self, final: bool) -> list[str]:
        pending = self.text[self._emitted:]
        # A sentence ends at . ! ? followed by whitespace -- not at end of
        # buffer mid-stream ("3." might be "3.5").
        ends = [m.end() for m in re.finditer(r"[.!?]+[\"')]*\s", pending)]
        cut = len(pending) if final else (ends[-1] if ends else 0)
        if not cut:
            return []
        out = [x.strip() for x in re.split(r"(?<=[.!?])\s+", pending[:cut]) if x.strip()]
        self._emitted += cut
        return out


def demo() -> None:
    Reply = build_reply_model(["nod", "think"])
    instance = Reply(reply="Got it!", tone_action="nod")
    assert instance.tone_action == "nod"
    assert list(Reply.model_json_schema()["properties"])[0] == "tone_action"

    full = '{"tone_action": "nod", "reply": "Hi there! It\'s 3.5 degrees.\\nBrr \\"cold\\" \\u00e9h? Ok"}'
    rs, got = ReplyStream(), []
    for ch in full:  # worst case: one character per delta
        got += rs.feed(ch)
    got += rs.finish()
    assert rs.tone_action == "nod", rs.tone_action
    assert got == ["Hi there!", "It's 3.5 degrees.", 'Brr "cold" \u00e9h?', "Ok"], got
    rs2 = ReplyStream()
    assert rs2.feed('{"tone_action": "think", "reply": "One. Two') == ["One."]
    assert rs2.finish() == ["Two"]
    try:
        Reply(reply="x", tone_action="not-a-real-action")
        raise AssertionError("expected validation error")
    except Exception:
        pass


if __name__ == "__main__":
    demo()
    print("reply_schema: ok")
