"""Pluggable "pick one option" engine. Callers (mind.py) depend only on this
interface: a Decider takes state + a question + {option: description} and
returns a Decision, or None if it can't/won't decide (off, unreachable,
bad reply). None always means "caller falls back to its old behavior", so
no engine is ever a hard dependency.

Add an engine (e.g. Laya): write a module exposing `make() -> Decider | None`
and add its name to _PROVIDERS. Nothing else changes.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

_PROVIDERS = {"jev": "common.jev", "local": "common.local_decider"}  # name -> module exposing make()


@dataclass(frozen=True)
class Decision:
    choice: str
    confidence: float  # 0..1, engine-calibrated


Decider = Callable[[dict, str, dict[str, str]], "Decision | None"]


def load(name: str) -> Decider | None:
    """The named engine, or None if name is empty/unknown/unconfigured."""
    if name not in _PROVIDERS:
        return None
    return importlib.import_module(_PROVIDERS[name]).make()


def demo() -> None:
    assert load("") is None and load("nope") is None


if __name__ == "__main__":
    demo()
    print("decider: ok")
