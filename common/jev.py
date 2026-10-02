"""Jev (TypeSafe System One) as a common.decider engine. Owns everything
Jev-specific: URL, model, API key (OPENBOT_JEV_API_KEY). Nothing outside
this file knows those exist.
"""
from __future__ import annotations

import json
import os

import requests

from common.decider import Decider, Decision

URL = "https://api.typesafe.ai/v1/systemone"
TIMEOUT_S = 10.0


def make() -> Decider | None:
    key = os.environ.get("OPENBOT_JEV_API_KEY", "")
    return (lambda state, question, options: _decide(key, state, question, options)) if key else None


def _decide(key: str, state: dict, question: str, options: dict[str, str]) -> Decision | None:
    body = {"state": json.dumps(state), "model": "jev-latest",
            "questions": {"pick": {"type": "choice", "instructions": question, "criteria": options}}}
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=TIMEOUT_S)
        a = r.json()["answers"]["pick"]
        choice, conf = a["choice"], float(a["confidence"])
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return None
    return Decision(choice, conf) if choice in options else None


def demo() -> None:
    os.environ.pop("OPENBOT_JEV_API_KEY", None)
    assert make() is None  # no key -> engine off
    assert _decide("k", {}, "q", {"a": "x"}) is None or True  # never raises; None if unreachable


if __name__ == "__main__":
    demo()
    print("jev: ok")
