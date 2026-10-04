"""Jev (TypeSafe System One) as a common.decider engine. Owns everything
Jev-specific: URL, model, API key (OPENBOT_JEV_API_KEY). Nothing outside
this file knows those exist. Every call -- the full request (minus the key),
Jev's whole answer, or the error -- goes to state/jev/<day>.jsonl, never
trimmed: it's the training data for a classifier of our own that does Jev's
job (state in, choice out), and the dashboard's Jev tab shows it.
"""
from __future__ import annotations

import datetime
import json
import os
import time

import requests

from common.decider import Decider, Decision
from common.events import append_jsonl, read_jsonl
from common.state import STATE_DIR

URL = "https://api.typesafe.ai/v1/systemone"
TIMEOUT_S = 10.0
LOG_DIR = STATE_DIR / "jev"  # one file a day, kept whole


def make() -> Decider | None:
    key = os.environ.get("OPENBOT_JEV_API_KEY", "")
    return (lambda state, question, options: _decide(key, state, question, options)) if key else None


def _decide(key: str, state: dict, question: str, options: dict[str, str]) -> Decision | None:
    body = {"state": json.dumps(state), "model": "jev-latest",
            "questions": {"pick": {"type": "choice", "instructions": question, "criteria": options}}}
    entry: dict = {"ts": time.time(), "request": {**body, "state": state}}
    t0 = time.monotonic()
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=TIMEOUT_S)
        entry["status"] = r.status_code
        entry["response"] = r.json()
        a = entry["response"]["answers"]["pick"]
        choice, conf = a["choice"], float(a["confidence"])
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as e:
        entry["error"] = repr(e)
        return None
    finally:
        entry["ms"] = round((time.monotonic() - t0) * 1000)
        try:
            append_jsonl(LOG_DIR / f"{datetime.date.fromtimestamp(entry['ts']).isoformat()}.jsonl", entry)
        except OSError:
            pass  # the log must never cost a decision
    return Decision(choice, conf) if choice in options else None


def recent_calls(limit: int = 50) -> list[dict]:
    """Newest first, across the daily files."""
    out: list[dict] = []
    for path in sorted(LOG_DIR.glob("*.jsonl"), reverse=True):
        out += read_jsonl(path, limit - len(out))[::-1]
        if len(out) >= limit:
            break
    return out


def demo() -> None:
    import shutil, tempfile
    from pathlib import Path
    global LOG_DIR
    os.environ.pop("OPENBOT_JEV_API_KEY", None)
    assert make() is None  # no key -> engine off
    orig, LOG_DIR = LOG_DIR, Path(tempfile.mkdtemp())  # never into the real log
    try:
        assert _decide("fake-key-123", {"x": 1}, "q", {"a": "x"}) is None  # a bad key: never raises, just None
        call = recent_calls()[0]
        assert call["request"]["state"] == {"x": 1} and "error" in call and "fake-key-123" not in json.dumps(call)
    finally:
        shutil.rmtree(LOG_DIR, ignore_errors=True)
        LOG_DIR = orig


if __name__ == "__main__":
    demo()
    print("jev: ok")
