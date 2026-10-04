"""Our own decision engine, distilled from Jev's (common/jev.py): trained on the
logged calls in state/jev/*.jsonl -- the state Jev was shown in, Jev's
probabilities out (soft labels) -- one small model per kind of question (its
set of options: text/stay_quiet, or the mind's actions). Plain multinomial
logistic regression over hashed words, numpy only, so it runs on the robot.

    python3 -m common.local_decider train   # (re)train from everything logged; prints how often it agrees with Jev
    OPENBOT_TEXT_DECIDER=local / OPENBOT_DECIDER=local   # then use it instead of Jev

ponytail: bag of hashed words + a linear model; if it agrees too little with
Jev, the upgrade is a small text encoder's embeddings -- same data, same plug.
"""
from __future__ import annotations

import json
import re
import sys
import zlib

import numpy as np

from common.decider import Decider, Decision
from common.events import read_jsonl
from common.state import STATE_DIR

MODEL_PATH = STATE_DIR / "local_decider.json"
DIMS = 4096  # hashed feature space


def _words(state: dict, prefix: str = "") -> list[str]:
    """"his_mood_now=bored", "what_just_happened:moving" ... -- a key's own value
    as one feature, plus every word under it, so both "mood is bored" and
    "something is moving" count."""
    out = []
    for key, value in state.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out += _words(value, f"{name}.")
            continue
        text = json.dumps(value) if not isinstance(value, str) else value
        if len(text) < 30:
            out.append(f"{name}={text.lower()}")
        out += [f"{name}:{w}" for w in re.findall(r"[a-z]+", text.lower())]
    return out


def features(state: dict) -> np.ndarray:
    x = np.zeros(DIMS)
    for w in _words(state):
        x[zlib.crc32(w.encode()) % DIMS] += 1.0
    return np.log1p(x)


def _kind(options) -> str:
    return "|".join(sorted(options))


def dataset() -> dict[str, tuple[list[str], np.ndarray, np.ndarray, list[str]]]:
    """Per kind of question: (option names, X, Y soft labels, Jev's choices) from every answered call."""
    rows: dict[str, list] = {}
    for path in sorted((STATE_DIR / "jev").glob("*.jsonl")):
        for c in read_jsonl(path, 10**9):
            q = ((c.get("request") or {}).get("questions") or {}).get("pick") or {}
            a = ((c.get("response") or {}).get("answers") or {}).get("pick") or {}
            if c.get("status") != 200 or not a.get("probabilities") or not q.get("criteria"):
                continue
            rows.setdefault(_kind(q["criteria"]), []).append((c["request"]["state"], a["probabilities"], a["choice"]))
    out = {}
    for kind, items in rows.items():
        names = kind.split("|")
        X = np.array([features(s) for s, _, _ in items])
        Y = np.array([[p.get(n, 0.0) for n in names] for _, p, _ in items])
        out[kind] = (names, X, Y / Y.sum(axis=1, keepdims=True), [ch for _, _, ch in items])
    return out


def _fit(X: np.ndarray, Y: np.ndarray, epochs: int = 300, lr: float = 0.5, l2: float = 1e-3) -> np.ndarray:
    W = np.zeros((X.shape[1] + 1, Y.shape[1]))
    Xb = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(epochs):
        P = _softmax(Xb @ W)
        W -= lr * (Xb.T @ (P - Y) / len(X) + l2 * W)
    return W


def _softmax(Z: np.ndarray) -> np.ndarray:
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def _predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _softmax(np.hstack([X, np.ones((len(X), 1))]) @ W)


def train() -> dict:
    """Fits every kind on all its data, after measuring agreement with Jev on the newest 20%."""
    models, report = {}, {}
    for kind, (names, X, Y, choices) in dataset().items():
        split = int(len(X) * 0.8)
        held = None
        if len(X) >= 20 and split < len(X):
            guess = _predict(_fit(X[:split], Y[:split]), X[split:]).argmax(axis=1)
            held = float(np.mean([names[g] == c for g, c in zip(guess, choices[split:])]))
        models[kind] = {"options": names, "W": _fit(X, Y).tolist()}
        report[kind] = {"examples": len(X), "agrees_with_jev_on_newest_20pct": held,
                        "jev_choices": {n: choices.count(n) for n in names}}
    MODEL_PATH.write_text(json.dumps({"dims": DIMS, "models": models}))
    return report


def make() -> Decider | None:
    try:
        models = json.loads(MODEL_PATH.read_text())["models"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    weights = {kind: (m["options"], np.array(m["W"])) for kind, m in models.items()}

    def decide(state: dict, question: str, options: dict[str, str]) -> Decision | None:
        model = weights.get(_kind(options))
        if model is None:  # a kind of question it was never trained on
            return None
        names, W = model
        p = _predict(W, features(state)[None, :])[0]
        return Decision(names[int(p.argmax())], float(p.max()))

    return decide


def demo() -> None:
    import shutil, tempfile
    from pathlib import Path
    global MODEL_PATH, STATE_DIR
    orig, test_dir = (MODEL_PATH, STATE_DIR), Path(tempfile.mkdtemp())
    STATE_DIR, MODEL_PATH = test_dir, test_dir / "local_decider.json"
    try:
        (test_dir / "jev").mkdir()
        options = {"text": "...", "stay_quiet": "..."}
        with open(test_dir / "jev" / "2026-10-03.jsonl", "w") as f:  # Jev texts on a mood change, else stays quiet
            for i in range(60):
                change = i % 3 == 0
                state = {"his_mood_before": "bored", "his_mood_now": "curious" if change else "bored",
                         "what_just_happened": ["something is moving"]}
                probs = {"text": 0.9, "stay_quiet": 0.1} if change else {"text": 0.1, "stay_quiet": 0.9}
                f.write(json.dumps({"status": 200, "request": {"state": state, "questions": {"pick": {"criteria": options}}},
                                    "response": {"answers": {"pick": {"choice": max(probs, key=probs.get),
                                                                      "probabilities": probs}}}}) + "\n")
        report = train()["stay_quiet|text"]
        assert report["examples"] == 60 and report["agrees_with_jev_on_newest_20pct"] == 1.0, report
        decide = make()
        assert decide({"his_mood_before": "bored", "his_mood_now": "curious"}, "q", options).choice == "text"
        assert decide({"his_mood_before": "bored", "his_mood_now": "bored"}, "q", options).choice == "stay_quiet"
        assert decide({}, "q", {"speak": "", "wait": ""}) is None  # never trained on this kind
    finally:
        MODEL_PATH, STATE_DIR = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        print(json.dumps(train(), indent=2))
    else:
        demo()
        print("local_decider: ok")
