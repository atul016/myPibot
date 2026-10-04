"""The one-call cognition client: a local OpenAI-compatible server (MLX
Serve on the Mac, `/v1/chat/completions`), no fallback ladder,
classified failures instead of a bare exception raised from deep inside a
lifecycle hook. Same "one call, no ladder" simplicity SPARK settled on
after real incidents from a cold-start fallback amplifying contention --
here that's moot (no cloud/resident-session alternative exists to fall
back to anyway), so the classification is purely for callers to react
sensibly (speak a fallback line vs. retry vs. defer), not a safety feature.
"""
from __future__ import annotations

import base64
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from .state import STATE_DIR, atomic_write

# Where the LLM server was last FOUND, if not where config says -- the Mac's
# DHCP address has moved before, and the Pi can't resolve its .local name
# (the Mac doesn't answer mDNS). Shared by every process via state/.
FOUND_PATH = STATE_DIR / "llm_url.json"
REDISCOVER_EVERY_S = 300.0
CONNECT_TIMEOUT_S = 3.0
_last_scan = 0.0

OFFLINE = "offline"
TIMEOUT = "timeout"
BAD_RESPONSE = "bad_response"
AVAILABLE = "available"


@dataclass
class CognitionResult:
    status: str  # AVAILABLE | OFFLINE | TIMEOUT | BAD_RESPONSE
    text: str = ""
    error: str = ""
    truncated: bool = False  # finish_reason == "length" -- text ends mid-thought, not by choice


# A small token budget is enough that a JSON-schema-
# constrained reply to an open-ended request ("tell me a story") can run
# out of room and still close out syntactically valid JSON with a
# mid-sentence `reply` value -- confirmed live: "there lived brilliant
# scientist named." with no error, because the JSON itself was well-formed.
# Generous enough for a few sentences without giving a small model room to
# ramble past what a spoken reply should be anyway.
DEFAULT_NUM_PREDICT = 400


def ask(base_url: str, model: str, prompt: str, *, system: str | None = None,
        json_schema: dict | None = None, timeout_s: float = 60.0,
        num_predict: int = DEFAULT_NUM_PREDICT, image_jpeg: bytes | None = None,
        history: list[dict] | None = None, temperature: float | None = None) -> CognitionResult:
    """history: prior {"role", "content"} chat turns, placed between the
    system prompt and this prompt -- system + history first keeps the
    server's prefix cache warm across a conversation. image_jpeg: one
    camera frame, sent as an OpenAI-style data URI. temperature: None = the
    server's own; low for a decision that should come out the same each time."""
    payload = _payload(model, prompt, system, json_schema, num_predict, image_jpeg, history)
    if temperature is not None:
        payload["temperature"] = temperature
    url = _known(base_url)
    try:
        resp = _post(url, payload, timeout_s)
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
        found = rediscover(base_url, model)  # the server moved? (rate-limited LAN scan)
        if not found or found == url:
            return CognitionResult(OFFLINE, error=str(e))
        try:
            resp = _post(found, payload, timeout_s)
        except requests.exceptions.RequestException as e2:
            return CognitionResult(OFFLINE, error=str(e2))
    except requests.exceptions.Timeout:
        return CognitionResult(TIMEOUT, error=f"timed out after {timeout_s}s")
    except requests.exceptions.RequestException as e:
        return CognitionResult(OFFLINE, error=str(e))

    if resp.status_code != 200:
        return CognitionResult(BAD_RESPONSE, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        choice = resp.json()["choices"][0]
        text = choice["message"]["content"] or ""
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return CognitionResult(BAD_RESPONSE, error=f"malformed response: {e}")
    return CognitionResult(AVAILABLE, text=text, truncated=choice.get("finish_reason") == "length")


def stream(base_url: str, model: str, prompt: str, *, system: str | None = None,
           json_schema: dict | None = None, timeout_s: float = 60.0,
           num_predict: int = DEFAULT_NUM_PREDICT, history: list[dict] | None = None,
           image_jpeg: bytes | None = None) -> Iterator[str]:
    """Same request as ask(), but yields the reply's text as it's generated
    -- so a caller can start speaking the first sentence while the rest is
    still being written. Yields nothing at all if the server can't be
    reached or errors (the caller checks for an empty reply)."""
    payload = {**_payload(model, prompt, system, json_schema, num_predict, image_jpeg, history), "stream": True}
    url = _known(base_url)
    try:
        try:
            resp = _post(url, payload, timeout_s, stream=True)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
            found = rediscover(base_url, model)
            if not found or found == url:
                return
            resp = _post(found, payload, timeout_s, stream=True)
        if resp.status_code != 200:
            return
        for raw in resp.iter_lines():  # bytes: requests decodes text/event-stream as Latin-1 ("It's" -> "Itâs")
            line = raw.decode("utf-8", "replace")
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta
    except requests.exceptions.RequestException:
        return  # mid-stream drop: the caller speaks whatever arrived


def _payload(model: str, prompt: str, system: str | None, json_schema: dict | None, num_predict: int,
             image_jpeg: bytes | None, history: list[dict] | None) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}] if system else []
    messages += history or []
    content: Any = prompt
    if image_jpeg:
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {
                       "url": "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()}}]
    messages.append({"role": "user", "content": content})
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": num_predict}
    if json_schema:
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "reply", "schema": json_schema}}
    return payload


def _post(url: str, payload: dict, timeout_s: float, stream: bool = False) -> requests.Response:
    # Short connect timeout: an absent host must fail in seconds, not after a
    # 45s read timeout, so rediscovery kicks in quickly.
    return requests.post(f"{url.rstrip('/')}/chat/completions", json=payload,
                         timeout=(CONNECT_TIMEOUT_S, timeout_s), stream=stream)


def _known(base_url: str) -> str:
    try:
        return json.loads(FOUND_PATH.read_text()).get(base_url) or base_url
    except (OSError, json.JSONDecodeError, AttributeError):
        return base_url


def rediscover(base_url: str, model: str) -> str | None:
    """Scans this machine's /24 for an OpenAI-compatible server on the same
    port that serves `model`; remembers and returns its URL. At most once
    per REDISCOVER_EVERY_S per process."""
    global _last_scan
    if time.time() - _last_scan < REDISCOVER_EVERY_S:
        return None
    _last_scan = time.time()
    u = urlparse(base_url)
    port = u.port or 80
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet sent -- just picks our LAN address
            prefix = s.getsockname()[0].rsplit(".", 1)[0]
    except OSError:
        return None

    def probe(i: int) -> str | None:
        ip = f"{prefix}.{i}"
        try:
            socket.create_connection((ip, port), timeout=0.4).close()
            url = f"{u.scheme}://{ip}:{port}{u.path}"
            return url if model in requests.get(f"{url.rstrip('/')}/models", timeout=2).text else None
        except (OSError, requests.RequestException):
            return None

    with ThreadPoolExecutor(64) as pool:
        found = next((url for url in pool.map(probe, range(1, 255)) if url), None)
    if found:
        atomic_write(FOUND_PATH, json.dumps({base_url: found}))
        print(f"cognition: LLM server found at {found} (configured: {base_url})")
    return found


def demo() -> None:
    # No live LLM server here -- confirms an unreachable host returns a
    # classified OFFLINE/TIMEOUT result rather than raising, which is the
    # contract every caller (wake-listen's reactive turn, mind's
    # reflection) depends on.
    global REDISCOVER_EVERY_S, _last_scan
    REDISCOVER_EVERY_S, _last_scan = 1e9, time.time()  # no LAN scan from a self-check
    result = ask("http://192.0.2.1:11234/v1", "does-not-matter", "hi", timeout_s=1.0)
    assert result.status in (OFFLINE, TIMEOUT)
    assert list(stream("http://127.0.0.1:1/v1", "does-not-matter", "hi", timeout_s=1.0)) == []


if __name__ == "__main__":
    demo()
    print("cognition: ok")
