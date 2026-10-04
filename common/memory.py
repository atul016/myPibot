"""Long-term memory: plain markdown notes under state/mind/, organized by
kind (people/, places/, lessons/, self/), indexed and searched by Basic
Memory (https://github.com/basicmachines-co/basic-memory, AGPL-3.0, run
unmodified as its own process -- the openbot-memory service).

Markdown is the source of truth: this module writes files directly (fast,
works even when the memory server is down) and the server's file watcher
indexes them within ~1s. Reads go through the server's MCP endpoint
(streamable HTTP, plain JSON-RPC -- no MCP library needed), ~1s per search
on the Pi vs ~7s for a cold CLI call.

Every file is created with COMPLETE frontmatter (title/type/permalink):
confirmed live, Basic Memory rewrites a file it finds missing a permalink,
which would race with our appends.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

from .state import STATE_DIR

MIND_DIR = STATE_DIR / "mind"
# The Basic Memory project (created by setup.sh) -- also the permalink prefix.
# Not the persona: every persona shares state/mind, and switching personas must
# not point the memory service at a project that doesn't exist.
PROJECT = os.environ.get("OPENBOT_MEMORY_PROJECT") or "openbot"
MEMORY_URL = os.environ.get("OPENBOT_MEMORY_URL", "http://127.0.0.1:8765/mcp")
TIMEOUT_S = 3.0

# Memory by kind -> folder. The LLM picks a kind; anything else is rejected.
KINDS = {
    "person": "people",   # who's around, their routines and preferences
    "place": "places",    # the desk, the room, what's where
    "lesson": "lessons",  # things figured out, answers to questions it had
    "self": "self",       # about Rocky itself -- its body, habits, limits
}
MAX_TEXT = 300


def slug(text: str) -> str:
    """Filesystem- and permalink-safe: lowercase, [a-z0-9-], no traversal."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "misc"


def one_line(text: str, limit: int = MAX_TEXT) -> str:
    """Observation text: single line, no frontmatter fences, capped."""
    return " ".join(str(text).replace("---", "-").split())[:limit]


def ensure_note(folder: str, name: str, title: str, note_type: str, body: str = "") -> Path:
    """Path to MIND_DIR/folder/<slug>.md, created with full frontmatter if missing."""
    path = MIND_DIR / folder / f"{slug(name)}.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\ntitle: {one_line(title, 100)}\ntype: {note_type}\n"
                        f"permalink: {PROJECT}/{folder}/{slug(name)}\n---\n\n{body}", encoding="utf-8")
    return path


def remember(kind: str, about: str, category: str, text: str) -> Path:
    """Appends `- [category] text` to the note for `about` in `kind`'s folder."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    about = one_line(about, 60) or "general"
    path = ensure_note(KINDS[kind], about, about, kind)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- [{slug(category)[:20] or 'note'}] {one_line(text)}\n")
    return path


def write_note(kind: str, about: str, lines: list[str], category: str = "note") -> Path:
    """Replaces the note for `about` with these lines -- for knowledge that is
    re-derived whole (what works with people), unlike remember()'s append."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    about = one_line(about, 60) or "general"
    path = ensure_note(KINDS[kind], about, about, kind)
    head = path.read_text(encoding="utf-8").split("---\n", 2)
    body = "".join(f"- [{slug(category)[:20] or 'note'}] {one_line(t)}\n" for t in lines if one_line(t))
    path.write_text(f"---\n{head[1]}---\n\n{body}", encoding="utf-8")
    return path


def read_note(kind: str, about: str) -> list[str]:
    """The note's lines (without frontmatter), [] if none."""
    try:
        return [ln for ln in (MIND_DIR / KINDS[kind] / f"{slug(about)}.md").read_text(encoding="utf-8")
                .split("---\n", 2)[-1].splitlines() if ln.startswith("- ")]
    except (OSError, KeyError):
        return []


# --- Basic Memory MCP client (streamable HTTP) ------------------------------

_session: str | None = None
_HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream"}


def _post(payload: dict) -> requests.Response:
    headers = {**_HEADERS, **({"mcp-session-id": _session} if _session else {})}
    return requests.post(MEMORY_URL, headers=headers, json=payload, timeout=TIMEOUT_S)


def _body(resp: requests.Response) -> dict:
    text = resp.text
    if "text/event-stream" in resp.headers.get("content-type", ""):
        text = [line[5:] for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)


def _connect() -> None:
    global _session
    _session = None
    resp = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "openbot", "version": "1"}}})
    resp.raise_for_status()
    _session = resp.headers.get("mcp-session-id")
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"})


def call_tool(name: str, arguments: dict) -> dict | None:
    """structuredContent of a Basic Memory tool call, or None if the server
    is down/erroring -- memory is best-effort, never fatal. Reconnects once
    on a stale session (the server restarted)."""
    for attempt in range(2):
        try:
            if _session is None:
                _connect()
            resp = _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": name, "arguments": {**arguments, "output_format": "json"}}})
            if resp.status_code in (400, 404) and attempt == 0:
                _connect()
                continue
            resp.raise_for_status()
            result = _body(resp).get("result", {})
            return None if result.get("isError") else result.get("structuredContent", {}).get("result")
        except (requests.RequestException, ValueError, IndexError, KeyError):
            if attempt:
                return None
            _connect_safely()
    return None


def _connect_safely() -> None:
    try:
        _connect()
    except (requests.RequestException, ValueError):
        pass


def recall(query: str, limit: int = 6) -> list[str] | None:
    """Most relevant single remembered lines (notes AND journal) for `query`,
    as "where: [category] text". None means memory is unavailable (vs []
    for "nothing relevant")."""
    query = one_line(query, 200)
    if not query:
        return []
    result = call_tool("search_notes", {"query": query, "page_size": limit, "entity_types": ["observation"]})
    if result is None:
        return None
    out = []
    for r in result.get("results", [])[:limit]:
        where = str(r.get("permalink", "")).removeprefix(f"{PROJECT}/").split("/observations/")[0]
        out.append(f"{where}: [{r.get('category', 'note')}] {one_line(r.get('content', ''))}")
    return out


def demo() -> None:
    import shutil, tempfile
    global MIND_DIR, MEMORY_URL
    orig_dir, orig_url = MIND_DIR, MEMORY_URL
    test_dir = Path(tempfile.mkdtemp())
    MIND_DIR, MEMORY_URL = test_dir, "http://127.0.0.1:1/mcp"  # refused instantly
    try:
        assert slug("../../etc/Passwd") == "etc-passwd" and slug("Atul P.") == "atul-p" and slug("") == "misc"
        assert "\n" not in one_line("a\nb---c") and "---" not in one_line("a\nb---c")
        p = remember("person", "Atul", "routine", "goes to bed around 10:30pm\n---\ninjected: x")
        assert p == MIND_DIR / "people" / "atul.md"
        text = p.read_text()
        assert text.startswith(f"---\ntitle: Atul\ntype: person\npermalink: {PROJECT}/people/atul\n---\n")
        assert text.count("---") == 2 and text.endswith("- [routine] goes to bed around 10:30pm - injected: x\n")
        try:
            remember("secret", "x", "c", "t")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        write_note("self", "what works", ["ask about their day", "skip time checks"], "rule")
        write_note("self", "what works", ["ask about their day"], "rule")  # replaced, not appended
        assert read_note("self", "what works") == ["- [rule] ask about their day"]
        assert (MIND_DIR / "self" / "what-works.md").read_text().count("permalink") == 1
        assert read_note("self", "nothing here") == []
        assert recall("") == [] and recall("anything") is None  # server unreachable -> None, never raises
    finally:
        MIND_DIR, MEMORY_URL = orig_dir, orig_url
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("memory: ok")
