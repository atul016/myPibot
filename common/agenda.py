"""What Rocky is working on and waiting for.

Goals -- a short to-do list of open questions ("what was that noise?"),
kept in state/mind/agenda.md so it's readable/searchable like any note.
At most MAX_GOALS open; resolving one saves "question -> conclusion" as a
lesson (memory.remember), which is where new ideas come from.

Reminders / watches / rest -- self-scheduled wake-ups, kept in the
session (operational, not knowledge):
  remind_me(in_minutes, about)   wake me later with this on my mind
  watch(for_kind, about)         when <surprise kind> next happens, remember why I cared
  rest(minutes)                  no idle reflections until then (surprises still wake me)
A reminder also rests until it's due -- that's what stops a dark, empty
room from producing an idle "staring into the void" thought every 5 min.

The list logic is pure functions over plain lists; the thin load/save
wrappers at the bottom touch files.
"""
from __future__ import annotations

import datetime as dt
import re
import time

from . import memory, state
from .state import atomic_write

MAX_GOALS = 3
GOAL_TTL_S = 48 * 3600
MAX_REMINDERS = 5
MAX_WATCHES = 5
WATCH_TTL_S = 12 * 3600
REMIND_MINUTES = (1, 720)
WATCHABLE = {"approach", "leave", "sound", "picked_up", "put_down", "battery_low", "scene", "lights_on", "lights_off",
             "person_arrived", "person_left", "stranger"}

_GOAL = re.compile(r"^- \[(open|done|dropped)\] (\d{4}-\d\d-\d\d \d\d:\d\d) \| (.*)$")


# --- goals: pure ---------------------------------------------------------------

def parse_goals(text: str) -> list[dict]:
    goals = []
    for m in map(_GOAL.match, text.splitlines()):
        if m:
            goals.append({"status": m[1], "since": m[2], "text": m[3]})
    return goals


def render_goals(goals: list[dict]) -> str:
    return "".join(f"- [{g['status']}] {g['since']} | {g['text']}\n" for g in goals)


def open_goals(goals: list[dict]) -> list[dict]:
    return [g for g in goals if g["status"] == "open"]


def add_goal(goals: list[dict], text: str, now: dt.datetime) -> list[dict]:
    """Adds an open goal; drops the oldest open one if already at MAX_GOALS.
    A near-duplicate of an open goal is ignored."""
    text = memory.one_line(text, 200)
    if not text or any(g["text"].lower() == text.lower() for g in open_goals(goals)):
        return goals
    goals = [dict(g) for g in goals]
    opened = open_goals(goals)
    if len(opened) >= MAX_GOALS:
        opened[0]["status"] = "dropped"
    return goals + [{"status": "open", "since": f"{now:%Y-%m-%d %H:%M}", "text": text}]


def resolve_goal(goals: list[dict], number: int, conclusion: str) -> tuple[list[dict], dict | None]:
    """number is 1-based into open_goals(). Returns (goals, resolved goal or None)."""
    opened = open_goals(goals)
    if not 1 <= number <= len(opened):
        return goals, None
    target = opened[number - 1]
    goals = [dict(g) for g in goals]
    for g in goals:
        if g == target:
            g["status"] = "done"
            g["text"] = f"{target['text']} -> {memory.one_line(conclusion, 200)}"
            return goals, target
    return goals, None


def expire_goals(goals: list[dict], now: dt.datetime) -> list[dict]:
    out = [dict(g) for g in goals]
    for g in out:
        since = dt.datetime.strptime(g["since"], "%Y-%m-%d %H:%M")
        if g["status"] == "open" and (now - since).total_seconds() > GOAL_TTL_S:
            g["status"] = "dropped"
    return out[-30:]  # done/dropped history kept short


# --- reminders / watches: pure ----------------------------------------------------

def add_reminder(reminders: list[dict], minutes: int, about: str, now: float) -> list[dict]:
    minutes = max(REMIND_MINUTES[0], min(REMIND_MINUTES[1], int(minutes)))
    about = memory.one_line(about, 200)
    kept = [r for r in reminders if r["about"].lower() != about.lower()]
    return sorted(kept + [{"at": now + minutes * 60, "about": about}], key=lambda r: r["at"])[:MAX_REMINDERS]


def due(reminders: list[dict], now: float) -> tuple[list[dict], list[dict]]:
    """(due, still pending)."""
    return [r for r in reminders if r["at"] <= now], [r for r in reminders if r["at"] > now]


def add_watch(watches: list[dict], for_kind: str, about: str, now: float) -> list[dict]:
    if for_kind not in WATCHABLE:
        raise ValueError(f"can only watch for {sorted(WATCHABLE)}")
    kept = [w for w in watches if w["until"] > now and w["for"] != for_kind]
    return (kept + [{"for": for_kind, "about": memory.one_line(about, 200), "until": now + WATCH_TTL_S}])[-MAX_WATCHES:]


def match_watches(watches: list[dict], kinds: set[str], now: float) -> tuple[list[dict], list[dict]]:
    """(triggered by these surprise kinds, still waiting). Expired ones drop."""
    live = [w for w in watches if w["until"] > now]
    return [w for w in live if w["for"] in kinds], [w for w in live if w["for"] not in kinds]


# --- file/session wrappers ----------------------------------------------------------

def _goals_path():
    return memory.MIND_DIR / "agenda.md"


def load_goals() -> list[dict]:
    try:
        return parse_goals(_goals_path().read_text(encoding="utf-8"))
    except OSError:
        return []


def save_goals(goals: list[dict]) -> None:
    path = _goals_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, f"---\ntitle: Agenda\ntype: agenda\npermalink: {memory.PROJECT}/agenda\n---\n\n"
                       + render_goals(goals))


def session_lists() -> tuple[list[dict], list[dict], float]:
    s = state.load_session()
    return s.get("reminders", []), s.get("watches", []), float(s.get("rest_until", 0))


def demo() -> None:
    now = dt.datetime(2026, 10, 1, 22, 0)
    g = add_goal([], "What was that noise?", now)
    g = add_goal(g, "what was that noise?", now)  # duplicate ignored
    g = add_goal(add_goal(g, "Is Atul home?", now), "Why is it dark?", now)
    assert [x["text"] for x in open_goals(g)] == ["What was that noise?", "Is Atul home?", "Why is it dark?"]
    g = add_goal(g, "Where is the mug?", now)  # 4th -> oldest open dropped
    assert g[0]["status"] == "dropped" and len(open_goals(g)) == 3
    assert parse_goals(render_goals(g)) == g
    g2, done = resolve_goal(g, 2, "the lights are off for the night")
    assert done["text"] == "Why is it dark?" and len(open_goals(g2)) == 2
    assert any(x["text"] == "Why is it dark? -> the lights are off for the night" for x in g2)
    assert resolve_goal(g, 9, "x")[1] is None
    assert all(x["status"] != "open" for x in expire_goals(g, now + dt.timedelta(days=3)))

    t = 1000.0
    r = add_reminder([], 600, "check if the lights are on", t)
    r = add_reminder(r, 99999, "way too far", t)          # clamped to 720 min
    r = add_reminder(r, 5, "Check if the lights are on", t)  # same reminder -> rescheduled, not duplicated
    assert [x["about"] for x in r] == ["Check if the lights are on", "way too far"]
    assert r[1]["at"] == t + 720 * 60
    fired, pending = due(r, t + 5 * 60)
    assert len(fired) == 1 and len(pending) == 1

    w = add_watch([], "lights_on", "want to see the room", t)
    trig, rest = match_watches(w, {"sound"}, t + 1)
    assert trig == [] and len(rest) == 1
    trig, rest = match_watches(w, {"lights_on"}, t + 1)
    assert trig[0]["about"] == "want to see the room" and rest == []
    assert match_watches(w, {"lights_on"}, t + WATCH_TTL_S + 1) == ([], [])  # expired
    try:
        add_watch([], "the moon", "x", t)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    demo()
    print("agenda: ok")
