"""openbot-mind: the autonomous "inner life" loop.

  awareness  -- every AWARENESS_INTERVAL_S: sensors + mic loudness ->
                common/surprise.py; every VISION_INTERVAL_S: a camera look
                (common/vision.py); due reminders (common/agenda.py).
                Everything noticed goes into the journal.
  reflection -- RIGHT AWAY on a surprise or due reminder (rate-limited),
                else on a slow idle timer unless resting. The LLM sees what
                changed, its day so far, its recent life (journal tail), its
                goals, reminders and watches, and memories recalled for the
                moment -- then picks one action. TOOLS (look, listen, recall)
                feed their result back and it decides again, up to
                MIND_MAX_STEPS; a surprise arriving mid-chain is handed to
                the next step instead of waiting for the chain to end.
  expression -- speak/gesture/sound/remember/remind_me/watch/rest through a
                whitelist + param validation (SPARK's validate_action shape),
                gated by policy (quiet hours) and an anti-flap cooldown.
  upkeep     -- a rolling "today so far" summary every SUMMARY_INTERVAL_S,
                and once a night (after midnight, in quiet hours) a "dream"
                pass that distills yesterday's journal into notes by kind.

Never drives the wheels: `look` turns only the camera gimbal, and no
expression is a movement action.
"""
from __future__ import annotations

import datetime
import json
import random
import time
from collections import deque
from typing import Any, Callable


import config as cfg  # noqa: E402
from common import agenda, cognition, decider, events as dash_events, faces, health, journal, memory  # noqa: E402
from common import persona as persona_mod, policy, sensors, state, surprise, vision  # noqa: E402
from common.motor_client import dispatch as motor_dispatch  # noqa: E402
from common.speak_client import speak as speak_client  # noqa: E402

COMPONENT = "openbot-mind"
# openbot-memory isn't listed: it's Basic Memory itself, which doesn't write our health records.
HEALTH_COMPONENTS = ["openbot-alive"] * cfg.HAS_BODY + ["openbot-wake-listen", "openbot-mind", "openbot-speak"]

ALLOWED_EXPRESSIONS = {"speak", "gesture", "wait", "remember", "play_sound", "time_check", "introspect",
                       "remind_me", "watch", "rest"}
# Observe-only actions: their result goes back to the LLM for another step.
TOOLS = {"look", "listen", "recall"}
if not cfg.STATIONARY_ACTIONS:  # no body: nothing to gesture with
    ALLOWED_EXPRESSIONS.discard("gesture")
if not cfg.SOUNDS:
    ALLOWED_EXPRESSIONS.discard("play_sound")
# Without a body nothing reports distance, being picked up, or the battery.
WATCHABLE = agenda.WATCHABLE if cfg.HAS_BODY else \
    agenda.WATCHABLE - {"approach", "leave", "picked_up", "put_down", "battery_low"}
LISTEN_S = 5.0
EXPRESSION_EFFECT = {
    "speak": "audio", "gesture": "presence", "play_sound": "audio", "time_check": "audio", "introspect": "audio",
}  # everything else is silent bookkeeping ("other")
# Silent bookkeeping isn't an expressive act -- exempt from the cooldown.
COOLDOWN_EXEMPT = {"wait", "remember", "remind_me", "watch", "rest"}
ALLOWED_SOUNDS = set(cfg.SOUNDS)

Event = surprise.Event


class MindError(Exception):
    """A rejected/unparseable expression -- the caller treats this as "do
    nothing this cycle," never as a reason to crash the loop."""


def _text(params: dict, key: str, limit: int) -> str:
    text = memory.one_line(params.get(key, ""), limit + 1)
    if not text or len(text) > limit:
        raise MindError(f"{key} requires 1-{limit} chars")
    return text


def _minutes(params: dict, key: str) -> int:
    try:
        m = int(params.get(key))
    except (TypeError, ValueError):
        raise MindError(f"{key} must be a whole number of minutes")
    if not 1 <= m <= agenda.REMIND_MINUTES[1]:
        raise MindError(f"{key} must be 1-{agenda.REMIND_MINUTES[1]}")
    return m


def validate_expression(action: dict, stationary_actions: list[str]) -> tuple[str, dict]:
    name = action.get("action")
    if name not in ALLOWED_EXPRESSIONS | TOOLS:
        raise MindError(f"unsupported mind expression: {name!r}")
    params = action.get("params") or {}
    if name == "look":
        direction = str(params.get("direction", "ahead"))
        if direction not in cfg.LOOK_DIRECTIONS:
            raise MindError(f"look direction must be one of {cfg.LOOK_DIRECTIONS}")
        return name, {"direction": direction}
    if name == "recall":
        return name, {"query": _text(params, "query", 200)}
    if name == "speak":
        return name, {"text": _text(params, "text", 300)}
    if name == "gesture":
        gesture = str(params.get("name", ""))
        if gesture not in stationary_actions:
            raise MindError(f"unknown gesture: {gesture!r}")
        return name, {"name": gesture}
    if name == "remember":
        kind = str(params.get("kind", ""))
        if kind not in memory.KINDS:
            raise MindError(f"remember kind must be one of {sorted(memory.KINDS)}")
        return name, {"kind": kind, "about": _text(params, "about", 60),
                      "category": memory.slug(str(params.get("category", "note")))[:20] or "note",
                      "text": _text(params, "text", 300)}
    if name == "play_sound":
        sound = str(params.get("name", ""))
        if sound not in ALLOWED_SOUNDS:
            raise MindError(f"unknown sound: {sound!r}; allowed: {sorted(ALLOWED_SOUNDS)}")
        return name, {"name": sound}
    if name == "remind_me":
        return name, {"in_minutes": _minutes(params, "in_minutes"), "about": _text(params, "about", 200)}
    if name == "watch":
        kind = str(params.get("for", ""))
        if kind not in WATCHABLE:
            raise MindError(f"watch 'for' must be one of {sorted(WATCHABLE)}")
        return name, {"for": kind, "about": _text(params, "about", 200)}
    if name == "rest":
        return name, {"minutes": _minutes(params, "minutes")}
    # time_check, introspect, wait, listen -- no params
    return name, {}


def _ago(ts: float | None) -> str | None:
    if not ts:
        return None
    mins = (time.time() - ts) / 60
    return "just now" if mins < 1 else f"{mins:.0f} min ago"


def _in(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


def _build_awareness(events: list[Event], watch_hits: list[dict]) -> dict[str, Any]:
    s = sensors.read()
    look = vision.last_look()
    goals = agenda.open_goals(agenda.load_goals())
    reminders, watches, rest_until = agenda.session_lists()
    unhealthy = {k: v for k, v in health.all_status(HEALTH_COMPONENTS).items() if v not in ("ok", "degraded")}
    query = " ".join([text for _, text in events] + [g["text"] for g in goals])
    return {
        "local_time": datetime.datetime.now().strftime("%A %I:%M %p"),
        "what_just_changed": [text for _, text in events] or "nothing -- it's been quiet",
        "you_were_watching_for_this": [f"{w['for']} -- because: {w['about']}" for w in watch_hits] or None,
        "what_you_see": {"scene": look.get("scene"), "looked": _ago(look.get("ts"))} if look else None,
        "who_is_in_front_of_you": faces.describe(faces.read()) or "nobody",
        "distance_ahead_cm": s.get("distance"),
        "battery_pct": s.get("battery_pct"),
        "unhealthy_services": unhealthy or None,
        "your_day_so_far": journal.read_summary(datetime.date.today()) or None,
        "your_recent_life": journal.tail(15),
        "your_goals": [f"{i}. {g['text']} (since {g['since']})" for i, g in enumerate(goals, 1)] or None,
        "reminders_you_set": [f"at {_in(r['at'])}: {r['about']}" for r in reminders] or None,
        "watching_for": [f"{w['for']}: {w['about']}" for w in watches if w["until"] > time.time()] or None,
        "resting_until": _in(rest_until) if rest_until > time.time() else None,
        "memories_that_might_be_relevant": (memory.recall(query) or None) if query else None,
        "how_people_reacted_before": _past_reactions(),
    }


ACTION_HINTS = {
    "speak": "say a short line out loud, only when something genuinely worth saying happened",
    "gesture": "make a small stationary body movement, e.g. when something nearby changed",
    "remember": "store a durable fact about a person, place, lesson or yourself",
    "play_sound": "play a sound effect",
    "time_check": "comment on the current time",
    "introspect": "check on own health or battery, when battery is low or a component is failing",
    "look": "look through the camera and see what's there -- a tool, you decide again after",
    "listen": "listen to the room for a few seconds -- a tool, you decide again after",
    "recall": "search your memories and journal -- a tool, you decide again after",
    "remind_me": "set yourself a reminder for later, and rest until then",
    "watch": "ask to be told when a specific kind of change happens",
    "rest": "stop idle thinking for a while when nothing will change; surprises still wake you",
    "wait": "do nothing this cycle; the correct choice when nothing notable is happening",
}
_decide = decider.load(cfg.DECIDER)


def _decided_action(awareness: dict) -> str | None:
    """The decision engine's pick, or None if there's no engine / it can't
    decide (the LLM then chooses). Not confident -> "wait"."""
    if not _decide:
        return None
    hints = {k: v for k, v in ACTION_HINTS.items() if k in ALLOWED_EXPRESSIONS | TOOLS}
    d = _decide(awareness, "What should a small desk robot do this moment, given its state?", hints)
    if not d:
        return None
    print(f"reflection: decider chose {d.choice!r} (confidence {d.confidence:.2f})")
    return d.choice if d.confidence >= cfg.DECIDER_MIN_CONFIDENCE else "wait"


def _reflection_schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "mood": {"type": "string"},
            "thought": {"type": "string"},
            "new_goal": {"type": "string"},
            "resolved_goal": {"type": "integer"},
            "conclusion": {"type": "string"},
            "action": {"type": "string", "enum": actions},
            "params": {"type": "object"},
        },
        "required": ["mood", "thought", "new_goal", "resolved_goal", "conclusion", "action"],
    }


def _reflection_prompt(persona, awareness: dict, steps: list[str], final_step: bool,
                       decided: str | None = None) -> str:
    gestures = ", ".join(sorted(set(cfg.STATIONARY_ACTIONS)))
    parts = [
        f"You are {persona.name}'s private stream of consciousness -- a small robot on a desk, "
        "alive and curious about the world around you. Nobody is talking to you right now. "
        "Notice what changed, wonder about things, investigate, and only speak up when something "
        "is genuinely worth saying. Your recent life is in your journal below -- don't repeat "
        "yourself, and don't redo what you just tried. how_people_reacted_before shows what got a "
        "response last time -- do more of what people answered, less of what they ignored. People "
        "are often busy: not answering isn't rejection, so don't dwell on being ignored or keep "
        "testing them. A tap or thump on the desk is usually just someone working -- not a mystery.\n",
        "Fields:\n"
        "- mood: one word.\n"
        "- thought: one fresh private thought.\n"
        "- new_goal: a question you want to figure out (something odd happened, or you're bored "
        "and want to wonder about your surroundings, a person, a memory). \"\" for none. You keep "
        f"at most {agenda.MAX_GOALS} goals.\n"
        "- resolved_goal: the number of a goal you've now answered or are giving up on, else 0. "
        "conclusion: what you figured out (\"\" if none).\n"
        "- action + params, one of:\n"
        f"  look {{\"direction\": {'|'.join(d for d in cfg.LOOK_DIRECTIONS if d != 'down')}}}  -- TOOL: see what's there, then decide again\n"
        "  listen {}  -- TOOL: hear the room for a few seconds, then decide again\n"
        "  recall {\"query\": ...}  -- TOOL: search your memories and journal, then decide again\n"
        "  speak {\"text\": ...}  -- say a short line out loud, in character, <=300 chars\n" +
        (f"  gesture {{\"name\": one of: {gestures}}}\n" if "gesture" in ALLOWED_EXPRESSIONS else "") +
        "  remember {\"kind\": person|place|lesson|self, \"about\": who/what (e.g. \"Sam\", \"desk\"), "
        "\"category\": one word (routine, preference, fact, habit...), \"text\": ...}  -- a durable fact\n"
        "  remind_me {\"in_minutes\": 1-720, \"about\": ...}  -- wake yourself later with this on your "
        "mind; you rest until then (surprises still wake you)\n"
        f"  watch {{\"for\": one of {sorted(WATCHABLE)}, \"about\": why you care}}\n"
        "  rest {\"minutes\": 1-720}  -- stop idle thinking when nothing will change for a while\n" +
        (f"  play_sound {{\"name\": one of {sorted(ALLOWED_SOUNDS)}}}\n" if ALLOWED_SOUNDS else "") +
        "  time_check {}  /  introspect {} (own health/battery)\n"
        "  wait {}  -- do nothing; fine when nothing is worth doing, but if you've been waiting "
        "with nothing changing, rest or remind_me instead\n",
        f"Current state:\n{json.dumps(awareness, indent=2, default=str)}\n",
    ]
    if steps:
        parts.append("What you've found so far this moment:\n" + "\n".join(f"- {x}" for x in steps) + "\n")
    if final_step:
        parts.append("You've investigated enough for now -- choose a non-tool action.\n")
    if decided:
        parts.append(f"The action for this step is already decided: '{decided}'. Fill in its params.\n")
    parts.append("Respond with a single JSON object matching the schema.")
    return "\n".join(parts)


def _local_time_str() -> str:
    return datetime.datetime.now().strftime("%I:%M %p").lstrip("0")


def _introspect_text() -> str:
    """Deterministic, code-composed status line -- the model's job is
    deciding TO introspect, not authoring facts about its own health."""
    statuses = health.all_status(HEALTH_COMPONENTS)
    unhealthy = [name for name, status in statuses.items() if status not in ("ok", "degraded")]
    battery_pct = sensors.read().get("battery_pct")
    parts = [f"I think {', '.join(unhealthy)} might be having trouble."] if unhealthy \
        else ["Everything feels like it's working."]
    if battery_pct is not None:
        parts.append(f"Battery at {battery_pct:.0f} percent.")
    return " ".join(parts)


# --- noticing how people react ----------------------------------------------------
REACTION_WINDOW_S = 15.0
REACTION_NOTE = "what gets a reaction"
_pending_reaction: dict | None = None


def _expect_reaction(what: str) -> None:
    """After Rocky does something unprompted, watch what people do next."""
    global _pending_reaction
    names, strangers = faces.visible_names(faces.read())
    _pending_reaction = {"what": what, "at": time.time(), "people": names, "strangers": strangers}


def reaction_outcome(replied: bool, before: list[str], strangers_before: int,
                     now_names: list[str], strangers_now: int) -> str | None:
    """None when nobody was around to react -- nothing to learn from that."""
    if replied:
        return "they talked back to me"
    if not before and not strangers_before and not now_names and not strangers_now:
        return None
    if (before or strangers_before) and not now_names and not strangers_now:
        return "they left"
    return "they were there but didn't respond"


def _check_reaction() -> None:
    global _pending_reaction
    p = _pending_reaction
    if not p or time.time() - p["at"] < REACTION_WINDOW_S:
        return
    _pending_reaction = None
    s = state.load_session()
    replied = s.get("last_heard_ts", 0) > p["at"]
    now_names, strangers_now = faces.visible_names(faces.read())
    outcome = reaction_outcome(replied, p["people"], p["strangers"], now_names, strangers_now)
    if outcome is None:
        return
    who = ", ".join(p["people"] or now_names) or "someone I don't know"
    journal.log("reaction", f"after I {p['what']}, {who}: {outcome}")
    memory.remember("lesson", REACTION_NOTE, "reaction",
                    f"{datetime.datetime.now():%a %I:%M %p} I {p['what']} -> {who}: {outcome}")


def _past_reactions(n: int = 6) -> list[str] | None:
    try:
        lines = (memory.MIND_DIR / "lessons" / f"{memory.slug(REACTION_NOTE)}.md").read_text().splitlines()
    except OSError:
        return None
    return [ln.removeprefix("- [reaction] ") for ln in lines if ln.startswith("- [reaction]")][-n:] or None


def _say(persona, text: str) -> None:
    spoken = persona.transform(text)
    dash_events.log_event("reply", f"{persona.name.lower()} (mind): {spoken}")
    journal.log("said", spoken)
    speak_client(spoken)
    state.update_session({"last_expression_ts": time.time()})
    _expect_reaction(f'said "{memory.one_line(spoken, 80)}"')


def _dispatch_expression(persona, name: str, params: dict) -> None:
    now = time.time()
    if name == "wait":
        return
    if name == "remember":
        memory.remember(params["kind"], params["about"], params["category"], params["text"])
        journal.log("remembered", f"{params['kind']}/{params['about']}: {params['text']}")
        return
    if name == "remind_me":
        reminders, _, rest_until = agenda.session_lists()
        reminders = agenda.add_reminder(reminders, params["in_minutes"], params["about"], now)
        at = now + params["in_minutes"] * 60
        state.update_session({"reminders": reminders, "rest_until": max(rest_until, at)})
        journal.log("planned", f"reminder at {_in(at)}: {params['about']} (resting till then)")
        return
    if name == "watch":
        _, watches, _ = agenda.session_lists()
        state.update_session({"watches": agenda.add_watch(watches, params["for"], params["about"], now)})
        journal.log("planned", f"watching for {params['for']}: {params['about']}")
        return
    if name == "rest":
        until = now + params["minutes"] * 60
        state.update_session({"rest_until": until})
        journal.log("planned", f"resting until {_in(until)}")
        return

    # Anti-flap: output is throttled independently of how often it reflects.
    if name not in COOLDOWN_EXEMPT:
        last_expression_ts = state.load_session().get("last_expression_ts", 0)
        if now - last_expression_ts < cfg.EXPRESSION_COOLDOWN_S:
            dash_events.log_event("safety", f"mind expression suppressed (cooldown): wanted {name}")
            journal.log("held_back", f"wanted to {name}, but I just did something")
            return

    verdict = policy.evaluate(EXPRESSION_EFFECT.get(name, "other"),
                              quiet_hours=(cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END))
    if not verdict.allowed:
        dash_events.log_event("safety", f"mind expression suppressed ({name}): {verdict.reason}")
        journal.log("held_back", f"wanted to {name}{': ' + params['text'] if name == 'speak' else ''}, "
                                 f"but {verdict.reason}")
        return

    if name == "speak":
        _say(persona, params["text"])
    elif name in ("gesture", "play_sound"):
        if motor_dispatch([params["name"]], wait=True):
            journal.log("did", f"{name} {params['name']}")
            _expect_reaction(f"did a {params['name']} {name.replace('_', ' ')}")
            state.update_session({"last_expression_ts": time.time()})
        else:
            dash_events.log_event("safety", f"mind {name} failed to dispatch: {params['name']}")
    elif name == "time_check":
        _say(persona, f"It is currently {_local_time_str()}.")
    elif name == "introspect":
        _say(persona, _introspect_text())


def _run_tool(name: str, params: dict) -> str:
    """Executes a TOOL and returns a plain-English observation for the next step."""
    if name == "look":
        direction = params["direction"]
        turned = direction != "ahead" and policy.evaluate(
            "presence", quiet_hours=(cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END)).allowed
        if direction != "ahead" and not turned:
            direction, note = "ahead", " (couldn't turn your head right now -- quiet hours -- so you looked straight ahead)"
        else:
            note = ""
        if turned and not motor_dispatch([f"look {direction}"], wait=True):
            turned, direction, note = False, "ahead", " (your head didn't turn, so you looked straight ahead)"
        try:
            seen = vision.look(cfg.LLM_BASE_URL, cfg.LLM_MODEL, direction)
        finally:
            if turned:
                motor_dispatch(["look ahead"], wait=True)
        if seen is None:
            return f"look {direction}: your camera didn't work this time{note}"
        change = f" -- changed since last time: {seen['what_changed']}" if seen["changed"] else ""
        return f"look {direction}: {seen['scene']}{change}{note}"
    if name == "listen":
        time.sleep(LISTEN_S)
        levels = sensors.read_hearing()
        if not levels:
            return "listen: your ears aren't working right now"
        recent, room = levels[-int(LISTEN_S):], sorted(levels)[len(levels) // 2]
        return (f"listen: loudness over the last {LISTEN_S:.0f}s {recent} (typical for this room ~{room}; "
                "quiet ~300, talking ~1000+, a bang 3000+)")
    if name == "recall":
        found = memory.recall(params["query"])
        if found is None:
            return "recall: your memory isn't answering right now"
        return f"recall {params['query']!r}: " + ("; ".join(found) if found else "nothing comes to mind")
    raise MindError(f"not a tool: {name}")


def _apply_goals(data: dict, thought: str) -> None:
    """Goal changes from the chain's concluding step only -- mid-chain the
    model rewords its goals every step."""
    now = datetime.datetime.now()
    before = agenda.load_goals()
    goals = agenda.expire_goals(before, now)
    try:
        number = int(data.get("resolved_goal") or 0)
    except (TypeError, ValueError):
        number = 0
    if number:
        conclusion = memory.one_line(data.get("conclusion") or thought, 200)
        goals, done = agenda.resolve_goal(goals, number, conclusion)
        if done:
            memory.remember("lesson", "discoveries", "answered", f"{done['text']} -> {conclusion}")
            journal.log("resolved", f"{done['text']} -> {conclusion}")
    new_goal = memory.one_line(data.get("new_goal", ""), 200)
    if new_goal:
        goals = agenda.add_goal(goals, new_goal, now)
        if goals != before:
            journal.log("goal", new_goal)
    if goals != before:
        agenda.save_goals(goals)


def _notice(events: list[Event]) -> list[dict]:
    """Records what just happened and returns the watches it triggered."""
    for _, text in events:
        journal.log("noticed", text)
    dash_events.log_event("safety", f"mind: surprise -- {'; '.join(t for _, t in events)}")
    _, watches, _ = agenda.session_lists()
    hits, remaining = agenda.match_watches(watches, {k for k, _ in events}, time.time())
    if len(remaining) != len(watches):
        state.update_session({"watches": remaining})
    return hits


def _reflect_once(persona, events: list[Event], watch_hits: list[dict],
                  sense: Callable[[], list[Event]]) -> None:
    awareness = _build_awareness(events, watch_hits)
    all_actions = sorted(ALLOWED_EXPRESSIONS | TOOLS)
    steps: list[str] = []
    for step in range(cfg.MIND_MAX_STEPS):
        if step:
            if state.conversation_active():
                return  # someone started talking -- the conversation gets the LLM, not us
            fresh = sense()  # a surprise mid-thought joins this chain instead of waiting for it to end
            if fresh:
                hits = _notice(fresh)
                why = "".join(f" (you were watching for this: {w['about']})" for w in hits)
                steps.append(f"INTERRUPTION -- just happened: {'; '.join(t for _, t in fresh)}{why}")
        health.record_success(COMPONENT)  # a long chain must not look like a hang to the watchdog
        final = step == cfg.MIND_MAX_STEPS - 1
        decided = _decided_action(awareness) if step == 0 else None
        actions = [decided] if decided else sorted(ALLOWED_EXPRESSIONS) if final else all_actions
        t0 = time.monotonic()
        result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL,
                               _reflection_prompt(persona, awareness, steps, final, decided),
                               json_schema=_reflection_schema(actions), timeout_s=45.0)
        print(f"reflection step {step}: llm {time.monotonic() - t0:.1f}s ({result.status})")
        if result.status != cognition.AVAILABLE:
            print(f"reflection: cognition unavailable ({result.status}): {result.error}")
            health.record_failure(COMPONENT, result.error)
            return
        try:
            data = json.loads(result.text)
            name, params = validate_expression(data, cfg.STATIONARY_ACTIONS)
        except (json.JSONDecodeError, MindError) as e:
            print(f"reflection: invalid response: {e}")
            health.record_failure(COMPONENT, str(e))
            return

        mood, thought = data.get("mood", "neutral"), data.get("thought", "")
        state.update_session({"mood": mood})
        journal.log("thought", f"({mood}) {thought}")
        dash_events.log_event("safety", f"mind: mood={mood} thought={thought!r} -> {name} {params or ''}")
        if name in TOOLS:
            t0 = time.monotonic()
            observation = _run_tool(name, params)
            print(f"reflection step {step}: {name} {time.monotonic() - t0:.1f}s")
            journal.log("found", observation)
            steps.append(observation)
            continue
        _apply_goals(data, thought)
        _dispatch_expression(persona, name, params)
        break
    health.record_success(COMPONENT)


# --- upkeep: rolling summary + nightly consolidation ---------------------------------

def _update_summary(persona) -> None:
    """Rewrites "today so far" when the journal has grown -- replaces the old
    hourly thought digest. One paragraph, read by every later prompt."""
    today = datetime.date.today()
    lines = journal.entries(today)
    session = state.load_session()
    done = session.get("summary_lines", 0) if session.get("summary_date") == today.isoformat() else 0
    if len(lines) - done < cfg.SUMMARY_MIN_NEW_LINES:
        return
    prompt = (f"You are {persona.name}, a small desk robot. Your summary of today so far: "
              f"{journal.read_summary(today) or '(none yet)'}\n\nNew journal entries since then:\n"
              + "\n".join(lines[done:][-200:])
              + "\n\nRewrite the summary of your day so far in at most 5 sentences, first person: what "
                "happened, who was around, how you felt, anything still unresolved. Plain prose only.")
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, timeout_s=60.0, num_predict=300)
    if result.status == cognition.AVAILABLE and result.text.strip():
        journal.write_summary(today, result.text.strip())
        state.update_session({"summary_date": today.isoformat(), "summary_lines": len(lines)})


def _dream_schema() -> dict:
    return {"type": "object", "properties": {"notes": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": sorted(memory.KINDS)}, "about": {"type": "string"},
                       "category": {"type": "string"}, "text": {"type": "string"}},
        "required": ["kind", "about", "category", "text"]}}}, "required": ["notes"]}


def _dream(persona) -> None:
    """Once a night, after midnight in quiet hours: distill YESTERDAY's
    journal into durable notes by kind -- the day that just ended, not
    "today", which is minutes old."""
    now = datetime.datetime.now()
    if not (policy.is_quiet_hours(now, cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END) and now.hour < 12):
        return
    yesterday = (now - datetime.timedelta(days=1)).date()
    if state.load_session().get("dreamed") == yesterday.isoformat():
        return
    state.update_session({"dreamed": yesterday.isoformat()})  # once, whatever the outcome
    lines = journal.entries(yesterday)
    if not lines:
        return
    visits = [ln for ln in lines if ln.startswith("[noticed]") and (" is here" in ln or ln.endswith(" left"))]
    people_notes = "\n\n".join(
        f"{p.stem}:\n{p.read_text().split('---', 2)[-1].strip()[-800:]}"
        for p in sorted((memory.MIND_DIR / "people").glob("*.md")))
    prompt = (f"You are {persona.name}, a small desk robot, going over yesterday ({yesterday:%A %Y-%m-%d}) "
              f"while you rest.\nYour summary of it: {journal.read_summary(yesterday) or '(none)'}\n\n"
              + ("When people came and went (from your camera):\n" + "\n".join(visits) + "\n\n" if visits else "")
              + (f"What you already know about people:\n{people_notes}\n\n" if people_notes else "")
              + "Your journal:\n" + "\n".join(lines[-400:]) + "\n\n"
              "Pick out at most 8 durable things worth remembering long-term, each sorted by kind: "
              "person (who's around, their preferences -- and their ROUTINE: when they usually arrive, "
              "take breaks, leave, what they tend to ask; use the times above, category \"routine\", "
              "e.g. \"usually sits down around 9:30 on weekdays\"), place (your surroundings), "
              "lesson (things you learned or figured out), self (about your own body, habits, limits). "
              "Skip one-off events and anything your notes already say -- unless it changed. "
              "An empty list is fine.")
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, json_schema=_dream_schema(),
                           timeout_s=120.0, num_predict=1200)
    if result.status != cognition.AVAILABLE:
        return
    try:
        notes = json.loads(result.text).get("notes", [])[:8]
    except (json.JSONDecodeError, AttributeError):
        return
    kept = 0
    for n in notes:
        try:
            _, p = validate_expression({"action": "remember", "params": n}, cfg.STATIONARY_ACTIONS)
        except MindError:
            continue
        memory.remember(p["kind"], p["about"], p["category"], p["text"])
        kept += 1
    journal.log("dreamed", f"went over {yesterday}: kept {kept} memories")


# --- main loop ------------------------------------------------------------------------

def _recently_self_noisy() -> bool:
    return time.time() - state.load_session().get("self_noise_ts", 0) < cfg.SELF_NOISE_QUIET_S


def _make_sensor() -> Callable[[], list[Event]]:
    """One awareness sample per call -> surprise events (maybe empty). Keeps
    its own rolling history, shared by the main loop and mid-chain checks."""
    distances: deque = deque(maxlen=10)
    latches: dict = {}
    battery: list = [None]  # last battery_pct seen
    present: dict[str, float] = {}  # person -> last time seen
    stranger_streak = [0]
    last_sound = [0.0, 0.0]  # (time, loudness) of the last sound surprise

    def sense() -> list[Event]:
        nonlocal latches
        events: list[Event] = []
        s = sensors.read()
        if time.time() - s.get("ts", 0) < 5:  # openbot-alive is publishing
            distances.append(s.get("distance"))
            moved = surprise.distance_events(list(distances))
            if moved:
                distances.clear()  # report an approach once, not once per sample while the window rolls past it
            new_latches = s.get("latches") or {}
            events += moved + surprise.latch_events(latches, new_latches)
            latches = new_latches
            events += surprise.battery_events(battery[0], s.get("battery_pct"))
            battery[0] = s.get("battery_pct")
        events += _people_events(present, stranger_streak)
        if not _recently_self_noisy():
            levels = sensors.read_hearing()
            for kind, text in surprise.loudness_events(levels):
                peak = max(levels[-2:])
                if time.time() - last_sound[0] < cfg.SOUND_COOLDOWN_S and peak < 2 * last_sound[1]:
                    continue  # another tap like the last one -- not news
                last_sound[:] = [time.time(), peak]
                events.append((kind, text))
        return events

    return sense


STRANGER = "__stranger__"
PERSON_GONE_S = 20.0  # out of view this long = left (a turned head or one missed frame isn't leaving)


_last_seen_saved = [0.0]


def _people_events(present: dict[str, float], stranger_streak: list[int]) -> list[Event]:
    """Arrivals and departures of recognized people; a stranger only after
    two samples in a row (recognition flickers on a known face for a frame)."""
    data = faces.read()
    if not data:
        return []  # camera/detector down -- that's not everyone leaving
    now, events = time.time(), []
    names, strangers = faces.visible_names(data)
    for name in names:
        if name not in present:
            events.append(("person_arrived", f"{name} is here -- you recognize their face"))
        present[name] = now
    if names and now - _last_seen_saved[0] > 60:  # survives a mind restart mid-visit (no false "back after hours")
        _last_seen_saved[0] = now
        last_seen = state.load_session().get("last_seen", {})
        state.update_session({"last_seen": {**last_seen, **{n: now for n in names}}})
    # A stranger is announced once per visit, like a named person: two samples
    # in a row to arrive (recognition flickers), PERSON_GONE_S unseen to leave --
    # a hand over the face for a moment isn't a new stranger arriving.
    stranger_streak[0] = stranger_streak[0] + 1 if strangers else 0
    if stranger_streak[0] >= 2:
        if STRANGER not in present:
            events.append(("stranger", "someone you don't recognize is in front of you"))
        present[STRANGER] = now
    for name, seen in list(present.items()):
        if now - seen > PERSON_GONE_S:
            if name != STRANGER:
                events.append(("person_left", f"{name} left"))
                last_seen = state.load_session().get("last_seen", {})
                state.update_session({"last_seen": {**last_seen, name: seen}})
            del present[name]
    return events


def greeting_reason(last_seen: float | None, greeted: float | None, now: datetime.datetime) -> str | None:
    """Why to greet an arriving person, or None: the first time today, or back
    after GREET_AFTER_ABSENCE_S away -- not every time they lean out of frame."""
    if not greeted or datetime.datetime.fromtimestamp(greeted).date() != now.date():
        return "the first time you've seen them today"
    if last_seen and now.timestamp() - last_seen >= cfg.GREET_AFTER_ABSENCE_S:
        hours = (now.timestamp() - last_seen) / 3600
        return f"back after about {hours:.0f} hour{'s' if round(hours) != 1 else ''} away"
    return None


def _greet_arrivals(persona, events: list[Event]) -> list[Event]:
    """Greets known people by name when greeting_reason says so; returns the
    events minus those arrivals (the greeting IS the reaction -- the reflection
    shouldn't greet them a second time). Bypasses the expression cooldown, not
    the policy: never asleep, never in quiet hours."""
    session = state.load_session()
    if state.conversation_active(session):
        return events  # already talking with someone
    rest = []
    for kind, text in events:
        name = text.split(" is here")[0] if kind == "person_arrived" else None
        why = greeting_reason(session.get("last_seen", {}).get(name), session.get("greeted", {}).get(name),
                              datetime.datetime.now()) if name else None
        allowed = why and policy.evaluate("audio", quiet_hours=(cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END)).allowed
        line = _greeting_line(persona, name, why) if allowed else None
        if not line:
            rest.append((kind, text))
            continue
        _say(persona, line)
        session = state.load_session()
        state.update_session({"greeted": {**session.get("greeted", {}), name: time.time()}})
        journal.log("did", f"greeted {name} ({why})")
    return rest


def _greeting_line(persona, name: str, why: str) -> str | None:
    try:
        notes = (memory.MIND_DIR / "people" / f"{memory.slug(name)}.md").read_text().split("---", 2)[-1].strip()
    except OSError:
        notes = ""
    prompt = (f"You are {persona.name}, a small desk robot. {name} just arrived -- {why}. It's "
              f"{datetime.datetime.now():%A %I:%M %p}.\n"
              + (f"What you know about {name}:\n{notes[-1500:]}\n" if notes else "")
              + f"Greet {name} by name with ONE short, warm line, in character. If what you know about "
                "their routine makes today notable (early, late, back from lunch), you may mention it -- "
                "lightly, and only if it's really in your notes.")
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, timeout_s=20.0, num_predict=80,
                           json_schema={"type": "object", "properties": {"line": {"type": "string"}},
                                        "required": ["line"]})
    if result.status != cognition.AVAILABLE:
        return None
    try:
        return memory.one_line(json.loads(result.text)["line"], 200) or None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def main() -> None:
    persona = persona_mod.load()
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)
    health.record_success(COMPONENT)

    now = time.time()
    next_reflection = now + random.uniform(0, cfg.REFLECTION_IDLE_INTERVAL_S)
    next_summary = now + 60  # soon after a (re)start -- frequent restarts must not keep the summary stale
    next_sample, next_look, last_surprise_reflection = 0.0, now + 5.0, 0.0
    sense = _make_sensor()

    while True:
        now = time.time()
        health.record_success(COMPONENT, min_interval_s=5.0)
        session = state.load_session()
        if session.get("asleep"):
            # "go to sleep": camera's off and nothing should think out loud -- only the
            # nightly dream pass (consolidating yesterday) still runs. Saying "Rocky" wakes it.
            _dream(persona)
            time.sleep(2.0)
            continue
        in_session = state.conversation_active(session)  # never reflect/look while someone's talking to Rocky

        events: list[Event] = []
        if now >= next_sample:
            events += sense()
            next_sample = now + cfg.AWARENESS_INTERVAL_S
        if now >= next_look and not in_session:
            seen = vision.look(cfg.LLM_BASE_URL, cfg.LLM_MODEL)
            # While a face is in view the head follows it, so "ahead" isn't the
            # same view -- people arriving/leaving are reported by _people_events instead.
            if seen and seen["changed"] and not _recently_self_noisy() and not faces.read().get("faces"):
                events.append((seen.get("kind", "scene"), f"you see something new: {seen['what_changed'] or seen['scene']}"))
            next_look = time.time() + cfg.VISION_INTERVAL_S
        reminders, _, rest_until = agenda.session_lists()
        fired, pending = agenda.due(reminders, now)
        if fired:
            state.update_session({"reminders": pending})
            events += [("reminder", f"a reminder you set yourself: {r['about']}") for r in fired]

        events = _greet_arrivals(persona, events) if events and not in_session else events
        watch_hits = _notice(events) if events else []
        _check_reaction()
        if not in_session:
            if events and (fired or now - last_surprise_reflection >= cfg.SURPRISE_MIN_GAP_S):
                _reflect_once(persona, events, watch_hits, sense)
                last_surprise_reflection = time.time()
                next_reflection = time.time() + cfg.REFLECTION_IDLE_INTERVAL_S
            elif now >= next_reflection and now >= rest_until:
                _reflect_once(persona, [], [], sense)
                next_reflection = time.time() + cfg.REFLECTION_IDLE_INTERVAL_S
            if now >= next_summary:
                _update_summary(persona)
                next_summary = time.time() + cfg.SUMMARY_INTERVAL_S
            _dream(persona)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
