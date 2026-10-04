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
                gated by policy (asleep) and an anti-flap cooldown.
  texting    -- after each reflection, with WhatsApp set up: the decision
                engine (Jev), not the LLM or a rule, decides whether the
                moment is worth texting its person; the LLM then writes it
                and openbot-chat sends it.
  upkeep     -- a rolling "today so far" summary every SUMMARY_INTERVAL_S,
                and once a day (first thing after midnight) a "dream"
                pass that distills yesterday's journal into notes by kind.

The wheels: `look` turns only the camera gimbal; the only wheel moves the
mind may choose are config.PLAYFUL_ACTIONS -- the body's short, bounded,
cliff-checked playful moves -- as gestures. Driving anywhere is a spoken
command, never a reflection.
"""
from __future__ import annotations

import datetime
import json
import random
import threading
import time
from collections import deque
from typing import Any, Callable


import config as cfg  # noqa: E402
from common import agenda, cognition, contacts, decider, events as dash_events, faces, health, journal, memory  # noqa: E402
from common import hearing, objects, outcomes, tools  # noqa: E402
from common import persona as persona_mod, policy, react, sensors, state, surprise, vision  # noqa: E402
from common.motor_client import dispatch as motor_dispatch  # noqa: E402
from common.speak_client import speak as speak_client  # noqa: E402

COMPONENT = "openbot-mind"
# openbot-memory isn't listed: it's Basic Memory itself, which doesn't write our health records.
HEALTH_COMPONENTS = ["openbot-alive"] * cfg.HAS_BODY + ["openbot-ears", "openbot-wake-listen", "openbot-mind",
                                                        "openbot-speak"]

ALLOWED_EXPRESSIONS = {"speak", "gesture", "wait", "remember", "play_sound", "time_check", "introspect",
                       "remind_me", "watch", "rest", "wish", "sleep", "go_to", "explore"}
if not cfg.CAN_DRIVE:
    ALLOWED_EXPRESSIONS -= {"go_to", "explore"}
EXPLORE_S = (10, 120)
INVITE_S = 20.0  # after the mind speaks, wake-listen listens this long for an answer -- no wake word needed
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
    "go_to": "motion", "explore": "motion", "sleep": "presence",
}  # everything else is silent bookkeeping ("other")
# Silent bookkeeping isn't an expressive act -- exempt from the cooldown.
COOLDOWN_EXEMPT = {"wait", "remember", "remind_me", "watch", "rest", "wish", "sleep",
                   # A gesture is the model's call, whenever it likes: nothing else moves the body
                   # when nobody's talking to it (no canned fidgets) -- only speech is throttled.
                   "gesture"}
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
        if gesture not in stationary_actions and gesture not in cfg.PLAYFUL_ACTIONS:
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
    if name == "wish":
        return name, {"text": _text(params, "text", 200)}
    if name in ("go_to", "explore"):
        if vision.last_look().get("scene") == vision.DARK_SCENE:
            raise MindError("too dark to drive")
        if name == "go_to":
            return name, {"target": _text(params, "target", 40)}
        try:
            secs = int(params.get("seconds", 30))
        except (TypeError, ValueError):
            raise MindError("seconds must be a whole number")
        return name, {"seconds": max(EXPLORE_S[0], min(EXPLORE_S[1], secs))}
    if name == "sleep":
        if not any(n.startswith(("dark", "tired")) for n in _needs()):
            raise MindError("not tired: sleep only when it's dark and quiet or the battery is low")
        if faces.read().get("faces"):
            raise MindError("someone is right in front of you -- not a time to sleep")
        return name, {}
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
    s, session = sensors.read(), state.load_session()
    head = s.get("head")
    look = vision.last_look()
    goals = agenda.open_goals(agenda.load_goals())
    reminders, watches, rest_until = agenda.session_lists()
    unhealthy = {k: v for k, v in health.all_status(HEALTH_COMPONENTS).items() if v not in ("ok", "degraded")}
    query = " ".join([text for _, text in events] + [g["text"] for g in goals])
    return {
        "local_time": datetime.datetime.now().strftime("%A %I:%M %p"),
        "weather_outside": tools.weather(),
        "you_are_driving_right_now": True if s.get("driving") else None,
        "your_needs": _needs() or None,
        "goal_to_pursue_now": goals[0]["text"] if goals else None,
        "what_you_have_learned_works_with_people": _what_works() or None,
        "people_routines_you_know": _routines() or None,
        "people_last_seen": _last_seen() or None,
        "what_just_changed": [text for _, text in events] or "nothing -- it's been quiet",
        "you_were_watching_for_this": [f"{w['for']} -- because: {w['about']}" for w in watch_hits] or None,
        "what_you_see": {"scene": look.get("scene"), "looked": _ago(look.get("ts"))} if look else None,
        "where_your_head_points": vision.head_words(head["pan"], head["tilt"]) if head else None,
        "which_way_you_face": vision.facing_words(),
        "you_were_moved": f"{session['moved_how']} {_ago(session['moved_ts'])} -- you may face another way now"
                          if time.time() - session.get("moved_ts", 0) < 1800 else None,
        "who_is_in_front_of_you": faces.describe(faces.read()) or "nobody",
        "things_in_view": objects.names(objects.read()),
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


def _needs() -> list[str]:
    """Code-derived drives from the body's own state -- the model decides what
    to do about them (ask to be charged, rest, sleep, say it can't see)."""
    s, needs = sensors.read(), []
    pct = s.get("battery_pct")
    if pct is not None and pct < 20:
        needs.append(f"tired: battery at {pct:.0f}% -- you'd like to be charged; rest more, move less")
    look = vision.last_look()
    if look.get("scene") == vision.DARK_SCENE and max((hearing.read() or {}).get("levels", [])[-5:] or [0]) < 600:
        needs.append("dark and quiet: a good time to sleep (the `sleep` action) -- unless something is going on")
    unhealthy = [k for k, v in health.all_status(HEALTH_COMPONENTS).items() if v not in ("ok", "degraded")]
    if "openbot-camera" in unhealthy:
        needs.append("you can't see right now (camera trouble) -- worth saying if someone's around")
    return needs


WHAT_WORKS = "what works"


def _what_works() -> list[str]:
    return [ln.split("] ", 1)[-1] for ln in memory.read_note("self", WHAT_WORKS)][-6:]


def _routines() -> list[str]:
    """Routine lines from people notes ("usually sits down around 9:30") -- so the
    model can expect people, and notice when a routine is broken."""
    out = []
    for path in sorted((memory.MIND_DIR / "people").glob("*.md")):
        lines = [ln.split("] ", 1)[-1] for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("- [routine]")]
        out += [f"{path.stem}: {ln}" for ln in lines[-2:]]
    return out[:8]


def _last_seen() -> dict[str, str]:
    seen = state.load_session().get("last_seen", {})
    return {name: _ago(ts) for name, ts in sorted(seen.items(), key=lambda kv: -kv[1])[:4] if _ago(ts)}


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
    "wish": "write down something you wish you could do or have -- your developer reads these",
    "sleep": "go to sleep for the night when it's dark and quiet, or you're very low on battery",
    "go_to": "drive to something you can see, to look at it up close",
    "explore": "drive around for a while to see what's there",
}
_decide = decider.load(cfg.DECIDER)
# Texting first: the engine decides whether a moment is worth a WhatsApp text (not the LLM, not a rule).
_text_decider = decider.load(cfg.TEXT_DECIDER) if cfg.CHAT_ALLOW else None
TEXT_OPTIONS = {
    "text": "Text them now: this is one of those moments -- or something else they'd really want to know right now.",
    "stay_quiet": "Keep it to himself: same mood as before, the usual motion or noise, or he already told them.",
}
TEXT_FLOOR_S = 10 * 60  # however keen the engine: one text per 10 min at most, so a loop can't flood them
_TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}, "photo": {"type": "boolean"}},
                "required": ["text", "photo"]}
_texting = threading.Lock()  # one decision at a time -- a slow engine never stalls the mind loop


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
            "if_a_hand_reaches_out_gesture": {"type": "string", "enum": cfg.TONE_ACTIONS or ["none"]},
            "if_a_hand_reaches_out_say": {"type": "string"},
        },
        "required": ["mood", "thought", "new_goal", "resolved_goal", "conclusion", "action",
                     "if_a_hand_reaches_out_gesture", "if_a_hand_reaches_out_say"],
    }


def _reflection_prompt(persona, awareness: dict, steps: list[str], final_step: bool,
                       decided: str | None = None) -> str:
    gestures = cfg.describe_actions(cfg.STATIONARY_ACTIONS)
    parts = [
        f"You are {persona.name}'s private stream of consciousness -- a small robot on a desk, "
        "alive and curious about the world around you. Nobody is talking to you right now. "
        "Notice what changed, wonder about things, investigate, and only speak up when something "
        "is genuinely worth saying. Your recent life is in your journal below -- don't repeat "
        "yourself, and don't redo what you just tried. how_people_reacted_before shows what got a "
        "response last time -- do more of what people answered, less of what they ignored. People "
        "are often busy: not answering isn't rejection, so don't dwell on being ignored or keep "
        "testing them. A tap or thump on the desk is usually just someone working -- not a mystery. "
        "Your head turns by itself to follow faces and glance around, and people sometimes pick you up "
        "and turn you: a different view usually means you're looking another way (where_your_head_points, "
        "you_were_moved) -- not that the room changed. When which_way_you_face is \"not sure\", looking around "
        "(left, right) helps you recognize where you are. "
        "Your body never moves on its own (your head follows faces, that's all): every nod, shake, "
        "stretch or thinking pose is yours to choose, here -- a small gesture now and then, fitting your "
        "mood, is how you show you're alive, even when nothing happened. Speaking is for when there's "
        "something worth saying -- and when you speak, you'll hear an answer for a few seconds without "
        "them saying your name, so questions work. goal_to_pursue_now is yours: take one concrete step "
        "toward it this moment (look, listen, recall, go somewhere, ask) -- goals are for pursuing, not "
        "keeping. your_needs are your body talking: act on them. what_you_have_learned_works_with_people "
        "and people_routines_you_know are hard-won -- use them to expect people, notice a broken routine, "
        "and choose things that got a response.\n",
        "Fields:\n"
        "- mood: one word.\n"
        "- thought: one fresh private thought.\n"
        "- new_goal: a question you want to figure out (something odd happened, or you're bored "
        "and want to wonder about your surroundings, a person, a memory). \"\" for none. You keep "
        f"at most {agenda.MAX_GOALS} goals.\n"
        "- resolved_goal: the number of a goal you've now answered or are giving up on, else 0. "
        "conclusion: what you figured out (\"\" if none).\n"
        "- if_a_hand_reaches_out_gesture / if_a_hand_reaches_out_say: decide NOW, in your current mood, what "
        "you'd do and say if someone holds a hand or fist out to you in the next few minutes -- so you can "
        "react instantly. A gesture (any of the above, including the wheel ones) and a word or two (or \"\").\n"
        "- action + params, one of:\n"
        f"  look {{\"direction\": {'|'.join(d for d in cfg.LOOK_DIRECTIONS if d != 'down')}}}  -- TOOL: see what's there, then decide again\n"
        "  listen {}  -- TOOL: hear the room for a few seconds, then decide again\n"
        "  recall {\"query\": ...}  -- TOOL: search your memories and journal, then decide again\n"
        "  speak {\"text\": ...}  -- say a short line out loud, in character, <=300 chars\n" +
        (f"  gesture {{\"name\": one of: {gestures}" + (f"; or, on the wheels, when it fits: "
         f"{cfg.describe_actions(cfg.PLAYFUL_ACTIONS)}" if cfg.PLAYFUL_ACTIONS else "")
         + "}\n" if "gesture" in ALLOWED_EXPRESSIONS else "") +
        "  remember {\"kind\": person|place|lesson|self, \"about\": who/what (e.g. \"Sam\", \"desk\"), "
        "\"category\": one word (routine, preference, fact, habit...), \"text\": ...}  -- a durable fact\n"
        "  remind_me {\"in_minutes\": 1-720, \"about\": ...}  -- wake yourself later with this on your "
        "mind; you rest until then (surprises still wake you)\n"
        f"  watch {{\"for\": one of {sorted(WATCHABLE)}, \"about\": why you care}}\n"
        "  rest {\"minutes\": 1-720}  -- stop idle thinking when nothing will change for a while\n"
        "  wish {\"text\": ...}  -- something you wish you could do or have; your developer reads these\n"
        "  sleep {}  -- for the night: everything off until someone says your name and \"wake up\". Only when "
        "it's dark and quiet with nobody around, or you're very low on battery\n" +
        ("  go_to {\"target\": a thing you can see, e.g. \"pink toy\", \"person\"}  -- drive to it and look up "
         "close (on the table it stops at the edge; never in the dark)\n"
         f"  explore {{\"seconds\": {EXPLORE_S[0]}-{EXPLORE_S[1]}}}  -- drive around to see what's there\n"
         if "go_to" in ALLOWED_EXPRESSIONS else "") +
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
REACTION_NOTE = outcomes.REACTION_NOTE
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
    outcomes.record(p["what"], f"{who}: {outcome}", kind="unprompted", decided_ts=p["at"])


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
    # Open ears: wake-listen listens for INVITE_S without the wake word, so a question gets its answer.
    state.update_session({"last_expression_ts": time.time(), "invite_until": time.time() + INVITE_S,
                          "invite_text": spoken})
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
    if name == "wish":
        memory.remember("self", "wishes", "wish", f"{datetime.datetime.now():%Y-%m-%d} {params['text']}")
        journal.log("wished", params["text"])
        return

    # Anti-flap: output is throttled independently of how often it reflects.
    if name not in COOLDOWN_EXEMPT:
        last_expression_ts = state.load_session().get("last_expression_ts", 0)
        if now - last_expression_ts < cfg.EXPRESSION_COOLDOWN_S:
            dash_events.log_event("safety", f"mind expression suppressed (cooldown): wanted {name}")
            journal.log("held_back", f"wanted to {name}, but I just did something")
            return

    effect = "motion" if name == "gesture" and params["name"] in cfg.PLAYFUL_ACTIONS else EXPRESSION_EFFECT.get(name, "other")
    verdict = policy.evaluate(effect)
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
            if name == "play_sound":  # a gesture doesn't start the speech cooldown
                state.update_session({"last_expression_ts": time.time()})
        else:
            dash_events.log_event("safety", f"mind {name} failed to dispatch: {params['name']}")
    elif name == "time_check":
        _say(persona, f"It is currently {_local_time_str()}.")
    elif name == "sleep":
        journal.log("did", "went to sleep on my own -- dark and quiet, nobody around")
        dash_events.log_event("wake", "went to sleep (own decision)")
        state.update_session({"asleep": True})
        motor_dispatch(["look down"], wait=True)
    elif name in ("go_to", "explore"):
        from common.motor_client import navigate
        task = {"approach": params["target"]} if name == "go_to" else {"explore": params["seconds"]}
        ok, err = navigate(task)
        what = f"to the {params['target']}" if name == "go_to" else f"around for {params['seconds']}s"
        if ok:
            journal.log("did", f"decided to drive {what}")
            _expect_reaction(f"drove {what}")
            state.update_session({"last_expression_ts": time.time()})
        else:
            journal.log("held_back", f"wanted to drive {what}, but: {err or 'the body did not answer'}")
    elif name == "introspect":
        _say(persona, _introspect_text())


QUICK_REACT_GAP_S = 5.0  # between two reactions to a hand -- a second bump 5s later is a second bump
TOUCH_PLAN_FRESH_S = 15 * 60  # a reaction decided this recently (in each reflection) is used as-is: no LLM wait
_last_quick = [0.0]


def _react_to_touch(persona, events: list[Event]) -> list[Event]:
    """Something came right up to Rocky (within SAFE_DISTANCE): react NOW -- a
    gesture and maybe a word, the LLM's pick -- even mid-conversation: a
    hand held out is someone engaging, like speech. The slow
    reflection (rate-limited, paused while talking) would miss it. Returns the
    events minus the one reacted to."""
    touch = [t for k, t in events if k == "approach"]
    s = sensors.read()
    close = (s.get("distance") is not None and s["distance"] <= cfg.SAFE_DISTANCE) or \
        any(s.get("latches", {}).get(k) for k in ("danger", "caution"))
    if not touch or not close or time.time() - _last_quick[0] < QUICK_REACT_GAP_S \
            or state.load_session().get("asleep") or s.get("driving"):
        return events
    _last_quick[0] = time.time()
    plan = state.load_session().get("touch_plan") or {}
    if plan.get("gesture") in cfg.TONE_ACTIONS and time.time() - plan.get("ts", 0) < TOUCH_PLAN_FRESH_S:
        line, tone = persona.transform(plan.get("say", "")), plan["gesture"]  # decided ahead: instant
    else:
        line, tone = react.line(
            persona, touch[0] + " -- probably a hand or fist held out to you. React in a word or two (or say "
                f"nothing: an empty reply), and pick your gesture -- also allowed now: {cfg.describe_actions(cfg.PLAYFUL_ACTIONS)}.",
            "", timeout_s=8.0)
    tone = tone or (cfg.PLAYFUL_ACTIONS[0] if cfg.PLAYFUL_ACTIONS else None)  # brain offline: still a nudge
    if tone and motor_dispatch([tone]):
        journal.log("did", f"{tone} -- something came right up to me")
        dash_events.log_event("safety", f"touch reaction: {tone}" + (f" / {line!r}" if line else ""))
    if line:
        journal.log("said", line)
        speak_client(line)
    state.update_session({"last_expression_ts": time.time()})
    _expect_reaction(f"reacted to something coming close with a {tone}" if tone else "noticed something close")
    return [(k, t) for k, t in events if k != "approach"]


def _run_tool(name: str, params: dict) -> str:
    """Executes a TOOL and returns a plain-English observation for the next step."""
    if name == "look":
        direction = params["direction"]
        turned = direction != "ahead" and policy.evaluate(
            "presence").allowed
        if direction != "ahead" and not turned:
            direction, note = "ahead", " (couldn't turn your head right now, so you looked straight ahead)"
        else:
            note = ""
        if turned and not motor_dispatch([f"look {direction}"], wait=True):
            turned, direction, note = False, "ahead", " (your head didn't turn, so you looked straight ahead)"
        try:
            seen = vision.look(cfg.LLM_BASE_URL, cfg.LLM_MODEL, direction,
                               head=cfg.LOOK_ANGLES.get(direction) if turned else None)
        finally:
            if turned:
                motor_dispatch(["look ahead"], wait=True)
        if seen is None:
            return f"look {direction}: your camera didn't work this time{note}"
        change = f" -- changed since last time: {seen['what_changed']}" if seen["changed"] else ""
        found = f" -- {seen['bearings']}" if seen.get("bearings") else ""
        if found:
            journal.log("found", seen["bearings"])
        return f"look {direction}: {seen['scene']}{change}{note}{found}"
    if name == "listen":
        time.sleep(LISTEN_S)
        heard = hearing.read()
        if heard is None or not heard["levels"]:
            return "listen: your hearing service (openbot-ears) isn't answering right now -- that's not silence"
        n, levels = int(LISTEN_S), heard["levels"]
        recent, room = levels[-n:], sorted(levels)[len(levels) // 2]
        return (f"listen: {hearing.describe(heard['sounds'][-n:])}; loudness over the last {n}s {recent} "
                f"(typical for this room ~{room}; quiet ~300, talking ~1000+, a bang 3000+)")
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
    mood_before = state.load_session().get("mood")
    steps: list[str] = []
    for step in range(cfg.MIND_MAX_STEPS):
        if step:
            if state.conversation_active():
                return  # someone started talking -- the conversation gets the LLM, not us
            fresh = _react_to_touch(persona, sense())  # a surprise mid-thought joins this chain instead of waiting
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
        plan = {"gesture": data.get("if_a_hand_reaches_out_gesture"), "say": memory.one_line(
            data.get("if_a_hand_reaches_out_say", ""), 80), "ts": time.time()}
        state.update_session({"mood": mood, "thought": memory.one_line(thought, 200),
                              "touch_plan": plan if plan["gesture"] in cfg.TONE_ACTIONS else None})
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
        if _text_decider:
            threading.Thread(target=_consider_texting, daemon=True,
                             args=(persona, events, steps, mood_before, mood, thought, name)).start()
        break
    health.record_success(COMPONENT)


def _consider_texting(persona, events: list[Event], steps: list[str], mood_before: str | None,
                      mood: str, thought: str, action: str) -> None:
    """Should this moment be a WhatsApp text to its person? The engine decides
    (every question and answer is in the dashboard's Jev tab); on "text" the
    LLM writes it and openbot-chat sends it (session["text_out"]). Replies to
    their texts never come here -- those are openbot-chat's own."""
    if not _texting.acquire(blocking=False):
        return
    try:
        s = state.load_session()
        if time.time() - s.get("last_text_ts", 0) < TEXT_FLOOR_S:
            return
        if all(state.texting_paused(n, s) for n in cfg.CHAT_ALLOW):
            return  # they asked for a break from its texts (skills/pause_texting): don't even ask
        happened = [t for _, t in events] or "nothing new -- an idle moment"
        to = contacts.names(cfg.CHAT_ALLOW)  # linked like faces: "My name is Atul" texted once
        seen, _ = faces.visible_names(faces.read())
        moment = {
            "local_time": datetime.datetime.now().strftime("%A %I:%M %p"),
            "what_just_happened": happened,
            "what_he_found_out": steps or None,
            "his_mood_before": mood_before,
            "his_mood_now": mood,
            "his_thought": thought,
            "what_he_chose_to_do": action,
            "who_he_sees": faces.describe(faces.read()) or "nobody",
            "the_text_would_go_to": to or "someone he doesn't know by name yet",
            "they_are_right_in_front_of_him": bool(set(to) & set(seen)),
            "he_last_texted_them": _ago(s.get("last_text_ts")) or "never",
            "they_last_texted_him": _ago(s.get("chat_last_heard_ts")) or "never",
            "how_his_last_texts_went": [r for r in outcomes.recent(12) if r.startswith("texted")][-3:] or None,
            "his_open_questions": [g["text"] for g in agenda.open_goals(agenda.load_goals())] or None,
        }
        who = " and ".join(to) or "his person"
        decided_ts = time.time()  # joins this choice's outcome (common/outcomes.py) to its Jev call
        d = _text_decider(moment, f"{persona.name} is a small robot at home; {who} may be away and wants a WhatsApp "
                          f"text from him only when {cfg.CHAT_TEXT_WHEN}. Otherwise he stays quiet. Should he "
                          f"text {who} about this moment, right now?", TEXT_OPTIONS)
        if not d or d.choice != "text":  # Jev's call, whatever its confidence -- 0.6 had blocked a 70% "text"
            return
        prompt = journal.inject(
            f"You are {persona.name}, a small robot at home, and you've decided to text "
            f"{' and '.join(to) or 'your person'} on WhatsApp, on your own -- they may be away. "
            f"What just happened: {happened}. You feel {mood}; you're "
            f"thinking: {thought}\n"
            + (f"Things you've been wondering about: {'; '.join(moment['his_open_questions'])} -- if they could "
               "answer one, ask them. " if moment["his_open_questions"] else "") +
            "Write that text: a line or two, like a friend texting, no greeting needed. "
            "Don't say you're moving or going anywhere. photo: true attaches what your camera sees right now -- "
            "for something they'd want to see.", query=thought)
        result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, json_schema=_TEXT_SCHEMA,
                               timeout_s=45.0, num_predict=200)
        try:
            out = json.loads(result.text) if result.status == cognition.AVAILABLE else {}
            text = persona.transform(memory.one_line(str(out.get("text", "")), 500))
        except (json.JSONDecodeError, AttributeError):
            return
        if not text:
            return
        photo, now = out.get("photo") is True, time.time()
        state.update_session({"text_out": {"text": text, "photo": photo, "ts": now, "decided_ts": decided_ts},
                              "last_text_ts": now})
        journal.log("texted", text + (" [with a photo]" if photo else ""))
        dash_events.log_event("reply", f"{persona.name.lower()} texts (Jev: {d.confidence:.2f}): {text}")
    finally:
        _texting.release()


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
    """Once a day, first thing after midnight: distill YESTERDAY's
    journal into durable notes by kind -- the day that just ended, not
    "today", which is minutes old."""
    now = datetime.datetime.now()
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
    kept: list[str] = []
    for n in notes:
        try:
            _, p = validate_expression({"action": "remember", "params": n}, cfg.STATIONARY_ACTIONS)
        except MindError:
            continue
        memory.remember(p["kind"], p["about"], p["category"], p["text"])
        kept.append(f"- [{p['kind']}/{p['about']}] {p['text']}")
    journal.log("dreamed", f"went over {yesterday}: kept {len(kept)} memories")
    # The night's dream as its own note (state/mind/dreams/<date>.md): what it kept, readable
    # later -- in the dashboard's Dreams tab, Obsidian, or memory.recall.
    with memory.ensure_note("dreams", yesterday.isoformat(), f"Dream about {yesterday}", "dream",
                            f"Summary of the day: {journal.read_summary(yesterday) or '(none)'}\n\n").open("a") as f:
        f.write("\n".join(kept) + ("\n" if kept else "- (nothing worth keeping)\n"))
    _distill_what_works(persona)


def _distill_what_works(persona) -> None:
    """Rewrites self/what-works.md from the reaction log: 3-6 rules about what
    gets a response from people and what doesn't. Rewritten whole each night,
    so it tracks the people it lives with rather than piling up."""
    reactions = _past_reactions(40)
    if not reactions or len(reactions) < 4:
        return
    prompt = (f"You are {persona.name}, a small desk robot. Here is what happened after things you did "
              "unprompted (what you did -> how people reacted):\n" + "\n".join(reactions)
              + "\n\nWrite 3-6 short rules for yourself about what works with these people and what doesn't "
                "(what to do more of, what to drop, what times are good or bad). Plain, specific, first person.")
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, timeout_s=60.0, num_predict=300,
                           json_schema={"type": "object", "properties": {"rules": {"type": "array",
                                        "items": {"type": "string"}}}, "required": ["rules"]})
    if result.status != cognition.AVAILABLE:
        return
    try:
        rules = [memory.one_line(r, 200) for r in json.loads(result.text)["rules"]][:6]
    except (json.JSONDecodeError, KeyError, TypeError):
        return
    if rules:
        memory.write_note("self", WHAT_WORKS, rules, "rule")
        journal.log("learned", f"what works with people: {'; '.join(rules)}")


# --- main loop ------------------------------------------------------------------------

def _recently_self_noisy() -> bool:
    return time.time() - state.load_session().get("self_noise_ts", 0) < cfg.SELF_NOISE_QUIET_S


HEAD_SETTLE_S = 2.0  # the camera's "motion" this soon after its own head moved is the head, not the room


def _make_sensor() -> Callable[[], list[Event]]:
    """One awareness sample per call -> surprise events (maybe empty). Keeps
    its own rolling history, shared by the main loop and mid-chain checks."""
    distances: deque = deque(maxlen=10)
    latches: dict = {}
    battery: list = [None]  # last battery_pct seen
    present: dict[str, float] = {}  # person -> last time seen
    streaks: dict[str, int] = {}  # person -> samples in a row seen
    last_sound = [0.0, 0.0]  # (time, loudness) of the last sound surprise
    last_motion = [0.0]
    was_driving = [False]

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
            lifted = surprise.latch_events(latches, new_latches)
            events += moved + lifted
            latches = new_latches
            how = "picked up" if any(k in ("picked_up", "put_down") for k, _ in lifted) else \
                "drove" if bool(s.get("driving")) != was_driving[0] else None
            was_driving[0] = bool(s.get("driving"))
            if how:  # the body moved: which way it faces is unknown until a look recognizes its map
                vision.lost_bearings()
                state.update_session({"moved_ts": time.time(), "moved_how": how})
            events += surprise.battery_events(battery[0], s.get("battery_pct"))
            battery[0] = s.get("battery_pct")
        events += _people_events(present, streaks)
        head_still = time.time() - ((s.get("head") or {}).get("moved_ts") or 0) > HEAD_SETTLE_S
        carried = (s.get("latches") or {}).get("cliff") or \
            time.time() - state.load_session().get("moved_ts", 0) < HEAD_SETTLE_S  # in the air, or just set down
        if head_still and not carried and not _recently_self_noisy() and not faces.read().get("faces") \
                and time.time() - last_motion[0] > 60:
            moving = surprise.motion_events(sensors.read_motion())  # a face in view is already an event
            if moving:
                last_motion[0] = time.time()
                events += moving
        if not _recently_self_noisy():
            heard = hearing.read() or {"levels": [], "sounds": []}
            levels = heard["levels"]
            for kind, text in surprise.loudness_events(levels):
                peak = max(levels[-2:])
                if time.time() - last_sound[0] < cfg.SOUND_COOLDOWN_S and peak < 2 * last_sound[1]:
                    continue  # another tap like the last one -- not news
                last_sound[:] = [time.time(), peak]
                name = heard["sounds"][len(levels) - 2 + levels[-2:].index(peak)]  # openbot-ears named that second
                events.append((kind, f"{text} -- it sounded like: {name}" if name and name != hearing.OWN_VOICE
                               else text))
        return events

    return sense


STRANGER = "__stranger__"
# Hysteresis both ways -- recognition in dim light flickers: a face recognized in
# ONE sample was announced "is here" and, 20 s later, "left" (every 1-3 min all day).
ARRIVE_SAMPLES = 2    # seen this many samples in a row = arrived
PERSON_GONE_S = 90.0  # out of view this long = left (eyes on the laptop for a minute isn't leaving)


_last_seen_saved = [0.0]


def _people_events(present: dict[str, float], streaks: dict[str, int]) -> list[Event]:
    """Arrivals and departures of recognized people, and of "a stranger" (one
    per visit): ARRIVE_SAMPLES in a row to arrive, PERSON_GONE_S unseen to leave."""
    data = faces.read()
    if not data:
        return []  # camera/detector down -- that's not everyone leaving
    now, events = time.time(), []
    names, strangers = faces.visible_names(data)
    seen_now = names + [STRANGER] * bool(strangers)
    for who in list(streaks):
        if who not in seen_now:
            del streaks[who]
    for who in seen_now:
        streaks[who] = streaks.get(who, 0) + 1
        if who not in present and streaks[who] >= ARRIVE_SAMPLES:
            events.append(("stranger", "someone you don't recognize is in front of you") if who == STRANGER
                          else ("person_arrived", f"{who} is here -- you recognize their face"))
        if who in present or streaks[who] >= ARRIVE_SAMPLES:
            present[who] = now
    if names and now - _last_seen_saved[0] > 60:  # survives a mind restart mid-visit (no false "back after hours")
        _last_seen_saved[0] = now
        last_seen = state.load_session().get("last_seen", {})
        state.update_session({"last_seen": {**last_seen, **{n: now for n in names}}})
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
    the policy: never asleep."""
    session = state.load_session()
    if state.conversation_active(session):
        return events  # already talking with someone
    rest = []
    for kind, text in events:
        name = text.split(" is here")[0] if kind == "person_arrived" else None
        why = greeting_reason(session.get("last_seen", {}).get(name), session.get("greeted", {}).get(name),
                              datetime.datetime.now()) if name else None
        allowed = why and policy.evaluate("audio").allowed
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
        lost = vision.last_look().get("lost")
        if lost:  # just moved: look again soon (the head glances about) -- that's how it finds its bearings
            next_look = min(next_look, now + (3.0 if not lost.get("tries") else 20.0))
        if now >= next_look and not in_session:
            seen = vision.look(cfg.LLM_BASE_URL, cfg.LLM_MODEL)
            if seen and seen.get("bearings"):
                journal.log("found", seen["bearings"])
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

        events = _react_to_touch(persona, events) if events else events  # a hand in front: now, talking or not
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
        tools.refresh_weather()  # every WEATHER_EVERY_S; every prompt reads the cached line
        time.sleep(0.5)


if __name__ == "__main__":
    main()
