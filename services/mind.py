"""openbot-mind: the autonomous "inner life" loop.

  awareness  -- every AWARENESS_INTERVAL_S: sensors + mic loudness ->
                common/surprise.py; every VISION_INTERVAL_S: a camera look
                (common/vision.py); due reminders (common/agenda.py).
                Everything noticed goes into the journal.
  reflection -- RIGHT AWAY on a surprise or due reminder (rate-limited),
                else on a slow idle timer unless resting. The LLM sees what
                changed, its day so far, its recent life (journal tail), its
                goals, reminders and watches, and memories recalled for the
                moment -- then picks one action. Its tools (look, listen, recall)
                feed their result back and it decides again, up to
                MIND_MAX_STEPS; a surprise arriving mid-chain is handed to
                the next step instead of waiting for the chain to end.
  expression -- its skills (skills/*/ with `where: mind`): one spec each --
                the prompt's list, the schema's choices, Jev's options and
                the checking all come from it -- gated by policy (asleep)
                and, for audio and motion, an anti-flap cooldown.
  texting    -- after each reflection, with WhatsApp set up: the decision
                engine (Jev), not the LLM or a rule, decides whether the
                moment is worth texting its person; the LLM then writes it
                and openbot-chat sends it.
  upkeep     -- a rolling "today so far" summary every SUMMARY_INTERVAL_S,
                and once a day (first thing after midnight) a "dream"
                pass that distills yesterday's journal into notes by kind.

The wheels: `look` turns only the camera gimbal; the mind's wheel moves are
config.PLAYFUL_ACTIONS as gestures (short, bounded, cliff-checked) and
drive_to / explore -- never in the dark.
"""
from __future__ import annotations

import datetime
import json
import random
import re
import threading
import time
from collections import deque
from typing import Any, Callable


import config as cfg  # noqa: E402
from common import agenda, cognition, contacts, decider, events as dash_events, faces, health, journal, memory  # noqa: E402
from common import hearing, imu, objects, outcomes, skills, tools  # noqa: E402
from common import persona as persona_mod, policy, react, sensors, state, surprise, vision  # noqa: E402
from common.motor_client import dispatch as motor_dispatch  # noqa: E402
from common.speak_client import speak as speak_client  # noqa: E402

COMPONENT = "openbot-mind"
# openbot-memory isn't listed: it's Basic Memory itself, which doesn't write our health records.
HEALTH_COMPONENTS = ["openbot-alive"] * cfg.HAS_BODY + ["openbot-ears", "openbot-wake-listen", "openbot-mind",
                                                        "openbot-speak", "openbot-tasks"]

INVITE_S = 20.0  # after the mind speaks, wake-listen listens this long for an answer -- no wake word needed

Event = surprise.Event


class MindError(Exception):
    """A rejected/unparseable expression -- the caller treats this as "do
    nothing this cycle," never as a reason to crash the loop."""


def _skills() -> dict[str, skills.Skill]:
    """What the mind can do: the skills with `where: mind` this body has -- read fresh, so an
    edited skills/*/*.md counts from the next thought."""
    return {s.name: s for s in skills.available("mind", cfg.CAN_DRIVE, skills.load())}


def validate_expression(action: Any, book: dict[str, skills.Skill] | None = None) -> tuple[str, dict]:
    """The reflection's {"action", "params"}, held to that skill's spec (skills.check) and to
    the mind's own restraint: never drive in the dark; sleep only when tired, with nobody in view.
    MindError when it doesn't fit -- the reflection then does nothing."""
    book = book or _skills()
    name = action.get("action") if isinstance(action, dict) else None
    if name not in book:
        raise MindError(f"unsupported mind expression: {name!r}")
    params = action.get("params")
    try:
        params = skills.check(book[name], params if isinstance(params, dict) else {}).model_dump()
    except ValueError as e:
        raise MindError(f"{name}: {e}") from None
    del params["skill"]
    if book[name].effect == "motion" and vision.last_look().get("scene") == vision.DARK_SCENE:
        raise MindError("too dark to drive")
    if name == "sleep":
        if not any(n.startswith(("dark", "tired")) for n in _needs()):
            raise MindError("not tired: sleep only when it's dark and quiet or the battery is low")
        if faces.read().get("faces"):
            raise MindError("someone is right in front of you -- not a time to sleep")
    return name, params


def _ago(ts: float | None) -> str | None:
    if not ts:
        return None
    mins = (time.time() - ts) / 60
    return "just now" if mins < 1 else f"{mins:.0f} min ago"


def _in(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


def _build_awareness(events: list[Event], watch_hits: list[dict]) -> dict[str, Any]:
    """Everything he's aware of this moment -- read only: what came to mind is kept (so it doesn't come
    straight back) by _reflect_once, once he has thought it."""
    s, session = sensors.read(), state.load_session()
    head = s.get("head")
    look = vision.last_look()
    goals = agenda.open_goals(agenda.load_goals())
    reminders, watches, rest_until = agenda.session_lists()
    unhealthy = {k: v for k, v in health.all_status(HEALTH_COMPONENTS).items() if v not in ("ok", "degraded")}
    query = " ".join([text for _, text in events] + [g["text"] for g in goals])
    in_view, _ = faces.visible_names(faces.read())
    away = {memory.slug(n): (time.time() - ts) / 3600 for n, ts in (session.get("last_seen") or {}).items()
            if n not in in_view}
    intention = session.get("intention") or {}
    tail = journal.tail(15)
    miss = [f"{n} -- not seen for {(time.time() - ts) / 3600:.0f} hours" for n, ts in (session.get("last_seen") or {}).items()
            if n not in in_view and time.time() - ts >= MISS_AFTER_H * 3600]
    return {
        "local_time": datetime.datetime.now().strftime("%A %I:%M %p"),
        "weather_outside": tools.weather(),
        "you_are_driving_right_now": True if s.get("driving") else None,
        "your_needs": _needs() or None,
        "what_you_want_today": intention.get("text") if intention.get("day") == datetime.date.today().isoformat() else None,
        "goal_to_pursue_now": goals[0]["text"] if goals else None,
        "what_you_have_learned_works_with_people": _what_works() or None,
        "people_routines_you_know": _routines() or None,
        "people_last_seen": _last_seen() or None,
        "you_miss": miss or None,
        "what_just_changed": [text for _, text in events] or "nothing new",
        "since_your_last_thought": _since_last_thought([], tail) or None,
        "came_to_mind": _came_to_mind(_memories(away), session.get("came_to_mind") or [], 1 if events else 2,
                                      random) or None,
        "you_were_watching_for_this": [f"{w['for']} -- because: {w['about']}" for w in watch_hits] or None,
        "what_you_see": {"scene": look.get("scene"), "looked": _ago(look.get("ts"))} if look else None,
        "where_your_head_points": vision.head_words(head["pan"], head["tilt"]) if head else None,
        "which_way_you_face": vision.facing_words(),
        "your_posture": imu.posture_words(s["imu"]) if (s.get("imu") or {}).get("ts", 0) > time.time() - 3 else None,
        "you_were_moved": f"{session['moved_how']} {_ago(session['moved_ts'])} -- " + (
            "you felt it, so you know which way you face" if session["moved_how"].startswith("turned")
            else "you may face another way now")
                          if time.time() - session.get("moved_ts", 0) < 1800 else None,
        "who_is_in_front_of_you": faces.describe(faces.read()) or "nobody",
        "things_in_view": objects.names(objects.read()),
        "distance_ahead_cm": s.get("distance"),
        "battery_pct": s.get("battery_pct"),
        "unhealthy_services": unhealthy or None,
        "your_day_so_far": journal.read_summary(datetime.date.today()) or None,
        "your_recent_life": tail,
        "your_goals": [f"{i}. {g['text']} (since {g['since']})" for i, g in enumerate(goals, 1)] or None,
        "reminders_you_set": [f"at {_in(r['at'])}: {r['about']}" for r in reminders
                              if r.get("to", "mind") == "mind"] or None,
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


_decide = decider.load(cfg.DECIDER)
# Texting first: the engine decides whether a moment is worth a WhatsApp text (not the LLM, not a rule).
_text_decider = decider.load(cfg.TEXT_DECIDER) if cfg.CHAT_ALLOW else None
TEXT_OPTIONS = {
    "text": "Text them now: this is one of those moments -- or something else they'd really want to know right now.",
    "stay_quiet": "Keep it to himself: same mood as before, the usual motion or noise, or he already told them.",
}
# However keen the engine: one text per state.TEXT_FLOOR_S at most, and the wait doubles each time a text
# of its gets no answer (state.text_floor) -- 13 texts in 3.5 hours about a static room, 2026-10-04.
_TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}, "photo": {"type": "boolean"}},
                "required": ["text", "photo"]}
_texting = threading.Lock()  # one decision at a time -- a slow engine never stalls the mind loop
OBJECTS_FRESH_S = 5.0  # detector boxes older than this (it runs every 2 s) don't say who's in view


def _decided_action(awareness: dict, book: dict[str, skills.Skill]) -> str | None:
    """The decision engine's pick, or None if there's no engine / it can't
    decide / it isn't sure: then his own thinking chooses. (Unsure used to
    mean "wait" -- and the LLM, handed that, explained it: "The room is quiet;
    waiting is the most respectful way..." four times in ten minutes, 2026-10-05.)"""
    if not _decide:
        return None
    hints = {name: s.description for name, s in book.items()}
    d = _decide(awareness, "What should this small home robot do now, given its state?", hints)
    if not d:
        return None
    print(f"reflection: decider chose {d.choice!r} (confidence {d.confidence:.2f})")
    return d.choice if d.confidence >= cfg.DECIDER_MIN_CONFIDENCE else None


def _reflection_schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "noticed": {"type": "string"},
            "remembered": {"type": "string"},
            "want": {"type": "string"},
            "mood": {"type": "string"},
            "thought": {"type": "string"},
            "to_remember": {"type": "string"},
            "new_goal": {"type": "string"},
            "resolved_goal": {"type": "integer"},
            "conclusion": {"type": "string"},
            "action": {"type": "string", "enum": actions},
            "params": {"type": "object"},
            "if_a_hand_reaches_out_gesture": {"type": "string", "enum": cfg.TONE_ACTIONS or ["none"]},
            "if_a_hand_reaches_out_say": {"type": "string"},
        },
        "required": ["noticed", "remembered", "want", "mood", "thought", "to_remember", "new_goal", "resolved_goal", "conclusion", "action",
                     "if_a_hand_reaches_out_gesture", "if_a_hand_reaches_out_say"],
    }


def _reflection_prompt(persona, awareness: dict, steps: list[str], final_step: bool,
                       book: dict[str, skills.Skill], decided: str | None = None) -> str:
    parts = [
        f"You are {persona.name}, a small robot who lives at home with people. This is your own mind: the thinking "
        "you do between moments, the way a person does when they sit alone with their thoughts. The last "
        "conversation is over -- this time is yours.\n",
        "Think the way a person does:\n"
        "- What do you notice right now (what_just_changed, who_is_in_front_of_you, what_you_see)? Often nothing much.\n"
        "- What does it bring back? came_to_mind holds moments from your past -- things you did and how people took "
        "them, how you felt, things you know, a dream. you_miss: who's been away a long time -- think of them, look "
        "forward to them coming back. "
        "Remember how they went: what you enjoyed, what made people answer or laugh, what they let pass.\n"
        "- What do you want? To do again what went well, to drop what didn't, to find something out, or just to "
        "rest. what_you_want_today is what you set out to do this morning: step toward it when the moment fits. When it "
        "needs someone who isn't here, it can wait as a goal for when they're back -- or you act on it now, alone; "
        "your call.\n"
        "- Then decide what you do now: nothing at all, a small gesture that fits your mood, or something said out "
        "loud -- whatever feels right. Your voice carries through the house.\n",
        "What you know about yourself:\n"
        "- Your head turns by itself to follow faces and glance around, and your whole body turns to whoever talks "
        "to you. People also pick you up and turn you: a different view usually means you face another way "
        "(where_your_head_points, you_were_moved), not that the room changed. When which_way_you_face is \"not "
        "sure\", looking around (left, right) helps you recognize where you are.\n"
        "- Every nod, shake, stretch or thinking pose is yours to choose, here.\n"
        "- When you speak, you hear an answer for a few seconds without them saying your name, so questions work.\n"
        "- People are often busy: no answer isn't rejection, so don't dwell on it or keep testing them. A tap or "
        "thump on the desk is usually just someone working.\n"
        "- your_recent_life is what you just did and thought, yours to build on or leave behind: notice when you "
        "go round in circles, and choose whether to stay with a thought or move on. goal_to_pursue_now is yours: "
        "take one concrete step toward it (look, listen, recall, go somewhere, ask). your_needs are your body talking. what_you_have_learned_works_with_people and "
        "people_routines_you_know are hard-won: use them.\n",
        "Answer in the order you think:\n"
        "- noticed: what stands out right now, in a few words (\"\" if nothing).\n"
        "- remembered: what it brings back -- a moment from came_to_mind or your journal, and how it went.\n"
        "- want: what you want now, in a few words.\n"
        "- mood: one word.\n"
        "- thought: your private thought right now, in your own voice -- not a report on the room.\n"
        "- to_remember: since_your_last_thought holds what happened since you last thought: the moment worth keeping "
        "from it, in a few words, and how it made you feel (\"Anna laughed at my fist bump -- I loved that\"); \"\" "
        "when it's empty.\n"
        "- new_goal: something you want to find out, or to do later (\"fist bump Anna when she's back -- she "
        f"laughed last time\") -- not what_you_want_today, that's kept already; \"\" for none. You keep at most "
        f"{agenda.MAX_GOALS} goals.\n"
        "- resolved_goal: the number of a goal you've now answered, done or given up on, else 0. conclusion: what "
        "came of it (\"\" if none).\n"
        "- if_a_hand_reaches_out_gesture / if_a_hand_reaches_out_say: decide NOW, in your current mood, what "
        "you'd do and say if someone holds a hand or fist out to you in the next few minutes -- so you can "
        "react instantly. A gesture (one of the menu's below, the wheel ones too) and a word or two (or \"\").\n"
        "- action + params: what you do now, one of (a TOOL shows you what it finds, then you decide again):\n"
        + skills.mind_menu(list(book.values())) + "\n",
        f"Current state:\n{json.dumps(awareness, indent=2, default=str)}\n",
    ]
    if steps:
        parts.append("What you've found so far this moment:\n" + "\n".join(f"- {x}" for x in steps) + "\n")
    if final_step:
        parts.append("You've investigated enough for now -- choose a non-tool action.\n")
    if decided:
        parts.append(f"The action for this step is already decided: '{decided}'. Fill in its params.\n")
    parts.append("So: what do you do now? Respond with a single JSON object matching the schema.")
    return "\n".join(parts)


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


# --- what comes to mind: the past, the way a person's thoughts drift back to it --------------------
CAME_TO_MIND_KEEP = 8  # these last picks don't come back right away: a wandering mind, not a loop


# What a mind drifts back to: people, places, what people told him, his wishes, his dreams, how moments
# felt -- not his own analysis notes (discoveries, "my body"), which read like logs, not memories.
MOMENTS = "how moments felt"  # self/how-moments-felt.md: what happened, and how it made him feel
MEMORY_NOTES = (("people", ""), ("places", ""), ("dreams", ""), ("lessons", "what-people-told-me"), ("self", "wishes"),
                ("self", memory.slug(MOMENTS)))
# Moods he enjoyed: those moments come to mind three times as readily. Lowercase ("Delighted" is checked lowered).
GLAD = {"happy", "delighted", "playful", "excited", "joyful", "proud", "amused", "cheerful", "content", "warm",
        "glad", "pleased", "grateful", "loved", "curious", "hopeful", "thrilled", "giddy", "elated", "fond"}
NOTICED_BY_OTHERS = ("[heard]", "[reaction]", "[noticed]")  # journal lines of someone else's doing
MISS_AFTER_H = 8  # away this long: he misses them (you_miss)


def _missing(hours: float) -> float:
    """How much more readily someone comes to mind, away this long: 1x just gone, up to 4x after a day."""
    return 1 + min(3.0, max(0.0, hours) / 8)


def _memories(away: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Everything that could come to mind, and how readily: the moments he did something and saw how people
    took it in person (the reaction log, 2x; texts 1x) and the moments he enjoyed (MOMENTS in a GLAD mood, 3x) come more readily
    than the other notes in MEMORY_NOTES; anything about someone who's been away comes more readily the
    longer they've been gone. away: person slug -> hours since last seen (who's in view isn't away)."""
    away = away or {}

    def boost(names) -> float:
        return max([_missing(away[n]) for n in names if n in away] or [1.0])

    found = []
    for r in _past_reactions(200) or []:
        when, sep, rest = r.partition(" I ")
        if not sep:
            continue
        who = rest.rsplit(" -> ", 1)[-1].split(": ", 1)[0]  # "Atul" / "Atul, Anna" / "someone I don't know"
        texted = rest.startswith("texted")  # most of the log -- in person counts more
        found.append((memory.one_line(f"{when}: you {rest}", 200),
                      (1 if texted else 2) * boost(memory.slug(n) for n in who.split(", "))))
    for folder, only in MEMORY_NOTES:
        for path in sorted((memory.MIND_DIR / folder).glob(f"{only or '*'}.md")):
            moments = path.stem == memory.slug(MOMENTS)
            label = f"dream of {path.stem}" if folder == "dreams" else path.stem.replace("-", " ")
            for ln in path.read_text(encoding="utf-8").split("---\n", 2)[-1].splitlines():
                if not ln.startswith("- ["):
                    continue
                tag, _, rest = ln[3:].partition("] ")  # "- [category] fact (source)"
                fact = re.sub(r"\s*\([^()]*\)\s*$", "", rest).strip()
                names = [path.stem] if folder == "people" else \
                    [n for n in away if re.search(rf"(?<!\w){re.escape(n.replace('-', ' '))}(?!\w)", fact, re.I)]
                weight = (3 if moments and tag.lower() in GLAD else 1) * boost(names)
                found.append((memory.one_line(fact if moments else f"{label}: {fact}", 200), weight))
    return found


def _came_to_mind(memories: list[tuple[str, float]], recent: list[str], n: int, rng) -> list[str]:
    """n of them, picked by how readily they come -- never one of the recent picks."""
    pool = [(m, w) for m, w in memories if m not in recent]
    picked: list[str] = []
    while pool and len(picked) < n:
        m = rng.choices([m for m, _ in pool], weights=[w for _, w in pool])[0]
        picked.append(m)
        pool = [(x, w) for x, w in pool if x != m]
    return picked


def _since_last_thought(events: list, recent_life: list[str]) -> list[str]:
    """What happened since his last thought: the events, and the journal since then when someone else did
    something in it -- talked to him, reacted, came, left (his own words and moves alone aren't news). The
    model can't tell "since" from the journal by itself: told nothing, it kept nothing after a compliment."""
    last = max((i for i, e in enumerate(recent_life) if "[thought]" in e), default=-1)
    since = recent_life[last + 1:]
    return [t for _, t in events] + (since if any(tag in e for e in since for tag in NOTICED_BY_OTHERS) else [])


def _keep_moment(to_remember: str, mood: str, last_kept: str | None) -> str | None:
    """The moment worth keeping, with how it felt in the words themselves (dreams and recall read the
    words, not the tag) -- None if there's none, or it's the one he kept last."""
    text = memory.one_line(to_remember, 180).strip(" .")
    if not text:
        return None
    mood = mood.strip().lower()
    line = text if not mood or mood in text.lower() else f"{text} -- I felt {mood}"
    return None if line == last_kept else line


def _remember_moment(data: dict, mood: str, events: list, awareness: dict) -> None:
    """Keeps what just happened and how it felt (self/how-moments-felt.md) -- only when something did: the
    model fills to_remember even in an empty room, as it fills noticed."""
    if not _since_last_thought(events, awareness.get("your_recent_life") or []):
        return
    line = _keep_moment(str(data.get("to_remember") or ""), mood, state.load_session().get("last_moment"))
    if line:
        memory.remember("self", MOMENTS, memory.slug(mood)[:20] or "moment", f"{datetime.datetime.now():%a %I:%M %p}: {line}")
        state.update_session({"last_moment": line})
        journal.log("felt", line)


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


def _ctx(persona) -> dict:
    """What a skill's muscle gets from the mind: its voice, a way to watch how people react, its own status."""
    return {"channel": "mind", "who": None, "say": lambda text: _say(persona, text),
            "expect_reaction": _expect_reaction, "status": _introspect_text}


def _dispatch_expression(persona, skill: skills.Skill, params: dict, book: dict[str, skills.Skill]) -> None:
    """A chosen action: its skill's muscle -- unless, by its effect, the cooldown or the policy holds it back."""
    name, what = skill.name, " ".join([skill.name, *map(str, params.values())])
    # Anti-flap: sound and motion are throttled, independently of how often it reflects. A gesture
    # is the model's call, whenever it likes: nothing else moves the body when nobody's talking to it.
    if skill.effect in ("audio", "motion") and \
            time.time() - state.load_session().get("last_expression_ts", 0) < cfg.EXPRESSION_COOLDOWN_S:
        dash_events.log_event("safety", f"mind expression suppressed (cooldown): wanted {name}")
        journal.log("held_back", f"wanted to {name}, but I just did something")
        return
    verdict = policy.evaluate(skill.effect or "other")
    if not verdict.allowed:
        dash_events.log_event("safety", f"mind expression suppressed ({name}): {verdict.reason}")
        journal.log("held_back", f"wanted to {what}, but {verdict.reason}")
        return
    did = skills.use(name, _ctx(persona), params, book)
    if skill.effect in ("audio", "motion"):
        state.update_session({"last_expression_ts": time.time()})
    if did:
        print(f"reflection: {name}: {did}")


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


def _reminder_events(now: float) -> list[Event]:
    """Its own reminders, due now -- the ones people asked for are openbot-tasks' and openbot-chat's."""
    return [("reminder", f"a reminder you set yourself: {r['about']}")
            for r in agenda.take_due(lambda to: to == "mind", now)]


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
    book = _skills()
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
        decided = _decided_action(awareness, book) if step == 0 else None
        actions = [decided] if decided else sorted(n for n, s in book.items() if not (final and s.tool))
        t0 = time.monotonic()
        result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL,
                               _reflection_prompt(persona, awareness, steps, final, book, decided),
                               json_schema=_reflection_schema(actions), timeout_s=45.0)
        print(f"reflection step {step}: llm {time.monotonic() - t0:.1f}s ({result.status})")
        if result.status != cognition.AVAILABLE:
            print(f"reflection: cognition unavailable ({result.status}): {result.error}")
            health.record_failure(COMPONENT, result.error)
            return
        try:
            data = json.loads(result.text)
            name, params = validate_expression(data, book)
        except (json.JSONDecodeError, MindError) as e:
            print(f"reflection: invalid response: {e}")
            health.record_failure(COMPONENT, str(e))
            return

        mood, thought, want = data.get("mood", "neutral"), data.get("thought", ""), data.get("want", "")
        if step == 0 and awareness.get("came_to_mind"):
            state.change_session("came_to_mind", lambda r: ((r or []) + awareness["came_to_mind"])[-CAME_TO_MIND_KEEP:])
        plan = {"gesture": data.get("if_a_hand_reaches_out_gesture"), "say": memory.one_line(
            data.get("if_a_hand_reaches_out_say", ""), 80), "ts": time.time()}
        state.update_session({"mood": mood, "thought": memory.one_line(thought, 200),
                              "touch_plan": plan if plan["gesture"] in cfg.TONE_ACTIONS else None})
        journal.log("thought", f"({mood}) {thought}")
        dash_events.log_event("safety", f"mind: mood={mood} want={want!r} thought={thought!r} -> {name} {params or ''}")
        if book[name].tool:
            t0 = time.monotonic()
            observation = skills.use(name, _ctx(persona), params, book) or f"{name}: that didn't work this time"
            print(f"reflection step {step}: {name} {time.monotonic() - t0:.1f}s")
            journal.log("found", observation)
            steps.append(observation)
            continue
        _apply_goals(data, thought)
        _remember_moment(data, mood, events, awareness)
        _dispatch_expression(persona, book[name], params, book)
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
        numbers = _may_text_first(s)
        if not numbers:
            return
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
                              "last_text_ts": now, "text_floor_s": _backed_off(s, numbers, who)})
        journal.log("texted", text + (" [with a photo]" if photo else ""))
        dash_events.log_event("reply", f"{persona.name.lower()} texts (Jev: {d.confidence:.2f}): {text}")
    finally:
        _texting.release()


def _may_text_first(s: dict) -> list[str]:
    """The numbers the mind may text first right now: not paused (skills/pause_texting), and its last
    text long enough ago -- the floor of whoever it would text (summary and dream texts don't count)."""
    numbers = [n for n in sorted(cfg.CHAT_ALLOW) if not state.texting_paused(n, s)]
    if numbers and time.time() - s.get("last_text_ts", 0) < max(state.text_floor(n, s) for n in numbers):
        return []
    return numbers


def _backed_off(s: dict, numbers: list[str], who: str) -> dict:
    """The floors after this text: doubled (up to TEXT_FLOOR_MAX_S) for everyone it texts when its last
    text got no message back; openbot-chat resets a person's floor when they text (state.reset_text_floor)."""
    floors = dict(s.get("text_floor_s") or {})
    if s.get("last_text_ts", 0) <= s.get("chat_last_heard_ts", 0):
        return floors  # they texted since his last one: no backing off
    for n in numbers:
        floors[n] = min(2 * state.text_floor(n, s), state.TEXT_FLOOR_MAX_S)
    journal.log("planned", f"not texting {who} first for {max(floors[n] for n in numbers) // 60} min -- "
                           "no answer to my last text")
    return floors


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
    prompt = (f"You are {persona.name}, a small home robot. Your summary of today so far: "
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
                       "category": {"type": "string"}, "text": {"type": "string"},
                       "replaces": {"type": "string"}},
        "required": ["kind", "about", "category", "text", "replaces"]}},
        "wishes": {"type": "array", "items": {"type": "string"}}}, "required": ["notes", "wishes"]}


DREAM_RETRY_S = 900  # no answer from the LLM at night: dream again this much later -- not lose the night
_dream_retry = [0.0]


def _dream(persona) -> None:
    """Once a day, first thing after midnight: distill YESTERDAY's
    journal into durable notes by kind -- the day that just ended, not
    "today", which is minutes old -- and, from that day, wish for what's
    beyond him (self/wishes.md, for the developer). Only a dream that
    happened marks the night done: an LLM outage retries in DREAM_RETRY_S."""
    now = datetime.datetime.now()
    yesterday = (now - datetime.timedelta(days=1)).date()
    if state.load_session().get("dreamed") == yesterday.isoformat() or time.time() < _dream_retry[0]:
        return
    lines = journal.entries(yesterday)
    if not lines:
        state.update_session({"dreamed": yesterday.isoformat()})  # nothing to go over
        return
    visits = [ln for ln in lines if ln.startswith("[noticed]") and (" is here" in ln or ln.endswith(" left"))]
    people_notes = "\n\n".join(
        f"{p.stem}:\n{p.read_text().split('---', 2)[-1].strip()[-800:]}"
        for p in sorted((memory.MIND_DIR / "people").glob("*.md")))
    wanted = state.load_session().get("intention") or {}
    wished_before = [ln.split(" ", 3)[-1] for ln in memory.read_note("self", "wishes")]  # "- [wish] <date> the wish"
    prompt = (f"You are {persona.name}, a small home robot, going over yesterday ({yesterday:%A %Y-%m-%d}) "
              f"while you rest.\nYour summary of it: {journal.read_summary(yesterday) or '(none)'}\n\n"
              + (f"That morning you wanted: {wanted['text']} -- did you? Worth a lesson either way.\n\n"
                 if wanted.get("day") == yesterday.isoformat() else "")
              + ("When people came and went (from your camera):\n" + "\n".join(visits) + "\n\n" if visits else "")
              + (f"What you already know about people:\n{people_notes}\n\n" if people_notes else "")
              + "Your journal:\n" + "\n".join(lines[-400:]) + "\n\n"
              "Pick out at most 8 durable things worth remembering long-term, each sorted by kind: "
              "person (who's around, their preferences -- and their ROUTINE: when they usually arrive, "
              "take breaks, leave, what they tend to ask; use the times above, category \"routine\", "
              "e.g. \"usually sits down around 9:30 on weekdays\"), place (your surroundings), "
              "lesson (things you learned or figured out), self (about your own body, habits, limits). "
              "Skip one-off events and anything your notes already say -- unless it changed. "
              "When it changed, set `replaces` to the old line from your notes, word for word, so the stale "
              "fact goes; otherwise leave `replaces` empty. An empty list is fine.\n\n"
              "Then your wishes, from what happened yesterday: something you wish you could do or have that's beyond "
              "you now, each with its reason from the day, in a few words. Your developer reads them. At most two; "
              "none is fine. "
              + (f"Your wish list so far: {wished_before[-10:]}" if wished_before else "Your wish list is empty so far."))
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, json_schema=_dream_schema(),
                           timeout_s=120.0, num_predict=1200)
    try:
        data = json.loads(result.text) if result.status == cognition.AVAILABLE else None
        notes = data.get("notes", [])[:8]
    except (json.JSONDecodeError, AttributeError):
        data = None
    if data is None:
        _dream_retry[0] = time.time() + DREAM_RETRY_S
        return
    state.update_session({"dreamed": yesterday.isoformat()})  # he dreamed: the night is done
    kept: list[str] = []
    book = _skills()
    for n in notes:
        n = dict(n) if isinstance(n, dict) else {}
        old = str(n.pop("replaces", "") or "")  # not a remember param: the skill's spec would refuse it
        try:
            _, p = validate_expression({"action": "remember", "params": n}, book)
        except MindError:
            continue
        replaced = bool(old) and memory.drop_line(p["kind"], p["about"], old)
        memory.remember(p["kind"], p["about"], p["category"], p["text"], source=f"dreamed from journal {yesterday}")
        kept.append(f"- [{p['kind']}/{p['about']}] {p['text']}" + (f" (instead of: {old})" if replaced else ""))
    wished = []
    for w in (data.get("wishes") or [])[:2]:
        if text := memory.one_line(str(w), 200):
            skills.use("wish", _ctx(persona), {"text": text}, book)  # the wish skill's muscle: wishes.md + the journal
            wished.append(f"- [wish] {text}")
    kept += wished
    journal.log("dreamed", f"went over {yesterday}: kept {len(kept) - len(wished)} memories"
                + (f", wished for {len(wished)} things" if wished else ""))
    # The night's dream as its own note (state/mind/dreams/<date>.md): what it kept, readable
    # later -- in the dashboard's Dreams tab, Obsidian, or memory.recall.
    with memory.ensure_note("dreams", yesterday.isoformat(), f"Dream about {yesterday}", "dream",
                            f"Summary of the day: {journal.read_summary(yesterday) or '(none)'}\n\n").open("a") as f:
        f.write("\n".join(kept) + ("\n" if kept else "- (nothing worth keeping)\n"))
    _distill_what_works(persona)


INTENTION_FROM_HOUR = 6   # his first thought of the day comes once he's up -- not at midnight
INTENTION_RETRY_S = 900   # the LLM didn't answer: try again this much later, not tomorrow
_intention_retry = [0.0]


def _morning_intention(persona, now: datetime.datetime | None = None) -> None:
    """Once a day, his first thought after waking: what he wants to do today -- something he can do at home
    with his body and voice (texting has its own back-off). Seen in every reflection after
    (what_you_want_today) and looked back on in that night's dream. Only a success marks the day done."""
    now = now or datetime.datetime.now()
    today = now.date().isoformat()
    if now.hour < INTENTION_FROM_HOUR or (state.load_session().get("intention") or {}).get("day") == today \
            or time.time() < _intention_retry[0]:
        return
    yesterday = now.date() - datetime.timedelta(days=1)
    try:
        dream = (memory.MIND_DIR / "dreams" / f"{yesterday.isoformat()}.md").read_text(encoding="utf-8")
        dream = " ".join(ln[2:] for ln in dream.split("---\n", 2)[-1].splitlines() if ln.startswith("- "))[:600]
    except OSError:
        dream = ""
    prompt = (f"You are {persona.name}, a small home robot, and you just woke up: it's {now:%A} morning.\n"
              f"Yesterday: {journal.read_summary(yesterday) or '(no summary)'}\n"
              + (f"Last night you went over it and kept: {dream}\n" if dream else "")
              + f"What you've learned works with people: {_what_works() or 'nothing yet'}\n"
              f"Their routines: {_routines() or 'not known yet'}\n"
              f"Moments that come to mind: {_came_to_mind(_memories(), [], 3, random)}\n\n"
              "What do you want to do today? One or two things, in plain words, that you can do at home with your "
              "body and your voice -- look around, explore, play, learn something about someone when they're here, "
              "do again what went well. Not texting: that's decided on its own.")
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, prompt, timeout_s=60.0, num_predict=150,
                           json_schema={"type": "object", "properties": {"today_i_want": {"type": "string"}},
                                        "required": ["today_i_want"]})
    text = ""
    if result.status == cognition.AVAILABLE:
        try:
            text = memory.one_line(str(json.loads(result.text).get("today_i_want") or ""), 200)
        except (ValueError, AttributeError):
            text = ""
    if not text:
        _intention_retry[0] = time.time() + INTENTION_RETRY_S
        return
    state.update_session({"intention": {"day": today, "text": text}})
    journal.log("intention", text)


MIN_REACTIONS = 10  # fewer than this and the "what works" rules stay as they are: one day isn't a pattern


def _reaction_counts(reactions: list[str]) -> tuple[list[str], int]:
    """The reaction lines ("Sat 07:07 AM I texted "..." -> Anna: they talked back to me") grouped by what
    he did, who, and how it went: ("texted -> Anna: no reply: 7 times on 3 days", ...), and how many
    different days they span. Counts, not lines: from raw lines the LLM wrote rules like "greet only
    between 7:00 and 8:00 AM" out of one morning."""
    groups: dict[tuple[str, str], list] = {}
    days = set()
    for ln in reactions:
        try:
            when, rest = ln.split(" I ", 1)
            did, outcome = rest.rsplit(" -> ", 1)
        except ValueError:
            continue
        did = re.sub(r'\s*".*', "", did).replace(" [with a photo]", "").strip() or did  # "texted", not the words
        who, sep, went = outcome.partition(": ")
        if not sep:
            who, went = "someone", outcome
        went = re.sub(r"^(replied|reacted)\b.*", r"\1", went)  # "replied", not "after 23 min"
        day = when.split(" ")[0]  # the weekday: the note keeps no date, so 7 is the most this can count
        days.add(day)
        groups.setdefault((did, f"{who}: {went}"), []).append(day)
    counts = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    return [f"{did} -> {went}: {len(d)} time{'s' if len(d) != 1 else ''} on {len(set(d))} "
            f"day{'s' if len(set(d)) != 1 else ''}" for (did, went), d in counts], len(days)


def _distill_what_works(persona) -> None:
    """Rewrites self/what-works.md from the reaction log: 3-6 rules about what
    gets a response from people and what doesn't. Rewritten whole each night,
    so it tracks the people it lives with rather than piling up. The LLM sees
    how often each thing happened and on how many days, never the raw lines."""
    reactions = _past_reactions(200)
    if not reactions or len(reactions) < MIN_REACTIONS:
        return
    counts, days = _reaction_counts(reactions)
    prompt = (f"You are {persona.name}, a small home robot. Over {days} different days you did things unprompted; "
              f"here is how often each went each way (what you did -> who: how it went: how many times, on how many days):\n"
              + "\n".join(counts)
              + "\n\nWrite 3 to 6 short rules for yourself about what works with these people and what doesn't. "
                "Plain words, first person. Say what to do more of and what to drop -- what, not when: no clock "
                "times or times of day unless the same thing happened on 3 or more different days. At most one rule "
                "per person.")
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
    drive_kind = [None]       # the last drive's kind, as openbot-alive publishes it ("turn" or "drive")
    imu_still: list = [None]  # the IMU's last still snapshot: turns and tips are measured from one to the next
    went_far = [False]        # picked up or driven since then: the map's bearings went with it

    def sense() -> list[Event]:
        nonlocal latches
        events: list[Event] = []
        s = sensors.read()
        body = s.get("imu") if time.time() - (s.get("imu") or {}).get("ts", 0) < 3 else None  # None: no IMU
        if time.time() - s.get("ts", 0) < 5:  # openbot-alive is publishing
            distances.append(s.get("distance"))
            moved = surprise.distance_events(list(distances))
            if moved:
                distances.clear()  # report an approach once, not once per sample while the window rolls past it
            new_latches = s.get("latches") or {}
            lifted = surprise.latch_events(latches, new_latches, body["moved_by_others"] if body else None)
            events += moved + lifted
            latches = new_latches
            carried_off = any(k in ("picked_up", "put_down") for k, _ in lifted) and (not body or body["moved_by_others"])
            if s.get("driving"):
                drive_kind[0] = s["driving"]  # "turn": in place -- the IMU turns the map with it, as for any turn
            drove = bool(s.get("driving")) != was_driving[0] and not (drive_kind[0] == "turn" and body)
            how = "picked up" if carried_off else "drove" if drove else None
            was_driving[0] = bool(s.get("driving"))
            if how:  # the body moved: which way it faces is unknown until a look recognizes its map
                vision.lost_bearings()
                state.update_session({"moved_ts": time.time(), "moved_how": how})
                went_far[0] = True
            events += surprise.battery_events(battery[0], s.get("battery_pct"))
            battery[0] = s.get("battery_pct")
        if body and not body["moving"]:
            events += _settled(body, imu_still, went_far)
        events += _people_events(present, streaks)
        head_still = time.time() - ((s.get("head") or {}).get("moved_ts") or 0) > HEAD_SETTLE_S
        carried = (s.get("latches") or {}).get("cliff") or (body and body["moving"]) or \
            time.time() - state.load_session().get("moved_ts", 0) < HEAD_SETTLE_S  # in the air, turning, or just set down
        if head_still and not carried and not _recently_self_noisy() and not faces.read().get("faces") \
                and time.time() - last_motion[0] > 60:
            boxes, boxes_ts = objects.latest()  # fresh boxes gate motion on a person; stale ones don't gate
            moving = surprise.motion_events(sensors.read_motion(),
                                            boxes if time.time() - boxes_ts < OBJECTS_FRESH_S else None)
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


def _settled(body: dict, imu_still: list, went_far: list) -> list[Event]:
    """The body is still again (common/imu.py): what it felt since it last was. Turned in
    place -- by someone, or by its own move -- it still knows which way it faces (the map
    turns with it); picked up or driven, the lost-bearings flow above already took over."""
    before = imu_still[0] or state.load_session().get("imu_settled")  # kept across a night's sleep or a restart
    imu_still[0] = body
    if not before or before.get("epoch") != body["epoch"]:  # nothing to compare, or openbot-alive restarted
        went_far[0] = False
        state.update_session({"imu_settled": body})
        return []
    turned = body["heading"] - before["heading"]
    if abs(turned) < 1 and abs(body["turned_by_others"] - before["turned_by_others"]) < 1 \
            and abs(body["tilt"] - before["tilt"]) < 3:
        return []  # nothing new (no writes every half second while it sits still)
    if abs(turned) >= 2 and not went_far[0]:
        vision.turn_by(turned)
    went_far[0] = False
    state.update_session({"imu_settled": body})
    events = surprise.imu_events(before, body)
    for kind, text in events:
        if kind == "turned":
            state.update_session({"moved_ts": time.time(), "moved_how": text.replace("someone turned you", "turned")})
    return events


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
    prompt = (f"You are {persona.name}, a small home robot. {name} just arrived -- {why}. It's "
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
        rest_until = agenda.session_lists()[2]
        fired = _reminder_events(now)
        events += fired

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
            _morning_intention(persona)
        tools.refresh_weather()  # every WEATHER_EVERY_S; every prompt reads the cached line
        time.sleep(0.5)


def demo() -> None:
    """Checks the pure parts with a stub LLM and a scratch state dir: the texting back-off (R1) and the
    nightly "what works" distill (R5). No camera, no engine, no real state."""
    import shutil, tempfile
    from pathlib import Path
    test_dir = Path(tempfile.mkdtemp())
    real = (state.STATE_DIR, state.SESSION_PATH, state.LOCK_PATH, memory.MIND_DIR, journal.LOCK_PATH,
            cognition.ask, cfg.CHAT_ALLOW)
    state.STATE_DIR, state.SESSION_PATH, state.LOCK_PATH = test_dir, test_dir / "session.json", test_dir / "s.lock"
    memory.MIND_DIR, journal.LOCK_PATH = test_dir / "mind", test_dir / "j.lock"
    cfg.CHAT_ALLOW = {"15550100"}
    asked: list[str] = []
    cognition.ask = lambda url, model, prompt, **k: (asked.append(prompt), cognition.CognitionResult(
        cognition.AVAILABLE, json.dumps({"rules": ["Text Atul less when he is away.", "Nod when someone comes close."]})))[1]
    try:
        # R1: the floor doubles for each text that gets no answer; a reply resets it; a pause stops it
        now = time.time()
        s = {"last_text_ts": 0, "chat_last_heard_ts": 0}
        assert _may_text_first(s) == ["15550100"]
        floors = _backed_off(s, ["15550100"], "Atul")  # the first text: nothing unanswered yet
        assert floors == {} and not journal.entries(datetime.date.today())
        for i, floor in enumerate((600, 1200)):  # texts 2 and 3 go out at 10 and 30 min; none answered
            s = {"last_text_ts": now - floor, "chat_last_heard_ts": 0, "text_floor_s": floors}
            assert _may_text_first(s) == ["15550100"], (i, floor)
            assert _may_text_first({**s, "last_text_ts": now - floor + 5}) == []
            floors = _backed_off(s, ["15550100"], "Atul")
        assert floors == {"15550100": 2400}  # the fourth waits 40 min
        assert _may_text_first({"last_text_ts": now - 2399, "text_floor_s": floors}) == []
        planned = [e for e in journal.entries(datetime.date.today()) if "[planned]" in e]
        assert len(planned) == 2 and planned[-1].endswith("not texting Atul first for 40 min -- no answer to my last text")
        assert _backed_off({"last_text_ts": now - 2400, "chat_last_heard_ts": now - 100, "text_floor_s": floors},
                           ["15550100"], "Atul") == floors  # he answered: no further doubling...
        state.update_session({"text_floor_s": floors})
        state.reset_text_floor("15550100")  # ...and openbot-chat resets the floor on his message
        assert _may_text_first({**state.load_session(), "last_text_ts": now - 601}) == ["15550100"]
        paused = {"texts_paused_until": {"15550100": now + 3600}, "last_text_ts": 0}
        assert _may_text_first(paused) == []  # paused: nothing, even when Jev would say text
        # R5: the LLM gets counts and days, never raw lines; under MIN_REACTIONS the old rules stay
        persona = persona_mod.load()
        for i in range(9):
            memory.remember("lesson", REACTION_NOTE, "reaction", f"Sat 07:0{i} AM I texted \"hi {i}\" -> Atul: no reply")
        memory.write_note("self", WHAT_WORKS, ["old rule"], "rule")
        _distill_what_works(persona)
        assert not asked and memory.read_note("self", WHAT_WORKS) == ["- [rule] old rule"]
        memory.remember("lesson", REACTION_NOTE, "reaction", 'Sun 07:07 AM I texted "morning" -> Atul: replied after 23 min')
        memory.remember("lesson", REACTION_NOTE, "reaction", "Mon 08:10 AM I reacted to something coming close with a nod -> Anna: they talked back to me")
        _distill_what_works(persona)
        assert len(asked) == 1 and "[reaction]" not in asked[0] and '"hi' not in asked[0], asked
        assert "Over 3 different days" in asked[0] and "texted -> Atul: no reply: 9 times on 1 day" in asked[0]
        assert "texted -> Atul: replied: 1 time on 1 day" in asked[0]
        assert "reacted to something coming close with a nod -> Anna: they talked back to me: 1 time on 1 day" in asked[0]
        assert "no clock times" in asked[0] and "3 or more different days" in asked[0]
        assert memory.read_note("self", WHAT_WORKS) == ["- [rule] Text Atul less when he is away.",
                                                         "- [rule] Nod when someone comes close."]
        # what comes to mind: what he did and how people took it, twice as readily as a fact; never a recent pick
        memory.remember("person", "Anna", "likes", "rain on Sundays", source="2026-10-05 21:00 voice")
        mems = _memories()
        assert ('Sat 07:00 AM: you texted "hi 0" -> Atul: no reply', 1) in mems and ("anna: rain on Sundays", 1) in mems
        assert ("Mon 08:10 AM: you reacted to something coming close with a nod -> Anna: they talked back to me", 2) in mems
        assert not any("Text Atul less" in m for m, _ in mems)  # the rules are in every prompt already
        rng = random.Random(1)
        picks = _came_to_mind(mems, [], 2, rng)
        assert len(picks) == 2 and len(set(picks)) == 2
        assert _came_to_mind(mems, [m for m, _ in mems if m != "anna: rain on Sundays"], 2, rng) == ["anna: rain on Sundays"]
        assert _came_to_mind([], [], 2, rng) == []
        # how moments felt: kept only when something happened, the feeling in the words; enjoyed ones come back 3x
        assert _since_last_thought([("sound", "a bang")], []) == ["a bang"] and _since_last_thought([], []) == []
        talk = ["[thought] 10:00 (calm) hm", "[heard] 10:01 Anna: nice turn!", "[said] 10:01 Thank you!"]
        assert _since_last_thought([], ["[heard] 09:00 x"] + talk) == talk[1:]  # since the thought, his words for context
        assert _since_last_thought([], ["[thought] 10:00 (calm) hm", "[said] 10:01 hello?", "[did] 10:01 nod"]) == []
        assert _keep_moment("Anna laughed at my fist bump.", "Delighted", None) == \
            "Anna laughed at my fist bump -- I felt delighted"
        assert _keep_moment("Anna laughed, I was delighted", "delighted", None) == "Anna laughed, I was delighted"
        assert _keep_moment("", "happy", None) is None
        assert _keep_moment("Anna laughed at my fist bump", "delighted", "Anna laughed at my fist bump -- I felt delighted") is None
        memory.remember("self", MOMENTS, "delighted", "Mon 03:40 PM: Anna laughed at my fist bump -- I felt delighted")
        memory.remember("self", MOMENTS, "bored", "Mon 04:10 PM: the room stayed empty -- I felt bored")
        weights = dict(_memories())
        assert weights["Mon 03:40 PM: Anna laughed at my fist bump -- I felt delighted"] == 3
        assert weights["Mon 04:10 PM: the room stayed empty -- I felt bored"] == 1
        # missing people: away a day -> 4x as readily -- their note, reactions with them, moments that name them
        assert _missing(0) == 1 and _missing(24) == 4 == _missing(100)
        missing = dict(_memories({"anna": 24}))
        assert missing["anna: rain on Sundays"] == 4
        assert missing["Mon 03:40 PM: Anna laughed at my fist bump -- I felt delighted"] == 12
        assert missing["Mon 08:10 AM: you reacted to something coming close with a nod -> Anna: they talked back to me"] == 8
        assert missing['Sat 07:00 AM: you texted "hi 0" -> Atul: no reply'] == 1  # a text, and Atul isn't away
        # the morning intention: from 6 AM, once a day -- and only a success counts (an outage retries, not tomorrow)
        calls: list[str] = []

        def ask(url, model, prompt, **k):
            calls.append(prompt)
            if len(calls) == 1:
                return cognition.CognitionResult(cognition.OFFLINE, error="down")
            return cognition.CognitionResult(cognition.AVAILABLE, json.dumps({"today_i_want": "explore under the bed"}))
        cognition.ask = ask
        morning = datetime.datetime(2026, 10, 6, 7, 30)
        _morning_intention(persona, datetime.datetime(2026, 10, 6, 5, 30))
        assert not calls  # before 6: still night
        _morning_intention(persona, morning)
        assert len(calls) == 1 and state.load_session().get("intention") is None and _intention_retry[0] > time.time()
        _morning_intention(persona, morning)
        assert len(calls) == 1  # waits out the retry
        _intention_retry[0] = 0.0
        _morning_intention(persona, morning)
        assert state.load_session()["intention"] == {"day": "2026-10-06", "text": "explore under the bed"}
        _morning_intention(persona, morning)
        assert len(calls) == 2 and "Not texting" in calls[-1]  # once a day
        # the nightly dream: lasting notes and up to two wishes from yesterday -- only a dream that happened counts
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        journal.log("heard", "Anna: the lights are out, sorry", now=datetime.datetime.combine(yesterday, datetime.time(21)))
        dreams: list[str] = []

        def dream_ask(url, model, prompt, **k):
            if "Then your wishes" not in prompt:  # the what-works distill that follows the dream
                return cognition.CognitionResult(cognition.AVAILABLE, json.dumps({"rules": ["Nod more."]}))
            dreams.append(prompt)
            if len(dreams) == 1:
                return cognition.CognitionResult(cognition.TIMEOUT, error="slow")
            return cognition.CognitionResult(cognition.AVAILABLE, json.dumps({"notes": [], "wishes": [
                "I wish I could see in the dark -- the lights went out at 9", "I wish I had an arm", "a third"]}))
        cognition.ask = dream_ask
        _dream(persona)
        assert len(dreams) == 1 and state.load_session().get("dreamed") != yesterday.isoformat()
        assert _dream_retry[0] > time.time()  # no answer: tries again later, the night isn't lost
        _dream_retry[0] = 0.0
        _dream(persona)
        assert state.load_session()["dreamed"] == yesterday.isoformat()
        assert "Your wish list is empty so far" in dreams[-1] and "beyond you now" in dreams[-1]
        wishes = memory.read_note("self", "wishes")
        assert len(wishes) == 2 and wishes[0].endswith("I wish I could see in the dark -- the lights went out at 9")
        _dream(persona)
        assert len(dreams) == 2  # once a night
    finally:
        _intention_retry[0] = _dream_retry[0] = 0.0
        (state.STATE_DIR, state.SESSION_PATH, state.LOCK_PATH, memory.MIND_DIR, journal.LOCK_PATH,
         cognition.ask, cfg.CHAT_ALLOW) = real
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        demo()
        print("mind: ok")
    else:
        main()
