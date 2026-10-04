"""openbot-chat: text the persona on WhatsApp. The robot logs in as a
*linked device* of a spare WhatsApp number -- the way WhatsApp Web does --
through neonize (Python over whatsmeow's Go core). It only connects out, so
the robot needs no public address. Unofficial, though: against WhatsApp's
terms, and bans are permanent -- use a number you can afford to lose.

Pairing, once: start the service and scan the QR code it prints
    journalctl -u openbot-chat -f -o cat -a
from the spare phone (WhatsApp > Settings > Linked devices > Link a device).
The login lives in ~/.openbot-whatsapp/ (0700: it's a credential -- and not
under state/, which the dashboard serves to anyone on the network). That phone
must open WhatsApp at least every 14 days, or WhatsApp logs this device out;
the service then restarts and prints a fresh QR code.

Answers only OPENBOT_CHAT_ALLOW's numbers, only one-to-one -- whoever can
text it can see through its camera. A message is one turn: the same persona
prompt, journal and memories as a spoken one, plus a fresh camera frame
(never while asleep: the camera is off then), and the LLM decides whether
to send that photo back. A photo they send replaces the camera frame: it
looks at theirs. "My name is Atul" texted links the number to Atul, like a
face (common/contacts.py): his messages are then Atul's, with what Rocky
knows about Atul. It never moves: nobody may be watching the table edge, so
tone_action is ignored and driving words are just talk. The exact commands
/be-quiet, /go-to-sleep and /wake-up are carried out by wake-listen as if
they had been said (commands.SLASH).

It also texts first: how its day went at OPENBOT_CHAT_SUMMARY_AT, last
night's dream at OPENBOT_CHAT_DREAM_AT, and whatever the mind's decision
engine says is worth a text (services/mind.py, session["text_out"]). How
each of those went -- a reply and how fast, an emoji reaction, or nothing in
REPLY_WINDOW_S -- is an outcome it learns from (common/outcomes.py).
"""
from __future__ import annotations

import datetime
import io
import json
import logging
import queue
import re
import threading
import time
from pathlib import Path

import common.system  # noqa: F401 -- os.getlogin shim: config's picarx import needs it under systemd

import config as cfg  # noqa: E402
from common import cognition, events as dash_events, health, journal, memory, persona as persona_mod  # noqa: E402
from common import agenda, commands, contacts, outcomes, reply_schema, sensors, state, tools, vision  # noqa: E402
from movement.keywords import detect_movement_keyword  # noqa: E402
from neonize.client import NewClient  # noqa: E402
from neonize.events import ConnectedEv, DisconnectedEv, LoggedOutEv, MessageEv  # noqa: E402
from neonize.utils.jid import Jid2String, build_jid  # noqa: E402
from neonize.utils.log import log as neonize_log  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import Field, ValidationError  # noqa: E402

COMPONENT = "openbot-chat"
SESSION_DIR = Path.home() / ".openbot-whatsapp"
MAX_HISTORY_MESSAGES = 16  # per person -- a conversation, bounded like a spoken session's
PHOTO_MAX_SIDE = 1024  # px a sent photo is shrunk to: 1600x1200 took 2.4s to describe, 0.9s at this size
TEXT_FRESH_S = 600  # a text the mind decided on longer ago than this (chat was down) isn't sent
DIRECT = ("s.whatsapp.net", "lid")  # one-to-one chats; not groups (g.us), status updates (broadcast), channels
# In the SYSTEM prompt: the body's own prompt says it moves when asked, and the
# same rule only in the message lost to that -- "drive forward" got "Okay, on it!".
CHAT_RULE = ("\n\nRight now you're chatting by WhatsApp text: a text never moves your body or changes your "
             "mode -- only words said out loud next to you do, or these exact texted commands: /go-to-sleep, "
             "/be-quiet, /wake-up. So never say you're moving any part of yourself -- your head too: turning, "
             "looking somewhere else (\"look away\" got \"Okay, I looking away\") -- or driving, exploring, coming, "
             "or going to sleep, or about to: if they ask for that, tell them to say it to you out loud.")

# Head moves aren't commands at all (it only looks around on its own), but with the rule alone
# "look left" still got "Okay, looking left." half the time -- so they're matched here too.
HEAD_MOVE = re.compile(r"\b(look|turn|face|glance)\s+(to\s+(the\s+)?)?(left|right|up|down|away|around|back|behind)\b"
                       r"|\b(turn|move|rotate|tilt)\s+(your\s+)?(head|camera|face)\b", re.I)

# Texts it started, by WhatsApp message id: replies and emoji reactions to them are outcomes (common/outcomes.py).
_sent: dict[str, dict] = {}
_sent_lock = threading.Lock()  # the worker sends, whatsmeow's thread sees reactions, the notifier times out

_connected = threading.Event()  # logged in and online -- the only thing that counts as healthy
_inbox: queue.Queue = queue.Queue()  # whatsmeow's callback thread must not wait out an LLM turn


def is_direct(source) -> bool:
    """A one-to-one message from someone else -- not its own (every reply
    it sends echoes back), not a group, a status update or a channel."""
    return not source.IsFromMe and source.Chat.Server in DIRECT


def sender_number(source, pn_for_lid) -> str | None:
    """The sender's phone number, country code first, digits only -- or None.
    WhatsApp now addresses many chats by a private "LID" instead: the number
    then rides along in SenderAlt, or comes from the session's LID map."""
    for jid in (source.Sender, source.SenderAlt):
        if jid.Server == "s.whatsapp.net" and jid.User:
            return jid.User
    found = pn_for_lid(source.Sender) if source.Sender.Server == "lid" else None
    return found.User if found and found.Server == "s.whatsapp.net" else None


def texted_command(text: str) -> str | None:
    """"/wakeup", "/wake up", "/Wake-Up" -> WAKE: "/wakeup" (no hyphen) got "System rebooting.
    Vision online." from the LLM while it stayed asleep."""
    squash = lambda t: re.sub(r"[\s_-]+", "", t.strip().lower())  # noqa: E731
    return {squash(k): v for k, v in commands.SLASH.items()}.get(squash(text))


def slash_situation(cmd: str, s: dict) -> tuple[str, bool]:
    """(what a texted command means right now, for the reply; whether
    wake-listen has anything to do -- already asleep/awake, or no
    conversation to end: nothing)."""
    if cmd == commands.SLEEP:
        return ("you're already asleep -- say so, sleepily.", False) if s.get("asleep") else \
            ("you're going to sleep now: everything off until someone says your name and \"wake up\", or "
             "texts /wake-up. A short, sleepy goodnight.", True)
    if cmd == commands.WAKE:
        return ("you're waking up -- one short, sleepy word.", True) if s.get("asleep") else \
            ("you're already awake -- say so.", False)
    return ("the conversation you're having out loud at home ends now; you'll still be around in the "
            "background. Acknowledge it briefly.", True) if s.get("in_session") else \
        ("you weren't talking with anyone at home -- nothing to end. Say so briefly.", False)


def build_chat_reply(Reply):
    class ChatReply(Reply):
        send_photo: bool = Field(description="True to send them the attached camera photo, with your reply as "
                                             "its caption -- whenever they want to see for themselves.")
        look_up: str = Field("", description="A topic to look up in the encyclopedia before you answer -- only for "
                                             "a factual question you can't answer well yourself. Otherwise empty.")
        remind_at: str = Field("", description="If they asked you to remind them of something later: when, as "
                                               "HH:MM (24-hour clock). Otherwise empty.")
        remind_about: str = Field("", description="What to remind them of, when remind_at is set.")
        learned: str = Field("", description="Something lasting their message taught you -- about your home, a "
                                             "person, or yourself -- in one sentence. Otherwise empty.")
        answered_goal: int = Field(0, description="The number of one of your open questions their message "
                                                  "answered, else 0.")
    return ChatReply


def as_jpeg(data: bytes) -> bytes:
    """A photo they sent, as a JPEG no bigger than PHOTO_MAX_SIDE."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def _ask(persona, ChatReply, turn: str, query: str, history: list[dict], image: bytes | None):
    """One LLM turn as the persona, by WhatsApp: (the parsed reply or None, the
    text to send, the raw JSON -- "" when the LLM couldn't be reached: the text
    is then a fallback)."""
    system = persona.system_prompt_template(cfg.ALLOWED_ACTIONS, cfg.STATIONARY_ACTIONS,
                                            cfg.describe_actions(cfg.STATIONARY_ACTIONS)) + CHAT_RULE
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, journal.inject(turn, query=query), system=system,
                           json_schema=ChatReply.model_json_schema(), history=history, image_jpeg=image)
    if result.status != cognition.AVAILABLE or not result.text.strip():
        print(f"{COMPONENT}: cognition {result.status}: {result.error}")
        return None, "I can't reach my brain right now -- the LLM server's unreachable.", ""
    try:
        out = ChatReply.model_validate_json(result.text)
        return out, persona.transform(out.reply), result.text
    except ValidationError:  # not the JSON shape asked for -- send what came back rather than nothing
        print(f"{COMPONENT}: reply parse failed; raw text: {result.text!r}")
        return None, persona.transform(result.text.strip()), result.text


def reply_to(persona, ChatReply, who: str, text: str, history: list[dict],
             sent: bytes | None = None, known: bool = True):
    """One exchange: (the text to send back, a camera photo to send with it or
    None, the parsed reply or None -- for its reminder and what it learned).
    `who`: the person this number is linked to, or (known=False) just the name
    WhatsApp shows. `sent`: a photo they texted -- the LLM sees it instead of
    the camera, and it's never sent back. A texted command goes to wake-listen.
    A factual question may get one encyclopedia lookup first. Commits the
    exchange to `history`."""
    s = state.load_session()
    asleep, cmd = bool(s.get("asleep")), texted_command(text)
    photo = None if asleep or sent or cmd else vision.capture()
    if known:
        turn = f"{who} is texting you on WhatsApp -- they may not be with you, and can't see your gestures. "
        notes = [ln.split("] ", 1)[-1] for ln in memory.read_note("person", who)[-6:]]
        if notes:
            turn += f"What you know about {who}: {' '.join(notes)} "
    else:
        turn = (f"Someone you don't know yet is texting you on WhatsApp (it shows the name \"{who}\") -- if it "
                "fits, ask who they are. ")
    turn += "Text back like a friend would: short.\n"
    if cmd:
        situation, act = slash_situation(cmd, s)
        if act:  # wake-listen carries it out, as if it had been said
            state.update_session({"remote_command": cmd, "remote_command_ts": time.time()})
        turn += f"They texted the command {text.strip()}: {situation}\n"
    elif sent:
        turn += "They sent you the attached photo -- it's theirs, not your camera's. Look at it and react.\n"
    elif photo:
        turn += ("The attached photo is what your camera sees right now: describe it only if they ask what you "
                 "see or what's going on there. When they want to see for themselves (\"show me\", \"send a pic\", "
                 "\"let me see\"), set send_photo -- it goes to them with your reply as its caption, so say "
                 "what's in it.\n")
    elif asleep:
        turn += ("You're asleep right now, so your camera is off: you can't see anything or send a photo, and "
                 "you mustn't say you're awake or can see. If they want something you'd need to be awake for, "
                 "tell them to text /wake-up.\n")
    else:
        turn += "Your camera isn't working right now -- you can't see anything or send a photo.\n"
    battery = sensors.read().get("battery_pct")
    turn += (f"Your body right now: {'asleep' if asleep else 'awake'}"
             + (f", battery {battery:.0f}%" if battery is not None else "") + " -- don't make up any other status.\n")
    # The exact spoken commands, matched in code like the voice turn does: with the rule alone,
    # "come here" still got "On my way!" once in three.
    mode = None if cmd else commands.parse(text, extra_sleep=persona.sleep_words)
    if mode in (commands.SLEEP, commands.STOP_SESSION):
        turn += (f"By text, only the exact command {'/go-to-sleep' if mode == commands.SLEEP else '/be-quiet'} "
                 "does that -- it is NOT happening now. Tell them to send it.\n")
    elif mode or (not cmd and (commands.navigation(text) or detect_movement_keyword(text))):
        turn += (f"\"{text}\" is something only words said out loud next to you can do -- it is NOT happening. "
                 "Don't say it is: tell them to say it to you out loud.\n")
    elif not cmd and HEAD_MOVE.search(text):
        turn += (f"\"{text}\" asks you to move your head -- you can't on request, by text or out loud (you only "
                 "look around on your own). It is NOT happening: don't say it is.\n")
    goals = [] if cmd else agenda.open_goals(agenda.load_goals())
    if goals:
        turn += ("Questions you've been wondering about -- if their message answers one, set answered_goal to its "
                 "number and learned to the answer: " + " ".join(f"{i}. {g['text']}" for i, g in enumerate(goals, 1))
                 + "\n")
    turn += ("If they ask you to remind them of something later, set remind_at and remind_about. For a factual "
             "question you can't answer well yourself, set look_up to a topic to look up first.\n"
             f"\nTheir message: {text}")
    out, reply, raw = _ask(persona, ChatReply, turn, text, history, sent or photo)
    if out and out.look_up.strip():
        found = tools.lookup(out.look_up)
        journal.log("found", f"looked up \"{out.look_up}\": {memory.one_line(found or 'nothing', 200)}")
        again = _ask(persona, ChatReply, turn + f"\n\nYou looked up \"{out.look_up}\": {found or 'nothing found'}"
                     " -- now answer them with it, short.", text, history, sent or photo)
        if again[0]:
            out, reply, raw = again
    if raw:
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": raw}]
        del history[:-MAX_HISTORY_MESSAGES]
    return reply, photo if out and out.send_photo else None, out


def _act_on(out, number: str, who: str) -> None:
    """What a reply decided beyond its words: a reminder to text later, an answer to remember."""
    at = tools.remind_time(out.remind_at) if out.remind_at.strip() and out.remind_about.strip() else None
    if at:
        about = memory.one_line(out.remind_about, 200)
        state.update_session({"text_reminders": state.load_session().get("text_reminders", [])
                              + [{"at": at, "about": about, "number": number}]})
        journal.log("planned", f"remind {who} at {datetime.datetime.fromtimestamp(at):%a %H:%M}: {about}")
    learned = memory.one_line(out.learned, 300)
    if learned:
        memory.remember("lesson", "what people told me", "told", f"{who} told me: {learned}")
        journal.log("learned", f"{who} told me: {learned}")
    if out.answered_goal:
        goals, done = agenda.resolve_goal(agenda.load_goals(), out.answered_goal, learned or f"{who} told me")
        if done:
            agenda.save_goals(goals)
            memory.remember("lesson", "discoveries", "answered", f"{done['text']} -> {learned or who + ' told me'}")
            journal.log("resolved", f"{done['text']} -> {learned or who + ' told me'}")


def _dream_note(day: datetime.date) -> str:
    """That night's dream note (the mind writes it after midnight), or ""."""
    try:
        note = (memory.MIND_DIR / "dreams" / f"{day.isoformat()}.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    return note.split("---\n", 2)[-1].strip()


def _pn_for_lid(client: NewClient, jid):
    try:
        return client.get_pn_from_lid(jid)
    except Exception:  # not in this session's LID map
        return None


def _on_message(client: NewClient, msg: MessageEv) -> None:
    source, m = msg.Info.MessageSource, msg.Message
    if m.HasField("reactionMessage") and is_direct(source):  # 👍 on one of its texts: feedback, not a turn
        _reacted(m.reactionMessage.key.ID, m.reactionMessage.text)
        return
    image = m.HasField("imageMessage")
    text = (m.conversation or m.extendedTextMessage.text or m.imageMessage.caption).strip()
    if not is_direct(source) or not (text or image):  # reactions, receipts, voice notes, videos: not answered
        return
    number = sender_number(source, lambda jid: _pn_for_lid(client, jid))
    if number not in cfg.CHAT_ALLOW:
        print(f"{COMPONENT}: ignored a message from {number or Jid2String(source.Sender)} (not in OPENBOT_CHAT_ALLOW)")
        return
    _inbox.put(("text", source.Chat, msg.Info.Pushname, number, text, m if image else None))


def _on_connected(_: NewClient, __: ConnectedEv) -> None:
    _connected.set()
    print(f"{COMPONENT}: on WhatsApp, answering {', '.join(sorted(cfg.CHAT_ALLOW))}")


def _on_disconnected(_: NewClient, __: DisconnectedEv) -> None:
    _connected.clear()  # whatsmeow reconnects by itself; offline past the watchdog -> restart


def _on_logged_out(_: NewClient, ev: LoggedOutEv) -> None:
    _connected.clear()  # the watchdog restarts the service, which prints a fresh QR code
    print(f"{COMPONENT}: WhatsApp logged this device out ({ev.Reason}) -- scan the new QR code")
    dash_events.log_event("safety", "WhatsApp logged the robot out -- scan the new QR code (journalctl -u openbot-chat)")


def _learn_number(number: str, text: str, photo: bool) -> None:
    """"My name is Atul" texted: this number is Atul's from now on -- like a face."""
    name = contacts.learn(number, text, photo)
    if name:
        journal.log("learned", f"{name} texts me on WhatsApp")
        if not any("[whatsapp]" in ln for ln in memory.read_note("person", name)):
            memory.remember("person", name, "whatsapp", f"{name} texts me on WhatsApp -- they told me it's them")


def _sent_text(message_id: str, number: str, text: str, photo: bool, kind: str, decided_ts: float | None) -> None:
    with _sent_lock:
        _sent[message_id] = {"number": number, "what": f'texted "{memory.one_line(text, 80)}"' + (" [with a photo]" if photo
                             else ""), "kind": kind, "sent_ts": time.time(), "decided_ts": decided_ts, "answered": False}


def _outcome(info: dict, outcome: str, **fields) -> None:
    who = contacts.name_of(info["number"]) or "they"
    outcomes.record(info["what"], f"{who}: {outcome}", kind=info["kind"], decided_ts=info["decided_ts"],
                    sent_ts=info["sent_ts"], **fields)


def _replied(number: str) -> None:
    """A message from them answers the texts it sent them that weren't answered yet."""
    with _sent_lock:
        waiting = [i for i in _sent.values() if i["number"] == number and not i["answered"]]
        for info in waiting:
            info["answered"] = True
    for info in waiting:
        after = time.time() - info["sent_ts"]
        _outcome(info, outcomes.reply_words(after), replied=True, after_s=round(after))


def _reacted(message_id: str, emoji: str) -> None:
    with _sent_lock:
        info = _sent.get(message_id)
    if info and emoji:  # "" is a reaction taken back
        _outcome(info, f"reacted {emoji}", reaction=emoji)


def _sweep() -> None:
    """Texts past the reply window: unanswered ones are "no reply"; all are let go."""
    now = time.time()
    with _sent_lock:
        old = [(k, i) for k, i in _sent.items() if now - i["sent_ts"] > outcomes.REPLY_WINDOW_S]
        for k, _ in old:
            del _sent[k]
    for _, info in old:
        if not info["answered"]:
            _outcome(info, outcomes.reply_words(None), replied=False)


def _answer(client: NewClient, persona, ChatReply, histories: dict, chat, pushname: str, number: str, text: str,
            image_msg) -> None:
    state.update_session({"chat_last_heard_ts": time.time()})  # the mind's texting decision sees it
    _replied(number)
    sent = as_jpeg(client.download_any(image_msg)) if image_msg else None
    if sent:
        text = f"[sent you a photo] {text}".strip()
    _learn_number(number, text, photo=sent is not None)
    name = contacts.name_of(number)
    who = name or pushname or "Someone"
    dash_events.log_event("wake", f"WhatsApp {who}: {text}")
    journal.log("heard", f"{who} (WhatsApp): {text}")
    reply, photo, out = reply_to(persona, ChatReply, who, text, histories.setdefault(number, []), sent,
                                 known=bool(name))
    if out:
        _act_on(out, number, who)
    if photo:
        client.send_image(chat, photo, caption=reply)
    else:
        client.send_message(chat, reply)
    dash_events.log_event("reply", f"WhatsApp reply: {reply}" + (" [photo]" if photo else ""))
    journal.log("said", f"(WhatsApp to {who}) {reply}" + (" [sent a photo]" if photo else ""))


def _send_all(client: NewClient, histories: dict, text: str, photo: bytes | None, kind: str,
              decided_ts: float | None = None, numbers: list[str] | None = None) -> None:
    """A text it starts, to everyone allowed (or just `numbers`) -- into each one's
    history, so their answer has its context, and watched for a reply or a
    reaction (an outcome)."""
    for number in numbers or sorted(cfg.CHAT_ALLOW):
        if photo:
            sent = client.send_image(build_jid(number), photo, caption=text)
        else:
            sent = client.send_message(build_jid(number), text)
        _sent_text(sent.ID, number, text, photo is not None, kind, decided_ts)
        history = histories.setdefault(number, [])
        history += [{"role": "user", "content": "(nobody texted -- you texted them first)"},
                    {"role": "assistant", "content": json.dumps({"tone_action": "none", "reply": text})}]
        del history[:-MAX_HISTORY_MESSAGES]
    dash_events.log_event("reply", f"WhatsApp, texted first: {text}" + (" [photo]" if photo else ""))


def _tell(client: NewClient, persona, ChatReply, histories: dict, kind: str, situation: str, fallback: str,
          numbers: list[str] | None = None) -> None:
    """A scheduled text -- the evening summary, the morning dream, a reminder -- in its own words."""
    to = " and ".join(contacts.names(numbers or cfg.CHAT_ALLOW)) or "your person"
    _, text, raw = _ask(persona, ChatReply, f"Nobody texted you -- you're texting {to} first, on WhatsApp. "
                        f"{situation}", situation, [], None)
    text = text if raw else persona.transform(fallback)
    _send_all(client, histories, text, None, kind, numbers=numbers)
    journal.log("texted", text)


def _worker(client: NewClient, persona, ChatReply) -> None:
    """Everything that talks to WhatsApp or the LLM, one at a time: replies,
    scheduled texts, and the texts the mind decided on."""
    histories: dict[str, list[dict]] = {}
    while True:
        kind, *args = _inbox.get()
        try:
            if kind == "text":
                _answer(client, persona, ChatReply, histories, *args)
            elif kind == "tell":
                _tell(client, persona, ChatReply, histories, *args)
            else:  # "send": the mind's own text, already written (and journaled) by openbot-mind
                text, with_photo, decided_ts = args
                photo = vision.capture() if with_photo and not state.load_session().get("asleep") else None
                _send_all(client, histories, text, photo, "jev", decided_ts)
        except Exception as e:  # one failed message must not end the chat
            print(f"{COMPONENT}: {kind} failed: {e!r}")
            health.record_failure(COMPONENT, repr(e))


def _notifier() -> None:
    """Texts it starts, queued for the worker so they never overlap a reply:
    the mind's (session["text_out"] -- its decision engine said yes), how
    the day went at CHAT_SUMMARY_AT, last night's dream at CHAT_DREAM_AT."""
    while True:
        time.sleep(2)
        if not _connected.is_set():
            continue
        s, now = state.load_session(), datetime.datetime.now()
        today, clock = now.date().isoformat(), now.strftime("%H:%M")
        _sweep()
        fired, waiting = agenda.due(s.get("text_reminders", []), time.time())
        if fired:
            state.update_session({"text_reminders": waiting})
            for r in fired:
                _inbox.put(("tell", "reminder", f"It's time for the reminder they asked you for: {r['about']}. Text it "
                            "to them, short and friendly.", f"Reminder: {r['about']}", [r["number"]]))
        out = s.get("text_out")
        if out:
            state.update_session({"text_out": None})
            if time.time() - out.get("ts", 0) < TEXT_FRESH_S:  # not one held over from a long outage
                _inbox.put(("send", out["text"], bool(out.get("photo")), out.get("decided_ts")))
        if clock >= cfg.CHAT_SUMMARY_AT and s.get("texted_summary") != today:
            state.update_session({"texted_summary": today})
            summary = journal.read_summary(now.date())
            if summary:
                _inbox.put(("tell", "summary", "It's evening: text them how your day went, in a few sentences. "
                            f"Your own summary of today: {summary}", summary))
        if clock >= cfg.CHAT_DREAM_AT and s.get("texted_dream") != today:
            dream = _dream_note(now.date() - datetime.timedelta(days=1))
            if dream:  # no note yet (the mind was off at midnight): sent once it's there
                state.update_session({"texted_dream": today})
                _inbox.put(("tell", "dream", "It's morning: tell them about last night's dream, in a few sentences. "
                            f"While you slept, you went over yesterday and kept this: {dream}", dream))


def _heartbeat() -> None:
    """Healthy only while on WhatsApp: unpaired, or offline longer than the
    watchdog allows -> systemd restarts it (a fresh QR code if unpaired)."""
    while True:
        if _connected.is_set():
            health.record_success(COMPONENT)
        time.sleep(30)


def main() -> None:
    if not cfg.CHAT_ALLOW:
        raise SystemExit("OPENBOT_CHAT_ALLOW is empty -- set the numbers allowed to text the robot")
    neonize_log.setLevel(logging.WARNING)  # connect() hands this level to whatsmeow; unset, it's chatty
    persona = persona_mod.load()
    ChatReply = build_chat_reply(reply_schema.build_reply_model(cfg.TONE_ACTIONS))
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.chmod(0o700)
    client = NewClient(str(SESSION_DIR / "session.db"))
    client.event(MessageEv)(_on_message)
    client.event(ConnectedEv)(_on_connected)
    client.event(DisconnectedEv)(_on_disconnected)
    client.event(LoggedOutEv)(_on_logged_out)
    threading.Thread(target=_worker, args=(client, persona, ChatReply), daemon=True).start()
    threading.Thread(target=_notifier, daemon=True).start()
    threading.Thread(target=_heartbeat, daemon=True).start()
    client.connect()  # blocks; until paired, prints a QR code every ~20s


def demo() -> None:
    from neonize.proto.Neonize_pb2 import JID, MessageSource
    pn, lid = (lambda u: JID(User=u, Server="s.whatsapp.net")), (lambda u: JID(User=u, Server="lid"))
    me, lookup = "919876543210", {"555": pn("919876543210")}.get
    by_lid = lambda jid: lookup(jid.User)  # noqa: E731
    assert sender_number(MessageSource(Chat=pn(me), Sender=pn(me)), by_lid) == me
    assert sender_number(MessageSource(Chat=lid("7"), Sender=lid("7"), SenderAlt=pn("15550100")), by_lid) == "15550100"
    assert sender_number(MessageSource(Chat=lid("555"), Sender=lid("555")), by_lid) == me  # from the LID map
    assert sender_number(MessageSource(Chat=lid("9"), Sender=lid("9")), by_lid) is None  # unknown: not anyone's number
    assert is_direct(MessageSource(Chat=pn(me), Sender=pn(me))) and is_direct(MessageSource(Chat=lid("7"), Sender=lid("7")))
    assert not is_direct(MessageSource(Chat=pn(me), Sender=pn(me), IsFromMe=True))  # its own replies echo back
    assert not is_direct(MessageSource(Chat=JID(User="1203", Server="g.us"), Sender=pn(me), IsGroup=True))
    assert not is_direct(MessageSource(Chat=JID(User="status", Server="broadcast"), Sender=pn(me)))  # never post a status
    assert texted_command(" /Go-To-Sleep ") == commands.SLEEP and texted_command("/be-quite") == commands.STOP_SESSION
    assert texted_command("/wakeup") == texted_command("/wake up") == texted_command("/Wake_Up") == commands.WAKE
    assert texted_command("go to sleep") is None  # words are just talk; only the exact command acts
    assert all(HEAD_MOVE.search(t) for t in ("Look away", "look left", "turn your head", "Look up!", "look to the right"))
    assert not any(HEAD_MOVE.search(t) for t in ("look at the door", "I looked away", "what do you see?", "turn on the light"))
    assert slash_situation(commands.SLEEP, {})[1] and not slash_situation(commands.SLEEP, {"asleep": True})[1]
    assert slash_situation(commands.WAKE, {"asleep": True})[1] and not slash_situation(commands.WAKE, {})[1]
    assert slash_situation(commands.STOP_SESSION, {"in_session": True})[1]
    assert not slash_situation(commands.STOP_SESSION, {})[1]  # no conversation to end
    calls, real = [], (outcomes.record, contacts.name_of)  # outcomes of the texts it starts
    outcomes.record, contacts.name_of = (lambda what, outcome, **f: calls.append((outcome, f))), (lambda n: "Atul")
    try:
        _sent_text("m1", me, "Is that your jacket?", False, "jev", 100.0)
        _reacted("m1", "👍")
        _replied(me)
        _replied(me)  # a second message isn't a second answer
        assert [c[0] for c in calls] == ["Atul: reacted 👍", "Atul: replied right away"] and calls[1][1]["replied"]
        _sent_text("m2", me, "Good night", False, "summary", None)
        _sent["m2"]["sent_ts"] -= outcomes.REPLY_WINDOW_S + 1
        _sweep()
        assert calls[-1][0] == "Atul: no reply" and "m2" not in _sent and "m1" in _sent
    finally:
        outcomes.record, contacts.name_of = real
        _sent.clear()
    png = io.BytesIO()
    Image.new("RGBA", (3000, 2000), (200, 0, 0, 255)).save(png, "PNG")
    small = Image.open(io.BytesIO(as_jpeg(png.getvalue())))
    assert small.format == "JPEG" and small.size == (1024, 683), (small.format, small.size)
    ChatReply = build_chat_reply(reply_schema.build_reply_model(["nod"]))
    assert list(ChatReply.model_json_schema()["properties"]) == ["tone_action", "reply", "send_photo", "look_up",
                                                              "remind_at", "remind_about", "learned", "answered_goal"]
    assert ChatReply.model_validate_json('{"tone_action": "nod", "reply": "Hi!", "send_photo": true}').send_photo


if __name__ == "__main__":
    import sys
    demo() if "--check" in sys.argv else main()
