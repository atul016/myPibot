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
knows about Atul. What it can do from a text is its skills (common/skills.py,
skills/*.md with `where: text`): the LLM reads them and picks -- pause its
texting, send a photo, set a reminder, look something up, go to sleep... --
and their muscles (skills/*.py) carry it out. The rest (driving, turning its
head) are listed as needing someone next to it: nobody may be watching the
table edge. tone_action is ignored.

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
import threading
import time
from pathlib import Path

import common.system  # noqa: F401 -- os.getlogin shim: config's picarx import needs it under systemd

import config as cfg  # noqa: E402
from common import cognition, events as dash_events, health, journal, memory, persona as persona_mod  # noqa: E402
from common import agenda, contacts, outcomes, reply_schema, sensors, skills, state, vision  # noqa: E402
from neonize.client import NewClient  # noqa: E402
from neonize.events import ConnectedEv, DisconnectedEv, LoggedOutEv, MessageEv  # noqa: E402
from neonize.utils.jid import Jid2String, build_jid  # noqa: E402
from neonize.utils.log import log as neonize_log  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import ValidationError  # noqa: E402

COMPONENT = "openbot-chat"
SESSION_DIR = Path.home() / ".openbot-whatsapp"
MAX_HISTORY_MESSAGES = 16  # per person -- a conversation, bounded like a spoken session's
PHOTO_MAX_SIDE = 1024  # px a sent photo is shrunk to: 1600x1200 took 2.4s to describe, 0.9s at this size
TEXT_FRESH_S = 600  # a text the mind decided on longer ago than this (chat was down) isn't sent
DIRECT = ("s.whatsapp.net", "lid")  # one-to-one chats; not groups (g.us), status updates (broadcast), channels
# In the SYSTEM prompt: a rule only in the message lost to the persona's own -- "drive forward" got "Okay, on it!".
CHAT_NOTE = ("\n\nRight now you're chatting by WhatsApp text. Your message tells you what you did about theirs: "
             "never say you're doing anything else, or about to -- the rest of what your body does needs someone "
             "next to you, out loud.")
# Step 1 of a turn: what to DO -- just the skills, its own short call (common/skills.py).
DECIDE = ("You're a small robot, and someone just texted you on WhatsApp. Here you only decide what to DO about "
          "their message: put each skill it asks for in `actions` -- none for plain chat, questions or news. Your "
          "words come after, separately. When they ask, do it -- it's their call.\n\nYour skills:\n")
DECIDE_TEMPERATURE = 0.1  # the same message should get the same decision; the words keep their own temperature

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


def as_jpeg(data: bytes) -> bytes:
    """A photo they sent, as a JPEG no bigger than PHOTO_MAX_SIDE."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def _ask(persona, Reply, turn: str, query: str, history: list[dict], image: bytes | None):
    """One LLM turn as the persona, by WhatsApp: (the parsed reply or None, the
    text to send, the raw JSON -- "" when the LLM couldn't be reached: the text
    is then a fallback). No list of physical actions: what it did is in `turn`."""
    system = persona.system_prompt_template([], cfg.STATIONARY_ACTIONS, cfg.describe_actions(cfg.STATIONARY_ACTIONS)) \
        + CHAT_NOTE
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, journal.inject(turn, query=query), system=system,
                           json_schema=Reply.model_json_schema(), history=history, image_jpeg=image)
    if result.status != cognition.AVAILABLE or not result.text.strip():
        print(f"{COMPONENT}: cognition {result.status}: {result.error}")
        return None, "I can't reach my brain right now -- the LLM server's unreachable.", ""
    try:
        out = Reply.model_validate_json(result.text)
        return out, persona.transform(out.reply), result.text
    except ValidationError:
        try:  # the LLM server doesn't always hold the JSON to its schema: the words are what matter
            return None, persona.transform(str(json.loads(result.text)["reply"])), result.text
        except (ValueError, KeyError, TypeError):  # not JSON at all -- send what came back rather than nothing
            print(f"{COMPONENT}: reply parse failed; raw text: {result.text!r}")
            return None, persona.transform(result.text.strip()), result.text


def _decide(book: dict, situation: str, text: str, history: list[dict]) -> tuple[list, list[str]]:
    """Step 1: which skills their message asks for -- a short LLM call that sees
    every skill the body has and answers only with them (common/skills.py).
    (the actions usable by text, the skills asked for that aren't)."""
    Decision = skills.decision_model(cfg.CAN_DRIVE, book)
    if Decision is None:
        return [], []
    result = cognition.ask(cfg.LLM_BASE_URL, cfg.LLM_MODEL, f"{situation}\nTheir message: {text}",
                           system=DECIDE + skills.menu(cfg.CAN_DRIVE, book), json_schema=Decision.model_json_schema(),
                           history=history[-6:], temperature=DECIDE_TEMPERATURE, num_predict=200)
    if result.status != cognition.AVAILABLE:
        print(f"{COMPONENT}: deciding: cognition {result.status}: {result.error}")
        return [], []
    return skills.read_decision(result.text, "text", cfg.CAN_DRIVE, book)


def reply_to(persona, Reply, ctx: dict, text: str, history: list[dict], sent: bytes | None = None,
             known: bool = True):
    """One exchange, in three steps like a person: decide what to do (_decide:
    just the skills), do it (their muscles; anything a text can't do is
    refused), then say it (the persona's own call, told what was done -- so its
    words can't claim what didn't happen). ctx["who"]: the person this number is
    linked to, or (known=False) just the name WhatsApp shows. `sent`: a photo they
    texted -- seen instead of the camera. Returns (the text to send back, the
    camera photo to send with it or None, the skills carried out, the ones
    refused) and commits the exchange to `history`."""
    who, s = ctx["who"], state.load_session()
    asleep = bool(s.get("asleep"))
    photo = ctx["photo"] = None if asleep or sent else vision.capture()
    battery = sensors.read().get("battery_pct")
    body = (f"Your body right now: {'asleep' if asleep else 'awake'}"
            + (f", battery {battery:.0f}%" if battery is not None else "") + " -- don't make up any other status.\n")
    camera = ("They sent you the attached photo -- it's theirs, not your camera's.\n" if sent else
              "Your camera is on: the attached photo is what it sees right now -- describe it only if they ask what "
              "you see or what's going on there.\n" if photo else
              "You're asleep, so your camera is off: you can't see anything or send a photo, and you mustn't say "
              "you're awake or can see.\n" if asleep else
              "Your camera isn't working right now -- you can't see anything or send a photo.\n")
    goals = agenda.open_goals(agenda.load_goals())
    questions = ("Your open questions: " + " ".join(f"{i}. {g['text']}" for i, g in enumerate(goals, 1)) + "\n"
                 if goals else "")

    book = skills.load()
    actions, refused = _decide(book, body + camera + questions, text, history)
    did = [r for a in actions if (r := _use(a, ctx))]
    did += [f"They asked you to {book[n].description[0].lower() + book[n].description[1:].rstrip('.')} -- you can, "
            "but only when someone asks you out loud, in person -- not by text: tell them so." for n in refused]

    if known:
        turn = f"{who} is texting you on WhatsApp -- they may not be with you, and can't see your gestures. "
        notes = [ln.split("] ", 1)[-1] for ln in memory.read_note("person", who)[-6:]]
        if notes:
            turn += f"What you know about {who}: {' '.join(notes)} "
    else:
        turn = (f"Someone you don't know yet is texting you on WhatsApp (it shows the name \"{who}\") -- if it "
                "fits, ask who they are. ")
    turn += ("Text back like a friend would: short. Texts are casual -- short forms and typos (\"der\" is "
             "\"there\", \"u\" is \"you\"), and a short text usually answers your own last one.\n")
    turn += camera + body + ("What you did about their message: " + " ".join(did) if did else
                             "You did nothing about their message -- don't say you're doing anything.") + "\n"
    turn += f"\nTheir message: {text}"
    _, reply, raw = _ask(persona, Reply, turn, text, history, sent or photo)
    if raw:
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": raw}]
        del history[:-MAX_HISTORY_MESSAGES]
    return reply, photo if ctx.get("send_photo") else None, actions, refused


def _use(action, ctx: dict) -> str | None:
    """One skill, carried out -- a failing one must not take the reply down with it."""
    try:
        return skills.run(action, ctx)
    except Exception as e:
        print(f"{COMPONENT}: skill {action.skill} failed: {e!r}")
        return None


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


def _answer(client: NewClient, persona, Reply, histories: dict, chat, pushname: str, number: str, text: str,
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
    ctx = {"channel": "text", "who": who, "number": number}
    reply, photo, actions, refused = reply_to(persona, Reply, ctx, text, histories.setdefault(number, []), sent,
                                              known=bool(name))
    if actions or refused:
        dash_events.log_event("reply", f"WhatsApp skills: used {[a.skill for a in actions]}"
                              + (f", can't by text: {refused}" if refused else ""))
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
    reaction (an outcome). Not to anyone who asked for a pause -- unless it's a
    reminder they asked for, or urgent (the battery's about to die)."""
    for number in numbers or sorted(cfg.CHAT_ALLOW):
        if kind not in ("reminder", "urgent") and state.texting_paused(number):
            continue
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


def _tell(client: NewClient, persona, Reply, histories: dict, kind: str, situation: str, fallback: str,
          numbers: list[str] | None = None) -> None:
    """A scheduled text -- the evening summary, the morning dream, a reminder -- in its own words."""
    to = " and ".join(contacts.names(numbers or cfg.CHAT_ALLOW)) or "your person"
    _, text, raw = _ask(persona, Reply, f"Nobody texted you -- you're texting {to} first, on WhatsApp. "
                        f"{situation}", situation, [], None)
    text = text if raw else persona.transform(fallback)
    _send_all(client, histories, text, None, kind, numbers=numbers)
    journal.log("texted", text)


def _worker(client: NewClient, persona, Reply) -> None:
    """Everything that talks to WhatsApp or the LLM, one at a time: replies,
    scheduled texts, and the texts the mind decided on."""
    histories: dict[str, list[dict]] = {}
    while True:
        kind, *args = _inbox.get()
        try:
            if kind == "text":
                _answer(client, persona, Reply, histories, *args)
            elif kind == "tell":
                _tell(client, persona, Reply, histories, *args)
            else:  # "send": the mind's own text, already written (and journaled) by openbot-mind
                text, with_photo, decided_ts, send_kind = args
                photo = vision.capture() if with_photo and not state.load_session().get("asleep") else None
                _send_all(client, histories, text, photo, send_kind, decided_ts)
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
                _inbox.put(("send", out["text"], bool(out.get("photo")), out.get("decided_ts"),
                            "urgent" if out.get("urgent") else "jev"))
        if all(state.texting_paused(n, s) for n in cfg.CHAT_ALLOW):
            continue  # they asked for a break: the summary and the dream wait until it's over
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
    Reply = reply_schema.build_reply_model(cfg.TONE_ACTIONS)
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.chmod(0o700)
    client = NewClient(str(SESSION_DIR / "session.db"))
    client.event(MessageEv)(_on_message)
    client.event(ConnectedEv)(_on_connected)
    client.event(DisconnectedEv)(_on_disconnected)
    client.event(LoggedOutEv)(_on_logged_out)
    threading.Thread(target=_worker, args=(client, persona, Reply), daemon=True).start()
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
    book = skills.load()  # by text: the decision may name any skill; only the texting ones get done
    acts, refused = skills.read_decision('{"actions": [{"skill": "pause_texting", "minutes": 60}, {"skill": '
                                         '"send_photo"}, {"skill": "move", "how": "forward"}]}', "text", True, book)
    assert [a.skill for a in acts] == ["pause_texting", "send_photo"] and refused == ["move"]
    assert "move(how)" in skills.menu(True, book) and "move(" not in skills.menu(False, book)


if __name__ == "__main__":
    import sys
    demo() if "--check" in sys.argv else main()
