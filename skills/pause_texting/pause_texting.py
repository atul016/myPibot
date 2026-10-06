"""pause_texting's muscle: openbot-chat and the mind hold back the texts it starts to this person
until then -- state.texting_paused(). Out loud: the speaker's own number if they're linked, else everyone."""
import datetime
import re
import time

from common import contacts, journal, state

# "Stop texting me for 1 hour" said plainly is a reflex (openbot-chat runs this skill when the decision
# missed it): on 2026-10-04 the decision step read it as a reminder to set, and two more texts went out.
# Only as a request, at the start of the message -- "I don't text much" or "why did you stop texting me?"
# aren't one. The LLM still handles every other wording.
ASKED = re.compile(r"^\W*(?:(?:please|pls|ok|okay|hey|rocky|can you|could you)\W+)*(?:stop|don'?t|do not|no more|quit)\b"
                   r"[^.!?]{0,30}?\b(?:text|texting|texts|message|messaging|messages)\b", re.I)
HOW_LONG = re.compile(r"(\d+)\s*(hour|hr|h\b|min)", re.I)
DEFAULT_MINUTES = 240  # the spec's own default, when they don't say how long


def asked(text: str) -> int | None:
    """Minutes, if `text` plainly asks for no more texts -- None if it doesn't."""
    if not ASKED.search(text):
        return None
    m = HOW_LONG.search(text)
    if not m:
        return DEFAULT_MINUTES
    return int(m.group(1)) * (60 if m.group(2).lower().startswith("h") else 1)


def run(ctx: dict, minutes: int) -> str:
    until = time.time() + minutes * 60 if minutes > 0 else 0.0
    ctx["texting_paused"] = until > 0  # openbot-chat records the text this answered as "asked me to stop texting"
    number = ctx.get("number") or next((n for n, name in contacts.load().items() if name == ctx.get("who")), "*")
    paused = dict(state.load_session().get("texts_paused_until") or {})
    paused[number] = until
    state.update_session({"texts_paused_until": paused})
    who = ctx.get("who") or "them"
    if not until:
        journal.log("planned", f"texting {who} first again -- they said it's fine")
        return "You'll feel free to text them first again."
    at = datetime.datetime.fromtimestamp(until)
    journal.log("planned", f"no texting {who if number != '*' else 'anyone'} first until {at:%a %H:%M} -- they asked")
    return f"You won't text them first until {at:%H:%M} -- your answers to their texts still go out."
