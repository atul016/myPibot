"""pause_texting's muscle: openbot-chat and the mind hold back the texts it starts to this person
until then -- state.texting_paused(). Out loud: the speaker's own number if they're linked, else everyone."""
import datetime
import time

from common import contacts, journal, state


def run(ctx: dict, minutes: int) -> str:
    until = time.time() + minutes * 60 if minutes > 0 else 0.0
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
