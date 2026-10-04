"""pause_texting's muscle: openbot-chat and the mind hold back the texts it starts to this
person (everyone, said out loud) until then -- state.texting_paused()."""
import datetime
import time

from common import journal, state


def run(ctx: dict, minutes: int) -> str:
    until = time.time() + minutes * 60 if minutes > 0 else 0.0
    paused = dict(state.load_session().get("texts_paused_until") or {})
    paused[ctx.get("number") or "*"] = until
    state.update_session({"texts_paused_until": paused})
    if not until:
        journal.log("planned", f"texting {ctx['who']} first again -- they said it's fine")
        return "You'll feel free to text them first again."
    at = datetime.datetime.fromtimestamp(until)
    journal.log("planned", f"no texting {ctx['who']} first until {at:%a %H:%M} -- they asked")
    return f"You won't text them first until {at:%H:%M} -- your answers to their texts still go out."
