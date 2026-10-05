"""remind's muscle: a reminder in the shared list (common/agenda.py) -- texted to them by openbot-chat
when asked for by text, said out loud at home by openbot-tasks when asked for out loud."""
import datetime

from common import agenda, journal, tools


def run(ctx: dict, at: str, about: str) -> str:
    when = tools.remind_time(at)
    to = ctx.get("number") if ctx["channel"] == "text" else "home"
    if not when or not to:
        return "You couldn't set that reminder: when it's for wasn't clear -- ask them."
    agenda.remind(when, about, to, ctx.get("who"))
    hhmm, how = f"{datetime.datetime.fromtimestamp(when):%H:%M}", "text them" if to != "home" else "say it out loud"
    journal.log("planned", f"remind {ctx.get('who') or 'them'} at {hhmm} ({how}): {about}")
    return f"You set a reminder: at {hhmm} you'll {how} -- {about}."
