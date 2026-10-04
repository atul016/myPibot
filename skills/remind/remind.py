"""remind's muscle: a text reminder for openbot-chat's notifier to send when it's due."""
import datetime

from common import journal, memory, state, tools


def run(ctx: dict, at: str, about: str) -> str:
    when, about = tools.remind_time(at), memory.one_line(about, 200)
    if not when or not about or not ctx.get("number"):
        return "You couldn't set that reminder: the time or what it's about wasn't clear -- ask them."
    state.update_session({"text_reminders": state.load_session().get("text_reminders", [])
                          + [{"at": when, "about": about, "number": ctx["number"]}]})
    journal.log("planned", f"remind {ctx['who']} at {datetime.datetime.fromtimestamp(when):%a %H:%M}: {about}")
    return f"You set a reminder: you'll text them at {datetime.datetime.fromtimestamp(when):%H:%M} about {about}."
