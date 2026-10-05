"""remind_me's muscle: a reminder to itself (common/agenda.py) -- back as a surprise when due; it rests until then."""
import datetime
import time

from common import agenda, journal, state


def run(ctx: dict, in_minutes: int, about: str) -> str:
    at = time.time() + in_minutes * 60
    agenda.remind(at, about)  # to: "mind"
    state.update_session({"rest_until": max(agenda.session_lists()[2], at)})
    when = datetime.datetime.fromtimestamp(at).strftime("%I:%M %p").lstrip("0")
    journal.log("planned", f"reminder at {when}: {about} (resting till then)")
    return f"You'll be reminded at {when}."
