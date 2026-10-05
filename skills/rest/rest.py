"""rest's muscle: no idle reflections until then (a surprise still starts one)."""
import datetime
import time

from common import journal, state


def run(ctx: dict, minutes: int) -> str:
    until = time.time() + minutes * 60
    state.update_session({"rest_until": until})
    when = datetime.datetime.fromtimestamp(until).strftime("%I:%M %p").lstrip("0")
    journal.log("planned", f"resting until {when}")
    return f"You're resting until {when}."
