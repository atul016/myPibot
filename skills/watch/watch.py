"""watch's muscle: a kind of surprise it cares about (common/agenda.py) -- when one comes, it's told why."""
import time

from common import agenda, journal, state


def run(ctx: dict, **params) -> str:  # its param is called `for`: a Python keyword
    kind, about = params["for"], params["about"]
    _, watches, _ = agenda.session_lists()
    state.update_session({"watches": agenda.add_watch(watches, kind, about, time.time())})
    journal.log("planned", f"watching for {kind}: {about}")
    return f"You'll be told when it's {kind}."
