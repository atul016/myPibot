"""sleep's muscle. By text: openbot-wake-listen does it at home (session["remote_command"]).
Out loud: driving stops now; after the goodnight is said (ctx["after"]), everything goes off and the
head goes down -- only "<name>, wake up" wakes it. The mind's own choice (it's tired: services/mind.py
checks): at once."""
import time

from common import events as dash_events, journal, state
from common.motor_client import cancel_navigation, dispatch


def _lie_down(own: bool = False) -> None:
    state.update_session({"asleep": True})
    dispatch(["look down"], wait=True)  # head down: visibly asleep
    journal.log("did", "went to sleep on my own -- dark and quiet, nobody around" if own else
                "went to sleep -- everything off, listening only for my name and \"wake up\"")
    dash_events.log_event("wake", "went to sleep" + " (own decision)" * own)


def run(ctx: dict) -> str:
    if state.load_session().get("asleep"):
        return "You're already asleep."
    if ctx["channel"] == "text":
        state.update_session({"remote_command": {"skill": "sleep"}, "remote_command_ts": time.time()})
        return "You're going to sleep now: camera, thinking and moving off until someone wakes you."
    if ctx["channel"] == "mind":
        _lie_down(own=True)
        return "You went to sleep."
    cancel_navigation()
    ctx.setdefault("after", []).append(_lie_down)
    ctx["end_session"] = True
    return "You're going to sleep now: everything off until they say your name and \"wake up\". Say a short, sleepy goodnight."
