"""end_conversation's muscle. By text: openbot-wake-listen ends the conversation at home. Out loud:
driving stops, the conversation ends after the reply -- it stays awake in the background."""
import time

from common import journal, state
from common.motor_client import cancel_navigation


def run(ctx: dict) -> str:
    if ctx["channel"] == "text":
        if not state.load_session().get("in_session"):
            return "There's no conversation going on out loud at home -- nothing to end."
        state.update_session({"remote_command": {"skill": "end_conversation"}, "remote_command_ts": time.time()})
        return "The conversation at home is ending; you stay awake in the background."
    cancel_navigation()  # never keep driving after "be quiet"
    journal.log("did", "ended the conversation (asked to be quiet) -- still around in the background")
    ctx["end_session"] = True
    return ("This conversation is over, but you'll still be around in the background -- saying your name starts a "
            "new one. Acknowledge it briefly.")
