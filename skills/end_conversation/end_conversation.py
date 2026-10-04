"""end_conversation's muscle, by text: openbot-wake-listen ends the conversation at home, as if "be quiet" was said."""
import time

from common import commands, state


def run(ctx: dict) -> str:
    if not state.load_session().get("in_session"):
        return "There's no conversation going on out loud at home -- nothing to end."
    state.update_session({"remote_command": commands.STOP_SESSION, "remote_command_ts": time.time()})
    return "The conversation at home is ending; you stay awake in the background."
