"""wake_up's muscle, by text: openbot-wake-listen carries it out at home as if it had been said."""
import time

from common import commands, state


def run(ctx: dict) -> str:
    if not state.load_session().get("asleep"):
        return "You're already awake."
    state.update_session({"remote_command": commands.WAKE, "remote_command_ts": time.time()})
    return "You're waking up."
