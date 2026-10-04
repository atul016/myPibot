"""sleep's muscle, by text: openbot-wake-listen carries it out at home as if it had been said."""
import time

from common import commands, state


def run(ctx: dict) -> str:
    if state.load_session().get("asleep"):
        return "You're already asleep."
    state.update_session({"remote_command": commands.SLEEP, "remote_command_ts": time.time()})
    return "You're going to sleep now: camera, thinking and moving off until someone wakes you."
