"""wake_up's muscle, by text: openbot-wake-listen wakes it at home. (Out loud, "<name>, wake up" is a
reflex of wake-listen's own: asleep, the wake word listens for nothing else.)"""
import time

from common import state


def run(ctx: dict) -> str:
    if not state.load_session().get("asleep"):
        return "You're already awake."
    state.update_session({"remote_command": {"skill": "wake_up"}, "remote_command_ts": time.time()})
    return "You're waking up."
