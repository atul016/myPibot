"""volume's muscle, by text: openbot-wake-listen turns it up or down at home, and says so out loud."""
import time

from common import commands, state


def run(ctx: dict, change: str) -> str:
    cmd = commands.LOUDER if change == "louder" else commands.SOFTER
    state.update_session({"remote_command": cmd, "remote_command_ts": time.time()})
    return f"You're turning your voice {'up' if change == 'louder' else 'down'} at home."
