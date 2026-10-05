"""volume's muscle. By text: openbot-wake-listen does it at home. Out loud: the speaker volume, 15% a step,
remembered across reboots (openbot-speak re-applies session["volume"])."""
import time

from common import journal, state


def run(ctx: dict, change: str) -> str:
    if ctx["channel"] == "text":
        state.update_session({"remote_command": {"skill": "volume", "change": change}, "remote_command_ts": time.time()})
        return f"You're turning your voice {'up' if change == 'louder' else 'down'} at home."
    from common.system import get_volume, set_volume
    current = get_volume() or 100
    if change == "louder" and current >= 100:
        return "You're already at full volume."
    new = set_volume(current + (15 if change == "louder" else -15))
    state.update_session({"volume": new})
    journal.log("did", f"volume {current}% -> {new}%")
    return f"You turned your volume {'up' if change == 'louder' else 'down'} to {new}%."
