"""play_sound's muscle: one of the body's sound effects (config.SOUNDS)."""
from common import journal
from common.motor_client import dispatch


def run(ctx: dict, name: str) -> str | None:
    if not dispatch([name], wait=True):
        return f"Your body didn't play the {name} sound."
    journal.log("did", f"played the {name} sound")
    ctx["expect_reaction"](f"played the {name} sound")
    return None
