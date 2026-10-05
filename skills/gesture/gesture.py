"""gesture's muscle: one of the body's gestures (config.TONE_ACTIONS) -- one on the wheels
(config.PLAYFUL_ACTIONS) only when motion is allowed (common/policy.py)."""
import config as cfg
from common import journal, policy
from common.motor_client import dispatch


def run(ctx: dict, name: str) -> str | None:
    if name in cfg.PLAYFUL_ACTIONS and not (verdict := policy.evaluate("motion")).allowed:  # a wheel move
        journal.log("held_back", f"wanted to {name}, but {verdict.reason}")
        return None
    if not dispatch([name], wait=True):
        return f"Your body didn't do the {name}."
    journal.log("did", f"gesture {name}")
    ctx["expect_reaction"](f"did a {name} gesture")
    return None
