"""move's muscle: one of the body's own moves (movement/actions.py) -- only those this body may do
here (config.ALLOWED_ACTIONS: dance is floor-only), and only if openbot-alive takes it (not mid-drive)."""
from common import journal
from common.motor_client import dispatch


def run(ctx: dict, how: str) -> str:
    import config as cfg
    if how not in cfg.ALLOWED_ACTIONS:
        return f"You can't {how} here -- not on this surface, or your body can't."
    if not dispatch([how]):
        return f"You couldn't {how} just now: you're busy driving, or your body isn't answering."
    journal.log("did", f"moved: {how}")
    return f"You're doing it right now: {how}."
