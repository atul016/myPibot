"""stop's muscle. (Said while driving, "stop" is also a reflex in openbot-wake-listen that doesn't
wait for this: the wheels stop first, then this says so.)"""
from common.motor_client import cancel_navigation


def run(ctx: dict) -> str:
    return "You've stopped." if cancel_navigation() or ctx.get("stopped") else "You weren't moving anyway."
