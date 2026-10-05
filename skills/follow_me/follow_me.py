"""follow_me's muscle: navigate.follow in openbot-alive -- it stops ~0.6m away, calls out if it loses them."""
from common import journal
from common.motor_client import navigate


def run(ctx: dict) -> str:
    ok, err = navigate({"follow": True})
    if not ok:
        return ("You can't follow now: you're already driving somewhere -- they'd have to say stop first."
                if "already" in err else "You can't follow right now: your wheels aren't answering.")
    journal.log("did", "started following them")
    return "You've just started following them -- they say stop to end it."
