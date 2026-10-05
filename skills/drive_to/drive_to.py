"""drive_to's muscle: a drive in openbot-alive (movement/navigate.approach, proven in sim/ first),
which says how it went when it ends."""
from common import journal
from common.motor_client import navigate


def run(ctx: dict, thing: str) -> str:
    ok, err = navigate({"approach": thing})
    if not ok:
        journal.log("held_back", f"wanted to drive to the {thing}, but: {err or 'the wheels did not answer'}")
        return ("You can't drive there now: you're already driving somewhere -- they'd have to say stop first."
                if "already" in err else "You can't drive right now: your wheels aren't answering.")
    journal.log("did", f"started driving to the {thing}")
    if ctx["channel"] == "mind":
        ctx["expect_reaction"](f"drove to the {thing}")
    return "You've just set off toward them." if thing == "person" else f"You've just set off toward the {thing}."
