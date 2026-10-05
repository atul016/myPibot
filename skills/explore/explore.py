"""explore's muscle: navigate.explore in openbot-alive (on a table it stops at the first edge and looks instead)."""
from common import journal
from common.motor_client import navigate


def run(ctx: dict, seconds: int) -> str:
    seconds = max(10, min(120, seconds))
    ok, err = navigate({"explore": seconds})
    if not ok:
        journal.log("held_back", f"wanted to explore, but: {err or 'the wheels did not answer'}")
        return ("You can't explore now: you're already driving somewhere."
                if "already" in err else "You can't explore right now: your wheels aren't answering.")
    journal.log("did", f"started driving around to explore for {seconds}s")
    if ctx["channel"] == "mind":
        ctx["expect_reaction"](f"drove around for {seconds}s")
    return "You've just set off to explore."
