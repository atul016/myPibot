"""finish_task's muscle: off the to-do list -- the one their words mean (common/agenda.find_task)."""
from common import agenda, journal


def run(ctx: dict, item: str) -> str:
    done = agenda.finish_task(item)
    if not done:
        left = "; ".join(t["item"] for t in agenda.load_tasks()) or "nothing"
        return f"Nothing on their to-do list matches \"{item}\" -- it has: {left}. Ask which they meant."
    journal.log("did", f"took off the to-do list: {done['item']}")
    return f"You took \"{done['item']}\" off their to-do list."
