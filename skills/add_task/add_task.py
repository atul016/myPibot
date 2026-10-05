"""add_task's muscle: the household to-do list (common/agenda.py), shared by text and voice."""
from common import agenda, journal


def run(ctx: dict, item: str) -> str:
    tasks = agenda.add_task(item, ctx.get("who"))
    journal.log("planned", f"to-do for {ctx.get('who') or 'them'}: {item}")
    return f"You put it on their to-do list: {item}. It has {len(tasks)} thing{'s' * (len(tasks) != 1)} on it now."
