"""resolve_question's muscle: one of the mind's open questions (common/agenda.py), answered."""
from common import agenda, journal, memory


def run(ctx: dict, number: int, answer: str) -> str | None:
    answer = memory.one_line(answer, 300) or f"{ctx['who']} told me"
    goals, done = agenda.resolve_goal(agenda.load_goals(), number, answer)
    if not done:
        return None
    agenda.save_goals(goals)
    memory.remember("lesson", "discoveries", "answered", f"{done['text']} -> {answer}")
    journal.log("resolved", f"{done['text']} -> {answer}")
    return f"Your question \"{done['text']}\" is answered: {answer}"
