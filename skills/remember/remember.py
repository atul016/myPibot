"""remember's muscle: a lasting note, in the memory the mind and every turn read (common/memory.py)."""
from common import journal, memory


def run(ctx: dict, fact: str) -> str | None:
    fact = memory.one_line(fact, 300)
    if not fact:
        return None
    memory.remember("lesson", "what people told me", "told", f"{ctx['who']} told me: {fact}")
    journal.log("learned", f"{ctx['who']} told me: {fact}")
    return f"You'll remember: {fact}"
