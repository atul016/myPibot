"""remember's muscle: a lasting note by kind, in the memory the mind and every turn read (common/memory.py) --
what someone told it says who."""
from common import journal, memory


def run(ctx: dict, kind: str, about: str, category: str, text: str) -> str:
    note = text if ctx["channel"] == "mind" else f"{ctx.get('who') or 'someone'} told me: {text}"
    memory.remember(kind, about, category, note)
    journal.log("remembered", f"{kind}/{about}: {note}")
    return f"You'll remember: {text}"
