"""remember's muscle: a lasting note by kind, in the memory the mind and every turn read (common/memory.py) --
what someone told it says who."""
import datetime

from common import journal, memory


def run(ctx: dict, kind: str, about: str, category: str, text: str) -> str:
    note = text if ctx["channel"] == "mind" else f"{ctx.get('who') or 'someone'} told me: {text}"
    memory.remember(kind, about, category, note, source=f"{datetime.datetime.now():%Y-%m-%d %H:%M} {ctx['channel']}")
    journal.log("remembered", f"{kind}/{about}: {note}")
    return f"You'll remember: {text}"
