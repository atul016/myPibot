"""wish's muscle: state/mind/self/wishes.md -- for the developer to read."""
import datetime

from common import journal, memory


def run(ctx: dict, text: str) -> str:
    memory.remember("self", "wishes", "wish", f"{datetime.datetime.now():%Y-%m-%d} {text}")
    journal.log("wished", text)
    return "You wrote it down."
