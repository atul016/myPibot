"""look_up's muscle: an encyclopedia search (common/tools.py) -- the reply answers with what it found."""
from common import journal, memory, tools


def run(ctx: dict, topic: str) -> str:
    found = tools.lookup(topic)
    journal.log("found", f"looked up \"{topic}\": {memory.one_line(found or 'nothing', 200)}")
    return f"You looked up \"{topic}\": {found or 'nothing found'} -- answer them with it, short."
