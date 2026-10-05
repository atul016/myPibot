"""recall's muscle (a tool): memory.recall over its notes and journal."""
from common import memory


def run(ctx: dict, query: str) -> str:
    found = memory.recall(query)
    if found is None:
        return "recall: your memory isn't answering right now"
    return f"recall {query!r}: " + ("; ".join(found) if found else "nothing comes to mind")
