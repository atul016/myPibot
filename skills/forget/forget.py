"""forget's muscle: memory.forget over every note, dream, summary and journal day, holding the
journal's lock (three processes append to it). Never journals WHAT was forgotten -- that would
write it straight back."""
from filelock import FileLock, Timeout

from common import journal, memory, state


def run(ctx: dict, what: str) -> str:
    try:
        with FileLock(str(journal.LOCK_PATH), timeout=10):
            removed = memory.forget(what)
    except Timeout:
        return "Your memory is busy right now -- ask them to say it again in a moment."
    if removed is None:
        return f"\"{what}\" is too vague to forget by -- ask them who or what exactly."
    if not removed:
        return (f"Nothing in your memory mentions the words \"{what}\" -- ask them what word you'd have used for it, "
                "before saying it's all gone.")
    journal.scrub(memory.forget_term(what))  # the reply to this, journaled next, mustn't write it back
    state.update_session({"summary_lines": 0})  # today's summary rebuilds from the cleaned journal
    journal.log("did", f"forgot something, as {ctx.get('who') or 'someone'} asked")
    return (f"You erased everything about it -- {removed} lines of notes and journal. Confirm it's gone, "
            "but don't name or repeat what it was.")
