"""list_tasks's muscle: the to-do list and the reminders people asked for (not the mind's own) -- never a number."""
import datetime

from common import agenda


def run(ctx: dict) -> str:
    tasks = [t["item"] + (f" (for {t['who']})" if t.get("who") else "") for t in agenda.load_tasks()]
    reminders, _, _ = agenda.session_lists()
    theirs = [f"at {datetime.datetime.fromtimestamp(r['at']):%a %H:%M}, {r['about']}"
              + (" (out loud)" if r["to"] == "home" else " (by text)") + (f" for {r['who']}" if r.get("who") else "")
              for r in reminders if r.get("to", "mind") != "mind"]
    return (f"Their to-do list: {'; '.join(tasks) or 'empty'}. "
            f"Reminders set for them: {'; '.join(theirs) or 'none'}.")
