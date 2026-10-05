"""The to-do list and one list of reminders, through the skills a turn uses (no LLM): each
reminder is delivered exactly once, by the one service it's for -- said out loud by
openbot-tasks (asked for out loud), texted by openbot-chat to that number (asked for by
text), back to openbot-mind as a thought (its own) -- and four processes changing the
session at once lose nothing. Speech stubbed; isolated state.

    cd ~/openbot && set -a && . ./openbot.env && set +a && python3 -m tests.test_reminders
"""
from tests._audio import isolate_state

isolate_state()

import time  # noqa: E402
from multiprocessing import Process  # noqa: E402

import services.chat as chat  # noqa: E402
import services.mind as mind  # noqa: E402
import services.tasks as tasks  # noqa: E402
from common import agenda, persona as persona_mod, skills, state  # noqa: E402

spoken: list[str] = []
tasks.speak_client = lambda text, persona=None: spoken.append(text)
persona, book = persona_mod.load(), skills.load()
by_voice = {"channel": "voice", "who": "Anna"}
by_text = {"channel": "text", "who": "Anna", "number": "15550100"}


def use(name: str, ctx: dict, params: dict) -> str:
    did = skills.use(name, ctx, params, book)
    assert did, f"{name} {params}: nothing done"
    return did


# reminders, set the way a turn sets them
assert "say it out loud" in use("remind", by_voice, {"at": "in 1 minute", "about": "check the oven"})
assert "text them" in use("remind", by_text, {"at": "in 1 minute", "about": "call mom"})
use("remind_me", {"channel": "mind"}, {"in_minutes": 1, "about": "look at the door"})
assert "wasn't clear" in use("remind", by_text, {"at": "soonish", "about": "x"})

# the to-do list, shared by text and voice
use("add_task", by_text, {"item": "buy milk"})
assert "2 things" in use("add_task", by_voice, {"item": "call the plumber"})
listed = use("list_tasks", by_voice, {})
assert all(w in listed for w in ("buy milk", "call the plumber", "check the oven", "out loud", "call mom", "by text"))
assert "look at the door" not in listed and "15550100" not in listed  # not the mind's own; never a number
assert "buy milk" in use("finish_task", by_text, {"item": "I bought the milk"})
assert "Nothing" in use("finish_task", by_text, {"item": "walk the dog"})
assert [t["item"] for t in agenda.load_tasks()] == ["call the plumber"]

# delivered: not before they're due, then each once, by its own service
now = time.time()
assert tasks.say_due(persona, now) == [] and mind._reminder_events(now) == []
chat._text_due_reminders(now)
assert chat._inbox.empty()
later = now + 61
said = tasks.say_due(persona, later)
assert len(said) == 1 and "oven" in said[0] and "Anna" in said[0] and spoken == said
chat._text_due_reminders(later)
kind, what, prompt, _, numbers = chat._inbox.get_nowait()
assert (kind, what, numbers) == ("tell", "reminder", ["15550100"]) and "call mom" in prompt and chat._inbox.empty()
assert mind._reminder_events(later) == [("reminder", "a reminder you set yourself: look at the door")]
assert tasks.say_due(persona, later) == [] and mind._reminder_events(later) == []
chat._text_due_reminders(later)
assert chat._inbox.empty() and agenda.session_lists()[0] == []


# several services changing the list at once: none of their changes lost
def bump(n: int) -> None:
    for _ in range(n):
        state.change_session("counter", lambda v: (v or 0) + 1)


procs = [Process(target=bump, args=(25,)) for _ in range(4)]
for p in procs:
    p.start()
for p in procs:
    p.join()
assert state.load_session()["counter"] == 100
print("test_reminders: ok")
