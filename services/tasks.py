"""openbot-tasks: says the reminders people asked for out loud, the moment they're due --
asleep or not (an alarm they set), whatever the mind is busy with (a reflection can
hold the mind's loop for a minute). The to-do list and every reminder live in the
session (common/agenda.py; skills remind, add_task, finish_task, list_tasks): the
ones asked for by text go out through openbot-chat, the mind's own come back to it
as a thought, and this loop owns the spoken ones.

    python3 -m services.tasks            the service
    python3 -m services.tasks --check    its check, in a throwaway state folder
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

if __name__ == "__main__" and "--check" in sys.argv:
    os.environ["OPENBOT_STATE_DIR"] = tempfile.mkdtemp(prefix="openbot-check-")  # never the robot's own state

import config as cfg  # noqa: E402
from common import agenda, events as dash_events, health, journal, persona as persona_mod  # noqa: E402
from common.speak_client import speak as speak_client  # noqa: E402

COMPONENT = "openbot-tasks"
TICK_S = 1.0


def say_due(persona, now: float) -> list[str]:
    """Each spoken reminder that's due, said once (agenda.take_due takes it off the list)."""
    said = []
    for r in agenda.take_due(lambda to: to == "home", now):
        line = persona.transform(f"{r['who']}, it's time: {r['about']}." if r.get("who") else
                                 f"Reminder: {r['about']}.")
        speak_client(line)
        journal.log("said", f"{line} (the reminder they asked for)")
        dash_events.log_event("reply", f"{persona.name.lower()} (reminder): {line}")
        said.append(line)
    return said


def main() -> None:
    persona = persona_mod.load()
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)
    while True:
        health.record_success(COMPONENT, min_interval_s=5.0)
        say_due(persona, time.time())
        time.sleep(TICK_S)


def demo() -> None:
    global speak_client
    spoken: list[str] = []
    speak_client = lambda text, persona=None: spoken.append(text)  # noqa: E731
    now = time.time()
    agenda.remind(now - 1, "check the oven", "home", "Anna")
    agenda.remind(now - 1, "call mom", "15550100", "Anna")  # texted: openbot-chat's
    agenda.remind(now - 1, "look at the door")              # the mind's own
    agenda.remind(now + 600, "stretch", "home")             # not yet
    persona = persona_mod.load()
    assert len(say_due(persona, now)) == 1 and say_due(persona, now) == []  # said once
    assert len(spoken) == 1 and "oven" in spoken[0] and "Anna" in spoken[0], spoken
    assert sorted(r["about"] for r in agenda.session_lists()[0]) == ["call mom", "look at the door", "stretch"]
    assert say_due(persona, now + 601) and "stretch" in spoken[-1]


if __name__ == "__main__":
    demo() if "--check" in sys.argv else main()
    print("tasks: ok") if "--check" in sys.argv else None
