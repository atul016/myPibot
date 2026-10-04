"""Arrival/leave hysteresis (services.mind._people_events): a one-sample
(mis)recognition isn't an arrival; a minute out of view isn't leaving.

    cd ~/openbot && python3 -m tests.test_people_events
"""
import tempfile, os

os.environ["OPENBOT_STATE_DIR"] = tempfile.mkdtemp()

import services.mind as m  # noqa: E402

clock = [1000.0]
m.time.time = lambda: clock[0]
present, streaks, events = {}, {}, []
for names in [["Atul"], ["Atul"], ["Atul", "Ana"], ["Atul"], [], ["Atul"]] + [[]] * 50:  # 2 s per sample
    m.faces.read = lambda n=names: {"faces": [{"name": x, "box": [0, 0, 1, 1]} for x in n]}
    events += m._people_events(present, streaks)
    clock[0] += 2.0  # not AWARENESS_INTERVAL_S: that became 0.5s, and 50 samples no longer outlasted PERSON_GONE_S
assert events == [("person_arrived", "Atul is here -- you recognize their face"),
                  ("person_left", "Atul left")], events  # no one-sample "Ana", no left at the 2 s gap
print("people events: ok")
