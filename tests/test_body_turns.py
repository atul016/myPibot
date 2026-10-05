"""The body's turns, as the mind settles them (services.mind._settled, common/imu.py
snapshots): turned in place by someone, the camera's view map turns with it and the
mind notices; a turn made while the mind wasn't looking (asleep, restarted) still
counts when it next settles; picked up or driven, the map is left to the lost-bearings
flow; openbot-alive restarting (a new count) starts over quietly. No hardware; isolated state.

    cd ~/openbot && python3 -m tests.test_body_turns
"""
from tests._audio import isolate_state

isolate_state()

import services.mind as mind  # noqa: E402
from common import state, vision  # noqa: E402

vision._save({"scene": "a desk", "changed": False, "what_changed": "", "direction": "ahead", "head": (0, 0)})


def still(heading: float, by_others: float, tilt: float = 2.0, epoch: float = 1.0) -> dict:
    return {"heading": heading, "turned_by_others": by_others, "tilt": tilt, "pitch": 0.0, "roll": 0.0,
            "moving": False, "moved_by_others": False, "epoch": epoch, "ts": 0.0}


memo, far = [None], [False]
assert mind._settled(still(0, 0), memo, far) == []                      # the first: just a baseline
events = mind._settled(still(90, 90), memo, far)                         # turned right by someone
assert events == [("turned", "someone turned you about 90 degrees to your right")], events
assert vision.last_look()["heading"] == 90.0 and "turned" in state.load_session()["moved_how"]
assert mind._settled(still(90.4, 90.4), memo, far) == []                 # sitting still: nothing new
memo = [None]                                                            # the mind restarted, or slept...
events = mind._settled(still(-2, -2), memo, far)                         # ...while it was turned back left
assert events == [("turned", "someone turned you about 90 degrees to your left")], events
assert vision.last_look()["heading"] == -2.0
assert mind._settled(still(-47, -2), memo, far) == []                    # its own move: map turns, no event
assert vision.last_look()["heading"] == -47.0
far[0] = True                                                            # picked up / drove meanwhile
mind._settled(still(40, 85), memo, far)
assert vision.last_look()["heading"] == -47.0 and far == [False]        # the map wasn't turned by guesswork
assert mind._settled(still(300, 300, epoch=2.0), memo, far) == []        # openbot-alive restarted: start over
assert mind._settled(still(300, 300, tilt=80, epoch=2.0), memo, far)[0][0] == "tipped"
print("test_body_turns: ok")
