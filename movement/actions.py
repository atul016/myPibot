"""Cliff-aware forward/turn, patched into picarx.preset_actions.actions_dict
at import time -- actions_dict is shared, module-level state in that
package, and ActionFlow.do_action() looks names up in it fresh each call,
so patching an entry here covers every caller. Hardware-generic (PiCar-X
capabilities), no persona dependency.

NOTE: a couple of functions below do `import config` INSIDE the function
body, deliberately -- this module must be imported before config.py
anywhere in the process, since config.py computes ALLOWED_ACTIONS/etc from
actions_dict at ITS OWN import time. A top-level `import config` here would
trigger that ordering bug against itself; deferring to call time (well
after the whole process has already imported config elsewhere) sidesteps
it. See config.py's own docstring for the full ordering constraint.
"""
import time

from picarx import Picarx
from picarx.preset_actions import actions_dict


def is_real_cliff(grayscale: list, cliff_reference: list) -> bool:
    """True only if ALL THREE grayscale channels read at/below
    cliff_reference individually. Deliberately does NOT trust
    car.get_cliff_status() (Picarx, third-party), which flags a cliff on
    ANY single channel <= reference: too permissive on this hardware --
    every confirmed false positive (a lone/paired outlier channel
    dropping while the rest read near baseline) satisfied that ANY check
    but not this ALL check.

    No magnitude floor on top of the all-channels rule -- a real edge can
    read anywhere from the 40s (partially hanging off) up to 160-193
    (fully lifted), so there's no magnitude band safely "too low to be
    real" left to filter on; the all-channels rule alone already rejects
    every confirmed glitch.
    """
    return all(v <= ref for v, ref in zip(grayscale, cliff_reference))


def _drive_while_checking_cliff(car: Picarx, duration: float, cliff_reference: list,
                                 stop_msg: str, extra_stop=None, stop_motors: bool = True) -> None:
    """car is assumed already driving. Polls the cliff sensor every 50ms
    for up to `duration`, returning early the instant a real edge is
    detected. try/finally guarantees the stop (and extra_stop, if given)
    runs even if a sensor call raises mid-pulse -- without it, an
    exception here both kills the caller's action thread and leaves the
    motors running with nothing left alive to stop them.

    stop_motors=False skips car.stop() -- for callers that immediately
    issue their own car.backward() right after this returns.
    """
    try:
        end_time = time.time() + duration
        while time.time() < end_time:
            grayscale = car.get_grayscale_data()
            if is_real_cliff(grayscale, cliff_reference):
                print(stop_msg)
                return
            time.sleep(0.05)
    finally:
        if stop_motors:
            car.stop()
        if extra_stop:
            extra_stop()


def _cliff_aware_forward(car: Picarx, duration: float = 1.0, speed: int = 5) -> None:
    """Replaces preset_actions.forward, which drives blind for a fixed 1s
    with no fall/cliff check at all -- unsafe for a desk robot. Polls the
    cliff sensor throughout and stops short instead of driving over an
    edge."""
    import config as cfg
    car.forward(speed)
    _drive_while_checking_cliff(car, duration, cfg.CLIFF_REFERENCE,
                                 "cliff ahead -- stopping instead of driving forward")


actions_dict["forward"] = _cliff_aware_forward


def _cliff_aware_turn(car: Picarx, direction: str, angle: int = 25,
                       duration: float = 0.6, speed: int = 30) -> None:
    """Steers and drives an arc, angle/speed capped conservatively, with
    the same cliff-polling safety as _cliff_aware_forward -- a turn can
    walk the car toward an edge just as easily as driving straight can."""
    import config as cfg
    car.set_dir_servo_angle(-angle if direction == "left" else angle)
    time.sleep(0.1)
    car.forward(speed)
    _drive_while_checking_cliff(
        car, duration, cfg.CLIFF_REFERENCE,
        f"cliff ahead -- stopping instead of turning {direction}",
        extra_stop=lambda: car.set_dir_servo_angle(0))


actions_dict["turn left"] = lambda car: _cliff_aware_turn(car, "left")
actions_dict["turn right"] = lambda car: _cliff_aware_turn(car, "right")


def _fist_bump(car: Picarx) -> None:
    """Greeting gesture -- PiCar-X has no arms, so a single decisive
    forward-lean-and-retreat is the closest physical analog."""
    import config as cfg
    car.reset()
    car.set_cam_tilt_angle(-15)
    time.sleep(0.15)
    car.forward(20)
    _drive_while_checking_cliff(car, 0.2, cfg.CLIFF_REFERENCE,
                                 "cliff ahead -- skipping the fist bump nudge", stop_motors=False)
    car.backward(20)
    time.sleep(0.2)
    car.stop()
    car.set_cam_tilt_angle(0)


actions_dict["fist bump"] = _fist_bump


def _bullfight_push(car: Picarx, push_duration: float = 0.3, push_speed: int = 40) -> None:
    """One forward charge, no retreat -- meant to keep firing, one push at
    a time, for as long as something stays within danger range. The
    "keep going" behavior lives in the safety-trigger loop (services/
    alive.py), which re-queues this action each time the previous one
    finishes and the distance is still in range; this function stays a
    single, bounded pulse regardless of how many times it's re-queued."""
    import config as cfg
    car.forward(push_speed)
    _drive_while_checking_cliff(car, push_duration, cfg.CLIFF_REFERENCE,
                                 "cliff ahead -- skipping the bullfight push")


actions_dict["bullfight"] = _bullfight_push


def _safety_backward(car: Picarx, duration: float = 0.8) -> None:
    """Danger-tier reaction -- registered as its own action name, not a
    preset_actions.backward override: that plain action drives at a
    hardcoded low speed, fine for a casual request, too slow for actually
    getting away from something close. Uses config.POWER instead."""
    import config as cfg
    car.backward(cfg.POWER)
    time.sleep(duration)
    car.stop()


actions_dict["safety backward"] = _safety_backward


# What each gesture is for -- the one place prompts take it from (config.GESTURE_GUIDE), so
# a persona or a service never has to name this body's actions. Another body: write its own.
GESTURE_GUIDE = {
    "celebrate": "something good or happy", "nod": "agreement, something good",
    "depressed": "something bad or sad", "shake head": "disagreement, something bad",
    "resist": "annoyed or defensive", "rub hands": "playful or amused",
    "think": "uncertain, working something out", "wave hands": "a greeting",
    "act cute": "being adorable on purpose", "twist body": "a little wiggle of excitement",
    "curious": "interest, a question", "happy": "joy", "excited": "big news",
    "sad": "sympathy", "shy": "embarrassed or flattered",
    "fist bump": "a friendly nudge forward -- when a hand is held out to you",
    "bullfight": "a playful charge -- when something is right in front of you",
    "dance": "a circle and a figure 8 -- when there's room on the floor",
}


# --- emotes: eased head poses (after adrianwedd/spark's px-emote) ------------------
# Camera gimbal only -- stationary, so they land in STATIONARY_ACTIONS and the LLM can
# pick them as a reply's tone or a reflection's gesture. (pan, tilt, ease_s, hold_s);
# pan + = right, tilt + = up. "thinking"/"idle"/"alert" from the original are skipped:
# the preset "think" and "look ahead" already cover them.
EMOTES = {
    "curious": (25, 18, 0.7, 0.5),   # + a small tilt-nod
    "happy":   (0, 12, 0.5, 0.0),    # + side-to-side sweep
    "excited": (0, 15, 0.4, 0.0),    # + rapid pan sweep
    "sad":     (-10, -20, 1.2, 1.0),
    "shy":     (-40, 5, 0.8, 0.6),
}


def _ease_head(car: Picarx, from_pan: float, from_tilt: float, to_pan: float, to_tilt: float, duration: float) -> None:
    steps = max(2, int(duration * 20))
    for i in range(steps + 1):
        t = i / steps
        car.set_cam_pan_angle(round(from_pan + (to_pan - from_pan) * t))
        car.set_cam_tilt_angle(round(from_tilt + (to_tilt - from_tilt) * t))
        if i < steps:
            time.sleep(duration / steps)


def _emote(name: str):
    pan, tilt, ease_s, hold_s = EMOTES[name]

    def act(car: Picarx) -> None:
        _ease_head(car, 0, 0, pan, tilt, ease_s)  # gestures start centred (the tracker/presets leave it there)
        if name == "happy":
            for _ in range(2):
                _ease_head(car, pan, tilt, 20, tilt, 0.25)
                _ease_head(car, 20, tilt, -20, tilt, 0.35)
                _ease_head(car, -20, tilt, 0, tilt, 0.25)
        elif name == "excited":
            last = pan
            for deg in (35, -35, 25, -25, 0):
                _ease_head(car, last, tilt, deg, tilt, 0.18)
                last = deg
        elif name == "curious":
            _ease_head(car, pan, tilt, pan, tilt + 5, 0.3)
            _ease_head(car, pan, tilt + 5, pan, tilt, 0.3)
        time.sleep(hold_s)
        _ease_head(car, 0 if name in ("happy", "excited") else pan, tilt, 0, 0, 0.5)  # back to centre, like every preset
    return act


for _name in EMOTES:
    actions_dict[_name] = _emote(_name)


def _dance(car: Picarx, speed: int = 28, circle_s: float = 3.0, eight_s: float = 2.0) -> None:
    """A circle, then a figure 8 (after spark's px-dance) -- wheels, so it's a
    MOVEMENT action: a spoken "dance", or the mind's pick when the robot lives
    on the floor (config.PLAYFUL_ACTIONS). Cliff-checked every 50ms like every
    drive here, but an arc meets a table edge at any angle: floor only."""
    import config as cfg
    car.set_cam_tilt_angle(10)
    for angle, duration in ((30, circle_s), (-30, eight_s), (30, eight_s)):
        car.set_dir_servo_angle(angle)
        time.sleep(0.15)
        car.forward(speed)
        _drive_while_checking_cliff(car, duration, cfg.CLIFF_REFERENCE, "cliff ahead -- stopping the dance")
        if is_real_cliff(car.get_grayscale_data(), cfg.CLIFF_REFERENCE):
            break
    car.stop()
    car.set_dir_servo_angle(0)
    car.set_cam_tilt_angle(0)


actions_dict["dance"] = _dance


# Head turns for openbot-mind's `look` tool -- camera gimbal only, never the
# wheels. Excluded from STATIONARY_ACTIONS in config.py: they're a sensing
# tool, not an emotional tone_action or an idle gesture.
LOOK_ANGLES = {"ahead": (0, 0), "left": (-45, 0), "right": (45, 0), "up": (0, 25), "down": (0, -25)}  # down = sleep pose


def _look(pan: int, tilt: int):
    def act(car: Picarx) -> None:
        car.set_cam_pan_angle(pan)
        car.set_cam_tilt_angle(tilt)
        time.sleep(0.4)  # let the servos settle before the camera captures
    return act


for _direction, (_pan, _tilt) in LOOK_ANGLES.items():
    actions_dict[f"look {_direction}"] = _look(_pan, _tilt)
