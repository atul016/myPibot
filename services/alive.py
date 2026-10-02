"""openbot-alive: owns the persistent Picarx() handle, permanently. Hosts
the always-on safety triggers (cliff/too-close -- need continuous, low-
latency access; staying in the same process as the permanent handle avoids
IPC on the highest-frequency, most safety-critical read), does small idle
gesture drift when nothing else has requested an action recently, and
listens on a Unix socket for gesture/movement requests from wake-listen and
mind (common/motor_client.py) -- neither of those constructs its own
Picarx(). A second, independent Picarx() in another process does not work
on this hardware: lgpio's GPIO line claims are exclusive per open chip
handle and are not released by stopping motors/ActionFlow (confirmed live:
'GPIO busy' from a second Picarx() while this process's instance was still
running). One persistent owner, everyone else asks it to act -- the same
privilege-boundary shape as services/speak.py for audio.

Runs as the normal user -- motors/sensors/GPIO need no sudo on this
hardware (only audio does; see services/speak.py).
"""
from __future__ import annotations

import json
import os
import random
import socket
import threading
import time

import common.system  # noqa: F401 -- os.getlogin() shim, side effect only, import before any Picarx()
import movement.bus  # noqa: F401 -- one lock around all I2C traffic (threads were corrupting ADC reads)

import config as cfg  # noqa: E402
from common import events as dash_events  # noqa: E402
from common import faces, health, journal, persona as persona_mod, policy, sensors, state  # noqa: E402
from common.bounded import run_bounded  # noqa: E402
from common.motor_client import SOCK_PATH  # noqa: E402
from common import speak_client as speak_client_mod  # noqa: E402
from common.speak_client import speak as speak_client  # noqa: E402
from movement.triggers import SafetyMonitor  # noqa: E402

if cfg.BODY != "picarx":  # this service IS the PiCar-X body; nothing to drive otherwise
    raise SystemExit(f"openbot-alive drives a PiCar-X, but OPENBOT_BODY={cfg.BODY!r}")

from picarx import Picarx  # noqa: E402
from picarx import preset_actions  # noqa: E402
from picarx.preset_actions import ActionFlow, ActionStatus  # noqa: E402
from robot_hat.device import get_battery_voltage  # noqa: E402 -- not robot_hat.utils' wrapper, which is deprecated

COMPONENT = "openbot-alive"
# Subset of STATIONARY_ACTIONS that reads as "idling," not "reacting to
# something" -- the actual reflective/expressive autonomous behavior lives
# in openbot-mind; this is just enough that the robot doesn't look frozen.
# ("curious" was here -- not an action in picar-x 2.1.x, and ActionFlow
# answers an unknown name by blocking its worker on an empty queue.)
IDLE_GESTURES = ["think", "nod"]
IDLE_INTERVAL_S = (15, 40)

# Confirmed live: get_battery_voltage() reads ~8.2V on this pack. A 2S
# Li-ion pack (2x 18650, the PiCar-X standard) runs ~8.4V full to ~6.0V at
# cutoff -- a calibration knob, not a precise fuel gauge: real cells sag
# under load and don't discharge linearly, so treat the percentage as a
# rough indicator, not exact. Retune against a real full-to-empty
# discharge trace if it reads misleadingly.
BATTERY_FULL_V = 8.4
BATTERY_EMPTY_V = 6.0
BATTERY_POLL_INTERVAL_S = 30.0


class _SpeakerMusic:
    """Stands in for robot_hat's Music inside ActionFlow: sound effects go to
    openbot-speak instead of this process opening the audio device itself.
    Two processes (root speak + this one) on one ALSA dmix device crashed
    alive's startup ("unable to mmap channels") whenever both restarted."""

    def sound_play_threading(self, path: str, volume: int = 100) -> None:  # volume: ponytail, aplay plays at system volume
        threading.Thread(target=speak_client_mod.play_sound, args=(path,), daemon=True).start()


preset_actions.Music = _SpeakerMusic  # before ActionFlow() is constructed


def _battery_percent(voltage: float) -> float | None:
    """None well above "full" -- that's a charger/external supply (or an ADC
    glitch: 11.2V was seen on this 8.4V pack), not a fuller battery."""
    if voltage > BATTERY_FULL_V + 0.5:
        return None
    pct = (voltage - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V) * 100
    return max(0.0, min(100.0, pct))


# --- driving with a purpose (movement/navigate.py, proven in sim/) -------------------
# While a drive runs, nothing else moves the head or wheels: no gestures from the
# socket, no face tracking, no fidgets, no proximity reflexes (the drive does its
# own guarding -- floor + ultrasonic every 50ms). "Stop" sets NAV_CANCEL.
NAVIGATING = threading.Event()
NAV_CANCEL = threading.Event()


def _start_navigation(car, monitor, task: dict) -> str:
    if NAVIGATING.is_set():
        return "already driving"
    NAV_CANCEL.clear()
    NAVIGATING.set()
    threading.Thread(target=_navigate, args=(car, monitor, task), daemon=True).start()
    return ""


def _navigate(car, monitor, task: dict) -> None:
    from movement import navigate
    from movement.real_body import RealBody

    body = RealBody(car, monitor, cfg, NAV_CANCEL)
    target = task.get("approach")
    dash_events.log_event("safety", f"driving: {'to the ' + target if target else 'exploring'}")
    try:
        body.look(0)  # driving pose (head down at the floor ahead) -- wherever the face tracker left it
        outcome = navigate.approach(body, target) if target else navigate.explore(body, int(task.get("explore", 60)))
    except Exception as e:  # never leave the motors running on a bug
        outcome = navigate.Outcome(False, f"something went wrong ({e})")
    finally:
        car.stop()
        car.set_dir_servo_angle(0)
        body.look(0)
        NAVIGATING.clear()
    if outcome.reason == "cancelled":
        line = "Stopped."
    elif target and outcome.done:
        line = f"Made it to the {target}!"
    elif outcome.seen:
        line = f"{outcome.reason[0].upper()}{outcome.reason[1:]}. I saw: {', '.join(outcome.seen)}."
    else:
        line = f"{outcome.reason[0].upper()}{outcome.reason[1:]}."
    dash_events.log_event("safety", f"driving done: {outcome.reason} ({outcome.steps} steps)")
    print(f"drive: done -- {outcome.reason} ({outcome.steps} rounds)")
    journal.log("did", f"drove {'to the ' + target if target else 'around exploring'}: {outcome.reason}")
    speak_client(line)


def _action_server(action_flow: ActionFlow, car, monitor) -> None:
    """Unix socket listener: {"actions": [...], "wait": bool} -> queues
    onto this process's own ActionFlow. Thread-safe alongside the main
    loop's own add_action() calls (idle drift, safety reactions) -- both
    already ran concurrently from separate threads in the design this
    replaces (the safety-reaction thread queuing actions while the main
    loop kept polling)."""
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o666)
    server.listen(8)

    while True:
        conn, _ = server.accept()
        try:
            raw = conn.recv(65536)
            req = json.loads(raw.decode())
            if req.get("navigate_cancel"):
                NAV_CANCEL.set()
                conn.sendall(json.dumps({"ok": True, "was_driving": NAVIGATING.is_set()}).encode())
                continue
            if req.get("navigate"):
                busy = _start_navigation(car, monitor, req["navigate"])
                conn.sendall(json.dumps({"ok": not busy, "error": busy}).encode())
                continue
            actions = req.get("actions") or []
            if actions and NAVIGATING.is_set():
                conn.sendall(json.dumps({"ok": False, "error": "driving"}).encode())
                continue  # a gesture mid-drive would fight the drive for the head and wheels
            wait = bool(req.get("wait", False))
            if actions:
                if not any(a.startswith("look ") for a in actions):
                    actions = [*actions, "look ahead"]  # gestures can end with the head turned; mind's camera expects "ahead"
                action_flow.add_action(*actions)
                if wait:
                    run_bounded(action_flow.wait_actions_done, 8.0)
            conn.sendall(json.dumps({"ok": True}).encode())
        except Exception as e:
            try:
                conn.sendall(json.dumps({"ok": False, "error": str(e)}).encode())
            except OSError:
                pass
        finally:
            conn.close()


def main() -> None:
    persona = persona_mod.load()
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)

    car = Picarx()
    action_flow = ActionFlow(car)
    action_flow.start()
    # Startup sound goes through the same quiet-hours gate as everything
    # else audible -- a 3am restart (watchdog, power blip) shouldn't honk.
    if policy.evaluate("audio", quiet_hours=(cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END)).allowed:
        action_flow.add_action("start engine")
    car.set_cliff_reference(cfg.CLIFF_REFERENCE)

    monitor = SafetyMonitor(car, action_flow, speak_client, cfg, persona)
    monitor.start_distance_watchdog()

    threading.Thread(target=_action_server, args=(action_flow, car, monitor), daemon=True).start()
    threading.Thread(target=_face_tracker, args=(car, action_flow), daemon=True).start()

    health.record_success(COMPONENT)
    next_idle_at = time.time() + random.uniform(*IDLE_INTERVAL_S)
    next_sensor_publish = 0.0
    next_sleep_check, asleep = 0.0, False
    next_battery_poll = 0.0
    battery_v: float | None = None
    battery_pct: float | None = None

    try:
        while True:
            if time.time() >= next_sleep_check:
                asleep = bool(state.load_session().get("asleep"))
                monitor.paused = asleep
                next_sleep_check = time.time() + 1.0
            if asleep:
                # "Go to sleep": no sensors, no cliff/proximity reflexes, no fidgets, no
                # battery reads -- only the action socket stays up (to wake the head).
                # Still healthy, though: asleep on purpose, not hung.
                health.record_success(COMPONENT, min_interval_s=5.0)
                time.sleep(0.5)
                continue
            if NAVIGATING.is_set():
                # The drive guards itself; reflexes (a bullfight push!) and fidgets would fight it.
                if time.time() >= next_sensor_publish:
                    sensors.publish(**monitor.snapshot(), battery_v=battery_v, battery_pct=battery_pct)
                    next_sensor_publish = time.time() + 1.0
                health.record_success(COMPONENT, min_interval_s=5.0)
                time.sleep(0.05)
                continue
            monitor.poll()
            if time.time() >= next_idle_at:
                # Never drift mid-conversation: the gesture's own servo
                # noise gets picked up by the mic while wake_listen.py is
                # recording the next turn, and can read as silence/
                # hallucinated garbage -- which wake_listen.py treats as
                # "end the session." Same "don't act while someone's
                # talking to you" rule openbot-mind's reflection already
                # follows, just missing here originally.
                quiet_verdict = policy.evaluate("presence", quiet_hours=(cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END))
                # ...and not while looking at someone: a fidget would look away from them.
                if quiet_verdict.allowed and not state.conversation_active() and not faces.read().get("faces"):
                    _idle_drift(action_flow)
                next_idle_at = time.time() + random.uniform(*IDLE_INTERVAL_S)
            if time.time() >= next_battery_poll:
                # Slow-changing; nowhere near the main loop's ~10ms cadence.
                try:
                    battery_v = get_battery_voltage()
                    battery_pct = _battery_percent(battery_v)
                except Exception as e:
                    print(f"battery read failed: {e}")
                next_battery_poll = time.time() + BATTERY_POLL_INTERVAL_S
            if time.time() >= next_sensor_publish:
                sensors.publish(**monitor.snapshot(), battery_v=battery_v, battery_pct=battery_pct)
                next_sensor_publish = time.time() + 1.0
            time.sleep(0.01)
    finally:
        action_flow.stop()
        car.stop()


TRACK_INTERVAL_S = 0.2   # matches openbot-camera's face-check rate
FACE_LOST_HOLD_S = 3.0   # keep looking where you were this long before drifting back to centre


def _face_tracker(car: Picarx, action_flow: ActionFlow) -> None:
    """Turns the head toward the biggest face openbot-camera sees -- the
    single most "alive" thing a robot can do. Only between gestures (a
    gesture owns the head, and ends re-centred by "look ahead") and never in
    quiet hours (servo whine)."""
    pan = tilt = 0.0
    last_seen = allowed_checked = 0.0
    allowed = False
    while True:
        time.sleep(TRACK_INTERVAL_S)
        now = time.time()
        if action_flow.status != ActionStatus.STANDBY or NAVIGATING.is_set():
            pan = tilt = 0.0  # the gesture (or drive) finishes with the head centred
            continue
        if now - allowed_checked > 5.0:  # policy reads the session file -- not at 5Hz
            allowed = policy.evaluate("presence", quiet_hours=(cfg.QUIET_HOURS_START, cfg.QUIET_HOURS_END)).allowed
            allowed_checked = now
        if not allowed:
            continue
        data = faces.read()
        seen = data.get("faces") or []
        if seen:
            last_seen = now
            nxt = faces.head_step(seen[0]["box"], (data["w"], data["h"]), pan, tilt)
        elif (pan or tilt) and now - last_seen > FACE_LOST_HOLD_S:
            nxt = (pan - max(-5.0, min(5.0, pan)), tilt - max(-5.0, min(5.0, tilt)))  # ease back to centre
        else:
            nxt = None
        if nxt:
            pan, tilt = nxt
            try:
                car.set_cam_pan_angle(pan)
                car.set_cam_tilt_angle(tilt)
            except Exception as e:
                print(f"face tracker: servo write failed: {e}")


def _idle_drift(action_flow: ActionFlow) -> None:
    # Re-center after: "think" leaves the head turned, and mind's periodic
    # "ahead" camera look would read the shifted view as a scene change.
    action_flow.add_action(random.choice(IDLE_GESTURES), "look ahead")
    state.mark_self_noise()
    dash_events.log_event("safety", "idle drift")


if __name__ == "__main__":
    main()
