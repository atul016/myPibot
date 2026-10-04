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
import subprocess
import threading
import time

import common.system  # noqa: F401 -- os.getlogin() shim, side effect only, import before any Picarx()
import movement.bus  # noqa: F401 -- one lock around all I2C traffic (threads were corrupting ADC reads)

import config as cfg  # noqa: E402
from common import events as dash_events  # noqa: E402
from common import faces, health, journal, persona as persona_mod, policy, react, sensors, state, surprise  # noqa: E402
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


def _shut_down(pct: float) -> None:
    """The battery is all but empty: say so, text its person (openbot-chat sends
    session["text_out"]), and power off cleanly before it browns out."""
    journal.log("did", f"battery at {pct:.0f}% -- shutting down before it dies")
    dash_events.log_event("safety", f"battery at {pct:.0f}% -- shutting down")
    state.update_session({"text_out": {"text": f"My battery is at {pct:.0f}% -- I'm shutting myself down now. "
                                               "Please charge me!", "photo": False, "ts": time.time(),
                                      "urgent": True}})  # goes out even if they asked for a break from its texts
    speak_client("Battery empty. Shutting down.")
    time.sleep(15)  # time for the WhatsApp text to go out
    subprocess.run(["sudo", "-n", "shutdown", "now"], check=False)


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


def _start_navigation(car, monitor, task: dict, action_flow) -> str:
    if NAVIGATING.is_set():
        return "already driving"
    NAV_CANCEL.clear()
    NAVIGATING.set()
    threading.Thread(target=_navigate, args=(car, monitor, task, action_flow), daemon=True).start()
    return ""


def _navigate(car, monitor, task: dict, action_flow) -> None:
    from movement import navigate
    from movement.real_body import RealBody

    body = RealBody(car, monitor, cfg, NAV_CANCEL)
    target, following = task.get("approach"), bool(task.get("follow"))
    dash_events.log_event("safety", "driving: following a person" if following
                          else f"driving: {'to the ' + target if target else 'exploring'}")
    try:
        if following:
            body.tilt = navigate.FOLLOW_TILT  # head up at the person, not down at the floor
        body.look(0)  # driving pose (head down at the floor ahead) -- wherever the face tracker left it
        outcome = (navigate.follow(body) if following else navigate.approach(body, target) if target
                   else navigate.explore(body, int(task.get("explore", 60))))
    except Exception as e:  # never leave the motors running on a bug
        outcome = navigate.Outcome(False, f"something went wrong ({e})")
    finally:
        car.stop()
        car.set_dir_servo_angle(0)
        body.look(0)
        NAVIGATING.clear()
    what = (f"followed someone: {outcome.reason}" if following
            else f"drove {'to the ' + target if target else 'around exploring'}: {outcome.reason}")
    dash_events.log_event("safety", f"driving done: {outcome.reason} ({outcome.steps} steps)")
    print(f"drive: done -- {outcome.reason} ({outcome.steps} rounds)")
    journal.log("did", what)
    # How it went is the LLM's to tell (in its mood); the facts come from the drive.
    situation = (f"You just stopped driving because they said stop." if outcome.reason == "cancelled"
                 else f"You were following them, and you've stopped: {outcome.reason}." if following
                 else f"You just made it to the {target}!" if target and outcome.done
                 else f"Your drive ended: {outcome.reason}." + (f" Along the way you saw: {', '.join(outcome.seen)}."
                                                              if outcome.seen else ""))
    line, tone = react.line(persona_mod.load(), situation + " Say how it went.",
                            "Stopped." if outcome.reason == "cancelled" else outcome.reason.capitalize() + ".")
    if tone:
        action_flow.add_action(tone)
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
                busy = _start_navigation(car, monitor, req["navigate"], action_flow)
                conn.sendall(json.dumps({"ok": not busy, "error": busy}).encode())
                continue
            actions = req.get("actions") or []
            if actions and NAVIGATING.is_set():
                conn.sendall(json.dumps({"ok": False, "error": "driving"}).encode())
                continue  # a gesture mid-drive would fight the drive for the head and wheels
            wait = bool(req.get("wait", False))
            if actions:
                if not any(a.startswith("look ") for a in actions) and not faces.read().get("faces"):
                    # Re-centre for mind's "ahead" camera look -- unless someone's in view: the
                    # face tracker takes the head straight back to them instead (no 0.4s detour).
                    actions = [*actions, "look ahead"]
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
    health.start_watchdog(COMPONENT, cfg.WATCHDOG_STALE_SEC, cfg.WATCHDOG_PING_INTERVAL)

    car = Picarx()
    action_flow = ActionFlow(car)
    action_flow.start()
    car.set_cliff_reference(cfg.CLIFF_REFERENCE)

    monitor = SafetyMonitor(car, action_flow, cfg)
    monitor.start_distance_watchdog()

    threading.Thread(target=_action_server, args=(action_flow, car, monitor), daemon=True).start()
    threading.Thread(target=_face_tracker, args=(car, action_flow), daemon=True).start()

    health.record_success(COMPONENT)
    next_sensor_publish = 0.0
    last_latches: dict = {}
    next_sleep_check, asleep = 0.0, False
    next_battery_poll = 0.0
    battery_v: float | None = None
    battery_pct: float | None = None
    battery_low_readings, shutdown_tried = 0, False

    try:
        while True:
            if time.time() >= next_battery_poll and not NAVIGATING.is_set():
                # Read even asleep: it sleeps all night, which is when the battery runs out.
                try:
                    battery_v = get_battery_voltage()
                    battery_pct = _battery_percent(battery_v)
                    battery_low_readings = surprise.battery_critical(battery_low_readings, battery_v, battery_pct)
                except Exception as e:  # a failed read is no reading -- never a low one
                    print(f"battery read failed: {e}")
                next_battery_poll = time.time() + BATTERY_POLL_INTERVAL_S
                if battery_low_readings:
                    print(f"battery low: {battery_pct:.0f}% ({battery_v:.2f}V), reading "
                          f"{battery_low_readings}/{surprise.SHUTDOWN_READINGS} before shutting down")
                if battery_low_readings >= surprise.SHUTDOWN_READINGS and not shutdown_tried:
                    shutdown_tried = True  # once: if sudo is refused, don't say goodbye every 90s
                    _shut_down(battery_pct)
            if time.time() >= next_sleep_check:
                asleep = bool(state.load_session().get("asleep"))
                monitor.paused = asleep
                next_sleep_check = time.time() + 1.0
            if asleep:
                # "Go to sleep": no sensors, no cliff/proximity reflexes, no fidgets -- only the
                # battery check (above) and the action socket (to wake the head) stay up.
                # Still healthy, though: asleep on purpose, not hung.
                health.record_success(COMPONENT, min_interval_s=5.0)
                time.sleep(0.5)
                continue
            if NAVIGATING.is_set():
                # The drive guards itself; reflexes (a bullfight push!) and fidgets would fight it.
                if time.time() >= next_sensor_publish:
                    sensors.publish(**monitor.snapshot(), battery_v=battery_v, battery_pct=battery_pct, driving=True,
                                    head=dict(HEAD))
                    next_sensor_publish = time.time() + 1.0
                health.record_success(COMPONENT, min_interval_s=5.0)
                time.sleep(0.05)
                continue
            monitor.poll()
            if monitor.snapshot()["latches"] != last_latches:  # a hand just arrived: publish NOW, not at the next tick
                last_latches = monitor.snapshot()["latches"]
                next_sensor_publish = 0.0
            if HEAD_MOVED.is_set():  # the head just started moving: say so before the motion reading does
                HEAD_MOVED.clear()
                next_sensor_publish = 0.0
            if time.time() >= next_sensor_publish:
                sensors.publish(**monitor.snapshot(), battery_v=battery_v, battery_pct=battery_pct, head=dict(HEAD))
                next_sensor_publish = time.time() + 1.0
            time.sleep(0.01)
    finally:
        action_flow.stop()
        car.stop()


# Where the head points (by the tracker's own writes) and when it last moved -- published with the
# sensors, so the mind knows a different view is its own glance. A move starting after stillness
# is published at once: the camera's motion reading would otherwise beat the next 1s publish.
HEAD = {"pan": 0.0, "tilt": 0.0, "moved_ts": 0.0}
HEAD_MOVED = threading.Event()

TRACK_INTERVAL_S = 0.2   # matches openbot-camera's face-check rate
FACE_LOST_HOLD_S = 3.0   # keep looking where you were this long before drifting back to centre
WANDER_DEG = (8.0, 5.0)  # idle gaze: small random glances around centre (pan, tilt) -- not a frozen stare
WANDER_EVERY_S = (3.0, 8.0)
EASE_DEG = faces.MAX_STEP_DEG  # per tick, toward wherever the head is heading -- same pace as following a face


def _ease(cur: float, target: float) -> float:
    return cur + max(-EASE_DEG, min(EASE_DEG, target - cur))


def _face_tracker(car: Picarx, action_flow: ActionFlow) -> None:
    """Turns the head toward the biggest face openbot-camera sees -- the
    single most "alive" thing a robot can do. Only between gestures (a
    gesture owns the head; every preset starts with car.reset() and ends
    centred) and never while asleep. After a gesture the
    gaze goes back to the person, not to centre; with nobody around the
    head glances about a little instead of staring straight ahead."""
    pan = tilt = 0.0            # where the head is (by our own writes)
    target = (0.0, 0.0)         # where it's heading when no face steers it
    last_seen = allowed_checked = next_wander = 0.0
    allowed, talking = False, False
    while True:
        time.sleep(TRACK_INTERVAL_S)
        now = time.time()
        if action_flow.status != ActionStatus.STANDBY or NAVIGATING.is_set():
            # The gesture (or drive) finishes with the head centred. Remember where the
            # person was, so the gaze returns to them as soon as the head is free.
            if pan or tilt:
                target = (pan, tilt) if now - last_seen < FACE_LOST_HOLD_S else (0.0, 0.0)
            pan = tilt = 0.0
            HEAD.update(pan=0.0, tilt=0.0, moved_ts=now)  # the gesture or drive is moving the head
            continue
        if now - allowed_checked > 5.0:  # policy reads the session file -- not at 5Hz
            allowed = policy.evaluate("presence").allowed
            talking = state.conversation_active()
            allowed_checked = now
        if not allowed:
            continue
        data = faces.read()
        seen = data.get("faces") or []
        if seen:
            last_seen = now
            nxt = faces.head_step(seen[0]["box"], (data["w"], data["h"]), pan, tilt)
            if nxt:
                target = nxt
        elif now - last_seen > FACE_LOST_HOLD_S:
            if target != (0.0, 0.0) and now - last_seen < 2 * FACE_LOST_HOLD_S:
                target = (0.0, 0.0)  # they left: drift back to centre first
            elif now >= next_wander and not talking:  # servo noise mid-recording reads as garbage
                target = (random.uniform(-WANDER_DEG[0], WANDER_DEG[0]), random.uniform(-WANDER_DEG[1], WANDER_DEG[1]))
                next_wander = now + random.uniform(*WANDER_EVERY_S)
        nxt = (_ease(pan, target[0]), _ease(tilt, target[1]))
        if nxt != (pan, tilt):
            pan, tilt = nxt
            if now - HEAD["moved_ts"] > 1.0:
                HEAD_MOVED.set()
            HEAD.update(pan=pan, tilt=tilt, moved_ts=now)
            try:
                car.set_cam_pan_angle(pan)
                car.set_cam_tilt_angle(tilt)
            except Exception as e:
                print(f"face tracker: servo write failed: {e}")


if __name__ == "__main__":
    main()
