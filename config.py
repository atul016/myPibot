"""Framework-level configuration: everything about *this physical robot*
that isn't persona identity (that lives in personas/<name>/). Env-var
overrides follow the OPENBOT_* convention throughout -- set them in
/etc/openbot/openbot.env (see openbot.env.example).

The body is a choice (OPENBOT_BODY):
  none   -- no robot: just a mic, a speaker and (optionally) a camera. It
            talks, sees, thinks and remembers, but never moves.
  picarx -- a SunFounder PiCar-X (openbot-alive drives it). Importing this
            module then also patches the car's action dict (movement.actions/
            sounds), so the action lists below always include those patches.
Another robot: write a service that answers the openbot-alive socket
(see README, "Adding your own robot") and list its actions here.
"""
from __future__ import annotations

import os

BODY = os.environ.get("OPENBOT_BODY", "none")
if BODY == "picarx":
    import movement.actions  # noqa: F401 -- patches actions_dict; must run before it's read below
    import movement.sounds  # noqa: F401
    from movement.actions import LOOK_ANGLES
    from picarx.preset_actions import actions_dict
elif BODY == "none":
    LOOK_ANGLES, actions_dict = {"ahead": (0, 0)}, {}
else:
    raise SystemExit(f"OPENBOT_BODY={BODY!r}: expected 'none' or 'picarx'")
HAS_BODY = BODY != "none"
CAN_DRIVE = BODY == "picarx"     # "go to the X" / "explore" / forward/back/turn
LOOK_DIRECTIONS = sorted(LOOK_ANGLES)  # the mind's `look` tool; just "ahead" without a head to turn
SOUNDS = ["honking", "start engine"] if BODY == "picarx" else []

# --- LLM: any OpenAI-compatible /v1 server (MLX Serve, Ollama, llama.cpp...) ---
# Default is a local Ollama. A bigger model on another machine: point this at
# it, and give that machine a fixed IP (a DHCP lease change breaks the bot).
LLM_BASE_URL = os.environ.get("OPENBOT_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("OPENBOT_LLM_MODEL", "qwen2.5vl:7b")  # must see images and follow a JSON schema

# --- Decision engine for the mind loop's action choice ------------------
# "" -> off (Ollama picks the action itself). Engine names: see
# common/decider.py. Below the confidence floor the mind loop waits.
DECIDER = os.environ.get("OPENBOT_DECIDER", "")
DECIDER_MIN_CONFIDENCE = float(os.environ.get("OPENBOT_DECIDER_MIN_CONFIDENCE", "0.6"))

# --- STT / mic ------------------------------------------------------------
STT_DEVICE = os.environ.get("OPENBOT_STT_DEVICE", "USB PnP Sound Device")
STT_LANGUAGE = os.environ.get("OPENBOT_STT_LANGUAGE", "en-us")  # en-in: Indian English; accent match beat model size, see README
# Whisper re-checks what Vosk wasn't sure of (common/stt.py). A model name
# (downloaded once) or a local model dir; "" -> Vosk only. base.en: ~2.3s per
# utterance on the Pi 5; small.en was 7-8s -- too slow for conversation.
WHISPER_MODEL = os.environ.get("OPENBOT_WHISPER_MODEL", "base.en")
# A whisper.cpp server on the LLM's host, much faster than the Pi's own CPU
# (tools/setup-mac-whisper.sh sets one up on a Mac). 0 -> local Whisper only.
MAC_WHISPER_PORT = int(os.environ.get("OPENBOT_MAC_WHISPER_PORT", "0"))
# Start/end of speech is decided by streaming Vosk (are there words?), not a
# loudness threshold -- the fixed one (900) silently broke when a fresh
# install reset the mic gain: speech peaked ~670 and follow-ups were never heard.


# --- Safety thresholds (properties of this hardware, not any persona) ----
POWER = int(os.environ.get("OPENBOT_SAFETY_POWER", "50"))
SAFE_DISTANCE = float(os.environ.get("OPENBOT_SAFE_DISTANCE", "10"))    # cm
DANGER_DISTANCE = float(os.environ.get("OPENBOT_DANGER_DISTANCE", "5"))  # cm
CLIFF_REFERENCE = [200, 200, 200]

SAFETY_ACTIONS = {"safety backward"}
MOVEMENT_ACTIONS = {"forward", "backward", "act cute", "twist body", "turn left", "turn right",
                     "fist bump", "bullfight"} & set(actions_dict)  # empty with no body
LOOK_ACTIONS = {name for name in actions_dict if name.startswith("look ")}  # mind's `look` tool, see movement/actions.py
STATIONARY_ACTIONS = sorted(set(actions_dict) - MOVEMENT_ACTIONS - SAFETY_ACTIONS - LOOK_ACTIONS)
# What the persona's prompt advertises the LLM "can perform" -- deliberately
# STATIONARY_ACTIONS | MOVEMENT_ACTIONS, not actions_dict|sounds_dict:
# sound-only entries have no schema field or keyword path that can ever
# select them, so listing them would advertise actions with no way to
# actually be performed.
ALLOWED_ACTIONS = sorted(set(STATIONARY_ACTIONS) | MOVEMENT_ACTIONS)

SAFETY_TRIGGERS_SPEAK = False  # physical safety action fires either way; this only gates the extra LLM round
CLIFF_TRIGGERS_MOVE = True
DANGER_TRIGGERS_MOVE = True
CAUTION_TRIGGERS_MOVE = True
# Playful proximity reactions (fist bump / bullfight + a line): at most one a
# minute -- at 10s, leaning in to talk got "Whoa there!" every few seconds.
REACTION_COOLDOWN_SEC = 60.0
BULLFIGHT_SEQUENCE = ["rub hands", "bullfight"]
FIST_BUMP_SEQUENCE = ["wave hands", "fist bump"]

# --- Sessions -------------------------------------------------------------
# A session lasts until "stop session" / "go to sleep" (common/commands.py).
# Talk over Rocky ("stop", "wait", "hold on") to cut it off mid-reply.
# Untested against live echo -- set OPENBOT_BARGE_IN=0 if Rocky keeps
# interrupting itself through its own speaker.
BARGE_IN_ENABLED = os.environ.get("OPENBOT_BARGE_IN", "1") != "0"

# --- Autonomous "mind" loop -----------------------------------------------
QUIET_HOURS_START = int(os.environ.get("OPENBOT_QUIET_START_H", "21"))
QUIET_HOURS_END = int(os.environ.get("OPENBOT_QUIET_END_H", "8"))
# How often openbot-mind samples sensors/hearing for surprises (common/
# surprise.py). A surprise triggers a reflection right away, rate-limited
# by SURPRISE_MIN_GAP_S; with nothing happening it still reflects every
# REFLECTION_IDLE_INTERVAL_S ("bored" -- a chance to wonder about something).
AWARENESS_INTERVAL_S = float(os.environ.get("OPENBOT_AWARENESS_INTERVAL_S", "2"))
# 60s, was 30: every desk tap and head turn became its own reflection.
SURPRISE_MIN_GAP_S = float(os.environ.get("OPENBOT_SURPRISE_MIN_GAP_S", "60"))
# After a loud-sound surprise, more of them are ignored this long unless at least
# twice as loud -- tapping on the desk made the mind obsess ("4041 loudness... violent impact").
SOUND_COOLDOWN_S = float(os.environ.get("OPENBOT_SOUND_COOLDOWN_S", "300"))
# Greet a known person by name on arrival: the first time each day, or back after this long away.
GREET_AFTER_ABSENCE_S = float(os.environ.get("OPENBOT_GREET_AFTER_ABSENCE_S", "7200"))
REFLECTION_IDLE_INTERVAL_S = float(os.environ.get("OPENBOT_REFLECTION_INTERVAL_S", "300"))
# After the robot itself speaks/gestures, ignore sound and vision
# surprises this long -- otherwise it hears its own voice and reacts to it.
SELF_NOISE_QUIET_S = float(os.environ.get("OPENBOT_SELF_NOISE_QUIET_S", "6"))
# Periodic camera look (common/vision.py) -- a changed scene is a surprise.
VISION_INTERVAL_S = float(os.environ.get("OPENBOT_VISION_INTERVAL_S", "90"))
# Tool steps (look/listen) one reflection may chain before it must act or wait.
MIND_MAX_STEPS = int(os.environ.get("OPENBOT_MIND_MAX_STEPS", "4"))
# How often openbot-mind is actually allowed to speak/gesture, independent
# of how often it reflects -- matches SPARK's own expression anti-flap.
# Was 900s when reflection only ran on a 5-min timer; now that a surprise
# (you walking up) triggers reflection, 15 min of forced silence swallowed
# exactly the reactions that make it feel alive. Raise if it's chatty.
EXPRESSION_COOLDOWN_S = float(os.environ.get("OPENBOT_EXPRESSION_COOLDOWN_S", "120"))
# Rolling "today so far" summary (state/mind/summaries/), rewritten when the
# journal has grown by at least SUMMARY_MIN_NEW_LINES. The durable "dream"
# consolidation into notes by kind runs once a night, in quiet hours.
SUMMARY_INTERVAL_S = float(os.environ.get("OPENBOT_SUMMARY_INTERVAL_S", "1800"))
SUMMARY_MIN_NEW_LINES = int(os.environ.get("OPENBOT_SUMMARY_MIN_NEW_LINES", "5"))

# --- Dashboard --------------------------------------------------------------
DASHBOARD_ENABLED = os.environ.get("OPENBOT_DASHBOARD_ENABLED", "1") != "0"
DASHBOARD_PORT = int(os.environ.get("OPENBOT_DASHBOARD_PORT", "8080"))

# --- Face tracking ----------------------------------------------------------
# Off by default -- vision/tracking.py has a documented live deadlock in
# vilib's own camera thread; flip on once that's hardened with a bounded
# wait around camera_start().
FACE_TRACKING_ENABLED = os.environ.get("OPENBOT_FACE_TRACKING", "0") == "1"

# --- Self-healing (common/health.py) ---------------------------------------
# Needs real margin above the slowest normal phase each service goes
# through -- a local LLM round has been observed taking 30s+, and a custom
# model can run far slower still. If a future model needs more, raise this
# (and each service's systemd WatchdogSec, kept above stale+interval).
WATCHDOG_STALE_SEC = float(os.environ.get("OPENBOT_WATCHDOG_STALE_SEC", "240"))
WATCHDOG_PING_INTERVAL = float(os.environ.get("OPENBOT_WATCHDOG_PING_INTERVAL", "10"))
