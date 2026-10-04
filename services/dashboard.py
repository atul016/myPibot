"""openbot-dashboard: reads state/*.json (common/state.py, events.py,
sensors.py, health.py) instead of a live Python object reference to
another process -- decoupled so a dashboard-side hang (the vilib camera
deadlock that froze the whole original assistant) can't take wake-word
listening down with it. Deliberately has no `config`/`picarx` import, so it
runs (and is testable) independent of any robot hardware being present.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import Flask, Response, send_file

from common import events as dash_events
from common import agenda, faces, health, jev, journal, memory, objects, sensors, state, stats, vision
from dashboard.page import PAGE

PORT = int(os.environ.get("OPENBOT_DASHBOARD_PORT", "8080"))
HEALTH_COMPONENTS = ["openbot-alive"] * (os.environ.get("OPENBOT_BODY", "none") != "none") + ["openbot-ears", "openbot-wake-listen", "openbot-mind", "openbot-speak", "openbot-camera"] \
    + ["openbot-chat"] * bool(os.environ.get("OPENBOT_CHAT_ALLOW"))
STATE_ROOT = state.STATE_DIR.resolve()

app = Flask(__name__)


def _snapshot() -> dict:
    return {
        "session": state.load_session(),
        "sensors": sensors.read(),
        "health": health.all_status(HEALTH_COMPONENTS),
        "events": dash_events.recent_events(),
    }


@app.route("/")
def index():
    return PAGE


@app.route("/api/state")
def api_state():
    return Response(json.dumps(_snapshot()), mimetype="application/json")


@app.route("/api/stream")
def api_stream():
    def _gen():
        while True:
            yield f"data: {json.dumps(_snapshot())}\n\n"
            time.sleep(0.4)

    return Response(_gen(), mimetype="text/event-stream")


@app.route("/api/mind")
def api_mind():
    """Rocky's inner life at a glance: the Mind tab."""
    import datetime as dt
    session = state.load_session()
    reminders, watches, rest_until = agenda.session_lists()
    now = time.time()
    try:
        reactions = [ln.removeprefix("- [reaction] ") for ln in
                     (memory.MIND_DIR / "lessons" / "what-gets-a-reaction.md").read_text().splitlines()
                     if ln.startswith("- [reaction]")][-8:]
    except OSError:
        reactions = []
    return Response(json.dumps({
        "mood": session.get("mood"),
        "mode": "asleep" if session.get("asleep") else "awake",
        "in_session": bool(session.get("in_session")),
        "who": faces.describe(faces.read()),
        "summary": journal.read_summary(dt.date.today()),
        "goals": agenda.open_goals(agenda.load_goals()),
        "reminders": [{"at": r["at"], "about": r["about"]} for r in reminders],
        "watches": [w for w in watches if w["until"] > now],
        "resting_until": rest_until if rest_until > now else None,
        "reactions": reactions,
        "journal": journal.tail(60),
    }), mimetype="application/json")


@app.route("/api/dreams")
def api_dreams():
    """The Dreams & wishes tab: each night's dream note (what it kept from the
    day), its wishes (self/wishes.md), and the rules it wrote itself about
    what works with people (self/what-works.md)."""
    dreams = []
    for path in sorted((memory.MIND_DIR / "dreams").glob("*.md"), reverse=True)[:7]:
        body = path.read_text(encoding="utf-8", errors="replace").split("---\n", 2)[-1].strip()
        dreams.append({"night": path.stem, "text": body})
    strip = lambda lines: [ln.split("] ", 1)[-1] for ln in lines]
    return Response(json.dumps({
        "dreams": dreams,
        "wishes": strip(memory.read_note("self", "wishes"))[::-1],
        "rules": strip(memory.read_note("self", "what works")),
    }), mimetype="application/json")


@app.route("/api/stats")
def api_stats():
    """The Stats tab: per day for the last week, newest first (common/stats.py)."""
    return Response(json.dumps(stats.week(7)), mimetype="application/json")


@app.route("/api/jev")
def api_jev():
    """The Jev tab: every question asked of Jev (common/jev.py) -- the whole
    request but the key -- and Jev's whole answer, newest first."""
    return Response(json.dumps(jev.recent_calls(50)), mimetype="application/json")


@app.route("/api/vision")
def api_vision():
    """What Rocky last saw: {scene, direction, ts, by_direction, ...}."""
    return Response(json.dumps(vision.last_look()), mimetype="application/json")


@app.route("/api/objects")
def api_objects():
    """The detector's latest boxes, for drawing over the live video -- asking keeps
    openbot-camera detecting on every frame while the Camera tab is open."""
    objects.want_fast()
    return Response(json.dumps(objects.read(max_age_s=2.0)), mimetype="application/json")


@app.route("/api/camera.mjpg")
def api_camera_stream():
    """openbot-camera's live MJPEG, relayed so the page needs only this one
    origin/port (a second port is a separate site to some browsers, and to
    anything tunnelling just the dashboard)."""
    import requests
    try:
        upstream = requests.get(f"{vision.CAMERA_URL}/mjpg", stream=True, timeout=(3, 10))
    except requests.RequestException:
        return Response("camera service unavailable", status=503)
    return Response(upstream.iter_content(chunk_size=16384),
                    mimetype=upstream.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"))


@app.route("/api/camera.jpg")
def api_camera():
    """The latest camera frame openbot-mind captured -- read-only: the
    dashboard never opens the camera itself (it'd contend with mind's own
    captures, and a dashboard-side camera hang is exactly what this split
    exists to prevent)."""
    if not os.path.isfile(vision.SNAPSHOT_PATH):
        return Response("no photo yet", status=404)
    return send_file(vision.SNAPSHOT_PATH, mimetype="image/jpeg", max_age=0)


@app.route("/api/files")
def api_files():
    """Every file openbot's services have generated under state/ -- the
    Files tab's list, read fresh on each request since new files (a
    health record for a service that just started, say) appear over
    time."""
    if not STATE_ROOT.is_dir():
        return Response(json.dumps([]), mimetype="application/json")
    names = sorted(str(p.relative_to(STATE_ROOT)) for p in STATE_ROOT.rglob("*") if p.is_file())
    return Response(json.dumps(names), mimetype="application/json")


def _within_state_root(path: Path) -> bool:
    try:
        path.relative_to(STATE_ROOT)
        return True
    except ValueError:
        return False


@app.route("/api/files/<path:name>")
def api_file_content(name):
    # Resolve and re-check containment -- name comes from the URL, and
    # Flask's <path:> converter allows "../" segments through unless this
    # is verified after resolving, not just string-prefix-checked before.
    target = (STATE_ROOT / name).resolve()
    if not _within_state_root(target) or not target.is_file():
        return Response(json.dumps({"error": "not found"}), status=404, mimetype="application/json")
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype="application/json")
    return Response(json.dumps({"name": name, "content": content}), mimetype="application/json")


def main() -> None:
    health.start_watchdog("openbot-dashboard", 60.0, 10.0)
    health.record_success("openbot-dashboard")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
