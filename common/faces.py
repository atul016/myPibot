"""Faces: who's in front of Rocky, and where.

openbot-camera runs FaceEngine on its own frames (~5/s) and publishes
state/faces.json; alive turns the head toward the biggest face
(head_step), the mind notices arrivals/departures, and conversations know
who they're talking to. Models are OpenCV's YuNet (detect, ~40ms on the Pi
5) + SFace (recognize, ~25ms), downloaded by setup.sh into
~/.openbot-models -- OpenCV 4.10 has both built in, no new dependency.

Known faces are a few 128-number SFace embeddings per name in
state/faces/<name>.npy -- local only, never leave the robot. A name is
learned when someone says "my name is X" while their face is visible
(services/wake_listen.py).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .state import STATE_DIR, atomic_write
from .persona import CURRENT

FACES_PATH = STATE_DIR / "faces.json"
KNOWN_DIR = STATE_DIR / "faces"
MODEL_DIR = Path(os.path.expanduser(os.environ.get("OPENBOT_MODEL_DIR", "~/.openbot-models")))
MATCH_COSINE = 0.363      # SFace's published cosine threshold for "same person"
MAX_SAMPLES = 20          # embeddings kept per person
STALE_S = 2.0             # faces.json older than this = camera/detector not running
# CPU budget: Whisper (speech-to-text) shares these 4 cores, and full-rate face
# work at full resolution took ~67% CPU and doubled Whisper's time. Detect on a
# half-size decode every frame; run the costlier recognition at most this often.
RECOGNIZE_EVERY_S = 1.0
# Identity sticks to a TRACKED face: once recognized, a face keeps its name while
# it stays in view (brief detector misses up to TRACK_GAP_S are bridged), even
# when a single check fails -- turning your head or holding up a phone used to
# make you "someone you don't recognize" and churn "Atul left"/"Atul is here".
TRACK_GAP_S = 2.0
TRACK_MOVE = 0.4  # fraction of the frame a face may move between checks and still be "the same face"

# Head tracking (alive). ov5647 at 640x480 sees roughly 54 x 41 degrees.
FOV_DEG = (54.0, 41.0)
DEADBAND = 0.08           # face within 8% of centre -> don't move (servo jitter)
GAIN = 0.5                # fraction of the error corrected per step (damping)
MAX_STEP_DEG = 8.0
PAN_LIMIT, TILT_LIMIT = 60.0, 30.0


def name_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


class FaceEngine:
    def __init__(self, width: int = 640, height: int = 480):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"), "",
                                             (width, height), 0.7)
        self.rec = cv2.FaceRecognizerSF.create(str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"), "")
        self.known: dict[str, "object"] = {}
        self._known_sig: list = []
        self._last_recognized = 0.0
        self._prev: list[tuple[float, float, str | None, float]] = []  # (cx, cy, name, similarity)
        self._prev_seen = 0.0
        self._embedding = None

    def _reload_known(self) -> None:
        import numpy as np
        files = sorted(KNOWN_DIR.glob("*.npy"))
        sig = [(p.name, p.stat().st_mtime) for p in files]  # names too: a rename or delete keeps the newest mtime
        if sig != self._known_sig:
            self.known = {p.stem: np.load(p) for p in files}
            self._known_sig = sig

    def analyze(self, jpeg: bytes) -> dict:
        """{w, h, faces: [{box: [x,y,w,h], score, name|None, similarity}], embedding (biggest face)}"""
        import time
        import numpy as np
        buf = np.frombuffer(jpeg, np.uint8)
        small = self.cv2.imdecode(buf, self.cv2.IMREAD_REDUCED_COLOR_2)  # JPEG decoded straight to half size
        h2, w2 = small.shape[:2]
        self.det.setInputSize((w2, h2))
        _, raw = self.det.detect(small)
        found = sorted(raw if raw is not None else [], key=lambda f: -f[2] * f[3])[:3]
        rows = []
        for f in found:  # back to full-resolution coordinates (box + 5 landmarks)
            f = f.copy()
            f[:14] *= 2
            rows.append(f)

        recognize = bool(rows) and time.time() - self._last_recognized >= RECOGNIZE_EVERY_S
        if recognize:
            self._reload_known()
            full = self.cv2.imdecode(buf, self.cv2.IMREAD_COLOR)
            self._last_recognized = time.time()
        faces, prev = [], []
        for i, f in enumerate(rows):
            box = [int(v) for v in f[:4]]
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            tracked = same_face(self._prev, cx, cy, w2 * 2, h2 * 2)
            if recognize:
                emb = self.rec.feature(self.rec.alignCrop(full, f))[0]
                name, sim = best_match(emb, self.known)
                if name is None and tracked and tracked[2]:
                    name = tracked[2]  # same face still in view: a failed check isn't a new person
                if i == 0:
                    self._embedding = emb
            else:  # between recognitions: whatever this tracked face was last time
                name, sim = (tracked[2], tracked[3]) if tracked else (None, 0.0)
            prev.append((cx, cy, name, sim))
            faces.append({"box": box, "score": round(float(f[-1]), 2), "name": name, "similarity": round(sim, 3)})
        now = time.time()
        if rows:
            self._prev, self._prev_seen = prev, now
        elif now - self._prev_seen > TRACK_GAP_S:
            self._prev = []  # gone long enough: whoever comes next is recognized afresh
        if not rows:
            self._embedding = None
        emb = self._embedding
        return {"w": w2 * 2, "h": h2 * 2, "faces": faces,
                "embedding": [round(float(x), 5) for x in emb] if emb is not None else None}


def same_face(prev: list, cx: float, cy: float, w: float, h: float):
    """The previously seen face (cx, cy, name, similarity) this one continues,
    if any: the nearest one that moved less than TRACK_MOVE of the frame."""
    near = min(prev, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2, default=None)
    if near and abs(near[0] - cx) < w * TRACK_MOVE and abs(near[1] - cy) < h * TRACK_MOVE:
        return near
    return None


def best_match(embedding, known: dict) -> tuple[str | None, float]:
    """(name, cosine similarity) of the closest known person, name None if
    nobody is close enough."""
    import numpy as np
    e = np.asarray(embedding, dtype=np.float32)
    e = e / (np.linalg.norm(e) + 1e-9)
    best, best_sim = None, 0.0
    for name, samples in known.items():
        s = np.asarray(samples, dtype=np.float32).reshape(-1, e.size)
        sims = (s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)) @ e
        if sims.max() > best_sim:
            best, best_sim = name, float(sims.max())
    return (best if best_sim >= MATCH_COSINE else None), best_sim


def publish(result: dict) -> None:
    atomic_write(FACES_PATH, json.dumps({**result, "ts": time.time()}))


def read() -> dict:
    """Latest faces.json, or {} if missing/stale."""
    try:
        data = json.loads(FACES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if time.time() - data.get("ts", 0) < STALE_S else {}


def visible_names(data: dict) -> tuple[list[str], int]:
    """(display names of recognized people, count of unrecognized faces)."""
    faces = data.get("faces", [])
    names = sorted({f["name"] for f in faces if f.get("name")})
    return [n.replace("-", " ").title() for n in names], sum(1 for f in faces if not f.get("name"))


def speaker(data: dict) -> str | None:
    """Who is talking: the one face in view, if recognized. With two faces
    or a stranger it's a guess, so None -- route 1 of voice-to-person (the
    face stands in for the voice); speaker-ID from audio is the upgrade."""
    faces = data.get("faces", [])
    if len(faces) == 1 and faces[0].get("name"):
        return faces[0]["name"].replace("-", " ").title()
    return None


def describe(data: dict) -> str | None:
    names, strangers = visible_names(data)
    if not names and not strangers:
        return None
    parts = names + ([f"{strangers} person{'s' if strangers > 1 else ''} you don't recognize"] if strangers else [])
    return ", ".join(parts)


def enroll(name: str, embedding: list[float]) -> int:
    """Adds one face sample for `name`; returns how many samples it now has."""
    import numpy as np
    KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    path = KNOWN_DIR / f"{name_slug(name)}.npy"
    samples = np.load(path) if path.exists() else np.zeros((0, len(embedding)), dtype=np.float32)
    samples = np.vstack([samples, np.asarray(embedding, dtype=np.float32)])[-MAX_SAMPLES:]
    np.save(path, samples)
    return len(samples)


# Phrase case-insensitive, name case-SENSITIVE (scoped (?i:...) flag).
NAME_INTRO = re.compile(r"\b(?i:my name is|call me|i am|i'm)\s+([A-Z][a-z]+)\b")


def introduced_name(transcript: str) -> str | None:
    """"My name is Atul" / "I'm Atul" -> "Atul". Case-sensitive on the name:
    Whisper capitalizes names, not "I am feeling tired"."""
    m = NAME_INTRO.search(transcript)
    return m.group(1) if m and m.group(1).lower() not in {CURRENT, "not", "just", "here", "back", "sorry"} else None


# "Not Ana, Anna." / "Her name is not Ana. Her name is Anna." -- fixes a misheard introduction.
CORRECTION = re.compile(r"\b(?i:not)\s+([A-Z][a-z]+)\W+(?:(?i:it['’]?s|that['’]?s|i['’]?m|(?:her|his|my) name is)\s+)?"
                        r"([A-Z][a-z]+)\b")
JUST_LEARNED = 3  # samples: only a name this new can be renamed by voice -- "Not Atul, Daddy!" must never rename Atul


def corrected_name(transcript: str) -> tuple[str, str] | None:
    """(old, new) if `transcript` corrects a name only just learned -- and renames its face file."""
    import numpy as np
    m = CORRECTION.search(transcript)
    if not m or name_slug(m[2]) in (CURRENT, name_slug(m[1])):  # "It's not Aria, it's Aria."
        return None
    src, dst = KNOWN_DIR / f"{name_slug(m[1])}.npy", KNOWN_DIR / f"{name_slug(m[2])}.npy"
    if not src.exists() or dst.exists() or len(np.load(src)) > JUST_LEARNED:
        return None
    os.replace(src, dst)
    return m[1], m[2]


def head_step(face_box: list[int], frame_wh: tuple[int, int], pan: float, tilt: float) -> tuple[float, float] | None:
    """Next (pan, tilt) to bring the face toward the centre, or None if it's
    already close enough. Pan + = right, tilt + = up (picarx convention);
    image y grows downward."""
    x, y, w, h = face_box
    ex = (x + w / 2) / frame_wh[0] - 0.5
    ey = (y + h / 2) / frame_wh[1] - 0.5
    if abs(ex) < DEADBAND and abs(ey) < DEADBAND:
        return None

    def step(err: float, fov: float) -> float:
        return max(-MAX_STEP_DEG, min(MAX_STEP_DEG, err * fov * GAIN))

    new_pan = max(-PAN_LIMIT, min(PAN_LIMIT, pan + (step(ex, FOV_DEG[0]) if abs(ex) >= DEADBAND else 0)))
    new_tilt = max(-TILT_LIMIT, min(TILT_LIMIT, tilt - (step(ey, FOV_DEG[1]) if abs(ey) >= DEADBAND else 0)))
    return round(new_pan, 1), round(new_tilt, 1)


def person_target(box: list[float], pan: float, tilt: float) -> tuple[float, float]:
    """Where to point the head at a person the object detector sees but whose face it doesn't:
    the middle of their box across, near its top -- where the face is (from the floor it's often
    above the frame, so the head tips up toward it). box: [left, top, right, bottom], fractions."""
    left, top, right, bottom = box
    ex, ey = (left + right) / 2 - 0.5, top + 0.1 * (bottom - top) - 0.5
    return (round(max(-PAN_LIMIT, min(PAN_LIMIT, pan + ex * FOV_DEG[0])), 1),
            round(max(-TILT_LIMIT, min(TILT_LIMIT, tilt - ey * FOV_DEG[1])), 1))


def demo() -> None:
    import shutil, tempfile
    import numpy as np
    assert person_target([0.4, 0.0, 0.6, 1.0], 0, 0) == (0.0, 16.4)      # straight ahead, head cut off: tip up
    assert person_target([0.7, 0.2, 0.9, 1.0], 10, 5) == (26.2, 14.0)    # to the right
    assert person_target([0.9, 0.0, 1.0, 1.0], 50, 0)[0] == PAN_LIMIT    # past the servo's reach: as far as it goes
    assert speaker({"faces": [{"name": "atul-p"}]}) == "Atul P"
    assert speaker({"faces": [{"name": "atul"}, {}]}) is None  # a stranger too: whose voice? a guess
    assert speaker({"faces": [{}]}) is None and speaker({}) is None
    global KNOWN_DIR, FACES_PATH
    orig = (KNOWN_DIR, FACES_PATH)
    test_dir = Path(tempfile.mkdtemp())
    KNOWN_DIR, FACES_PATH = test_dir / "faces", test_dir / "faces.json"
    try:
        a = np.zeros(128, dtype=np.float32); a[0] = 1
        b = np.zeros(128, dtype=np.float32); b[1] = 1
        assert enroll("Atul", list(a)) == 1 and enroll("Atul", list(a + 0.1)) == 2
        known = {p.stem: np.load(p) for p in KNOWN_DIR.glob("*.npy")}
        assert best_match(a, known)[0] == "atul"
        assert best_match(b, known)[0] is None  # orthogonal face = stranger
        publish({"w": 640, "h": 480, "faces": [{"box": [0, 0, 9, 9], "name": "atul"}, {"box": [1, 1, 5, 5], "name": None}]})
        assert describe(read()) == "Atul, 1 person you don't recognize"
        assert describe({}) is None
        assert introduced_name("Hi, my name is Atul.") == "Atul" and introduced_name("I'm Priya") == "Priya"
        assert introduced_name("I am feeling tired") is None and introduced_name(f"I'm {CURRENT.capitalize()}'s friend") is None
        assert introduced_name("This is Anna") is None  # might be someone else's face
        enroll("Ana", list(b))  # a misheard introduction...
        assert corrected_name("Also her name is not Ana. Her name is Anna.") == ("Ana", "Anna")
        assert (KNOWN_DIR / "anna.npy").exists() and not (KNOWN_DIR / "ana.npy").exists()
        assert corrected_name("It's not Anna, it's Anna.") is None and corrected_name("Not Ravi, Sam.") is None
        for _ in range(JUST_LEARNED):
            enroll("Atul", list(a))
        assert corrected_name("Not Atul, Daddy!") is None and (KNOWN_DIR / "atul.npy").exists()  # familiar: never renamed
        # face right of centre -> pan right; below centre -> tilt down; centred -> no move
        assert head_step([500, 200, 80, 80], (640, 480), 0, 0)[0] > 0
        assert head_step([280, 400, 80, 60], (640, 480), 0, 0)[1] < 0
        assert head_step([280, 200, 80, 80], (640, 480), 0, 0) is None
        assert head_step([630, 200, 10, 10], (640, 480), 59, 0)[0] == PAN_LIMIT  # clamped
        prev = [(320.0, 240.0, "atul", 0.5)]
        assert same_face(prev, 340, 250, 640, 480)[2] == "atul"   # moved a little: same face
        assert same_face(prev, 600, 250, 640, 480) is None         # other side of the frame: not them
    finally:
        KNOWN_DIR, FACES_PATH = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("faces: ok")
