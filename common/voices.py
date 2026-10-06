"""Voices: who's talking, by voice -- WeSpeaker's ResNet34 speaker embeddings
(trained on VoxCeleb), learned like faces: while Rocky sees exactly one face it
recognizes, what that person says adds to their voice samples (state/voices/
<name>.npy); when no face says who's talking, the voice can. Features follow
WeSpeaker's own: Kaldi-style 80-bin log-mel fbank, 25/10 ms Hamming frames on
16 kHz audio at int16 scale, then the mean over time subtracted.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .faces import MODEL_DIR, name_slug
from .state import STATE_DIR

MODEL = MODEL_DIR / "voxceleb_resnet34_LM.onnx"
VOICES_DIR = STATE_DIR / "voices"
RATE = 16000
MATCH = 0.55       # cosine similarity to call it that person's voice -- a calibration knob for this mic and room
MAX_SAMPLES = 20   # per person, newest kept
MIN_SECONDS = 1.0  # shorter than this, a voice print is a guess
NEAR_FACE = 0.12   # the one face's width as a fraction of the frame: closer than this, the voice is theirs
OBJECTS_FRESH_S = 5.0  # detector boxes older than this don't say how many people are in view
# "I'm not Atul", "I am not Atul", "I'm Atul's wife": the voice just learned as theirs wasn't.
NOT_ME = re.compile(r"\b(?i:i'?m not|i am not)\s+([A-Z][a-z]+)\b|\b(?i:i'?m|i am)\s+([A-Z][a-z]+)['’]s\s+"
                    r"(?i:wife|husband|partner|son|daughter|mother|father|mom|dad|brother|sister|friend)\b")


def _mel_banks(bins: int = 80, n_fft: int = 512, low: float = 20.0, high: float = RATE / 2) -> np.ndarray:
    mel = lambda f: 1127.0 * np.log1p(np.asarray(f) / 700.0)  # noqa: E731 -- Kaldi's mel scale
    edges = np.linspace(mel(low), mel(high), bins + 2)
    freqs = mel(np.arange(n_fft // 2) * RATE / n_fft)  # Kaldi leaves out the Nyquist bin
    left, centre, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up, down = (freqs - left) / (centre - left), (right - freqs) / (right - centre)
    return np.pad(np.maximum(0.0, np.minimum(up, down)), ((0, 0), (0, 1)))


_BANKS = _mel_banks()
_WINDOW = np.hamming(400)


def fbank(pcm16k: bytes) -> np.ndarray:
    """16 kHz 16-bit mono PCM -> (frames, 80) log-mel, mean-normalized over time."""
    x = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float64)
    n = 1 + (len(x) - 400) // 160
    if n < 1:
        return np.zeros((0, 80), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:n].copy()
    frames -= frames.mean(axis=1, keepdims=True)                           # remove DC offset
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()                          # pre-emphasis
    frames[:, 0] *= 1 - 0.97
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=512)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


class VoiceEngine:
    def __init__(self) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

    def embed(self, pcm16k: bytes) -> np.ndarray | None:
        """A unit-length voice print, or None for too little speech."""
        if len(pcm16k) < MIN_SECONDS * RATE * 2:
            return None
        e = self.session.run(None, {"feats": fbank(pcm16k)[None]})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


def enroll(name: str, emb: np.ndarray) -> int:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    path = VOICES_DIR / f"{name_slug(name)}.npy"
    samples = np.load(path) if path.exists() else np.zeros((0, len(emb)))
    samples = np.vstack([samples, emb])[-MAX_SAMPLES:]
    np.save(path, samples)
    return len(samples)


def can_teach(faces_data: dict, boxes: list[dict], boxes_ts: float) -> str | None:
    """Whose voice this is for sure -- the one known face in view, when the detector (fresh) also
    sees exactly one person and the face is near. Two people sitting together taught Rocky his
    wife's voice as Atul's (2026-10-04). None: don't learn from this."""
    faces = faces_data.get("faces", [])
    if len(faces) != 1 or not faces[0].get("name") or time.time() - boxes_ts > OBJECTS_FRESH_S:
        return None
    if sum(1 for o in boxes if o.get("name") == "person") != 1:
        return None
    if faces[0]["box"][2] < NEAR_FACE * faces_data.get("w", 640):
        return None
    return faces[0]["name"].replace("-", " ").title()


def speaker(face: str | None, heard: str | None, boxes: list[dict], boxes_ts: float) -> str | None:
    """Who's talking when this voice isn't one to learn from (can_teach said no): the voice's own match,
    else the one face in view -- unless more than one person is there: then it needn't be the one talking
    (his wife got called Atul, 2026-10-04). Alone, near or far, the face says who even before the voice
    print matches (prints taught on another mic score low)."""
    crowd = time.time() - boxes_ts < OBJECTS_FRESH_S and sum(o.get("name") == "person" for o in boxes) > 1
    return heard or (None if crowd else face)


def denied_name(text: str) -> str | None:
    """The known name `text` says this speaker is not -- "No, I'm not Atul" -> "Atul"."""
    m = NOT_ME.search(text)
    return (m.group(1) or m.group(2)) if m else None


def forget_last(name: str) -> bool:
    """Drops the newest voice sample taught for `name` (it was someone else's); False if there was none."""
    path = VOICES_DIR / f"{name_slug(name)}.npy"
    if not path.exists():
        return False
    samples = np.load(path)[:-1]
    if len(samples):
        np.save(path, samples)
    else:
        path.unlink()
    return True


def identify(emb: np.ndarray) -> tuple[str | None, float]:
    """(the best-matching known voice's name, or None below MATCH; its similarity)."""
    best, score = None, 0.0
    for path in VOICES_DIR.glob("*.npy"):
        s = float(np.max(np.load(path) @ emb))
        if s > score:
            best, score = path.stem.replace("-", " ").title(), s
    return (best if score >= MATCH else None), score


def demo() -> None:
    import shutil, tempfile
    from pathlib import Path
    global VOICES_DIR
    banks = _mel_banks()
    assert banks.shape == (80, 257) and banks[:, -1].sum() == 0 and (banks.sum(axis=1) > 0).all()
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 3000, RATE * 2)).astype(np.int16).tobytes()
    f = fbank(noise)
    assert f.shape == (198, 80) and abs(float(f.mean())) < 1e-3  # 2 s -> 198 frames, mean-normalized
    orig, test_dir = VOICES_DIR, Path(tempfile.mkdtemp())
    VOICES_DIR = test_dir
    try:
        a, b = np.eye(4)[0], np.eye(4)[1]
        assert enroll("Atul", a) == 1 and identify(a) == ("Atul", 1.0)
        assert identify(b)[0] is None  # a voice nobody taught it
        enroll("Atul", b)  # the wife's voice, learned as Atul's while they sat together
        assert identify(b) == ("Atul", 1.0)
        assert denied_name("No, I'm not Atul. I'm his wife") == "Atul" == denied_name("I am Atul's wife")
        assert denied_name("I'm not sure") is None and denied_name("I'm Atul") is None
        assert forget_last("Atul") and identify(b)[0] is None and identify(a) == ("Atul", 1.0)
        assert forget_last("Atul") and not forget_last("Atul") and identify(a)[0] is None  # none left: file gone
        one = {"w": 640, "faces": [{"name": "atul", "box": [100, 100, 120, 120]}]}
        now = time.time()
        assert can_teach(one, [{"name": "person"}], now) == "Atul"
        assert can_teach(one, [{"name": "person"}, {"name": "person"}], now) is None  # two people: whose voice?
        assert can_teach(one, [{"name": "person"}], now - 10) is None  # stale boxes: can't tell how many
        assert can_teach({"w": 640, "faces": [{"name": "atul", "box": [100, 100, 40, 40]}]}, [{"name": "person"}], now) is None
        assert can_teach({"w": 640, "faces": [{"name": "atul", "box": [0, 0, 120, 120]}, {"name": None, "box": [1, 1, 99, 99]}]},
                         [{"name": "person"}], now) is None
        two = [{"name": "person"}, {"name": "person"}]
        assert speaker("Atul", None, two, now) is None              # two people, a voice it doesn't know: no guess
        assert speaker("Atul", "Anna", two, now) == "Anna"          # the voice knows better than the face
        assert speaker("Atul", None, [{"name": "person"}], now) == "Atul"  # alone (or far): the face says who
        assert speaker("Atul", None, [], now) == "Atul" and speaker("Atul", None, two, now - 10) == "Atul"  # missed / stale
        assert speaker(None, None, [], now) is None
    finally:
        VOICES_DIR = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("voices: ok")
