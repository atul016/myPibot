"""Voices: who's talking, by voice -- WeSpeaker's ResNet34 speaker embeddings
(trained on VoxCeleb), learned like faces: while Rocky sees exactly one face it
recognizes, what that person says adds to their voice samples (state/voices/
<name>.npy); when no face says who's talking, the voice can. Features follow
WeSpeaker's own: Kaldi-style 80-bin log-mel fbank, 25/10 ms Hamming frames on
16 kHz audio at int16 scale, then the mean over time subtracted.
"""
from __future__ import annotations

import numpy as np

from .faces import MODEL_DIR, name_slug
from .state import STATE_DIR

MODEL = MODEL_DIR / "voxceleb_resnet34_LM.onnx"
VOICES_DIR = STATE_DIR / "voices"
RATE = 16000
MATCH = 0.55       # cosine similarity to call it that person's voice -- a calibration knob for this mic and room
MAX_SAMPLES = 20   # per person, newest kept
MIN_SECONDS = 1.0  # shorter than this, a voice print is a guess


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
    finally:
        VOICES_DIR = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("voices: ok")
