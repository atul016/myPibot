"""Speech-to-text speed and accuracy: the Pi's own Whisper (tiny.en, base.en)
vs the Mac's whisper-server (large-v3-turbo), on synthetic sentences.
Measured 2026-10-02: Mac ~0.2s; Pi base.en ~2.2s, tiny.en ~1.1s (with the
camera's face work running, the Pi got ~2x slower -- see Nice= in systemd/).

    cd ~/openbot && python3 -m tests.bench_whisper
"""
import time

import config as cfg
from common import stt
from tests._audio import say, seconds

SENTENCES = [
    "There is a delay between when I say something and when it gets into the loop.",
    "Quiet.",
    "Rocky, what can you see?",
]

engines = [("Mac large-v3-turbo", stt.RemoteWhisper(cfg.LLM_BASE_URL, cfg.MAC_WHISPER_PORT, None))]
for model in ["tiny.en", "base.en"]:
    engines.append((f"Pi {model}", stt.Whisper(model)))

clips = [(text, say(text)) for text in SENTENCES]
for name, engine in engines:
    for text, pcm in clips:
        engine.transcribe(pcm)  # warm up
        t = time.time()
        heard = engine.transcribe(pcm)
        print(f"{name:20} {time.time() - t:5.2f}s for {seconds(pcm):.1f}s audio -> {heard!r}")
