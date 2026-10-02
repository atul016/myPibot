"""Vision costs, on live camera frames (needs openbot-camera running):
  - LLM time to first word with vs without a photo (every conversation turn
    sends one; measured +0.1-0.2s here, ~0.4s with a full conversation prompt)
  - face detection + recognition per frame (measured ~40ms + ~25ms)

    cd ~/openbot && python3 -m tests.bench_vision
"""
import time

import config as cfg
from common import cognition, faces, vision

QUESTION = "What am I holding in my hand? Answer in one short sentence."
for label in ["no photo", "with photo"] * 2:
    jpeg = vision.capture() if label == "with photo" else None
    t, first, text = time.time(), None, ""
    for delta in cognition.stream(cfg.LLM_BASE_URL, cfg.LLM_MODEL, QUESTION,
                                  system="You are Rocky, a small desk robot.", image_jpeg=jpeg):
        first = first or time.time() - t
        text += delta
    print(f"{label:10} first word {first or float('nan'):.2f}s -> {text.strip()[:80]!r}")

engine = faces.FaceEngine()
for i in range(5):
    jpeg = vision.capture()
    engine._last_recognized = 0.0  # force the (costlier) recognition step every frame for the measurement
    t = time.time()
    result = engine.analyze(jpeg)
    print(f"frame {i}: {len(result['faces'])} face(s) {[f['name'] for f in result['faces']]} in {(time.time() - t) * 1000:.0f}ms")
    time.sleep(0.3)
