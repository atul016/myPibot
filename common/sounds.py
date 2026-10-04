"""Sounds: what a second of audio was -- Google's YAMNet (AudioSet's 521 kinds:
knock, doorbell, dog, speech, music, glass breaking...). openbot-ears runs it
on every second it hears (services/ears.py), and the mind asks ears what each
second sounded like. TFLite via ai-edge-litert.
"""
from __future__ import annotations

import csv

from .faces import MODEL_DIR

MODEL = MODEL_DIR / "yamnet.tflite"
NAMES = MODEL_DIR / "yamnet_class_map.csv"
RATE, WINDOW = 16000, 15600  # YAMNet hears 0.975 s of 16 kHz mono


class Classifier:
    def __init__(self) -> None:
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(MODEL))
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0]["index"], self.it.get_output_details()[0]["index"]
        with open(NAMES, encoding="utf-8") as f:
            self.names = [row[2] for row in list(csv.reader(f))[1:]]  # index,mid,display_name

    def classify(self, pcm16k: bytes) -> list[tuple[str, float]]:
        """16 kHz 16-bit mono PCM (its last WINDOW samples) -> the top 3 (name, score)."""
        import numpy as np
        x = np.frombuffer(pcm16k, dtype=np.int16)[-WINDOW:].astype(np.float32) / 32768.0
        x = np.pad(x, (WINDOW - len(x), 0))
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        scores = self.it.get_tensor(self.out)[0]
        return [(self.names[i], float(scores[i])) for i in scores.argsort()[::-1][:3]]


def demo() -> None:
    import math
    import struct
    if MODEL.exists() and NAMES.exists():  # a 1 kHz tone: YAMNet should hear a sine wave / beep
        tone = struct.pack(f"<{RATE}h", *(int(12000 * math.sin(2 * math.pi * 1000 * i / RATE)) for i in range(RATE)))
        top = Classifier().classify(tone)
        assert any(n in ("Sine wave", "Beep, bleep", "Whistle", "Tuning fork") for n, _ in top), top


if __name__ == "__main__":
    demo()
    print("sounds: ok")
