"""Rocky's voice: how long before the first sound (Piper + the Eridian chord
layer) for a short and a long sentence. Nothing is played. Needs root -- the
Piper voice lives where openbot-speak (root) downloaded it.
Measured 2026-10-02: 0.15s short, 0.9s for an 80-character sentence; the
speaker amp's ~0.55s switch-on is hidden by speak_client.warm().

    cd ~/openbot && sudo SDL_AUDIODRIVER=dummy python3 -m tests.bench_tts
"""
import time

from picarx.tts import Piper
from robot_hat import disable_speaker

from personas.rocky import voice

tts = Piper()
tts.set_model("en_US-lessac-low")
disable_speaker()  # constructing Piper switches the amp on -- this bench plays nothing

for text in ["Hey there!", "I see a person with dark hair, sitting in front of a desk, with a computer mouse."]:
    voice._mix_sentence(text, tts.piper)  # warm up
    t = time.time()
    mix = voice._mix_sentence(text, tts.piper)
    total = time.time() - t
    t = time.time()
    b"".join(chunk.audio_int16_bytes for chunk in tts.piper.synthesize(text))
    piper_only = time.time() - t
    print(f"{len(text):3d} chars: first sound after {total:.2f}s (Piper {piper_only:.2f}s, chord layer "
          f"{total - piper_only:.2f}s); plays {len(mix) / voice.SAMPLE_RATE:.1f}s")
