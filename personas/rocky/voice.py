"""Rocky's "Eridian chord language" -- speaks by playing short polyphonic
musical chords instead of words, matching Project Hail Mary's in-fiction
Rocky. Ported from
https://github.com/lahirumaramba/rocky/blob/main/synthesizer/rocky.py --
credit there for the lexicon/chord design. Persona-specific TTS overlay,
loaded only when the `rocky` persona is active (openbot-speak calls
`speak_rocky_overlay` as the persona's `speak_overlay` hook).

pygame's mixer is lazily inited and torn down PER CALL, not held open for
the whole process -- see speak_rocky_overlay()'s docstring. Caller (openbot-
speak) brackets enable_speaker()/disable_speaker() around this same as
Piper -- that's the robot_hat amp-power gate, independent of which audio
backend (Piper's own vs. this module's SDL/pygame) is doing playback.
"""
import hashlib
import re
import threading
import time

import numpy as np
import pygame

SAMPLE_RATE = 44100

_SOUND_CACHE: dict = {}


def _ensure_mixer() -> None:
    if pygame.mixer.get_init() is None:
        # 8192-sample buffer (~186ms), not pygame's 512-sample default
        # (~12ms) -- a 12ms buffer left almost no slack against normal
        # scheduling jitter between this project's other threads (watchdog
        # notifier, dashboard, STT) and caused audible ALSA underruns on
        # live hardware. Raise further (16384+) before assuming a recurring
        # underrun is something else -- USB audio devices can have their
        # own fixed period/latency quirks a software buffer can't fully
        # paper over.
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=8192)


def _release_mixer() -> None:
    if pygame.mixer.get_init() is not None:
        pygame.mixer.quit()
    _SOUND_CACHE.clear()


def _render_chord_wave(frequencies: list, duration: float, volume: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.zeros_like(t)
    envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))
    for f in frequencies:
        wave += (1.0 / len(frequencies)) * np.sin(2 * np.pi * f * t)
    wave *= envelope * volume
    return wave.astype(np.float32)


def generate_chord_sound(frequencies: list, duration: float = 0.3, volume: float = 0.5):
    cache_key = (tuple(sorted(frequencies)), duration, volume)
    if cache_key in _SOUND_CACHE:
        return _SOUND_CACHE[cache_key]
    audio_data = (_render_chord_wave(frequencies, duration, volume) * 32767).astype(np.int16)
    stereo_data = np.ascontiguousarray(np.column_stack((audio_data, audio_data)))
    sound = pygame.sndarray.make_sound(stereo_data)
    _SOUND_CACHE[cache_key] = sound
    return sound


def play_chord(frequencies: list, duration: float = 0.3, volume: float = 0.5) -> None:
    generate_chord_sound(frequencies, duration, volume).play()
    time.sleep(duration + 0.05)


ERIDIAN_LEXICON = {
    "amaze": [659.25, 830.61, 987.77],
    "happy": [783.99, 987.77, 1174.66],
    "yes": [523.25, 659.25, 783.99],
    "fist": [523.25, 659.25, 783.99],
    "bad": [220.00, 233.08, 277.18],
    "sad": [293.66, 349.23, 440.00],
    "sleep": [261.63, 311.13, 392.00],
    "danger": [698.46, 740.00, 783.99],
    "no": [349.23, 370.00, 392.00],
    "question": [440.00, 466.16],
    "grace": [493.88, 622.25, 739.99],
    "friend": [440.00, 554.37, 659.25],
    "astrophage": [880.00, 932.33, 987.77],
    "rocky": [
        [349.23, 440.00, 523.25], [440.00, 554.37, 659.25], [523.25, 659.25, 783.99],
        [587.33, 739.99, 880.00], [659.25, 830.61, 987.77], [783.99, 987.77, 1174.66],
    ],
}


def word_to_chord(word: str) -> list:
    hash_val = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return [200 + (hash_val % 700), 200 + ((hash_val >> 8) % 700), 200 + ((hash_val >> 16) % 700)]


def _word_chord(clean_word: str, is_exclamation: bool) -> tuple[list, float, float]:
    chord = ERIDIAN_LEXICON.get(clean_word) or word_to_chord(clean_word)
    duration, volume = 0.3, 0.5
    if is_exclamation or clean_word in ("amaze", "danger"):
        duration, volume = 0.2, 0.8
    return chord, duration, volume


def pre_warm_cache() -> None:
    for frequencies in ERIDIAN_LEXICON.values():
        if isinstance(frequencies[0], list):
            for chord in frequencies:
                generate_chord_sound(chord, duration=0.15, volume=0.5)
        else:
            generate_chord_sound(frequencies, duration=0.3, volume=0.5)
            generate_chord_sound(frequencies, duration=0.2, volume=0.8)
    generate_chord_sound(ERIDIAN_LEXICON["question"], duration=0.4, volume=0.7)


def _render_chord_bed(text: str) -> np.ndarray:
    segments = []
    words = text.lower().replace(",", "").replace(".", "").split()
    for word in words:
        is_question = "?" in word
        clean_word = word.replace("?", "").replace("!", "")
        if not clean_word:
            continue
        chord, duration, volume = _word_chord(clean_word, "!" in word)
        if isinstance(chord[0], list):
            for sequence_chord in chord:
                segments.append(_render_chord_wave(sequence_chord, 0.15, volume))
        else:
            segments.append(_render_chord_wave(chord, duration, volume))
        if is_question:
            segments.append(np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32))
            segments.append(_render_chord_wave(ERIDIAN_LEXICON["question"], 0.4, 0.7))
    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(segments)


def _resample(wave: np.ndarray, from_rate: int, target_len: int) -> np.ndarray:
    src_len = len(wave)
    if src_len == 0 or target_len <= 0:
        return np.zeros(max(target_len, 0), dtype=np.float32)
    src_x = np.linspace(0, 1, src_len, endpoint=False)
    dst_x = np.linspace(0, 1, target_len, endpoint=False)
    return np.interp(dst_x, src_x, wave).astype(np.float32)


def _split_sentences(text: str) -> list:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def _mix_sentence(text: str, piper_voice) -> np.ndarray:
    english_rate = piper_voice.config.sample_rate
    pcm_bytes = b"".join(chunk.audio_int16_bytes for chunk in piper_voice.synthesize(text))
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    english_i16 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    english = _resample(english_i16, english_rate, round(len(english_i16) * SAMPLE_RATE / english_rate))

    chord_bed = _render_chord_bed(text)
    chord_bed = _resample(chord_bed, SAMPLE_RATE, len(english))

    mix = english * 0.9 + chord_bed * 0.4
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak
    return mix


def _play_mix(mix: np.ndarray, should_stop=lambda: False) -> bool:
    """Plays one sentence; returns False if cut off by should_stop()
    (barge-in) -- polled every 50ms, so a stop lands within ~50ms."""
    if mix.size == 0:
        return True
    audio_data = (mix * 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack((audio_data, audio_data)))
    pygame.sndarray.make_sound(stereo).play()
    end = time.monotonic() + len(mix) / SAMPLE_RATE + 0.05
    while time.monotonic() < end:
        if should_stop():
            pygame.mixer.stop()
            return False
        time.sleep(0.05)
    return True


def speak_rocky_overlay(text: str, piper_voice, should_stop=lambda: False) -> None:
    """Speaks `text` as Piper English narration with the Eridian chord
    translation playing underneath, mixed per sentence and pipelined:
    sentence N+1 synthesizes on a background thread while sentence N
    plays, so only the first sentence's synthesis time blocks before any
    sound (mixing the whole reply upfront caused several seconds of dead
    silence on a multi-sentence reply before anything played)."""
    _ensure_mixer()
    try:
        sentences = _split_sentences(text)
        if not sentences:
            return

        next_mix = _mix_sentence(sentences[0], piper_voice)
        for i in range(len(sentences)):
            mix = next_mix
            producer = None
            result: dict = {}
            if i + 1 < len(sentences):
                def _produce(idx: int = i + 1) -> None:
                    result["mix"] = _mix_sentence(sentences[idx], piper_voice)

                producer = threading.Thread(target=_produce, name="speak_pipeline", daemon=True)
                producer.start()

            stopped = not _play_mix(mix, should_stop)

            if producer is not None:
                producer.join()
                next_mix = result["mix"]
            if stopped:
                return
    finally:
        _release_mixer()


def demo() -> None:
    chord = ERIDIAN_LEXICON["rocky"]
    assert isinstance(chord[0], list)
    assert word_to_chord("zzz") != word_to_chord("qqq")
    bed = _render_chord_bed("hello world")
    assert bed.size > 0
    resampled = _resample(bed, SAMPLE_RATE, 1000)
    assert len(resampled) == 1000


if __name__ == "__main__":
    demo()
    print("rocky/voice: ok")
