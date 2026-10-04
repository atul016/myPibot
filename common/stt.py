"""Speech-to-text in two roles. Vosk runs STREAMING while you talk
(services/wake_listen.py feeds it chunk by chunk) and decides WHEN you
started and stopped talking -- it hears words, which a loudness threshold
can't. Whisper then transcribes the recording for WHAT you said: Vosk's
small model heard "rocky come you tell me..." where Whisper got "can you
tell me..." right. Vosk's own transcript is the fallback only if no Whisper
can be asked -- when Whisper says it was noise, it was noise.

Measured on the Pi 5 (2026-10-02, beam 1, no timestamps): base.en ~2.1s
per utterance, tiny.en ~1.1s (same on clear speech; base handles accents
better), small.en 7-8s. Streaming Vosk's per-word confidences read 1.0 even
on wrong words, so "only ask Whisper when Vosk is unsure" doesn't work.

Both work on 44100Hz 16-bit mono PCM (mic_stream.ArecordStream),
resampled to 16kHz.
"""
from __future__ import annotations

import re
import struct

from .persona import CURRENT

VOSK_RATE = 16000
MIC_RATE = 44100

PHANTOM_PHRASES = {
    "thank you", "thanks for watching", "you", "the", "a", "and", "is",
    "bye", "so", "i'm sorry", "subscribe",
    # What the whisper-server writes (with PROMPT) for room noise or a mumble --
    # reproduced from plain noise; taken as the user 6 times on 2026-10-02.
    "you can see the next one", "see you in the next one", "you can see the camera",
    "you can see the light", "you can see the light on the screen", "you can see the surroundings",
}


def resample_pcm(data: bytes, in_rate: int, out_rate: int) -> bytes:
    """Linear-interpolation downsample of 16-bit mono PCM -- good enough
    for STT input (not archival audio), avoids a numpy/scipy dependency
    for this one conversion. Public: also used by wake_listen.py's wake-
    grammar recognizer, which needs the same 44100->16000 conversion."""
    samples = struct.unpack(f"<{len(data) // 2}h", data)
    if not samples:
        return b""
    ratio = out_rate / in_rate
    out_len = int(len(samples) * ratio)
    out = []
    for i in range(out_len):
        src = i / ratio
        lo = int(src)
        frac = src - lo
        hi = min(lo + 1, len(samples) - 1)
        out.append(int(samples[lo] * (1 - frac) + samples[hi] * frac))
    return struct.pack(f"<{len(out)}h", *out)


def reject_hallucination(text: str) -> str:
    """Shared post-filter: phantom phrases, non-ASCII-dominant text,
    repetitive word salad. Returns "" if `text` looks like a
    hallucination rather than real speech."""
    if not _words(text):  # "." / "- -": Whisper's transcript of noise has no words
        return ""
    stripped = text.lower().strip(".!? ")
    if stripped in PHANTOM_PHRASES:
        return ""
    ascii_text = text.encode("ascii", "ignore").decode("ascii").strip()
    if text != ascii_text and len(ascii_text) < len(text) * 0.5:
        return ""
    words = text.lower().split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.3:
        return ""
    return text


def vosk_text(results: list[dict]) -> str:
    """Joins Vosk Result() dicts into one filtered transcript."""
    return reject_hallucination(" ".join(r.get("text", "") for r in results if r.get("text")).strip())


class Whisper:
    # Names the model would otherwise mishear ("Rocket") -- Whisper's
    # initial_prompt biases spelling toward them.
    # Also the voice commands (common/commands.py): a spoken "quiet" came back
    # as "quite" without this hint.
    PROMPT = ("{0} is a small robot. People talk to {0}. "
              "Commands: be quiet, go to sleep, {0} wake up, stop session.").format(CURRENT.capitalize())

    def __init__(self, model: str):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, pcm_44100: bytes) -> str:
        import numpy as np
        resampled = resample_pcm(pcm_44100, MIC_RATE, VOSK_RATE)
        samples = np.frombuffer(resampled, dtype="<i2").astype("float32") / 32768.0
        segments, _info = self._model.transcribe(
            samples, language="en", beam_size=1, temperature=0, initial_prompt=self.PROMPT, without_timestamps=True,
            condition_on_previous_text=False, no_speech_threshold=0.6, vad_filter=False,
        )
        return reject_hallucination(" ".join(s.text.strip() for s in segments).strip())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def remove_echo(transcript: str, said: list[str]) -> str:
    """Drops from `transcript` the EXACT sentences Rocky itself said while the
    mic was open -- each whole sentence, in order, ignoring only case and
    punctuation (Whisper writes "Back away." for Rocky's "Back away!"). Never
    partial: a sentence is removed only if all of it is there. What's left is
    what the person said; "" if it was all Rocky. Also drops Whisper's
    non-speech tags ("[engine revving]", "(music)")."""
    transcript = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", transcript)
    tokens = re.findall(r"\S+", transcript)
    norm = [" ".join(_words(t)) for t in tokens]  # "away." -> "away", "--" -> ""
    keep = [bool(n) for n in norm]
    sentences = [_words(x) for line in said for x in re.split(r"(?<=[.!?])\s+", line)]
    for target in sorted(filter(None, sentences), key=len, reverse=True):  # longest first
        i = 0
        while i < len(tokens):
            # match `target` word by word starting at token i, skipping punctuation-only tokens
            j, k = i, 0
            while j < len(tokens) and k < len(target) and (not norm[j] or (keep[j] and norm[j] == target[k])):
                k += bool(norm[j])
                j += 1
            if k == len(target) and norm[i]:
                for x in range(i, j):
                    keep[x] = False
                i = j
            else:
                i += 1
    return " ".join(t for t, k in zip(tokens, keep) if k)


class RemoteWhisper:
    """Whisper on the Mac (whisper.cpp's whisper-server, GPU, large-v3-turbo):
    faster AND far more accurate than the Pi's base.en, which stays as the
    fallback for when the Mac is off or slow. The host follows wherever the
    LLM server was last found (cognition's LAN rediscovery) -- same Mac."""

    def __init__(self, llm_base_url: str, port: int, fallback: "Whisper | None", timeout_s: float = 4.0):
        self.llm_base_url, self.port, self.fallback, self.timeout_s = llm_base_url, port, fallback, timeout_s

    def _url(self) -> str:
        from urllib.parse import urlparse
        from .cognition import _known
        return f"http://{urlparse(_known(self.llm_base_url)).hostname}:{self.port}/inference"

    def transcribe(self, pcm_44100: bytes) -> str | None:
        """None: no Whisper to ask (use Vosk's words). "": Whisper heard only
        noise -- that verdict must not fall back to Vosk's guess at it."""
        import io, wave
        import requests
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(VOSK_RATE)
            w.writeframes(resample_pcm(pcm_44100, MIC_RATE, VOSK_RATE))
        try:
            resp = requests.post(self._url(), files={"file": ("speech.wav", wav.getvalue(), "audio/wav")},
                                 data={"response_format": "json", "temperature": "0", "prompt": Whisper.PROMPT},
                                 timeout=(1.5, self.timeout_s))
            resp.raise_for_status()
            return reject_hallucination(str(resp.json().get("text", "")).strip())
        except (requests.RequestException, ValueError) as e:
            print(f"stt: Mac whisper unavailable ({type(e).__name__}) -- using the Pi's")
            return self.fallback.transcribe(pcm_44100) if self.fallback else None


def demo() -> None:
    assert reject_hallucination("thank you") == ""
    assert reject_hallucination("okay okay okay okay okay okay") == ""
    assert reject_hallucination("turn left please") == "turn left please"
    assert reject_hallucination("") == ""
    assert reject_hallucination(".") == "" and reject_hallucination("You can see the next one.") == ""

    assert vosk_text([{"text": "turn left"}, {"text": ""}, {"text": "please"}]) == "turn left please"
    assert vosk_text([]) == "" and vosk_text([{"text": "thank you"}]) == ""  # phantom phrase

    # echo removal -- exact sentences. The real case: a reflex said "Back away!"
    # while the session listened, and it came back as the user's words.
    assert remove_echo("Back away. Stop. Stop.", ["Back away!"]) == "Stop. Stop."
    assert remove_echo("Back away.", ["Back away!"]) == ""                         # all Rocky -> nothing heard
    assert remove_echo("Hey, personal space! What is this in my hand?", ["Hey, personal space!"]) == \
        "What is this in my hand?"
    assert remove_echo("Back off now", ["Back away!"]) == "Back off now"            # not the whole sentence: untouched
    assert remove_echo("What time is it?", ["Beep boop!"]) == "What time is it?"
    assert remove_echo("[engine revving] Rocky, look left", []) == "Rocky, look left"
    assert remove_echo("Okay. Quiet now. Be quiet", ["Okay. Quiet now."]) == "Be quiet"  # per sentence of a line
    assert remove_echo("Quiet now be quiet", ["Okay. Quiet now."]) == "be quiet"     # mic caught only part of the line


if __name__ == "__main__":
    demo()
    print("stt: ok")
