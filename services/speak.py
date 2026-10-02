"""openbot-speak: the ONLY process that touches the speaker (and, on a
PiCar-X, its amp power: enable_speaker()/disable_speaker()). Runs as root --
on the PiCar-X only audio needs sudo (motors/sensors/GPIO don't; see
README's "Sudo required"). Every other service talks to this over a Unix socket
(common/speak_client.py) instead of running as root itself, and Piper's
voice model stays loaded once per persona instead of reloading per
utterance.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time

from pathlib import Path

from piper import PiperVoice
from piper.download_voices import download_voice

import common.persona as persona_mod
from common import health
from common.speak_client import PLAYING_FLAG, SAID_LOG, SOCK_PATH

COMPONENT = "openbot-speak"

# The Robot HAT's amp is off until asked -- whatever the body setting (a PiCar-X
# run with OPENBOT_BODY=none still speaks through it). An ordinary speaker is always on.
try:
    from robot_hat import enable_speaker, disable_speaker
except ImportError:
    enable_speaker = disable_speaker = lambda: None

VOICES_DIR = Path(os.environ.get("OPENBOT_PIPER_DIR", Path.home() / ".piper_models"))
_piper_cache: dict[str, PiperVoice] = {}
# One utterance at a time -- the amp/ALSA device isn't safely shareable
# across concurrent callers (same reasoning the original single-process
# version had no choice but to honor; here it's explicit).
_speak_lock = threading.Lock()
# Barge-in: set by a {"cmd": "stop"} request, which deliberately does NOT
# take _speak_lock (the utterance it's stopping holds it). Cleared at the
# start of each utterance.
_stop = threading.Event()
# The amp takes ~0.5s to switch on (measured: amp_enable=0.54s). A streamed
# reply arrives as several back-to-back utterances, so keep it on for a
# moment after each one instead of paying that gap between every sentence.
AMP_HOLD_S = 2.0
WARM_HOLD_S = 8.0  # a "warm" request keeps the amp ready this long for the reply
_amp_on = False
_amp_generation = 0


def _set_playing(on: bool) -> None:
    """See speak_client.PLAYING_FLAG. Best-effort: never blocks playback."""
    try:
        with open(PLAYING_FLAG, "w") as f:
            f.write("1" if on else "0")
        os.chmod(PLAYING_FLAG, 0o644)
    except OSError:
        pass


def _record_said(text: str, start: float, end: float) -> None:
    """See speak_client.SAID_LOG. Keeps the last 50 lines."""
    try:
        try:
            with open(SAID_LOG) as f:
                lines = f.readlines()[-49:]
        except OSError:
            lines = []
        lines.append(json.dumps({"text": text, "start": start, "end": end}) + "\n")
        with open(SAID_LOG, "w") as f:
            f.writelines(lines)
        os.chmod(SAID_LOG, 0o644)
    except OSError:
        pass


def _amp_acquire() -> None:
    """Call holding _speak_lock."""
    global _amp_on, _amp_generation
    _amp_generation += 1
    if not _amp_on:
        enable_speaker()
        _amp_on = True


def _amp_release_later(hold_s: float = AMP_HOLD_S) -> None:
    """Call holding _speak_lock: switch the amp off hold_s from now,
    unless another utterance has started by then."""
    generation = _amp_generation

    def _off() -> None:
        global _amp_on
        time.sleep(hold_s)
        with _speak_lock:
            if _amp_generation == generation and _amp_on:
                disable_speaker()
                _amp_on = False

    threading.Thread(target=_off, daemon=True).start()


def _get_piper(persona) -> PiperVoice:
    if persona.name not in _piper_cache:
        model = VOICES_DIR / f"{persona.piper_voice}.onnx"
        if not model.exists():
            VOICES_DIR.mkdir(parents=True, exist_ok=True)
            download_voice(persona.piper_voice, VOICES_DIR)  # once, needs internet
        _piper_cache[persona.name] = PiperVoice.load(str(model))
    return _piper_cache[persona.name]


def _say(voice: PiperVoice, text: str) -> None:
    """Streams Piper's audio to aplay, stopping between chunks on barge-in."""
    proc = subprocess.Popen(["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1",
                             "-r", str(voice.config.sample_rate)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for chunk in voice.synthesize(text):
            if _stop.is_set():
                proc.kill()
                return
            proc.stdin.write(chunk.audio_int16_bytes)
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait(timeout=30)


def _handle(conn: socket.socket) -> None:
    t_received = time.monotonic()
    try:
        raw = conn.recv(65536)
        req = json.loads(raw.decode())
        if req.get("cmd") == "warm":
            # Someone just finished talking: switch the amp on NOW (~0.55s) so it's
            # ready by the time the reply is -- not after it's written.
            conn.sendall(json.dumps({"ok": True}).encode())
            with _speak_lock:
                _amp_acquire()
                _amp_release_later(WARM_HOLD_S)
            return
        if req.get("cmd") == "stop":
            _stop.set()
            conn.sendall(json.dumps({"ok": True}).encode())
            return
        if req.get("cmd") == "sound":
            # Root process -- only ever plays .wav files from a picar-x/sounds folder
            # (whose home it's in depends on the caller; this process's own ~ is /root).
            path = os.path.realpath(str(req.get("path", "")))
            parts = path.split(os.sep)
            if parts[-3:-1] != ["picar-x", "sounds"] or not path.endswith(".wav") or not os.path.isfile(path):
                raise ValueError(f"not a sound effect: {path}")
            with _speak_lock:
                _amp_acquire()
                _set_playing(True)
                try:
                    subprocess.run(["aplay", "-q", path], timeout=15, check=False)
                finally:
                    _set_playing(False)
                    _amp_release_later()
            conn.sendall(json.dumps({"ok": True}).encode())
            return
        text = req["text"]
        persona = persona_mod.load(req.get("persona", persona_mod.CURRENT))
        t_before_piper = time.monotonic()
        piper = _get_piper(persona)
        t_before_speak = time.monotonic()
        with _speak_lock:
            _stop.clear()
            t_got_lock = time.monotonic()
            _amp_acquire()
            t_speaker_enabled = time.monotonic()
            _set_playing(True)
            said_start = time.time()
            try:
                if persona.speak_overlay is not None:
                    persona.speak_overlay(text, piper, should_stop=_stop.is_set)
                else:
                    _say(piper, text)
            finally:
                _set_playing(False)
                _record_said(text, said_start, time.time())
                _amp_release_later()
        t_done = time.monotonic()
        # Timing breakdown: which stage actually eats the "reaction to
        # speech" gap reported live -- persona/piper load, lock wait
        # (another utterance still playing), amp enable, or synthesis+
        # playback itself (this last one is inherent to speaking a longer
        # reply and not a bug, but the others would be). Logged via print
        # (this process runs as root; state/events.jsonl is atul-owned and
        # -- separately confirmed live -- root can't append to an
        # existing atul-owned file here despite having CAP_DAC_OVERRIDE,
        # an unexplained system-level quirk worth its own investigation).
        print(
            f"speak timing: piper_load={t_before_speak - t_before_piper:.2f}s "
            f"lock_wait={t_got_lock - t_before_speak:.2f}s "
            f"amp_enable={t_speaker_enabled - t_got_lock:.2f}s "
            f"synth_and_play={t_done - t_speaker_enabled:.2f}s "
            f"total={t_done - t_received:.2f}s"
        )
        conn.sendall(json.dumps({"ok": True}).encode())
        health.record_success(COMPONENT)
    except Exception as e:
        health.record_failure(COMPONENT, str(e))
        try:
            conn.sendall(json.dumps({"ok": False, "error": str(e)}).encode())
        except OSError:
            pass
    finally:
        conn.close()


def main() -> None:
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK_PATH)
    # World-writable: the daemon (running as root) is the privilege
    # boundary, not the socket -- every other service needs to connect as
    # a normal user, and none of them can do anything with GPIO/audio
    # beyond "ask this daemon to speak a string."
    os.chmod(SOCK_PATH, 0o666)
    server.listen(8)

    health.start_watchdog(COMPONENT, stale_after_s=60.0, ping_interval_s=10.0)

    # The volume someone asked for ("louder"/"softer") survives a reboot.
    # Read-only on purpose: this process is root, and load_session() can CREATE the
    # session file -- root-owned, it would lock every other service out of it.
    from common.state import SESSION_PATH
    from common.system import set_volume
    try:
        volume = json.loads(SESSION_PATH.read_text()).get("volume")
    except (OSError, json.JSONDecodeError):
        volume = None
    if volume:
        print(f"speaker volume restored to {set_volume(volume)}%")

    # Pre-warm the default persona's Piper model at boot -- confirmed live
    # a cold load takes ~7.5s (reading the ONNX model off disk), which
    # otherwise lands on whoever triggers the very first utterance after a
    # (re)start, not at a time anyone's actually waiting on it.
    t0 = time.monotonic()
    _get_piper(persona_mod.load())
    print(f"pre-warmed default persona's Piper model in {time.monotonic() - t0:.2f}s")

    health.record_success(COMPONENT)

    print(f"{COMPONENT} listening on {SOCK_PATH}")
    server.settimeout(10.0)  # wake up while idle: a quiet bot isn't a hung one
    while True:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            if _speak_lock.acquire(blocking=False):  # not while an utterance is stuck: let the watchdog see that
                _speak_lock.release()
                health.record_success(COMPONENT, min_interval_s=5.0)
            continue
        conn.settimeout(None)
        threading.Thread(target=_handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
