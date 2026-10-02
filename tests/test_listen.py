"""Listening end to end (services.wake_listen._listen) with a fake mic playing
synthetic speech, against the real Vosk model and the Mac's Whisper:

  - a two-phrase sentence is transcribed whole -- including its FIRST words
    (pre-roll once clipped "There is a delay..." to "Between...")
  - silence returns nothing, and gives up at the timeout
  - Rocky's own sentences are removed from what was heard (echo removal),
    while the person's words survive

Prints the end-of-speech -> transcript latency (target ~1.2s with the Mac's
Whisper). Needs the Mac's whisper-server (tools/setup-mac-whisper.sh); uses an isolated
state folder and a throwaway said-log.

    cd ~/openbot && python3 -m tests.test_listen
"""
from tests._audio import FakeMic, isolate_state, noise, say, seconds

isolate_state()

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import vosk  # noqa: E402

import config as cfg  # noqa: E402
import services.wake_listen as wl  # noqa: E402
from common import speak_client, stt  # noqa: E402

vosk.SetLogLevel(-1)
speak_client.SAID_LOG = tempfile.mktemp()  # never the real log
wl._wake_model = vosk.Model(wl._vosk_model_path())
wl._whisper = stt.RemoteWhisper(cfg.LLM_BASE_URL, cfg.MAC_WHISPER_PORT, None)

# 1. a full sentence with a pause in the middle; first words must survive
first, second = say("There is a delay between when I say something"), say("and when it gets into the loop.")
trailing = noise(4)
audio = noise(2) + first + noise(1.0) + second + trailing
t = time.time()
_, text = wl._listen(FakeMic(audio), onset_timeout_s=30)
latency = time.time() - t - (seconds(audio) - seconds(trailing))
print(f"heard {text!r} -- transcript {latency:.1f}s after speech ended")
assert text, "nothing transcribed -- is the Mac's whisper-server up? (tools/setup-mac-whisper.sh)"
assert "there is a delay" in text.lower(), "the start of the sentence was clipped"
assert "into the loop" in text.lower()

# 2. silence: nothing, and it gives up on time
t = time.time()
_, text = wl._listen(FakeMic(noise(3)), onset_timeout_s=2)
assert text == "" and time.time() - t < 4, (text, time.time() - t)
print("silence -> nothing, gave up on time")

# 3. echo removal: Rocky said X while the mic was open
for heard, rocky_said, expect in [("Back away. Stop. Stop.", "Back away!", "stop, stop"),
                                   ("Back away!", "Back away!", ""),
                                   ("What is this in my hand?", "Back away!", "what is this in my hand")]:
    now = time.time()
    with open(speak_client.SAID_LOG, "w") as f:
        f.write(json.dumps({"text": rocky_said, "start": now + 1, "end": now + 3}) + "\n")
    _, text = wl._listen(FakeMic(noise(1) + say(heard) + noise(3)), onset_timeout_s=10)
    print(f"mic heard {heard!r} while Rocky said {rocky_said!r} -> {text!r}")
    norm = " ".join(stt._words(text))
    assert norm == " ".join(stt._words(expect)), (text, expect)

os.remove(speak_client.SAID_LOG)
print("test_listen: ok")
