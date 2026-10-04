"""Face learning (services.wake_listen._learn_face): a misheard introduction
is renamed by voice, and only a real match on the ONE face in view becomes a
training sample -- a name carried by tracking, or a guess between two faces,
must never teach the wrong face.

    cd ~/openbot && python3 -m tests.test_learn_face
"""
import os, tempfile

os.environ["OPENBOT_STATE_DIR"] = tempfile.mkdtemp()

import numpy as np  # noqa: E402

import services.wake_listen as wl  # noqa: E402
from common import faces  # noqa: E402

atul, anna = np.eye(128, dtype=np.float32)[:2]
samples = lambda who: len(np.load(faces.KNOWN_DIR / f"{who}.npy"))
view = {"embedding": list(anna), "faces": [{"name": None, "similarity": 0.1, "box": [0, 0, 9, 9]}]}
faces.read = lambda: view

wl._learn_face("My name is Ana.")  # Whisper misheard "Anna"
assert samples("ana") == 1
wl._learn_face("Also her name is not Ana. Her name is Anna.")  # corrected right away -> renamed
assert samples("anna") == 1 and not (faces.KNOWN_DIR / "ana.npy").exists()

view["faces"][0].update(name="anna", similarity=0.5)  # a real but imperfect match: top up
wl._learn_face("What's that?")
assert samples("anna") == 2
view["faces"][0]["similarity"] = 0.3  # name only carried by tracking (no match): never a sample
wl._learn_face("What's that?")
assert samples("anna") == 2
view["faces"][0]["similarity"] = 0.5
view["faces"].append({"name": None, "similarity": 0.0, "box": [0, 0, 5, 5]})  # two faces: whose embedding? don't guess
wl._learn_face("What's that?")
wl._learn_face("I am Atul.")  # ...and an introduction is ambiguous too
assert samples("anna") == 2 and not (faces.KNOWN_DIR / "atul.npy").exists()
print("learn face: ok")
