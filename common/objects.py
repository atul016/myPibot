"""Objects: what's in front of Rocky, by name, without the LLM -- NanoDet-Plus
(OpenCV's model zoo, like the face models; COCO's 80 everyday kinds: person,
chair, cup, laptop, bed...). openbot-camera runs it on its own frames every
OBJECT_INTERVAL_S -> state/objects.json; the mind's awareness and every prompt
read the names, and a look's description is grounded by them. Pre/post-
processing follows the model zoo's nanodet.py demo.
"""
from __future__ import annotations

import json
import time

from .faces import MODEL_DIR
from .state import STATE_DIR, atomic_write

MODEL = MODEL_DIR / "object_detection_nanodet_2022nov.onnx"
OBJECTS_PATH = STATE_DIR / "objects.json"
OBJECT_INTERVAL_S = 2.0
STALE_S = 10.0
# Someone needs boxes NOW (following a person, the dashboard's live view): while this
# file was touched in the last FAST_FOR_S, openbot-camera detects on every new frame
# (~8-10/s on the Pi 5) instead of every OBJECT_INTERVAL_S. Expires on its own.
FAST_PATH = STATE_DIR / "objects_fast"
FAST_FOR_S = 3.0
COCO = [c.replace("_", " ") for c in (  # the 80 COCO kinds, in the model's order
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
    "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle "
    "wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake "
    "chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven "
    "toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()]


class Detector:
    SIZE, STRIDES, REG_MAX = 416, (8, 16, 32, 64), 7

    def __init__(self, score: float = 0.4, iou: float = 0.6) -> None:
        import cv2
        import numpy as np
        self.cv2, self.np, self.score, self.iou = cv2, np, score, iou
        self.net = cv2.dnn.readNet(str(MODEL))
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)  # BGR, as the model was trained
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        self.anchors = []
        for s in self.STRIDES:
            n = self.SIZE // s
            xv, yv = np.meshgrid(np.arange(n) * s, np.arange(n) * s)
            self.anchors.append(np.column_stack((xv.flatten() + 0.5 * (s - 1), yv.flatten() + 0.5 * (s - 1))))

    def detect(self, jpeg: bytes) -> list[dict]:
        """[{"name", "score", "box": [left, top, right, bottom] as fractions of the frame}], best first."""
        cv2, np = self.cv2, self.np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        scale = self.SIZE / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        top, left = (self.SIZE - nh) // 2, (self.SIZE - nw) // 2
        canvas = np.zeros((self.SIZE, self.SIZE, 3), np.float32)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage((canvas - self.mean) / self.std))
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes, scores = [], []
        for stride, cls, reg, anchors in zip(self.STRIDES, outs[::2], outs[1::2], self.anchors):
            cls, reg = cls.reshape(-1, len(COCO)), reg.reshape(-1, self.REG_MAX + 1)
            dist = (np.exp(reg) / np.exp(reg).sum(axis=1, keepdims=True)) @ np.arange(self.REG_MAX + 1)
            dist = dist.reshape(-1, 4) * stride
            boxes.append(np.column_stack((anchors - dist[:, :2], anchors + dist[:, 2:])).clip(0, self.SIZE))
            scores.append(cls)
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        best, conf = scores.argmax(axis=1), scores.max(axis=1)
        keep = cv2.dnn.NMSBoxes([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes.tolist()],
                                conf.tolist(), self.score, self.iou)
        found = []
        for i in sorted(np.array(keep).flatten().tolist(), key=lambda i: -conf[i]):
            x1, y1, x2, y2 = boxes[i]
            found.append({"name": COCO[best[i]], "score": round(float(conf[i]), 2),
                          "box": [round(float((x1 - left) / scale / w), 3), round(float((y1 - top) / scale / h), 3),
                                  round(float((x2 - left) / scale / w), 3), round(float((y2 - top) / scale / h), 3)]})
        return found


def publish(found: list[dict]) -> None:
    atomic_write(OBJECTS_PATH, json.dumps({"objects": found, "ts": time.time()}))


def latest() -> tuple[list[dict], float]:
    """(the newest boxes, when they were published) -- ([], 0.0) if there are none."""
    try:
        data = json.loads(OBJECTS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return [], 0.0
    return data.get("objects", []), data.get("ts", 0.0)


def read(max_age_s: float = STALE_S) -> list[dict]:
    """[] if missing or older than max_age_s (camera off, asleep, detector not running)."""
    found, ts = latest()
    return found if time.time() - ts < max_age_s else []


def want_fast() -> None:
    FAST_PATH.touch()


def fast_wanted() -> bool:
    try:
        return time.time() - FAST_PATH.stat().st_mtime < FAST_FOR_S
    except OSError:
        return False


def names(found: list[dict]) -> str | None:
    """"a person, a chair and 2 cups" -- None if nothing."""
    counts: dict[str, int] = {}
    for o in found:
        counts[o["name"]] = counts.get(o["name"], 0) + 1
    parts = [f"{n} {name}s" if n > 1 else f"a {name}" for name, n in counts.items()]
    return None if not parts else parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def demo() -> None:
    assert len(COCO) == 80 and COCO[0] == "person" and COCO[9] == "traffic light" and COCO[-1] == "toothbrush"
    assert "dining table" in COCO and "cell phone" in COCO
    assert names([{"name": "person"}, {"name": "cup"}, {"name": "cup"}]) == "a person and 2 cups"
    assert names([]) is None and names([{"name": "chair"}]) == "a chair"
    import os
    import shutil, tempfile
    from pathlib import Path
    global OBJECTS_PATH, FAST_PATH
    orig, test_dir = (OBJECTS_PATH, FAST_PATH), Path(tempfile.mkdtemp())
    OBJECTS_PATH, FAST_PATH = test_dir / "objects.json", test_dir / "objects_fast"
    try:
        assert not fast_wanted()
        want_fast()
        assert fast_wanted()
        os.utime(FAST_PATH, (time.time() - FAST_FOR_S - 1,) * 2)
        assert not fast_wanted()  # expires by itself
        publish([{"name": "person", "score": 0.9, "box": [0.1, 0.1, 0.5, 0.9]}])
        assert read()[0]["name"] == "person" and read(max_age_s=0.5)
        OBJECTS_PATH.write_text(json.dumps({"objects": [{"name": "cup"}], "ts": time.time() - 1}))
        assert read() and not read(max_age_s=0.5)  # fine for the mind, too old to steer by
    finally:
        OBJECTS_PATH, FAST_PATH = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("objects: ok")
