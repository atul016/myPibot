"""A streamed reply (services.wake_listen._reply_sentences) against the live LLM:
the tone gesture fires before the first sentence, sentences arrive one by one,
a fresh camera photo goes with the turn, and history is only touched on commit.
Motors stubbed (nothing moves), nothing is spoken; isolated state.

Prints when the gesture, the first sentence and the whole reply arrived
(measured ~1.0s to the first sentence with a photo attached).

    cd ~/openbot && python3 -m tests.test_reply_stream
"""
from tests._audio import isolate_state

isolate_state()

import time  # noqa: E402

import config as cfg  # noqa: E402
import services.wake_listen as wl  # noqa: E402
from common import persona as persona_mod, reply_schema  # noqa: E402

t0 = 0.0
gestures: list[float] = []
wl.motor_dispatch = lambda actions, wait=False: gestures.append(time.time() - t0) or True
persona = persona_mod.load()
Reply = reply_schema.build_reply_model(cfg.STATIONARY_ACTIONS)

for question in ["What am I holding in my hand?", "Tell me a short story about a robot and a cat."]:
    out, history, times = {}, [], []
    gestures.clear()
    t0 = time.time()
    for sentence in wl._reply_sentences(persona, Reply, question, history, out):
        times.append(time.time() - t0)
        print(f"  {times[-1]:.2f}s  {sentence}")
    assert times, "no reply at all -- is the LLM server up?"
    assert history == [], "_reply_sentences must not touch history (a speculative reply may be thrown away)"
    wl._commit_history(history, question, out)
    assert len(history) == 2 and history[0]["content"] == question
    assert out.get("text"), "the full reply text wasn't recorded"
    if gestures:
        assert gestures[0] <= times[0] + 0.05, "the tone gesture should fire before speech starts"
    print(f"{question!r}: gesture {gestures[0] if gestures else float('nan'):.2f}s, "
          f"first sentence {times[0]:.2f}s, whole reply {times[-1]:.2f}s\n")
print("test_reply_stream: ok")
