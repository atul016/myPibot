"""Bound a call/thread to a timeout instead of risking an unbounded hang --
guards third-party calls with no timeout of their own (ActionFlow.
wait_actions_done(), Vilib.camera_start(), etc). Pure stdlib.

A thread still running when its timeout expires is abandoned, not killed --
there's no safe way to force-kill a thread from outside in Python. Always
started as a daemon thread, so an abandoned one never blocks process exit.
"""
from __future__ import annotations

import threading
from typing import Callable


def run_bounded(fn: Callable[[], None], timeout: float) -> bool:
    """Run fn() on a helper thread; wait up to `timeout` seconds for it to
    finish. Returns True if it finished in time, False if abandoned."""
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive()


def join_bounded(thread: threading.Thread | None, timeout: float) -> None:
    """join() an already-running thread, bounded. No-op if thread is None."""
    if thread is not None:
        thread.join(timeout)


def demo() -> None:
    import time
    assert run_bounded(lambda: None, 1.0) is True
    assert run_bounded(lambda: time.sleep(5), 0.05) is False


if __name__ == "__main__":
    demo()
    print("bounded: ok")
