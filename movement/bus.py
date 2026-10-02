"""One lock around every I2C transaction to the Robot HAT, patched into
robot_hat at import time (same approach as movement/actions.py).

openbot-alive drives the HAT from several threads at once -- the face
tracker and gestures write servo angles, the safety loop reads the
grayscale ADC ~20x a second, the battery is read too. Interleaved, an ADC
read (a write of the channel, then a read) can land between another
thread's transfers and come back as 0: measured 2026-10-02, the grayscale
module read 0 on one channel 10-45 times a minute inside alive, and NEVER
in 160 reads from a standalone process. Serializing the bus fixes it at
the source instead of filtering bad readings downstream.

Import before constructing Picarx().
"""
import threading

from robot_hat.adc import ADC
from robot_hat.i2c import I2C

BUS_LOCK = threading.RLock()  # re-entrant: ADC.read itself calls write() then read()


def _locked(method):
    def wrapper(*args, **kwargs):
        with BUS_LOCK:
            return method(*args, **kwargs)
    wrapper.__wrapped__ = method
    return wrapper


for cls, names in ((I2C, ("write", "read", "mem_write", "mem_read")), (ADC, ("read",))):
    for name in names:
        method = cls.__dict__.get(name)
        if method is not None and not hasattr(method, "__wrapped__"):  # idempotent on re-import
            setattr(cls, name, _locked(method))
