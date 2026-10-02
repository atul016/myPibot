"""Sound-effect actions, patched into picarx.preset_actions.sounds_dict at
import time to fix a broken relative path.

preset_actions.honking/start_engine use a path ("../sounds/...") relative
to CWD, which only resolves correctly from ~/picar-x/example/ -- from this
project's own working directory it 404s (a FileNotFoundError caught in a
background thread and silently swallowed: the sound just never plays).
Same absolute-path fix as movement.actions, applied to both entries.
"""
import os

from picarx.preset_actions import sounds_dict

from common.system import SOUNDS_DIR


def _sound_action(filename: str, volume: int):
    def play(music) -> None:
        music.sound_play_threading(os.path.join(SOUNDS_DIR, filename), volume)
    return play


sounds_dict["honking"] = _sound_action("car-double-horn.wav", 100)
sounds_dict["start engine"] = _sound_action("car-start-engine.wav", 50)
