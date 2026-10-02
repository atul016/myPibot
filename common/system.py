"""System-level compatibility shim: os.getlogin() reads the controlling
terminal's utmp entry, which doesn't exist when a process is launched by
systemd (confirmed live: "OSError: [Errno -25] Unknown error -25" from
Picarx() under `systemctl start`, despite working fine over an interactive
SSH session, which does have a controlling tty) -- regardless of which user
the service runs as. Shimmed to the process's own real user instead of
fighting systemd for a fake tty.

Importing this module has a side effect: it monkey-patches os.getlogin.
Deliberate -- import it once, early, before any Picarx() is constructed.
"""
import os
import pwd

# SUDO_USER covers the ad hoc "sudo python3 services/speak.py" foreground-
# testing path; every systemd-launched service (User=atul or User=root, no
# actual `sudo` involved) falls through to the process's own real uid.
REAL_USER = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
SOUNDS_DIR = os.path.join(os.path.expanduser(f"~{REAL_USER}"), "picar-x", "sounds")

os.getlogin = lambda: REAL_USER

# (ALSA card, mixer control) for "louder"/"softer" -- `amixer -c <card> scontrols` lists them.
# Default: the PiCar-X HAT's speaker. A plain USB/3.5mm speaker is often "0:Master" or "0:PCM".
SPEAKER_CONTROL = tuple(os.environ.get("OPENBOT_MIXER", "2:robot-hat speaker").split(":", 1))


def get_volume() -> int | None:
    import re
    import subprocess
    try:
        out = subprocess.run(["amixer", "-c", SPEAKER_CONTROL[0], "sget", SPEAKER_CONTROL[1]],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\[(\d+)%\]", out)
    return int(m.group(1)) if m else None


def set_volume(pct: int) -> int:
    import subprocess
    pct = max(10, min(100, int(pct)))  # never all the way to silent: "softer" shouldn't make Rocky mute
    subprocess.run(["amixer", "-q", "-c", SPEAKER_CONTROL[0], "sset", SPEAKER_CONTROL[1], f"{pct}%"],
                   capture_output=True, timeout=5)
    return pct
