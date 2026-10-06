#!/usr/bin/env bash
# One-time setup on Linux (Raspberry Pi OS / Debian / Ubuntu). Run from this
# folder as your normal user (asks for the sudo password). Safe to re-run.
#   ./setup.sh            no robot: a mic, a speaker and (optionally) a camera
#   ./setup.sh --picarx   also SunFounder's PiCar-X libraries + speaker driver
# Then: edit openbot.env, and run  sudo systemd/install.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PICARX=0; [ "${1:-}" = "--picarx" ] && PICARX=1
PIP=(sudo pip3 install --break-system-packages)
python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' || {
  echo "OpenBot needs Python 3.12+: Raspberry Pi OS Trixie / Debian 13, or Ubuntu 24.04"; exit 1; }

echo "== apt packages"
sudo apt update
sudo apt install -y git curl unzip alsa-utils ffmpeg file pipx python3-pip espeak sox \
  python3-requests python3-flask python3-filelock python3-numpy python3-pygame \
  python3-pydantic python3-pil python3-opencv python3-usb

echo "== a reSpeaker XVF3800 mic array (optional): its direction finder, readable without root"
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="001a", MODE="0664", GROUP="plugdev"' \
  | sudo tee /etc/udev/rules.d/99-openbot-respeaker.rules >/dev/null
sudo usermod -aG plugdev "$USER"
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb

echo "== speech: Vosk + Whisper (hearing), Piper (voice)"
# --ignore-installed: pip can't uninstall Debian-owned deps (e.g. click) when upgrading them
"${PIP[@]}" --ignore-installed vosk faster-whisper piper-tts
# Older distros' apt packages are too old for what pip just installed (Ubuntu 24.04:
# pydantic 1.x; OpenCV 4.6, which can't load the face model or run on numpy 2).
python3 -c 'import pydantic, sys; sys.exit(int(pydantic.VERSION[0]) < 2)' || "${PIP[@]}" --ignore-installed 'pydantic>=2'
python3 -c 'import cv2, sys; sys.exit(tuple(map(int, cv2.__version__.split(".")[:2])) < (4, 8))' 2>/dev/null \
  || "${PIP[@]}" --ignore-installed opencv-python-headless
mkdir -p ~/.vosk_models
for m in vosk-model-small-en-us-0.15 vosk-model-small-en-in-0.4; do
  [ -d ~/.vosk_models/$m ] && continue
  curl -fL -o /tmp/$m.zip https://alphacephei.com/vosk/models/$m.zip
  unzip -q /tmp/$m.zip -d ~/.vosk_models && rm /tmp/$m.zip
done

echo "== WhatsApp chat (optional, README: \"Text it on WhatsApp\"): neonize"
"${PIP[@]}" neonize

[ -f "$DIR/openbot.env" ] || { cp "$DIR/openbot.env.example" "$DIR/openbot.env"; echo "   created openbot.env -- edit it"; }
[ $PICARX = 1 ] && sed -i -e 's/^OPENBOT_BODY=.*/OPENBOT_BODY=picarx/' \
  -e 's/^OPENBOT_MIXER=.*/OPENBOT_MIXER="2:robot-hat speaker"/' "$DIR/openbot.env"
PROJECT=$(set -a; . "$DIR/openbot.env"; echo "${OPENBOT_MEMORY_PROJECT:-openbot}")  # parsed the way install.sh does

echo "== long-term memory: Basic Memory (AGPL-3.0, run unmodified)"
pipx install basic-memory==0.23.2  # pinned: its CLI/MCP argument names drift between releases
export BASIC_MEMORY_NO_PROMOS=1
mkdir -p "$DIR/state/mind"
~/.local/bin/basic-memory project add "$PROJECT" "$DIR/state/mind" || true  # already added -> fine
~/.local/bin/basic-memory project default "$PROJECT"

echo "== face detection + recognition and object detection models (OpenCV model zoo)"
mkdir -p ~/.openbot-models
for f in face_detection_yunet/face_detection_yunet_2023mar.onnx face_recognition_sface/face_recognition_sface_2021dec.onnx \
         object_detection_nanodet/object_detection_nanodet_2022nov.onnx; do
  [ -s ~/.openbot-models/$(basename $f) ] || curl -fsSL -o ~/.openbot-models/$(basename $f) "https://github.com/opencv/opencv_zoo/raw/main/models/$f"
done
echo "== what a sound was (YAMNet, Google) and who's talking (WeSpeaker voice prints)"
"${PIP[@]}" ai-edge-litert
get() { [ -s ~/.openbot-models/$1 ] || curl -fsSL -o ~/.openbot-models/$1 "$2"; }
get yamnet.tflite https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
get yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
get voxceleb_resnet34_LM.onnx https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx

if [ $PICARX = 1 ]; then
  echo "== PiCar-X: robot-hat 2.5.x, vilib, picar-x 2.1.x"
  # Follows https://docs.sunfounder.com/projects/picar-x-v20/en/latest/python/install_all_modules.html
  sudo apt install -y python3-smbus python3-setuptools
  cd ~
  clone() { local d; d=$(basename "$2" .git); [ -d "$d" ] || git clone ${1:+-b "$1"} --depth 1 "$2"; }
  clone 2.5.x https://github.com/sunfounder/robot-hat.git && (cd robot-hat && sudo python3 install.py)
  clone "" https://github.com/sunfounder/vilib.git && (cd vilib && sudo python3 install.py)
  clone 2.1.x https://github.com/sunfounder/picar-x.git && (cd picar-x && "${PIP[@]}" .)
  # Picarx() keeps its servo calibration in /opt/picar-x; OpenBot runs it as this user, not root.
  sudo install -d -o "$USER" -g "$USER" -m 775 /opt/picar-x
  echo "== PiCar-X speaker driver (i2samp.sh -- answer its y/N prompts)"
  (cd robot-hat && sudo bash i2samp.sh)
  python3 -c "import picarx, robot_hat; print('picarx imports ok')"
fi

# Keep logs across reboots -- an unexplained reboot is undiagnosable otherwise. A drop-in,
# because Raspberry Pi OS forces RAM-only logs in its own (40-rpi-volatile-storage.conf).
sudo mkdir -p /var/log/journal /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\n' | sudo tee /etc/systemd/journald.conf.d/90-openbot.conf >/dev/null
sudo systemctl restart systemd-journald

# USB mic capture gain to max and persisted: a fresh install leaves it at ~69%, which
# halves speech level (peak -24 dBFS vs -14 at max, noise floor barely moves) and
# follow-ups went unheard. No-op if no USB mic is plugged in yet.
for card in $(arecord -l 2>/dev/null | sed -n 's/^card \([0-9]*\):.*USB.*/\1/p'); do
  amixer -q -c "$card" sset Mic 100% 2>/dev/null && echo "mic gain: card $card -> 100%"
done
sudo alsactl store
python3 -c "import vosk, piper, faster_whisper, cv2, flask, neonize; print('imports ok')"
echo
echo "Done. Next:"
echo "  1. Edit $DIR/openbot.env (the LLM server, your mic name from 'arecord -l')"
echo "  2. sudo $DIR/systemd/install.sh"
