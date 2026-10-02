#!/usr/bin/env bash
# Copy this folder from your computer to the robot (~/openbot/), when you
# edit code on one machine and run it on another. Not needed if you cloned
# OpenBot straight onto the robot.
#   ./deploy.sh user@robot-host        (or set OPENBOT_HOST)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/../config/pi.env" ] && source "$DIR/../config/pi.env"  # Walle's own setup
HOST="${1:-${OPENBOT_HOST:-${PI_HOST:-}}}"
: "${HOST:?usage: ./deploy.sh user@robot-host}"

# P: never delete the robot's own settings, or lgpio's pipe (it lives in the folder openbot-speak runs from)
rsync -a --delete \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='state/' --exclude='sim/out/' \
  --filter='P /openbot.env' --filter='P /.lgd-nfy*' \
  "$DIR/" "$HOST:~/openbot/"

echo "Deployed to $HOST:~/openbot/"
echo "Install/refresh services: ssh -t $HOST 'sudo ~/openbot/systemd/install.sh'"
