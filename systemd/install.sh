#!/usr/bin/env bash
# Installs/refreshes the OpenBot systemd units, filled in for this user and
# folder, plus /etc/openbot/openbot.env (from ../openbot.env). Run with sudo
# from the user that owns this folder:  sudo systemd/install.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:?run with sudo, as the user who owns $DIR}"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"

[ -f "$DIR/openbot.env" ] || { echo "No $DIR/openbot.env -- run setup.sh first (or copy openbot.env.example)"; exit 1; }
install -d /etc/openbot
install -m 600 "$DIR/openbot.env" /etc/openbot/openbot.env  # root-only: it can hold API keys (systemd reads it as root)
# shellcheck source=/dev/null
set -a; source "$DIR/openbot.env"; set +a

for f in "$DIR"/systemd/openbot-*.service; do
  sed -e "s#@USER@#$USER_NAME#g" -e "s#@HOME@#$USER_HOME#g" -e "s#@DIR@#$DIR#g" \
      -e "s#@PROJECT@#${OPENBOT_MEMORY_PROJECT:-openbot}#g" "$f" > "/etc/systemd/system/$(basename "$f")"
done
systemctl daemon-reload

units=(openbot-memory openbot-camera openbot-speak openbot-wake-listen openbot-mind openbot-dashboard)
if [ "${OPENBOT_BODY:-none}" = "none" ]; then
  systemctl disable --now openbot-alive 2>/dev/null || true  # no body to drive
else
  units+=(openbot-alive)
fi
if [ -n "${OPENBOT_CHAT_ALLOW:-}" ]; then
  units+=(openbot-chat)
else
  systemctl disable --now openbot-chat 2>/dev/null || true  # nobody may text it: no WhatsApp
fi
for unit in "${units[@]}"; do
  systemctl enable "$unit"
  systemctl restart "$unit"
done

echo "Installed (body: ${OPENBOT_BODY:-none}). Status: systemctl status ${units[*]}"
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
