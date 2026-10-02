#!/usr/bin/env bash
# Optional, for when the LLM runs on a Mac: run ON THE MAC to also give the
# robot fast ears there. Two launchd agents (start at login, restart if they die):
#   com.openbot.whisper-server -- whisper.cpp (GPU) on :8178; set OPENBOT_MAC_WHISPER_PORT=8178
#   com.openbot.keepawake      -- keeps the Mac from idle-sleeping WHILE the robot is on
#                                 (its brain and ears live here); lets it sleep when the robot's off.
#                                 No power settings changed. A closed lid still sleeps.
#   tools/setup-mac-whisper.sh <robot-ip>
# Undo either: launchctl unload ~/Library/LaunchAgents/<label>.plist && rm that file
set -euo pipefail
MODEL=~/.whisper-models/ggml-large-v3-turbo-q5_0.bin
PLIST=~/Library/LaunchAgents/com.openbot.whisper-server.plist

command -v whisper-server >/dev/null || brew install whisper-cpp
mkdir -p ~/.whisper-models
[ -s "$MODEL" ] || curl -fL -o "$MODEL" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.openbot.whisper-server</string>
  <key>ProgramArguments</key><array>
    <string>$(command -v whisper-server)</string>
    <string>-m</string><string>$MODEL</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>8178</string>
    <string>-l</string><string>en</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/whisper-server.log</string>
</dict></plist>
PLIST
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "whisper-server running on :8178 (log: /tmp/whisper-server.log)"

# --- keep awake while the robot is on ---------------------------------------------
PI_IP="${1:?usage: tools/setup-mac-whisper.sh <robot-ip>}"
AWAKE=~/Library/LaunchAgents/com.openbot.keepawake.plist
cat > "$AWAKE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.openbot.keepawake</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>-c</string>
    <string>while :; do if /usr/bin/curl -s -m3 -o /dev/null http://$PI_IP:8080/api/state; then /usr/bin/caffeinate -i -t 300; else sleep 60; fi; done</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
PLIST
launchctl unload "$AWAKE" 2>/dev/null || true
launchctl load "$AWAKE"
echo "keep-awake agent running: the Mac stays up while the robot ($PI_IP) is on"
