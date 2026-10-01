#!/bin/sh
# Start or resume the learning-rate scaling sweep, detached from any terminal.
# Safe to run repeatedly (the supervisor holds a lock); see scripts/lr_sweep.py.
set -eu
REPO=/home/kasm-user/Documents/CTM-Transformer
VENV=/home/kasm-user/.venvs/ctm-research
DRIVER=/home/kasm-user/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu
cd "$REPO"
mkdir -p research/results/lr_sweep "$HOME/.config/autostart"
[ -f "$HOME/.config/autostart/ctm-lr-sweep.desktop" ] || cat > "$HOME/.config/autostart/ctm-lr-sweep.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=CTM learning-rate sweep
Exec=$REPO/research/launch_lr_sweep.sh
X-GNOME-Autostart-enabled=true
DESKTOP
export LD_LIBRARY_PATH="$DRIVER${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO"
setsid nohup "$VENV/bin/python" -u -m scripts.lr_sweep \
  >> research/results/lr_sweep/supervisor.log 2>&1 < /dev/null &
echo "LR sweep supervisor started (pid $!); log: research/results/lr_sweep/supervisor.log"
