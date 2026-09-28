#!/bin/sh
# Start or resume the locked confirmation study, detached from any terminal.
# Safe to run repeatedly; see scripts/study_supervisor.py.
set -eu
REPO=/home/kasm-user/Documents/CTM-Transformer
VENV=/home/kasm-user/.venvs/ctm-research
DRIVER=/home/kasm-user/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu
cd "$REPO"
export LD_LIBRARY_PATH="$DRIVER${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
setsid nohup "$VENV/bin/python" -u -m scripts.supervise_sync_confirmation \
  >> research/results/sync_confirmation_v1/supervisor.log 2>&1 < /dev/null &
echo "Supervisor started (pid $!); log: research/results/sync_confirmation_v1/supervisor.log"
