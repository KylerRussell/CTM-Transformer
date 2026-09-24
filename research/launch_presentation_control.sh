#!/bin/sh
# Start or resume the presentation-control study, detached from any terminal.
# Safe to run repeatedly: the supervisor holds a lock, keeps completed cells,
# reruns interrupted cells from scratch and exits if the study is complete.
# The environment lives under $HOME, which survives container restarts; /tmp does not.
set -eu
REPO=/home/kasm-user/Documents/CTM-Transformer
VENV=/home/kasm-user/.venvs/ctm-research
DRIVER=/home/kasm-user/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu
cd "$REPO"
export LD_LIBRARY_PATH="$DRIVER${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
setsid nohup "$VENV/bin/python" -u -m scripts.supervise_presentation_control \
  >> research/results/presentation_control_v1/supervisor.log 2>&1 < /dev/null &
echo "Supervisor started (pid $!); log: research/results/presentation_control_v1/supervisor.log"
