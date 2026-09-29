#!/bin/sh
# Launch development recipe probes: one detached sequential queue per argument.
# Each argument is "device|probe args;probe args;...". Development only; not restart-safe.
set -eu
REPO=/home/kasm-user/Documents/CTM-Transformer
VENV=/home/kasm-user/.venvs/ctm-research
DRIVER=/home/kasm-user/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu
cd "$REPO"
export LD_LIBRARY_PATH="$DRIVER${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
i=0
for queue in "$@"; do
  i=$((i+1))
  device=${queue%%|*}; jobs=${queue#*|}
  log=research/results/recipe_probes/logs/queue$(date +%H%M%S)_$i.log
  setsid nohup sh -c '
    device=$1; shift
    IFS=";"
    for job in $*; do
      IFS=" "; echo "== $job"; "'"$VENV"'/bin/python" -u -m scripts.run_recipe_probe --device "$device" $job || echo "FAILED: $job"; IFS=";"
    done' queue "$device" "$jobs" >> "$log" 2>&1 < /dev/null &
  echo "queue $i on $device: $log"
done
