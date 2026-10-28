#!/bin/bash
# Stage 0 of research/LADDER_PLAN.md on one GPU: 0A + 0B (scripts/code_math_probe.py, queue N/2), then its half of 0C
# (scripts/code_math_tasks.py, 8 workers). Usage: setsid nohup scripts/run_stage0.sh N > research/results/code_math/gpuN.log 2>&1 &
set -u;gpu=$1;cd "$(dirname "$0")/..";source ~/.venvs/ctm-research/bin/activate
export PYTHONPATH=. LD_LIBRARY_PATH=$HOME/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu
echo "stage0 gpu $gpu start $(date -u +%FT%TZ)"
python -u -m scripts.code_math_probe --gpu "$gpu" --queue "$gpu/2";echo "probe exit $? $(date -u +%FT%TZ)"
mkdir -p research/results/code_math_tasks/logs
for w in 0 1 2 3 4 5 6 7;do k=$((gpu*8+w));python -u -m scripts.code_math_tasks --gpu "$gpu" --slice "$k/16" > research/results/code_math_tasks/logs/slice$k.log 2>&1 & done
wait;echo "tasks done $(date -u +%FT%TZ)"
