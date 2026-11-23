#!/bin/bash
# Phase P of scripts/reference_ctm.py, and more reference workers, once the LM revision queue is empty on both GPUs
# (each scripts.lm_revision worker logs "queue empty" when it exits normally).
cd /home/kasm-user/Documents/CTM-Transformer
export LD_LIBRARY_PATH=~/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu
L=research/results/reference_ctm/logs; REF=$HOME/Documents/ctm-reference; PY=$HOME/.venvs/ctm-official/bin/python
until grep -q "queue empty" research/results/lm_revision/logs/worker_gpu1.log 2>/dev/null && grep -q "queue empty" research/results/recall_memory/logs/stage2_watch.log 2>/dev/null; do sleep 600; done
echo "LM queue empty $(date)"
for g in 0 1; do for k in 1 2; do
  PYTHONPATH=.:$REF TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor_ref_g${g}_$k setsid nohup $PY -u -m scripts.reference_ctm --queue --gpu $g > $L/queue_late_g${g}_$k.log 2>&1 < /dev/null &
done; done
parity() {  # the official script, unchanged but for RUN, the log directory and the interpreter
  run=$1; gpu=$2; A=$PWD; sed -e "s/^RUN=1/RUN=$run/" -e "s#^LOG_DIR=.*#LOG_DIR=\"$PWD/research/runs/reference_ctm/parity_ctm_75_25_run$run\"#" -e "s#^python -m#$PY -m#" \
    $REF/tasks/parity/scripts/train_ctm_75_25.sh > $L/train_ctm_75_25_run$run.sh
  (cd $REF && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REF bash $A/$L/train_ctm_75_25_run$run.sh > $A/$L/parity_run$run.log 2>&1)
}
parity 1 0 & parity 2 1 & wait
parity 3 0
echo "phase P done $(date)"
