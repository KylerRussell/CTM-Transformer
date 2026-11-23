#!/bin/bash
# Waits for one group of runs to end, or to fail, then prints what happened and exits.
# Usage: watch.sh lm|r1|reference|parity
cd /home/kasm-user/Documents/CTM-Transformer
LM=research/results/lm_revision/logs/worker_gpu1.log; LM0=research/results/recall_memory/logs/stage2_watch.log
RL=research/results/reference_ctm/logs; RO=research/results/reference_ctm
fail_lm()  { ls research/runs/revision/*/failed 2>/dev/null; grep -h "exit [1-9]" $LM $LM0 2>/dev/null; }
fail_ref() { grep -l "Traceback\|Error" $RL/queue_*.log 2>/dev/null; }
fail_par() { grep -l "Traceback\|rror" $RL/parity_run*.log $RL/after_lm.log 2>/dev/null; }
while true; do
  case $1 in
    lm) if grep -q "queue empty" $LM 2>/dev/null && grep -q "queue empty" $LM0 2>/dev/null; then echo "LM revision queue finished $(date -u)"; exit 0; fi
        f=$(fail_lm); [ -n "$f" ] && { echo "LM revision FAILURE $(date -u): $f"; exit 1; } ;;
    r1) [ -f $RO/R1_LR.json ] && { echo "Reference CTM R1 done $(date -u): $(cat $RO/R1_LR.json)"; exit 0; }
        f=$(fail_ref); [ -n "$f" ] && { echo "Reference CTM FAILURE $(date -u): $f"; exit 1; } ;;
    reference) n=$(ls $RO/R[123]_*.json 2>/dev/null | wc -l); [ $n -ge 44 ] && { echo "Reference CTM R1-R3 done (44 runs) $(date -u)"; exit 0; }
        f=$(fail_ref); [ -n "$f" ] && { echo "Reference CTM FAILURE $(date -u) after $n runs: $f"; exit 1; } ;;
    parity) grep -q "phase P done" $RL/after_lm.log 2>/dev/null && { echo "Parity reproduction finished $(date -u)"; exit 0; }
        f=$(fail_par); [ -n "$f" ] && { echo "Parity reproduction FAILURE $(date -u): $f"; exit 1; } ;;
  esac
  sleep 300
done
