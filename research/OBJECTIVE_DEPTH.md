# Uniform versus dynamic aggregation across thought depth

## Plan recorded before new training — 2026-09-22

The user clarified that earlier CTM experiments showed little difference between uniform and dynamic temporal aggregation at T=4 but a large difference at T=16. The completed final/uniform/later-weighted T=4 experiment did not include dynamic aggregation and cannot test that statement. This experiment directly targets the objective-by-depth interaction.

### Four cells and controls

| Cell | Direct temporal objective | Trained T | Status at protocol freeze |
|---|---|---:|---|
| uniform_t4 | mean supervised-token CE over all ticks | 4 | Reuse frozen uniform run from temporal_ablation_v1 |
| dynamic_t4 | mean of per-token minimum-CE and maximum-certainty tick CE | 4 | New fresh run |
| uniform_t16 | mean supervised-token CE over all ticks | 16 | New fresh run |
| dynamic_t16 | same corrected dynamic objective | 16 | New fresh run |

Dynamic selections are made independently per supervised token, with equal weight on the two selected losses. The model's native entropy-based certainty is used for training. Ignore prompt/padding labels in both loss denominators. Monotonicity, distillation, curricula, and loss-weight schedules are disabled. The ignored-token correction was numerically verified before these runs; historical uncorrected masked-loss results would not be comparable.

Hold history length H=8, width/layers/head/synchronization settings, model parameter count, data, tokenizer, initialization seed 17, data seed 18, batch 32, optimizer, and update schedule fixed. Changing H with T would introduce another factor. In T=16 runs the FIFO eventually fills and evicts observations; T=4 does not fill it. This is part of the depth effect under fixed H, not a separate H comparison.

Each cell has 3,000 updates, 96,000 example presentations, 5,184,000 real input tokens, and 192,000 answer/EOS labels. AdamW peaks at 1e-3 after 30 warmup updates and decays to 1e-4. FP32 parameters/moments and BF16 autocast; full backpropagation through all ticks without checkpointing. T=16 has four times the nominal layer applications per example, so this is equal exposure, not equal compute. Profile actual time/memory and preserve all budgets.

The new shared-runner change admits masked dynamic aggregation and records its adaptive objective metadata. Uniform/final forward and optimizer math are unchanged. Reuse the completed uniform_t4 run and record its original source snapshot/hashes; do not claim identical runner file hashes across that historical boundary. Core model source is identical across all four cells.

### GPU feasibility and allocation

Five warmup plus twenty timed updates on the actual 32×54 answer-masked training batch measured median steps of 0.684 s (uniform T16) and 0.700 s (dynamic T16), with approximately 2.3 GiB peak allocated memory. Each projects about 35 minutes of training for 3,000 updates, excluding validation and evaluation; a short repeated-batch profile is not an exact completion forecast.

Run uniform_t16 on GPU 0. Run dynamic_t16 followed by dynamic_t4 on GPU 1. Both devices are RTX 3090s. Complete each fixed budget without tuning in response to partial results. Discard profiling model states; all experiment runs start fresh.

### Selection and outcomes

Select each checkpoint by minimum original-validation **final-trained-tick** answer/EOS CE, evaluated every 100 updates. Ties retain the earlier checkpoint. Primary contrasts are dynamic versus uniform within each depth. Report the interaction `(CE_dynamic16 − CE_uniform16) − (CE_dynamic4 − CE_uniform4)`, with an analogous accuracy contrast. This is a descriptive single-seed interaction; no significance claim or architecture ranking.

Evaluate only training/probe and original/all-start validation maps, never the held-out test split during tuning. All-start validation reuses the same maps and is secondary. At each checkpoint, report fixed-depth validation generation at T=1/2/4 and, for T16, T=8/16. These are diagnostics, not depth selection.

Also report **per-token minimum-entropy readout** for both objectives on original validation. Use the same FP32 entropy computed from each tick's logits; choose the earliest tick on ties, without gold labels. Rerun the full model on each generated prefix and select independently for the next token, including EOS. It uses all trained ticks and is not early stopping or a compute saving. This reflects dynamic aggregation's confidence component while avoiding an inference-policy advantage for one objective. Checkpoint selection remains final-tick CE for every cell. A future study could compare checkpoint-selection policies separately; this experiment does not optimize checkpoints for confidence readout.

### Checks and interpretation

Test both actual objectives and temporal-MLP gradients against masked per-token oracles at T4 and T16. Check confidence selection against the same label-free entropy rule and reject targets in the inference wrapper. Recheck shared-runner validation invariance. Audit initialization/config differences, all learning rates, budget counters, final-CE checkpoint selection, data/source/checkpoint hashes, and saved predictions.

A larger objective contrast at T16 supports depth sensitivity in this particular task/recipe; it does not establish generality or reproduce an unspecified historical setup. A null/reversed contrast motivates checking the earlier task, loss implementation, initialization, and optimizer rather than forcing the expected result. Keep H/initialization/optimizer changes separate. Before paper-level comparisons, replicate appropriately and give all architecture families equal tuning opportunities.

## Reproduction

Use the environment and CUDA library path in [README.md](README.md).

```sh
python -m pytest tests/test_objective_depth.py tests/test_temporal_runner.py -q
python -u -m scripts.run_objective_depth --device cuda:0 --cells uniform_t16
```

On GPU 1 independently:

```sh
python -u -m scripts.run_objective_depth --device cuda:1 --cells dynamic_t16 dynamic_t4
```

After those runs, evaluate the reused uniform T4 checkpoint with the same confidence policy, then generate the report:

```sh
python - <<'PYTHON'
import torch
from scripts.run_objective_depth import confidence_evaluation
torch.set_num_threads(4)
torch.cuda.set_device('cuda:0')
confidence_evaluation(
    'research/runs/temporal_ablation_v1/seed17/uniform/best.pt',
    'cuda:0', 'research/results/objective_depth_v1/uniform_t4.confidence.json')
PYTHON
MPLCONFIGDIR=/tmp/ctm-matplotlib python -m scripts.summarize_objective_depth
```

Training and confidence-evaluation commands require fresh output paths. To reproduce the reused cell, first follow [the T4 temporal-ablation protocol](TEMPORAL_ABLATION.md). The new snapshot records the prior source archive explicitly because the training-runner permissions changed between studies.

## Historical recipe search

The workspace contains `train_dual_gpu_teacher_rampmono.log` and `train_dual_gpu_teacher_dynamicaggregate.log`, whose headers specify 8 thought ticks, cached teacher distillation, 8-bit AdamW, batch 2 and sequence length 512. `logs/distill_12T_10k.log` instead specifies T12/H12 with width 1024 and 24 layers. These are different recipes from the small, undistilled controlled task here. The inspected logs did not identify a matched uniform/dynamic T16 comparison. This search does not establish that the earlier experiment is absent; its exact configuration remains unspecified.

## Required follow-up before paper-level claims

- Separate objective choice, training depth, history length, and inference readout in the experiment registry. A favorable or unfavorable result at T4 cannot settle the T16 question.
- Replicate the frozen four-cell comparison across additional initialization/data-order seeds before interpreting an interaction as reproducible. Keep the same validation rule and reserve the test split until recipe selection is complete.
- If confidence readout is the deployment target, study checkpoint selection for that policy explicitly with a prespecified validation criterion. The present checkpoints were selected for final-tick CE.
- Recover the earlier T16 recipe before claiming replication: task/data, T/H, temporal loss and masking, readout policy, initialization, optimizer/learning rate, distillation, and training budget all matter.
- Give the Transformer and recurrent-depth baselines comparable tuning budgets before architecture comparisons. Report equal-exposure and compute-matched results separately.

## Completed runs and subsequent clarification — 2026-09-23

All four 3,000-update cells are complete (uniform T4 reused). Fifteen pre-run checks passed; all 12,000 update records, exposure/LR budgets, checkpoint choices, and original-query consistency were audited. See [frozen-protocol results](results/objective_depth_v1/RESULTS.md).

The user's subsequent clarification makes best-across-ticks selection central. Final-tick checkpoint selection is therefore an incomplete assessment of the historical recipe. A separately labeled follow-up finds dynamic confidence accuracy of 54.7% (T4) and 70.3% (T16) at the fixed final update, compared with uniform 80.5% and 76.6%. This materially changes the interpretation of the original selected-checkpoint scores. See [implementation history, policy diagnosis, and next steps](results/objective_depth_v1/READOUT_FOLLOWUP.md). The exact earlier winning version remains unidentified; do not claim its replication or failure.
