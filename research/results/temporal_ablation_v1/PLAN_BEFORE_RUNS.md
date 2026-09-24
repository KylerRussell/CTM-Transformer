# CTM temporal-supervision ablation

## Plan recorded before training — 2026-09-22

The temporal audit found improving answer readouts across CTM ticks, intact gradients to every tick, and no previous auxiliary-supervision tuning in the shared runner. This experiment isolates direct CE weights across ticks. It does not test biological claims, CTM architecture superiority, or the full hyperparameter space.

| Recipe | Implementation | Direct CE weights at ticks 1–4 |
|---|---|---|
| `final` | `final_ce` | 0, 0, 0, 1 |
| `uniform` | `ramp_mono`, endpoints 1/1, penalty 0 | 1/4, 1/4, 1/4, 1/4 |
| `late` | `ramp_mono`, endpoints 0/1, penalty 0 | 0, 1/6, 1/3, 1/2 |

The `ramp_mono` implementation name is historical: monotonicity penalty is zero in both weighted recipes. Dynamic selection, distillation, depth curriculum, and all other extensions remain disabled. Its recent ignored-token fix is inactive in these three recipes.

### Fixed architecture, data, and training budget

All recipes start from fresh seed-17 initialization, with the same data/depth seeds and shuffled example order. Use the identical ordered variable-query eight-node one-hop dataset, 2,048 training maps and 128 original validation maps. Fully specified manifests are `research/configs/ctm_temporal_{final,uniform,late}_v1.json`; compared with `ctm_ordered_long_v1.json`, only objective type and ramp endpoints may differ.

Keep width 128, two layers, four thought ticks, eight-slot history, independent temporal MLPs, sparse-decay synchronization, shared head, batch 32, FP32 parameters/moments, BF16 autocast, AdamW, peak LR 1e-3, 30 warmup updates, cosine decay to 1e-4, and 3,000 updates. Each recipe consumes 96,000 examples, 5,184,000 real input tokens, and 192,000 answer/EOS labels. Auxiliary recipes apply some labels at multiple readouts; they do not see additional examples. Report this reuse and measured training time/peak memory. Equal update/token counts do not imply equal backward compute.

GPU 0 runs `final` followed by `late`; GPU 1 runs `uniform`. Both devices are RTX 3090s. Complete every budget regardless of intermediate outcomes. No change to model math is required; the shared runner now accepts finite nonnegative ramp weights with positive total and zero monotonicity penalty, and records per-tick supervised CE.

### Checks and selection rule

Before training, check actual weighted losses and gradients against masked per-tick token calculations, validate all recipes through the shared runner, reject invalid weight configurations, and verify objective-independent validation scores. The existing validation helper computes final-logit CE directly; preserve that behavior.

Select each recipe's checkpoint by minimum **original-validation final-tick answer/EOS CE**, evaluated every 100 updates including the final update, retaining the earlier checkpoint on exact ties. Rank recipes using that same metric; exact ties prefer final, then uniform, then late. Freeze the ranking after all three runs finish. Training loss is not comparable across these objectives.

Do not evaluate historical held-out test maps in this tuning milestone. Report original-validation greedy answer-plus-EOS accuracy, training and training-query probes, and all-eight-start validation accuracy with per-map all-correct counts. Original validation maps influenced checkpoint selection; expanded queries on them are secondary development measurements, not independent evidence. Report T=1/2/3/4 greedy validation accuracy at the same selected checkpoint as a diagnostic only; T=4 remains the inference policy and the only selection depth.

The earlier final-only run is a historical reproduction control. Report any difference in selected step, validation CE, and per-example outputs; do not silently substitute it for the fresh control. Source changes include the inactive dynamic-loss correction and new runner validation/logging. Test/core model math for the active final-only path is otherwise unchanged. Preserve source/config/data/checkpoint hashes, per-example outputs, curves, and all failed recipes.

### Interpretation planned before results

- Improvement in final-tick validation CE suggests objective sensitivity at this seed and budget, not a general claim about optimal CTM training.
- Better early readouts with worse final performance indicate a tradeoff under this schedule; they do not establish that intermediate supervision is always harmful.
- Similar results leave history/depth, synchronization initialization, optimizer, and curriculum choices unresolved.
- Aggregate success can hide incomplete per-map query coverage; retain all-start summaries.

The experiment is within CTM. Before a best-recipe architecture comparison, give each baseline an equal documented tuning budget with appropriate choices. Freeze any next intervention separately and keep future confirmatory data distinct from these development splits. Do not add seeds, longer depth, or another learning-rate trial inside this experiment.

## Reproduction

Use the environment and CUDA library path in [README.md](README.md).

```sh
python -m pytest tests/test_temporal_runner.py -q
```

Run independently:

```sh
python -u -m scripts.run_temporal_ablation --device cuda:0 --recipes final late
```

```sh
python -u -m scripts.run_temporal_ablation --device cuda:1 --recipes uniform
```
