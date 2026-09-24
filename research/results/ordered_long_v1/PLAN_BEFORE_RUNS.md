# Longer-budget ordered variable-query lookup

## Plan recorded before training — 2026-09-22

Fixed-slot copying reached 100% held-out accuracy for every model. The earlier ordered variable-query task reached only 24.2% / 38.3% / 21.9% at 600 updates (Transformer / recurrent depth / CTM). This experiment returns to that exact variable-query dataset with a longer, explicitly recorded optimization schedule.

### Frozen choices

- Reuse `research/data/ordered_pointer_v1` unchanged: 2,048 training maps, 128 validation maps, 128 held-out maps; eight-node cycles, one hop, sorted source labels, variable start queries. Train/validation/held-out maps are disjoint. Paired probes intentionally reuse maps.
- Use new fully specified `research/configs/{family}_ordered_long_v1.json` manifests. The only effective preset field changed from `*_algorithmic_v1.json` is `max_steps`: 600 → 3,000.
- Start from fresh initialization at seed 17, with the same independent data/depth seeds, example ordering policy, width 128, batch 32, architecture, and precision. Do not resume or initialize from fixed-copy checkpoints.
- Keep AdamW, peak learning rate 1e-3, warmup 30 updates, clipping, weight decay, and BF16 autocast with FP32 parameters/moments unchanged. Linear warmup reaches 1e-3 at update 30. Then cosine decay spans updates 30–3,000 and reaches 1e-4 at update 3,000. For completed update s > 30, LR = 1e-3 × [0.1 + 0.45 × (1 + cos(pi × (s−30)/2970))].
- Each run consumes 96,000 example presentations, 5,184,000 real input tokens, and 192,000 supervised answer/EOS tokens: five times the old exposure budget. Model parameter counts and compute remain unmatched.
- Validate every 100 updates. Select the checkpoint with the lowest validation answer/EOS cross-entropy (strict improvement; ties keep the earlier checkpoint), then evaluate all original splits with unrestricted greedy answer generation followed by EOS. Test scores do not select checkpoints or stop runs.
- CTM uses GPU 0; standard and recurrent-depth Transformers run sequentially on GPU 1. Both devices are RTX 3090s. Keep all three fixed budgets, regardless of intermediate results.

The new decay horizon changes learning rates before update 600. This is an extended-budget/schedule experiment, not a continuation or an isolated test of extra updates. Comparison with historical results is exploratory and reflects both changes. No curriculum, optimizer sweep, depth sweep, additional seed, or additional data is introduced.

### Development gate and probes

Primary gate: **at least 90% exact match on 128 held-out ordered variable-query maps**, including EOS. This establishes one-hop positional lookup in this setup, not permutation-invariant graph retrieval or multi-step composition. Uniform guessing among non-start nodes has expected accuracy 1/7.

Report training and validation scores, `test_id`, paired `test_shuffled`, and the familiar-map `train_probe`, `train_reordered`, `train_new_query` scores. Retain paired correctness counts and original-answer persistence under changed starts. Unlike the fixed-copy control, variable starts are present during training here. Preserve all per-example outputs, raw validation curves, hashes, selected checkpoints, and a verified source snapshot. Plot both old and new learning curves with their distinct schedules labeled.

### Interpretation planned before results

- If every model passes ordered lookup, test learning one-hop retrieval with shuffled edge presentation under a separately recorded budget before returning to path composition. Low shuffled transfer from an ordered-trained model does not establish inability to learn shuffled retrieval.
- If some models pass, record that the current development settings are adequate for those runs, without claiming superiority. Inspect learning/overfitting and query-dependence for the remaining runs before selecting another shared training intervention.
- If none passes, inspect query-conditioned attention/representations and define a separate curriculum or task-format experiment with equal tuning opportunities.
- A train/held-out gap points to a generalization problem under this setup; low train and held-out performance leaves optimization/learning unresolved. Decreasing loss at the cutoff does not establish convergence.

These are one-seed development results. No matched-budget main comparison or confirmatory replication is authorized as part of this milestone.

## Reproduction

Use the environment and CUDA library path in [README.md](README.md). Run the two commands independently:

```sh
python -u -m scripts.run_pointer_diagnostic --dataset_root research/data/ordered_pointer_v1 --mode ordered --config_tag ordered_long_v1 --device cuda:0 --families ctm --run_root research/runs/ordered_long_v1/seed17 --results_root research/results/ordered_long_v1
```

```sh
python -u -m scripts.run_pointer_diagnostic --dataset_root research/data/ordered_pointer_v1 --mode ordered --config_tag ordered_long_v1 --device cuda:1 --families transformer recurrent_depth --run_root research/runs/ordered_long_v1/seed17 --results_root research/results/ordered_long_v1
```
