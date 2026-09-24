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

To regenerate the report and audits after all runs finish:

```sh
MPLCONFIGDIR=/tmp/ctm-matplotlib python -m scripts.summarize_ordered_long
```

The report checks all 9,000 recorded learning rates and finite training losses/gradient norms, exact exposure counters, unchanged recorded training-source hashes, identical data/tokenizer/parameter counts, expected effective-config changes, best-validation checkpoint selection, and selected checkpoint hashes. Additional query-start breakdowns are descriptive diagnostics with small subgroup counts.

## Observed outcome

All three runs completed the 3,000-update budget on the two 3090s. Manifest/dataset preflight and all post-run audits passed; both GPUs are idle after evaluation. No model or training-kernel code changed. The only effective-config differences from the earlier runs are `max_steps` and the output directory.

| Model | Earlier ordered held-out (600) | Current ordered held-out (3,000) | Same maps shuffled | Selected update |
|---|---:|---:|---:|---:|
| Standard Transformer | 24.2% | 76.6% | 9.4% | 1,900 |
| Recurrent depth | 38.3% | 94.5% | 9.4% | 2,700 |
| CTM | 21.9% | 65.6% | 14.1% | 2,800 |

Recurrent depth passes the pre-run ordered lookup gate (121/128 correct). Transformer (98/128) and CTM (84/128) improve substantially over the shorter pilot but remain below the gate. The new decay horizon and larger exposure budget change together, so their individual contributions cannot be isolated. One seed, unequal parameter counts, and unequal compute prevent an architecture-superiority claim.

At the validation-selected checkpoints, training exact match is 86.9% / 96.6% / 82.4%. Changed-start accuracy on 256 familiar maps is 74.6% / 94.5% / 63.3%. The original answer persists on only 15 / 5 / 14 of these changed-start probes, unlike the complete query insensitivity of the fixed-copy-trained models. Variable-query learning has improved, but this does not reveal a unique mechanism. Ordered-to-shuffled transfer remains poor for all three; this does not test learning shuffled retrieval under sufficient training exposure.

The standard Transformer's validation CE reaches its minimum at update 1,900 and is higher at the end (0.385 → 0.519). CTM's best is near the end (0.470 at 2,800 versus 0.481 final); recurrent depth's is 0.116 at 2,700 versus 0.134 final. The curves do not justify assuming that another uniform increase in updates will solve the remaining gap. Both incomplete fitting and train/held-out differences remain relevant for the standard Transformer and CTM.

Per-start held-out counts suggest uneven lookup: the standard Transformer gets all A/C/E/F/G examples correct but only 1/11 for H, 6/15 for D, and 11/22 for B. CTM errors span several starts. These small, exploratory subgroups require validation-side follow-up rather than a claim of systematic failure.

**Next milestone:** diagnose query-dependent selection using the saved checkpoints. On training-probe and validation maps, evaluate all eight possible start nodes per map, retaining map-level pairing and distinguishing previously trained queries from changed queries. Compare error patterns and query sensitivity across all families; inspect the attention/readout path if those measurements identify a concrete failure to investigate. Treat this as a separate diagnostic, record its protocol before evaluation, and use validation for choosing the next intervention. Then define an equal validation-only tuning budget or a shared curriculum if warranted. Do not launch additional training, composition studies, or seed expansion from the current outcome alone. Future confirmatory data must remain separate from these repeatedly examined development splits.

See [full results and paired counts](results/ordered_long_v1/RESULTS.md), [learning curves](results/ordered_long_v1/learning_curves.png), and the [PDF figure](results/ordered_long_v1/learning_curves.pdf). The immutable [pre-run plan](results/ordered_long_v1/PLAN_BEFORE_RUNS.md), raw metrics, per-example predictions, selected checkpoint hashes, and `long_summary.json` audits are retained. `source_snapshot.zip` and its adjacent JSON preserve the verified source/config/environment/protocol/data-manifest snapshot; model checkpoints remain in `research/runs/ordered_long_v1/seed17`.

## Query diagnostic follow-up completed

The [all-start query diagnostic](QUERY_DIAGNOSTIC.md) is complete. All-start validation accuracy is 74.9% / 92.7% / 65.1% for Transformer / recurrent depth / CTM. Errors reveal concrete source-query weaknesses; the next milestone measures attention/readout behavior on frozen checkpoints with output-preservation checks.
