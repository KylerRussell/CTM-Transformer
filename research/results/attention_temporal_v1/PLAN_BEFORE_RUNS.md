# Attention and CTM temporal-training audit

## Protocol recorded before measurements — 2026-09-22

Continue the all-start diagnostic on the same 256 training-probe and 128 validation maps, all eight starts, using the same frozen seed-17 checkpoints. No training or checkpoint reselection. Validate hashes against the previous evaluations. Use GPU 0 for CTM and GPU 1 for both baselines.

Capture attention at the final prompt position (which predicts the answer), per head and per layer/call. CTM uses its actual attention-dropout output in eval mode. Baseline SDPA probabilities are reconstructed in FP32 from its actual Q/K tensors for the final query; compare the reconstructed attention output to the native output with maximum elementwise absolute error divided by (1 + abs(native)) <= 0.02. Return native outputs unchanged. Require bitwise-equal full logits between instrumented and native forwards for every batch and exact first-token agreement with prior greedy outputs. Preserve raw probability arrays, query/value/source positions, and all per-example metadata. Since previous outputs terminate correctly, this is an answer-position diagnostic; EOS-position attention is not analyzed.

Report correct-value attention, query-label attention, and total edge-value attention by event and correctness. Retain per-head measurements; head averages are descriptive. Heads can use source labels or contextual representations rather than directly attending to the answer value. Attention patterns do not establish a causal mechanism. In the standard Transformer, focal starts are B/D/H; in recurrent depth, G. Report all starts for every family. Canonical source label and position remain confounded.

CTM also exposes its actual shared-head logits at each of the four ticks. Report answer-token accuracy and probability of the correct answer by tick, separately from the prior answer-plus-EOS metric. Intermediate ticks were not directly supervised in these checkpoints. Do not select an inference depth from validation during this diagnostic.

## CTM temporal audit prompted by prior observations

The user's earlier observations suggest temporal learning choices may strongly affect CTM. Audit the implemented choices rather than assuming the current recipe is optimal or that biological motivation predicts performance. Distinguish within-forward thought ticks, FIFO history, loss weighting across ticks, and optimizer-update schedules.

Current preset: four ticks, eight-slot per-layer history, independent temporal MLPs, sparse-decay synchronization, shared head, final-tick CE, no monotonicity penalty, no distillation, no depth curriculum. The shared research runner explicitly rejects auxiliary temporal losses and depth curricula. Therefore the existing model comparison has not optimized those CTM-specific choices.

On a deterministic 32-example training-only batch (first four training-probe maps, all eight starts), perform a no-update gradient audit: measure gradients of final CE to each tick's output state and each temporal MLP's history input, plus initial-history and synchronization-decay parameters. Use FP32 for this gradient diagnostic and record it separately from the BF16 evaluation. Confirm that early ticks remain connected; gradient magnitudes alone do not prove adequate learning. Report how many history slots contain new observations versus learned initial history at each tick.

Compare the existing final, ramp, and dynamic-aggregate loss implementations numerically on the same frozen checkpoint and answer/EOS-masked batch. Validate losses against directly masked per-tick token losses; no training or hyperparameter selection. If an inactive loss path mishandles ignored tokens, record it as a blocker before using that objective. Historical pilot runs use final CE and must remain reproducible.

## Follow-up policy

Use findings to specify a temporal-objective ablation and, separately, history/depth or optimizer changes. A positive intervention must be trained from scratch and evaluated under recorded budgets; changing losses only at inference cannot establish improvement. Give recurrent-depth and ordinary Transformer baselines appropriate comparable tuning opportunities. Keep architecture-only and best-recipe comparisons distinct. Do not infer superiority from an untuned recipe, attention plots, or biological analogy.
