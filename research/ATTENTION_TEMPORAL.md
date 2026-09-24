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

## Observed findings

Three GPU capture checks passed. Across all 9,216 query forwards, instrumented and native full logits were bitwise identical, and final answer tokens matched the prior all-query evaluation. Maximum normalized SDPA reconstruction errors were 0.00352 (Transformer) and 0.00337 (recurrent depth), below the recorded 0.02 tolerance. CTM probabilities were captured directly. Attention arrays and per-head/per-query measurements are retained.

The standard Transformer's first-layer query-label attention averages only 0.019 for B and 0.040 for D, versus 0.553 across all validation queries; H is 0.234. These observations align with its B/D/H weakness, but do not prove its cause. Recurrent depth's G query receives 0.931 query-label attention in the prelude and still scores only 55.5%. Attending to the query token is not sufficient for correct selection. CTM's F queries receive substantial query attention and still fail frequently, so a uniform query-attention explanation is inadequate.

For CTM, mean attention on the correct value in the second layer grows from 0.077 to 0.120 across ticks on correctly answered validation queries, and from 0.008 to 0.029 on wrong answers. These are descriptive head averages; attention mass on literal value positions does not exhaust the computation.

| CTM tick | Validation answer-token accuracy | Mean probability of correct answer |
|---|---:|---:|
| 1 | 16.5% | 0.165 |
| 2 | 35.2% | 0.331 |
| 3 | 57.3% | 0.538 |
| 4 | 65.1% | 0.606 |

The measured refinement makes temporal training a useful hypothesis to test, not proof that a particular auxiliary objective will help. Early readouts were not directly supervised. Their scores are answer-token measurements rather than intermediate-tick answer-plus-EOS evaluations.

### What the current CTM recipe actually exercises

- The shared runner uses **only final CE**, fixed T=4, no temporal penalty/curriculum/distillation, and a shared readout. The nominal ramp and curriculum fields in the manifest are inactive. The runner rejects attempts to enable those extensions, so a config edit alone cannot launch an auxiliary-loss comparison.
- Each layer owns a FIFO history of length 8, reset from learned parameters for every forward. At the attention calculation on ticks 1–4, 0/1/2/3 slots contain new post-activation observations. At the temporal MLP, 1/2/3/4 pre-activation slots are new. The remainder are learned initial-history slots, not necessarily zeros after training. Therefore H=8 is not equivalent to observing eight thought steps in this preset.
- The checkpoint has nonzero learned pre-history initializers (norms 0.825 and 0.862), but both post-history initializers remain exactly zero and have zero gradient in the audit. This is consistent with the bilinear synchronization product having zero first derivative when both factors are zero. Initialization is another separate design choice to investigate, not evidence that the live recurrent history is disconnected.
- Sparse-decay synchronization additionally weights older observations by learned decay parameters. This differs from the explicit finite history length. It should be inspected/ablated separately rather than conflated with H or T.
- The active path retains the computation graph across ticks; the comments mentioning periodic history detachment do not describe the measured path. Final-CE gradients to tick outputs are nonzero: 0.046388 / 0.040092 / 0.038910 / 0.049654 on the predetermined FP32 audit batch. Full per-slot and relevant parameter gradient norms are saved. This establishes connectivity on that batch, not optimal temporal credit assignment.
- Optimizer time is distinct: the recent run uses 3,000 updates, 30 warmup updates, and cosine decay from 1e-3 to 1e-4. Earlier legacy training supports other schedules and even a data-phase switch to dynamic temporal aggregation. Those legacy choices were not exercised by these research pilots. Prior run/config details from the user can refine the next ablation.

### Correctness issue found and fixed

The inactive `dynamic_aggregate` objective averaged zero losses for ignored prompt positions together with the supervised labels. In this task, 2 out of 54 positions are supervised; it therefore returned 0.031152 instead of the masked reference 0.841108, a factor of 1/27. The same dilution affected its per-tick monotonicity statistic. This could change effective gradient scale when task/prompt length changes.

The implementation now excludes ignored tokens from both reductions and rejects a batch with no supervised tokens. Five actual-model tests verify final/ramp/dynamic objectives and readout-weight gradients against independently masked per-token calculations, including nonzero monotonicity weights and an entirely ignored example. The corrected checkpoint audit returns 0.841108 exactly; final CE, ramp CE, and measured final-CE temporal gradients are unchanged. No optimizer update was made. Existing final-CE experiment results are unaffected. Legacy dynamic-loss training with ignored targets requires explicit version tracking; use the preserved pre-fix source to reproduce it.

The initial regression-test attempt referenced the sequential readout as a single linear layer; after correcting that test reference, all five checks passed. The independent pre-fix numerical audit already demonstrated the denominator error. Initial test logs are retained for transparency.

## Proposed next experiment: isolate temporal supervision first

Before changing H, T, the optimizer, or the input task simultaneously, add a narrowly validated auxiliary-loss option to the shared runner and compare three CTM recipes from fresh initialization:

| Recipe | Direct CE weights at ticks 1/2/3/4 | Purpose |
|---|---|---|
| Final-only control | 0 / 0 / 0 / 1 | Reproduce the current training objective |
| Uniform tick supervision | 1/4 / 1/4 / 1/4 / 1/4 | Test direct early supervision |
| Later-weighted supervision | 0 / 1/6 / 1/3 / 1/2 | Test refinement with a free first tick |

Keep T=4, H=8, model/data/seed, 3,000 updates, batch size, optimizer, and LR schedule fixed. Set monotonicity, distillation, and dynamic selection weights to zero. Verify gradients and masked-token normalization in the actual shared runner. Preserve objective-independent final-tick validation CE for checkpoint selection: the existing `evaluate_loss` helper already computes CE directly from final logits, separately from the model's training loss. Add a regression check that this remains true when auxiliary objectives are enabled. Record measured compute because auxiliary supervision adds backward work. Select recipes using validation only; keep the existing held-out development scores out of recipe selection.

This first ablation tests CTM training sensitivity within a fixed architecture. It cannot rank CTM against untuned baselines. Before making a best-recipe comparison, give each baseline the same documented trial budget and enable comparable temporal supervision for recurrent depth where applicable. A regular Transformer has no thought axis; use its own appropriate tuning choices rather than inventing equivalent ticks. Subsequent experiments can isolate H/T (and the cost of longer unrolls), then monotonicity or corrected dynamic selection, then depth curricula. Do not bundle them into one purported architectural improvement.

No ablation training has been launched. These candidate weights may be refined if the user's earlier successful temporal-training recipe becomes available.

## Artifacts

See [results](results/attention_temporal_v1/RESULTS.md), [figure](results/attention_temporal_v1/attention_ticks.png), and [PDF](results/attention_temporal_v1/attention_ticks.pdf). Each family has raw `.npz` probabilities/readouts and a `.json` index with checkpoint/source hashes and output-preservation audits. `ctm_temporal.json` records the pre-fix audit; `ctm_temporal_after_fix.json` records the corrected path. Snapshots preserve both model-source versions and identify which measurements used each version.

Reproduce attention capture with `python -m scripts.run_attention_diagnostic --device cuda:0 --families ctm` (GPU 1 for `transformer recurrent_depth`) in fresh output directories. Run `python -m scripts.audit_ctm_temporal --output <fresh-path>` for the no-update audit and `python -m scripts.summarize_attention_temporal` to rebuild the report. The archived pre-fix source is required to recreate the historical incorrect dynamic-loss measurement.
