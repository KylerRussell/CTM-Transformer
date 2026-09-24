# All-start query-selection diagnostic

## Plan recorded before evaluation — 2026-09-22

Use the frozen validation-selected checkpoints from `ordered_long_v1`, seed 17: Transformer update 1,900, recurrent depth 2,700, CTM 2,800. Verify the exact checkpoint hashes against their earlier evaluations. No training, checkpoint reselection, depth sweep, architecture changes, or tuning occurs in this diagnostic.

Enumerate every start A–H on each of the existing 256 ordered training-probe maps and 128 ordered validation maps. Preserve the graph, canonical edge order, one-hop depth, prompt length, and source row order. Recompute the answer from each graph. This gives 2,048 training-map queries and 1,024 validation-map queries per model. Keep each start in a separate split so graph identity stays unique within a split; pair the eight splits by graph id.

One of eight queries per training map was presented during training; the other seven are new queries on a familiar map. Validation maps were excluded from gradient updates, but their original queries selected checkpoints. Mark these as original-validation versus changed-validation queries, never as trained queries or an independent test set. The 128 historical held-out maps are not evaluated in this diagnostic.

Use the existing greedy evaluator unchanged, BF16 autocast with FP32 weights, batch 32, trained depth, and exact answer plus EOS. CTM runs on GPU 0; both baselines run sequentially on GPU 1. Both devices are RTX 3090s. Source/file/checkpoint hashes and full predictions are retained.

## Analyses fixed before results

- Overall, per-start, and exposure-group exact match, with counts.
- Fraction of maps with all eight queries correct, and the distribution of 0–8 correct queries per map.
- Output sensitivity: count pairs of starts yielding identical generated output and termination status (28 pairs per map); report invalid outputs separately. Repeated predictions are not evidence that internal states are identical.
- Original-answer persistence: among seven changed queries per map, count exact generation of the original query's correct answer with EOS.
- Output-implied source confusion: each eight-node cycle has unique successor values. Invert the map to identify which source has the generated value as its successor. A correct output lands on the diagonal. Include an invalid-output column. This is a behavioral description, not an attention measurement or proof that the model retrieved that edge.
- Reproduction audit: compare the original-query predictions against the earlier evaluation, including answer, correctness, output, and EOS status. Investigate any discrepancies before drawing conclusions.

All eight queries share a map, so they are not independent experimental replicates. Report descriptive map-level summaries; no significance test or architecture ranking is planned. The checkpoints differ in size and compute, use one training seed, and validation influenced their selection.

Construction checks cover complete query coverage, pairing/disjointness, answer correctness, exposure labeling, reproducibility, immutability, missing-start rejection, and corrupted-label rejection. Aggregation checks use perfect and constant-output synthetic predictors to verify the sensitivity and source-confusion calculations independently of model output.

## Interpretation and next action

If errors concentrate on certain starts across validation maps, inspect the query-to-source selection path for those starts before picking a training change. Label and position are coupled in canonical order; this experiment cannot separate those effects. If new queries on training maps lag original trained queries, report the gap without attributing it uniquely to memorization. Broad errors call for an optimization/generalization inspection. Successful variable-query behavior alone does not establish arbitrary edge-order retrieval or multi-step reasoning.

Use this diagnostic to specify a concrete follow-up attention/readout check or equal validation-only tuning/curriculum budget. Any further intervention is a separate experiment. Keep future confirmatory data separate from these repeatedly examined development maps.

## Reproduction

Use the environment and CUDA library path in [README.md](README.md).

```sh
python -m pytest tests/test_query_diagnostic.py tests/test_query_analysis.py -q
python -m scripts.generate_query_diagnostic
```

Run independently on the two GPUs:

```sh
python -u -m scripts.run_query_diagnostic --device cuda:0 --families ctm
```

```sh
python -u -m scripts.run_query_diagnostic --device cuda:1 --families transformer recurrent_depth
```

To regenerate the report and figures:

```sh
MPLCONFIGDIR=/tmp/ctm-matplotlib python -m scripts.summarize_query_diagnostic
```

## Observed outcome

Four construction tests and four independent aggregation tests passed. All three GPU evaluations completed, totaling 9,216 generated answers. All 1,152 original-query outputs (384 per model) exactly reproduce the earlier predictions, answers, correctness, and EOS status. Checkpoint and dataset hashes match; no training or checkpoint reselection occurred.

| Model | All starts on training maps | All starts on validation maps | Validation maps with all eight correct |
|---|---:|---:|---:|
| Transformer | 74.9% | 74.9% | 0/128 |
| Recurrent depth | 94.0% | 92.7% | 64/128 |
| CTM | 67.3% | 65.1% | 0/128 |

All outputs are valid node labels followed by EOS. The remaining errors therefore concern which value is returned, rather than output formatting or termination. The all-eight metric is stricter than aggregate accuracy: even recurrent depth, which passes the earlier 90% average lookup gate, solves every start on only half these validation maps.

### Concrete behavioral findings

- **Transformer:** A, C, F, and G are correct on all 128 validation maps, while B/D/H reach only 33.6% / 43.8% / 32.8%. These three starts account for 243 of 257 validation errors. The predicted values overwhelmingly belong to successors of B, D, or H for those queries. Similar per-start weaknesses occur on training maps. This supports a targeted query-selection investigation; it does not distinguish label effects from positional effects.
- **Recurrent depth:** G reaches 55.5% and accounts for 57 of 75 validation errors. Most other starts are perfect or above 90%. Its aggregate score conceals an incomplete source-specific skill.
- **CTM:** E is perfect; B/F/H are below 50%, and errors occur across several other starts. The model shows broader incomplete lookup rather than a single failed start. Original trained queries score 83.6%, versus 65.0% for changed queries on the same training-map set. This descriptive exposure gap does not uniquely establish memorization and does not control every difference in query composition.

Identical-output start pairs occur on 8.5% / 2.1% / 7.6% of validation start pairs (Transformer / recurrent depth / CTM). Changed queries return the original query's correct answer on only 4.7% / 1.5% / 5.4%. Thus these checkpoints are responsive to query changes; their failures differ from the fully query-insensitive fixed-copy control. The source-confusion heatmaps describe generated answers, not measured attention or a proven internal mechanism.

### Next milestone

Perform a separately documented attention/readout diagnostic on the frozen checkpoints, using these training-probe and validation maps. Trace query-to-edge attention at the answer position across layers/recurrence steps for all starts, with the Transformer B/D/H cluster and recurrent-depth G as prespecified focal cases, and CTM evaluated under the same reporting policy. Compare successful and failed queries; verify that instrumentation preserves logits/predictions. If instrumentation uses recomputed attention probabilities, document numerical precision and verify equivalence of the underlying attention output before interpretation. Attention patterns alone are descriptive; a mechanism claim needs a controlled intervention.

Use that evidence to choose an equal validation-only optimization sweep or a shared query-coverage curriculum. Do not launch additional training, expand seeds, or claim multi-step reasoning yet. Retain the confounding between source label and canonical position, and reserve new data for future confirmatory comparisons.

See [full results](results/query_diagnostic_v1/RESULTS.md), [diagnostic figure](results/query_diagnostic_v1/query_patterns.png), and [PDF](results/query_diagnostic_v1/query_patterns.pdf). The immutable [pre-evaluation plan](results/query_diagnostic_v1/PLAN_BEFORE_RUNS.md), generated dataset, full per-query predictions, per-map summaries, exact reproduction audits, source hashes, and environment snapshot are retained. `source_snapshot.zip` preserves the verified source/config/protocol/data-manifest snapshot; the existing selected checkpoints remain under `research/runs/ordered_long_v1/seed17`.

## Follow-up completed

The [attention and CTM temporal audit](ATTENTION_TEMPORAL.md) is complete. It preserves native outputs, measures tick-wise refinement and gradients, and fixes an ignored-token denominator bug in an inactive temporal loss. The next proposed experiment isolates temporal supervision before changing history, depth, or optimizer settings.
