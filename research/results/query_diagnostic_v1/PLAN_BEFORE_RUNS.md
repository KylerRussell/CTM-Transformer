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
