# Ordered-edge pointer control

## Plan recorded before training — 2026-09-22

The earlier shuffled-map one-hop control did not establish lookup for any model, although all models could memorize 32 fixed examples. This experiment changes **only edge presentation order** in the one-hop datasets: source nodes always appear as A, B, …, H. Maps, split assignments, query starts, labels, row order, prompt lengths, tokenizer, models, and optimization remain the same.

The task now permits fixed-position lookup. Success establishes that simpler capability, not permutation-invariant graph retrieval. The primary development gate remains at least **90% exact match on 128 held-out maps**. This is a practical exploratory threshold, not a paper hypothesis or significance test.

All three families use the unchanged `{transformer,recurrent_depth,ctm}_algorithmic_v1.json` presets: width 128, initialization seed 17, batch 32, 600 updates, learning rate 1e-3, 30 warmup updates, BF16 autocast with FP32 weights and AdamW state. There are 2,048 training maps and 128 validation maps. Each run sees 19,200 example presentations in the same order. Best validation answer/EOS cross-entropy selects a checkpoint. Model parameters and compute remain unmatched.

GPU 0 runs CTM; GPU 1 runs the standard and recurrent-depth Transformers sequentially. No tuning, extra seeds, depth sweep, or longer budget is introduced.

## Paired probes

- `test_id`: held-out maps in canonical edge order; primary endpoint.
- `test_shuffled`: exactly the same held-out maps, queries, and answers in the original shuffled order. This tests order transfer without changing graph identity or prompt length.
- `train_probe`: the same 256 training maps used by the previous one-hop diagnostic, now ordered.
- `train_reordered`: shuffled versions of those training maps.
- `train_new_query`: the earlier changed-start queries, now ordered. This checks query-dependent lookup on familiar maps.

Only train, validation, and held-out graph identities are disjoint. Paired probes intentionally reuse the appropriate maps. If a source random presentation happens to equal canonical order, a one-position rotation makes its order probe different while preserving semantics. Four construction tests check pairing, identical source problems and example order, reproducibility, and rejection of an intervention that changes the query.

All scores use unconstrained greedy generation of the answer followed by EOS. Exact predictions, termination flags, source hashes, data hashes, selected checkpoint, and learning curves are retained. Historical runs and source archives remain separate.

## Interpretation planned before results

- Passing ordered lookup while failing shuffled transfer identifies a fixed-position skill and leaves arbitrary edge retrieval unresolved.
- Passing ordered and shuffled held-out evaluation would justify returning to path composition, followed by replication.
- Failing ordered lookup calls for a direct value-copy control and optimization inspection before spending more on composition or seed expansion.

Outcomes are observations at one seed and budget, not evidence of architecture superiority.

## Commands

Use the environment and CUDA library path from [README.md](README.md).

```sh
python -m scripts.generate_ordered_pointer \
  --source_root research/data/pointer_diagnostics_v1/onehop \
  --output_root research/data/ordered_pointer_v1
```

Run independently on the two GPUs; their model directories must be fresh:

```sh
python -u -m scripts.run_pointer_diagnostic \
  --dataset_root research/data/ordered_pointer_v1 --mode ordered --device cuda:0 \
  --families ctm --run_root research/runs/ordered_pointer_v1/seed17 \
  --results_root research/results/ordered_pointer_v1
```

```sh
python -u -m scripts.run_pointer_diagnostic \
  --dataset_root research/data/ordered_pointer_v1 --mode ordered --device cuda:1 \
  --families transformer recurrent_depth --run_root research/runs/ordered_pointer_v1/seed17 \
  --results_root research/results/ordered_pointer_v1
```

## Observed outcome

Four construction checks passed and all three GPU runs completed. Example, real-input-token, and supervision budgets match the earlier shuffled-training control exactly. No model reaches the pre-run 90% lookup gate.

| Model | Earlier shuffled training / held-out | Ordered training / ordered held-out | Ordered training / same held-out maps shuffled |
|---|---:|---:|---:|
| Standard Transformer | 14.1% | 24.2% | 15.6% |
| Recurrent depth | 8.6% | 38.3% | 13.3% |
| CTM | 14.8% | 21.9% | 8.6% |

The first two columns compare separately trained models on the same underlying queries. The last two columns are paired evaluations of each ordered-trained checkpoint on the exact same 128 held-out maps and queries. Presentation changed; the correct answer did not.

Ordering improves the observed held-out scores in this pilot, but it does not establish reliable fixed-position lookup. Shuffling reduces accuracy toward guessing, and query changes on training maps also produce low scores. The effect is consistent with partial use of positional regularities. It does not prove a particular learned mechanism, identify an architectural limit, or establish a model ranking. In particular, the baseline validation losses were still falling at the 600-update cutoff; convergence and adequate tuning have not been established.

**Next gate:** a direct value-copy control, using ordered maps and a fixed start node so the requested successor always occupies a fixed slot. Retain the maps/splits, prompt length, output vocabulary, and fixed-budget reporting. This separates copying a value from selecting its slot using a variable start query. If copying works but conditional lookup remains weak, inspect query dependence and consider a documented longer-budget or curriculum experiment with equal tuning opportunities. If copying fails, inspect the attention/readout and optimization path first. Keep all failed controls in the record; do not move to multi-step reasoning claims or expand seeds yet.

See [full results and paired counts](results/ordered_pointer_v1/RESULTS.md), [learning curves](results/ordered_pointer_v1/learning_curves.png), and the [PDF figure](results/ordered_pointer_v1/learning_curves.pdf). The immutable [pre-run plan](results/ordered_pointer_v1/PLAN_BEFORE_RUNS.md), per-example predictions, checkpoint hashes, and raw metrics are retained. `source_snapshot.zip` and its adjacent JSON preserve the verified source/config/environment/data-manifest snapshot; model checkpoints remain under the run directories.

To regenerate the report and figure:

```sh
python -m scripts.summarize_ordered_pointer \
  --results_root research/results/ordered_pointer_v1 \
  --run_root research/runs/ordered_pointer_v1/seed17 \
  --previous_results research/results/pointer_diagnostics_v1
```

## Follow-up completed

The [fixed-slot copy control](FIXED_POINTER.md) now passes at 100% held-out accuracy for all three families. Shuffled and unseen-query transfer remain weak. The next development experiment returns to variable-query lookup with an explicitly documented longer training budget.
