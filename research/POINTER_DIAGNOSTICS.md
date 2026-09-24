# Pointer-learning diagnostics v1

## Plan recorded before training — 2026-09-22

The first mixed-depth pilot stayed near guessing even for one-hop queries. The next experiment separates ability to fit fixed examples from ability to retrieve an edge on a held-out map. It uses the existing character tokenizer, eight-node maps, prompt format, optimizer, precision, and model presets.

Two conditions, each run for all three model families with initialization seed 17:

1. **Tiny-set fitting:** select 32 original training examples, four for each answer label, retaining the original mixture of path depths 1–3. Train for 600 updates and score the final checkpoint on the training examples. Validation remains disjoint and does not select this diagnostic's checkpoint.
2. **One-hop lookup:** keep all 2,048 original training maps, query start nodes, and shuffled edge orders, but change every path depth to 1 and recompute the answer. Make the same change to the original 128-example validation and 128-example held-out sets. Train for 600 updates and select the checkpoint with lowest validation answer/EOS cross-entropy.

Both use batch 32, learning rate 1e-3, 30 warmup updates, and the unchanged `{transformer,recurrent_depth,ctm}_algorithmic_v1.json` presets. Each run sees 19,200 example presentations. Tiny fitting therefore repeats the same 32 examples 600 times; it is intentionally a memorization diagnostic, not a generalization experiment. GPU 0 runs the tiny condition; GPU 1 runs the one-hop condition, with the three families sequential on each GPU.

Development gates chosen before these runs: at least 95% training exact match for tiny fitting; at least 90% held-out exact match to describe one-hop lookup as established. These are practical development gates, not preregistered paper hypotheses or significance thresholds. Parameter counts and compute are still unmatched. There is no hyperparameter search or seed expansion in this diagnostic.

## Paired interventions

After training, evaluate unchanged training examples alongside two controlled modifications:

- **Edge order:** shuffle the presentation of the same edges, keeping graph, start node, path depth, and answer fixed.
- **New query:** choose a different start node on the same graph, retaining presentation and path depth; recompute the endpoint. In a single cycle this changes the correct answer.

The tiny condition probes all 32 training maps. The one-hop condition uses a seeded subset of 256 training maps for paired interventions; full training-set accuracy is also recorded. Probe sets intentionally share graph identities with training and are labeled `train_probe`, `train_reordered`, and `train_new_query`. They are not held-out-map scores. Train, validation, and `test_id` still have disjoint maps.

The new data are derived deterministically from the original suite with seed 20260922 (tiny) and 20260923 (one-hop). Their manifests preserve the original manifest hash, split file hashes, generator source hash, counts, and intervention roles. Assertions verify that interventions change only the designated factor. The prior pilot's data, runs, and source archive remain intact.

## Evaluation and interpretation

All scores use the existing unconstrained greedy answer-plus-EOS exact match. Every prediction and termination flag is retained. The existing independent pointer oracle already establishes correct scoring on different sizes and path depths; the new construction tests verify the revised labels and paired controls.

- High tiny training accuracy with low modified-input accuracy supports memorization of the fixed examples.
- High held-out one-hop accuracy establishes single-edge retrieval in this particular representation and training regime.
- Failure of one-hop lookup means the mixed-depth pilot cannot isolate a failure of composition alone.
- Different outcomes across architectures at one seed and unmatched budgets are diagnostic observations, not superiority claims.

## Commands

Activate the environment and CUDA driver-library path in [README.md](README.md).

```sh
python -m scripts.generate_pointer_diagnostics \
  --source_root research/data/algorithmic_v1 \
  --output_root research/data/pointer_diagnostics_v1
```

In separate terminals:

```sh
python -u -m scripts.run_pointer_diagnostic \
  --dataset_root research/data/pointer_diagnostics_v1/tiny --mode tiny --device cuda:0 \
  --run_root research/runs/pointer_diagnostics_v1/tiny_seed17 \
  --results_root research/results/pointer_diagnostics_v1
```

```sh
python -u -m scripts.run_pointer_diagnostic \
  --dataset_root research/data/pointer_diagnostics_v1/onehop --mode onehop --device cuda:1 \
  --run_root research/runs/pointer_diagnostics_v1/onehop_seed17 \
  --results_root research/results/pointer_diagnostics_v1
```

Outputs require fresh directories/files. The training implementation is unchanged. The evaluator now accepts an explicit checkpoint-selection description so final-checkpoint tiny fitting cannot be confused with best-validation evaluation.

## Observed outcomes

All six runs completed. Four new construction tests passed, covering preservation of the original maps, balanced tiny selection, paired interventions, byte reproducibility, and rejection of interventions that change two factors. The GPU runs exercised the unchanged training path and the evaluator's explicit checkpoint-selection metadata.

| Model | Tiny training accuracy | Tiny reordered edges | Tiny new start | Held-out one-hop accuracy |
|---|---:|---:|---:|---:|
| Standard Transformer | 100.0% | 25.0% | 6.2% | 14.1% |
| Recurrent depth | 100.0% | 15.6% | 3.1% | 8.6% |
| CTM | 100.0% | 21.9% | 3.1% | 14.8% |

All models pass the tiny-fitting gate and fail the one-hop gate. The fitting result demonstrates that this training path can fit these 32 examples. Changing their presentation or query largely destroys accuracy. When a different start requires a different answer, the models still emit the old answer on 24/32, 21/32, and 21/32 examples respectively. This supports memorization of the fixed training examples rather than transferable graph lookup in these runs.

Held-out one-hop results remain near simple guessing references (12.5% uniform nodes; 14.3% excluding the start node). A training-majority-label predictor reaches 18.0% on this small one-hop held-out set. No architecture passes the retrieval control. Thus the earlier mixed-depth failure cannot be attributed specifically to multi-step composition, and these results are not evidence of a CTM-specific deficit or a comparative ranking. They concern one representation, seed, and training budget.

**Next experiment:** keep the same one-hop maps and queries, but place edges in canonical source-node order. This tests easier fixed-position lookup before returning to shuffled maps. Preserve the failed shuffled-map runs. If ordered lookup succeeds, study the transition to shuffled edge order as a representation/learning control. If ordered lookup also fails, add a direct value-copy control and inspect optimization before increasing reasoning depth, seeds, or model size. Do not launch paper-scale comparisons until held-out retrieval is established.

See the [full results](results/pointer_diagnostics_v1/RESULTS.md), [learning curves](results/pointer_diagnostics_v1/learning_curves.png), and [PDF figure](results/pointer_diagnostics_v1/learning_curves.pdf). The prior [pre-run plan](results/pointer_diagnostics_v1/PLAN_BEFORE_RUNS.md) is retained separately. Exact source, configuration, environment, and data-manifest snapshots are in `results/pointer_diagnostics_v1/source_snapshot.zip`, with hashes in its adjacent JSON. Model/optimizer checkpoints remain in the run directories.

To regenerate the report:

```sh
python -m scripts.summarize_pointer_diagnostics \
  --results_root research/results/pointer_diagnostics_v1 \
  --run_root research/runs/pointer_diagnostics_v1 \
  --dataset_root research/data/pointer_diagnostics_v1
```

The ordered-edge follow-up is now complete: [ordered lookup control](ORDERED_POINTER.md). Scores improve partially but no model passes the 90% gate. Direct fixed-slot value copying is the next diagnostic.
