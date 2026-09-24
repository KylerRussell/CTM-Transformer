# Fixed-slot value-copy control

## Plan recorded before training — 2026-09-22

The ordered-edge pilot did not reach reliable one-hop lookup. This control keeps its maps, split assignments, example order, canonical edge presentation, prompt lengths, tokenizer, and model/optimization presets, while changing every training, validation, and primary held-out query to `start A;steps 1`. The correct answer is now the successor token in the first edge `A>value`. Answers are recomputed, so this changes the query/label distribution relative to ordered lookup.

The primary development gate is **at least 90% exact match on 128 unseen maps**, including the generated EOS. This tests fixed-slot copying. It cannot establish arbitrary query selection, graph retrieval, composition, or model superiority. Because each map is a single eight-node cycle, A cannot point to itself; uniform guessing among the seven possible answers yields 1/7 expected accuracy. Empirical answer counts are preserved in the manifest.

Use the unchanged three `*_algorithmic_v1.json` presets, seed 17, batch 32, 600 updates, learning rate 1e-3, warmup 30, BF16 autocast with FP32 weights and AdamW. There are 2,048 training maps and 128 validation maps. Each run sees 19,200 examples, 1,036,800 real input tokens, and 38,400 supervised answer/EOS tokens. Select the checkpoint with lowest validation answer/EOS cross-entropy, checked every 100 updates. Parameters and compute remain unmatched. CTM runs on GPU 0; the standard and recurrent-depth Transformers run sequentially on GPU 1. Both devices are RTX 3090s.

## Paired probes

- `test_id`: ordered unseen maps, fixed query A; primary endpoint.
- `test_shuffled`: the same unseen maps and query A, with shuffled edge presentation.
- `test_new_query`: the same ordered unseen maps, querying B instead of A.
- `train_probe`, `train_reordered`, `train_new_query`: the corresponding interventions on 256 familiar training maps.

Query B is deliberately absent from training and validation. Its accuracy is an out-of-distribution probe, not a learning gate. In a cycle, different source nodes have different successors, so every query change changes the correct answer. Repeated predictions of the original A successor will be counted explicitly. The probe does not test variable-query learning under suitable training exposure.

Construction checks cover correct first-slot labels, retained map/split/order/length, paired interventions, graph disjointness, reproducibility, immutability, and rejection of semantically incorrect controls. Evaluation uses unrestricted greedy generation with answer plus EOS; retain per-example predictions, hashes, checkpoints, curves, and a source snapshot. No tuning, extra seeds, depth sweep, or longer budget is introduced within this control.

## Interpretation planned before results

- If copying passes while earlier variable-query lookup is weak, basic value copying is learnable under these presets. Next investigate query-conditioned selection with equal optimization opportunities; a documented longer-budget experiment or curriculum may be appropriate.
- If copying fails, inspect its learning curve and attention/readout/optimization path before increasing task complexity.
- Passing shuffled or unseen-query probes would suggest additional transfer, requiring follow-up rather than a mechanism claim.

Do not expand seeds or claim multi-step reasoning until held-out query-dependent retrieval is established. These are exploratory outcomes at one seed and budget.

## Reproduction

Use the environment and CUDA library path from [README.md](README.md).

```sh
python -m scripts.generate_fixed_pointer --source_root research/data/ordered_pointer_v1 --output_root research/data/fixed_pointer_v1
python -m pytest tests/test_fixed_pointer.py -q
```

Run these independently on the two GPUs with fresh output directories:

```sh
python -u -m scripts.run_pointer_diagnostic --dataset_root research/data/fixed_pointer_v1 --mode fixedcopy --device cuda:0 --families ctm --run_root research/runs/fixed_pointer_v1/seed17 --results_root research/results/fixed_pointer_v1
```

```sh
python -u -m scripts.run_pointer_diagnostic --dataset_root research/data/fixed_pointer_v1 --mode fixedcopy --device cuda:1 --families transformer recurrent_depth --run_root research/runs/fixed_pointer_v1/seed17 --results_root research/results/fixed_pointer_v1
```

## Observed outcome

Five construction checks passed. All three 600-update GPU runs completed, and each selected its step-600 checkpoint. Budget counters match the prior ordered-edge pilot exactly. Both GPUs are idle after evaluation.

| Model | Held-out copying | Same maps shuffled | Same maps query B |
|---|---:|---:|---:|
| Standard Transformer | 100.0% | 7.8% | 0.0% |
| Recurrent depth | 100.0% | 7.0% | 0.0% |
| CTM | 100.0% | 8.6% | 0.0% |

Every family passes the primary copying gate on all 128 unseen maps. All also achieve 100% on training and validation maps. When the held-out query changes from A to B, each model emits the original A successor with EOS on all 128 examples. This directly demonstrates query insensitivity for these interventions on fixed-query-trained checkpoints. It does not establish why variable-query training was difficult in the earlier experiment, or show that these architectures cannot learn query selection. Shuffled-edge transfer is also weak.

The common tokenization, answer generation, readout, and optimization path can support perfect fixed-slot copying under these presets. The remaining development issue is query-dependent selection on unfamiliar maps. This result narrows the problem; it does not establish reliable one-hop retrieval, composition, or an architecture ranking.

**Next experiment:** return to the ordered, variable-query dataset and prerecord a longer training budget for all three families. A concrete development candidate is 3,000 updates from fresh initialization with the same seed, model sizes, data, batch size, and peak learning rate; record the extended learning-rate schedule explicitly. Select checkpoints using validation only, retain learning curves, and evaluate the same held-out/paired probes at completion. Keep the original 600-update results as historical controls; the different decay horizon means this is not a pure continuation or an isolated update-count intervention. Do not change optimizer, add a curriculum, or sweep seeds in that same experiment. If lookup still fails, inspect query-conditioned attention/representations and define a separate curriculum experiment with equal tuning opportunities. Longer runs have not been launched.

See [full results](results/fixed_pointer_v1/RESULTS.md), [learning curves](results/fixed_pointer_v1/learning_curves.png), and the [PDF figure](results/fixed_pointer_v1/learning_curves.pdf). The immutable [pre-run plan](results/fixed_pointer_v1/PLAN_BEFORE_RUNS.md), per-example predictions, checkpoint hashes, and raw metrics are retained. The adjacent source snapshot verifies recorded source hashes and preserves code, configurations, environment, protocols, and data manifests. Checkpoints remain under `research/runs/fixed_pointer_v1/seed17`.

To regenerate the report:

```sh
python -m scripts.summarize_fixed_pointer --results_root research/results/fixed_pointer_v1 --run_root research/runs/fixed_pointer_v1/seed17 --previous_results research/results/ordered_pointer_v1
```

## Follow-up completed

The [3,000-update ordered variable-query experiment](ORDERED_LONG.md) is complete: held-out accuracy is 76.6% / 94.5% / 65.6% for Transformer / recurrent depth / CTM. Recurrent depth passes the ordered lookup gate; shuffled transfer remains weak. The next milestone diagnoses query selection on saved checkpoints before choosing further training.
