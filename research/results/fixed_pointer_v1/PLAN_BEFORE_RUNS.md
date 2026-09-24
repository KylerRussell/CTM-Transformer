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
