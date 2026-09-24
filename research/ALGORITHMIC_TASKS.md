# Controlled tasks and first GPU pilots

## Purpose

This milestone adds independently generated addition and pointer-chasing examples, fixed disjoint splits, answer-only training, and autoregressive exact-answer evaluation. It runs one development seed per task and model before committing to a larger study. These are exploratory datasets and results; the files named `test_id` are development evaluation sets, not the locked test set for a paper.

The suite is `research/data/algorithmic_v1`, generated with seed **20260921**. Its manifest records counts, generation strata, split seeds, file hashes, tokenizer rules, and the generator source hash. Files must match those hashes. Regeneration requires an empty directory. The same generator, seed, and code produce byte-identical data.

## Tasks and difficulty

### Decimal addition

Example: `add 19+81=` → `100`.

Both operands have the same digit length. The answer is the ordinary decimal sum, with no spaces or reasoning trace. Difficulty annotations record operand digit length and the number of decimal columns producing a carry, including a final carry into a new digit.

| Split | Examples | Operand digits | Carries |
|---|---:|---|---|
| Train | 2,080 | 32 one-digit; 2,048 two-digit | 0 or 1 |
| Validation | 136 | 8 one-digit; 128 two-digit | 0 or 1 |
| ID evaluation | 136 | 8 one-digit; 128 two-digit | 0 or 1 |
| Carry OOD | 256 | Two-digit | 2 |
| Length OOD | 256 | 128 three-digit; 128 four-digit | 0 or 1 |
| Joint OOD | 256 | 128 three-digit; 128 four-digit | At least 2 |

A semantic identity sorts the two operands: `a+b` and `b+a` cannot cross splits or appear twice within a split. The small one-digit quotas reflect the finite pool of 55 unordered pairs. Allowed carry counts are sampled by rejection from uniform operands, so carry strata are annotated but are not balanced. The manifest's quotas specify lengths and allowed carries, not equal counts for each carry value.

### Pointer chasing

Example format: `map A>B;B>D;D>C;C>A;start A;steps 2=` → `D`.

The actual training maps have eight labeled nodes, each with one outgoing edge, arranged as a single directed cycle. Edge presentation order is shuffled independently of the cycle. The query asks for the node reached after a specified number of steps. The model receives the complete map; success requires selecting and composing its edges.

| Split | Examples | Nodes | Path depth |
|---|---:|---:|---|
| Train | 2,048 | 8 | 1–3 |
| Validation | 128 | 8 | 1–3 |
| ID evaluation | 128 | 8 | 1–3 |
| Depth OOD | 256 | 8 | 4–6 |
| Length OOD | 256 | 12 | 1–3 |
| Joint OOD | 256 | 12 | 4–6 |

Within each split, requested path depths have counts differing by at most one. Depth OOD keeps graph size and prompt length fixed. Length OOD changes graph size while retaining the trained path depths.

An entire labeled successor map belongs to one split. Changing its start node, path depth, or presentation does not create a new split identity. This prevents the same map from being reused under another query in validation. These are held-out labeled cycles, not different graph topologies: all graphs of a given size are isomorphic. General graph reachability is outside this task's scope.

## Tokenization, masking, and scoring

Every model uses `algorithmic_char_v1`: a fixed 71-token vocabulary containing PAD, BOS, EOS, digits, letters, and prompt punctuation. There are no fitted tokenization rules and no pretrained embeddings. This is a separate algorithmic pilot configuration, not the 50,257-token language-model reference.

Each example is encoded as `BOS + prompt + answer + EOS`. The model sees the sequence without its last token; next-token targets mask every prompt position and every right-padding position with `-100`. Only answer characters and EOS contribute to the loss. Examples are never concatenated or allowed to attend to one another. Inputs are cropped to the longest real sequence in a batch. Oversized examples fail rather than silently truncating answers.

Gradient accumulation weights microbatch losses by their supervised-token counts. Training logs distinguish examples, real input tokens, padded token positions, and supervised tokens. Validation includes the last partial batch and reports token-weighted answer/EOS cross-entropy.

Exact match uses **unconstrained greedy generation from the prompt only**. No answer token, answer length, candidate list, or character restriction is supplied. A correct answer must match every answer token and terminate with EOS. The fixed generation budgets are six tokens for addition (up to five result digits plus EOS) and two for pointer chasing (one node plus EOS). Predictions and termination flags are saved per example. Teacher-forced token accuracy is not reported as task accuracy.

Every record's rendered prompt, identity, answer, and difficulty are checked against its structured problem. Split loading verifies file hashes; the pilot launcher checks semantic disjointness across every split before training. The shared training runner also rejects overlap between its training and validation files.

## Pilot settings

Presets: `configs/{transformer,recurrent_depth,ctm}_algorithmic_v1.json`.

| Model | Parameters | Computation |
|---|---:|---|
| Standard Transformer | 461,696 | Two independent decoder blocks |
| Recurrent-depth adaptation | 922,496 | Prelude 1, core 2 repeated four times, coda 1 |
| CTM | 544,839 | Two thought layers repeated four times; independent temporal MLPs and sparse synchronization |

Common settings: width 128, four heads, batch 32, 600 optimizer updates, learning rate 1e-3, 30 warmup updates, AdamW, clipping at 1, BF16 autocast with FP32 parameters/moments, learned token positions with a 128-token limit, and final-output loss. Seed **17** initializes the models, seed **18** orders examples, and seed **19** supplies thought-budget sampling when relevant. These presets use fixed depth. Each model sees 19,200 examples per task in the same order, with no hyperparameter search.

Addition runs sequentially through the three models on GPU 0; pointer chasing does the same on GPU 1. Validation occurs every 100 updates. `best.pt` is selected only by validation answer/EOS cross-entropy. Non-validation splits are evaluated after training. Model files and detailed logs are under `research/runs/algorithmic_v1`; `.pt` files are ignored by git. Reports and per-example predictions are under `research/results/algorithmic_v1`.

This is an equal-example development pilot, with unmatched parameter and compute budgets. It cannot establish which architecture is better. Single-seed uncertainty, architecture-specific learning rates, and recurrent normalization/initialization remain untested. Learned position embeddings at positions never seen in training are an additional confound for the length-OOD splits; those results are diagnostic, not a clean test of algorithmic length extrapolation.

## Commands

Activate the environment in [README.md](README.md). Generate a fresh suite:

```sh
python -m scripts.generate_algorithmic --output research/data/algorithmic_v1
```

Run in separate terminals, using fresh output directories:

```sh
python -u -m scripts.run_algorithmic_pilot \
  --dataset_root research/data/algorithmic_v1 --task addition --device cuda:0 \
  --run_root research/runs/algorithmic_v1/addition_seed17 \
  --results_root research/results/algorithmic_v1
```

```sh
python -u -m scripts.run_algorithmic_pilot \
  --dataset_root research/data/algorithmic_v1 --task pointer --device cuda:1 \
  --run_root research/runs/algorithmic_v1/pointer_seed17 \
  --results_root research/results/algorithmic_v1
```

For a standalone training run, the common launcher accepts `--data_format algorithmic`, the appropriate algorithmic manifest, and the task's `train.jsonl` and `validation.jsonl`. For later depth sweeps:

```sh
python -m scripts.eval_algorithmic \
  --checkpoint research/runs/algorithmic_v1/pointer_seed17/ctm/best.pt \
  --dataset_root research/data/algorithmic_v1 --task pointer --device cuda:0 \
  --depths 1,2,4,8 --output research/results/pointer_ctm_depth_sweep.json
```

Thought depth is distinct from pointer path depth. The initial run evaluates only the preset thought depth; this command describes a follow-up experiment.

## Checks and compatibility

Ten new GPU-backed checks cover known carries, semantic identities, independent arithmetic and rendered-map solutions, cross-split disjointness, byte reproducibility, masks and shift positions, padding and truncation, corruption/duplicate rejection, independent addition/pointer oracle generation and EOS, and actual masked training/checkpoint evaluation for each family. All 30 existing baseline/canonical regression checks also pass.

The shared runner is now recorded as `shared_research_v2`. Text mode retains its windowing policy; variable-length answer mode has explicit supervision and token counts. The language-model manifests remain unchanged. Use `eval_algorithmic` for these character-tokenizer checkpoints; the language-model harness's tiktoken/HuggingFace tokenizer path is separate.

## Observed pilot outcome and decision

All six runs completed. Every model saw exactly 19,200 examples per task; the per-task real-input and supervised-token counts also match across models. The [results table](results/algorithmic_v1/RESULTS.md) includes exact-match rates, difficulty groups, selected checkpoints, and training time. [Learning curves](results/algorithmic_v1/learning_curves.png) are also available as a [PDF](results/algorithmic_v1/learning_curves.pdf).

Addition ID exact match is 97.1% for the standard Transformer, 95.6% for recurrent depth, and 91.9% for CTM. Performance collapses on the unseen two-carry condition and longer operands. All three learning curves establish learning on the trained distribution, but the results do not establish general arithmetic algorithms or CTM superiority.

Pointer ID exact match is 14.1%, 9.4%, and 7.8%, respectively. Even one-hop ID examples remain near guessing. Both uniform-node guessing and the stronger rule that excludes the start node are included in the results notes. Separate exact solvers and a perfect graph-following generation oracle pass the scoring checks. This rules out the basic solution/scoring errors covered by those checks; it does not diagnose the optimization or representation problem.

**Next gate:** run a smaller one-hop lookup control, overfit a tiny fixed set of pointer examples, and separate content lookup from composition. Establish learning on fresh one-hop maps before increasing path depth or expanding seeds. For addition, isolate carry generalization and then address positional encoding before interpreting length extrapolation. Keep all current attempts in the record. Matched-compute variants, equal tuning budgets, multiple seeds, and a fresh locked test suite still follow these development gates.

To regenerate the report and figures after all runs finish:

```sh
python -m pip install -r research/requirements-plots.txt
python -m scripts.summarize_algorithmic_pilot \
  --results_root research/results/algorithmic_v1 \
  --run_root research/runs/algorithmic_v1 \
  --dataset_root research/data/algorithmic_v1
```

`algorithmic-environment.txt` records the installed versions. `results/algorithmic_v1/source_snapshot.zip` preserves the source and selected manifests used for this pilot; its adjacent JSON records hashes. Raw model/optimizer checkpoints remain in the run directories rather than inside that source archive.

The one-hop/tiny-set follow-up is now complete: [pointer diagnostic](POINTER_DIAGNOSTICS.md). All models memorize the tiny set but remain near guessing on held-out one-hop maps; ordered-edge lookup is the next control.
