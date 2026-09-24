# Transformer baselines and shared experiments

## Implemented milestone

Three families now share configuration loading, model construction, training, profiling, checkpoint loading, and conditional-likelihood evaluation:

| Family | Manifest | Structure |
|---|---|---|
| CTM | `configs/ctm_reference_v1.json` | Four thought layers, four ticks, temporal MLPs and synchronization |
| Standard Transformer | `configs/transformer_reference_v1.json` | Four distinct decoder blocks, each applied once |
| Recurrent-depth adaptation | `configs/recurrent_depth_reference_v1.json` | One prelude block, four shared core blocks repeated four times, one coda block |
| Variable-depth adaptation | `configs/recurrent_depth_variable_v1.json` | Same recurrent architecture; uniformly sample T=1…8 per optimizer update, evaluate at T=4 |

These are development configurations. Parameter matching, compute matching, and task-quality comparisons remain separate milestones. The variable-depth configuration is an exploratory training variant, not an additional independent architecture.

All presets use width 256, eight heads, a GPT-2 vocabulary of 50,257, learned absolute positions within 128 tokens, untied vocabulary weights, final-output cross-entropy, and no dropout. Both baselines use causal PyTorch scaled dot-product attention, SwiGLU with hidden width 768, and RMSNorm. The standard baseline uses pre-normalization. The recurrent baseline uses four sandwich norms per decoder block, query/key biases, input injection by a shared 512→256 projection each tick, and a shared final norm after the core loop and after the coda. Parameters are shared across recurrent ticks, with distinct blocks within the core.

The baseline head is a single width-to-vocabulary projection. CTM retains its latent-plus-token adapter and intermediate readout diagnostics. Those differences are intentional architecture differences and are included in measured costs. The later CTM scaffold control will isolate the temporal mechanisms within the same scaffold.

## Relationship to the recurrent-depth reference

The reference is [Geiping et al., arXiv:2502.05171v2, sections 3.1–3.3](https://arxiv.org/html/2502.05171v2), checked against the [authors' implementation](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py). This is an independent small-scale implementation, not a reproduction of Huginn-0125 or its benchmark scores.

| Design | Local adaptation |
|---|---|
| Prelude, repeated core, input injection, coda | Retained; layout (1,4,1), width 256 |
| Sandwich normalization and gated SiLU FFN | Retained |
| Position encoding | Learned absolute positions to align with the CTM development preset; no RoPE |
| Initial recurrent state | Deterministic zeros; no random initialization per forward |
| Vocabulary/head | GPT-2 tokenizer and untied head |
| Parameter initialization | Normal standard deviation 0.02; no reproduction of the reference scaling recipe |
| Training depth | Fixed four for the first control; separate bounded uniform 1…8 variant |
| Backpropagation | Full unrolling; no truncated-gradient training |

The deterministic initial state keeps scores independent of batch shape and request order. Its effect on quality must be studied before making a fidelity claim. Variable-depth training is supported, but its bounded uniform distribution differs from the reference's log-normal Poisson distribution. Published-model fidelity, rotary positions, random-state training, and the reference training recipe require explicit follow-up variants.

## Shared interface

```python
from ctm_transformer.research import load_research_config, build_model

config, identity = load_research_config('research/configs/recurrent_depth_reference_v1.json')
model = build_model(config)
result = model(input_ids, targets=shifted_targets, max_thought_steps=4)
loss, logits = result['loss'], result['logits']
```

Inputs and targets have the same shape. The data loader shifts targets once; the model does not shift them again. `-100` masks supervised targets. The standard Transformer accepts only T=1: change its layer count in a separate manifest to change depth. The recurrent model accepts any positive integer T, including depths unseen during training. This establishes execution support, not evidence of useful extrapolation. Its block applications are `1 + 4T + 1`; CTM has `4T`. These counts are not FLOPs.

Baseline manifests identify `model_family` both at top level and inside their configuration; disagreement fails. The original CTM manifest and digest remain unchanged. Old CTM checkpoints without a family tag still load as CTM. New checkpoints record the family, and baseline configurations must contain every field. Missing or unexpected weights remain errors. Tied-head consistency checks apply when tying is explicitly enabled.

## Shared training policy

`scripts/train_research.py` now calls `ctm_transformer/experiment.py` for all three families. It no longer delegates to the legacy CTM trainer. The runner:

- Reads separate UTF-8 training and validation files, tokenizes with tiktoken, and checks the frozen vocabulary size. Identical file contents are rejected.
- Uses contiguous windows of S input tokens and their S next-token targets. There is no EOS insertion or padding. The boundary target is the next window's first input. Incomplete trailing windows are dropped and counted.
- Shuffles training windows once per epoch; drops the last incomplete training batch. Validation includes partial batches and weights cross-entropy by target-token count.
- Uses dedicated seeded generators for data order and thought budgets. A different architecture or dropout consumption cannot change training-window order. With accumulation, one sampled depth applies to every microbatch in an optimizer update.
- Applies the same AdamW configuration, gradient clipping, accumulation, and BF16 autocast with FP32 weights/moments to every family. The learning rate warms up linearly over updates 1…100, then follows cosine decay to 10% of peak.
- Logs every update, including loss, learning rate, gradient norm, sampled T, input tokens, block applications, and synchronized wall time. Evaluates at the configured interval and on the final update.
- Writes `research_run.json`, a full tokenizer snapshot, `metrics.jsonl`, `best.pt`, `final.pt`, and `summary.json`. Records data/config/tokenizer/source hashes, package versions, seed, GPU, precision, and unique parameter categories. Refuses a nonempty output directory; resume is not implemented.

The data split is supplied by the caller: different files/hashes do not establish document independence or absence of overlap. Disjoint generated datasets and answer-only loading are now implemented; see [ALGORITHMIC_TASKS.md](ALGORITHMIC_TASKS.md). Raw-text mode continues to apply supervision to every token.

**Historical distinction:** the earlier canonical 101-step smoke used the legacy trainer's sliding windows and zero-indexed learning-rate schedule. New shared-runner experiments have a different data/schedule policy and must be versioned separately. Do not merge their learning curves. The model architecture and canonical manifest are unchanged.

The shared runner currently supports the basic final-CE CTM recipe. It rejects distillation, temporal-loss scheduling, two-phase/depth curricula, FEEC, Hyperloop, and custom CUDA scheduling. Use a deliberately extended runner for those experiments. An experiment registry and measured FLOP accounting are still outstanding; `token_block_applications` is an execution count, not a FLOP estimate.

## Commands on the two 3090s

Activate the environment from [README.md](README.md), including the pinned evaluation dependencies. From the repository root, use two independent terminals:

```sh
python -m scripts.profile_research \
  --config research/configs/transformer_reference_v1.json --device cuda:0 \
  --output research/results/transformer_profile_gpu0.json
```

```sh
python -m scripts.profile_research \
  --config research/configs/recurrent_depth_reference_v1.json --device cuda:1 \
  --output research/results/recurrent_depth_profile_gpu1.json
```

Each probe runs 5 warmup and 20 measured forward/backward/AdamW updates. The probe holds depth and learning rate fixed, including when passed the variable-depth preset. It measures synthetic training execution and excludes data loading, validation, and checkpoint writes. Both GPUs have independent 24 GiB memory budgets; neither model needs multi-GPU execution at this size.

After supplying real, separate text files, inspect a run with:

```sh
python -m scripts.train_research \
  --config research/configs/recurrent_depth_reference_v1.json \
  --data_path data/train.txt --eval_data_path data/validation.txt \
  --checkpoint_dir checkpoints/recurrent_depth_seed17 --device cuda:1 \
  --seed 17 --dry_run
```

Remove `--dry_run` to train. Swap the manifest to run CTM, the standard model, or the variable-depth adaptation with the same interface. These data paths are placeholders, not downloaded datasets.

The likelihood harness accepts the resulting `final.pt` for every family. Use `--thought_steps 1` for the standard Transformer or, for recurrent depth, `--t_sweep 1,2,4,8`. Generation tasks remain unsupported by this harness.

## Validation

`tests/test_baselines.py` contains 19 passing GPU checks covering causal prefixes and batching, position sensitivity, masked final-output loss, explicit recurrent unrolling and repeated input injection, checkpoint recomputation with dropout, tiny-batch learning, strict reload and adapter scores, fixed-depth rejection, seeded depth sampling, window packing, token-weighted validation with a partial batch, and actual shared-runner training/evaluation/checkpointing for all three families with gradient accumulation.

The 11 canonical checks and 27 evaluation checks also passed after integration. Reports: `results/baseline_checks_gpu0.xml` and `results/baseline_integration_gpu1.xml`. The shared-runner tests use reduced widths and temporary text fixtures. Full development configurations are exercised by the GPU profiling runs. This is targeted verification, not a claim that the entire legacy repository test suite passes.

## Preliminary full-preset measurements

Batch 4, sequence 128, width 256, BF16 autocast, FP32 AdamW:

| Model | Total parameters | Block applications | Tokens/s | Median update | Peak allocated memory |
|---|---:|---:|---:|---:|---:|
| Standard Transformer | 29,176,576 | 4 | 14,975 | 33 ms | 0.691 GiB |
| Recurrent depth, T=4 | 31,016,704 | 18 | 3,408 | 136 ms | 0.851 GiB |
| CTM, T=4 | 29,557,841 | 16 | 1,868 | 277 ms | 1.861 GiB |

Source records are `results/transformer_profile_gpu0.json`, `results/recurrent_depth_profile_gpu1.json`, and `results/ctm_shared_profile_gpu0.json`. These are feasibility measurements on the two 3090s, not matched-budget quality comparisons or a controlled speedup estimate. The first two probes ran concurrently; the CTM probe ran afterward. Re-profile on the same device without competing work when selecting final compute budgets. The models differ in layer applications, readout frequency, and other operations; no FLOP equivalence is implied.

## Across-tick selection extension — 2026-09-23

The current shared runner additionally supports uniform/ramped and dynamic CTM objectives, with a validated optional monotonic penalty schedule. Explicit `selection_readouts` enables policy-matched validation and checkpoint selection; the default remains final-tick scoring for older callers. `save_validation_checkpoints=True` preserves every evaluated checkpoint.

The recurrent baseline can return its shared coda/head predictions at each core tick via `return_all_logits=True`. The recurrent state itself is unchanged; coda outputs do not feed back into the core. Tests compare every returned tick with an independent truncated forward. The common confidence wrapper supports both recurrent families and selects per-token minimum FP32 entropy. This incurs additional coda computation for the baseline, recorded in inference latency.

The [current preliminary comparison](READOUT_COMPARISON.md) has approximately matched parameter counts and equal example/token exposure. It compares three declared new trials per family, with explicit caveats about unequal compute, prior tuning, and auxiliary-supervision differences. This supersedes the early milestone's final-CE-only runner restriction above.
