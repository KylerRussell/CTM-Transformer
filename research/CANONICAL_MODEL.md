# CTM reference v1

## Status and purpose

`configs/ctm_reference_v1.json` freezes a small, position-aware development model for implementing controlled comparisons. This is an executable starting point, not an optimized paper-scale training recipe. Architecture or optimizer changes require a separately named manifest; do not overwrite a manifest used for an experiment.

The loader in `ctm_transformer/research.py` requires every `CTMConfig` field explicitly, rejects unknown fields, and records a SHA-256 digest. Inactive options are included so new defaults cannot silently alter this configuration. Adding a dataclass field requires an explicit manifest migration.

## Architecture contract

| Component | Frozen choice |
|---|---|
| Vocabulary | GPT-2 tiktoken, 50,257 tokens |
| Token order | Learned absolute position embeddings; 128-token window |
| Width | Token and latent width 256; 8 attention heads |
| Recurrent computation | 4 distinct thought layers, reused for 4 thought ticks |
| Temporal history | FIFO of length 8 in each layer |
| Temporal MLPs | 256 distinct MLPs per layer, one per latent neuron; hidden width 16 |
| Synchronization | `sparse_decay`, 256 sampled neuron pairs per layer |
| Synapse | Existing MLP synapse and gated latent update |
| Across-layer routing | Sequential layer outputs; attention-based residual mixing disabled |
| Readout | Same head at every tick: concat latent and token representation → linear 512→256 → GELU → LayerNorm → linear 256→50,257 |
| Embedding tying | Off |
| Objective | Cross-entropy on the final tick only; monotonic penalty zero |
| Optional modules | FEEC, matrix streams, DSSA, Hyperloop, Engram, distillation, tick adapters, FiLM, and loop embeddings disabled |
| Dropout | 0 |

`nlm_groups=1` means one shared temporal MLP; `nlm_groups=256` with latent width 256 means independent neuron MLPs. Layers have their own parameters; ticks reuse those parameters. Each forward call initializes its temporal memory. The random synchronization pair indices are buffers saved in checkpoints; initialization is seeded.

Four ticks through four layers give 16 layer applications. That count alone is not a FLOP match to 16 ordinary Transformer layers: synchronization, the temporal MLPs, and readout have additional costs.

The shared head and absence of tick-indexed embeddings allow inference with larger thought budgets, such as T=8. This is an execution capability; usefulness at an unseen depth must be measured. The learned position table limits this preset to 128 input tokens. Longer-input generalization needs its own versioned positional-encoding experiment.

The model currently computes intermediate logits, cross-entropies, and certainty diagnostics even with `final_ce`. Only the final cross-entropy contributes to this preset's training objective, with gradients through earlier recurrent updates. Timing includes the intermediate diagnostics and readouts. A future final-readout-only optimization must be separately measured.

## Parameter accounting

The model has **29,557,841 total parameters**:

| Part | Parameters |
|---|---:|
| Token embedding | 12,865,792 |
| Shared readout, including adapter | 13,047,889 |
| Four thought layers | 3,610,624 |
| Position embedding, embedding norm, initial state | 33,536 |

Most parameters are in the vocabulary embedding and readout. Match and report the recurrent core separately when comparing capacity; matching total count alone could hide large core differences.

## Development training settings

- Batch 4 × sequence length 128; accumulation 1: 512 input tokens per optimizer update.
- AdamW, learning rate 3e-4, betas (0.9, 0.95), weight decay 0.1, gradient norm clipping 1.
- Shared runner's warmup/cosine schedule: 100 warmup steps, default 2,000 total steps. See [BASELINES.md](BASELINES.md) for the exact schedule and the distinction from the historical legacy-trainer smoke. This budget is for development, not a convergence claim.
- BF16 autocast with FP32 parameters and FP32 AdamW moments (`bf16_autocast=true`). Older configurations retain their previous manual low-precision casting behavior.
- Checkpointing, 8-bit optimizer, compilation, and custom Triton kernels disabled.
- Independent single-GPU runs; use the second 3090 for another model or seed. Each process has its own 24 GiB VRAM budget.

## Commands

Activate the environment and driver library path described in [README.md](README.md). Run from the repository root:

```sh
python -m scripts.profile_research \
  --config research/configs/ctm_reference_v1.json --device cuda:1 \
  --output research/results/canonical_profile_gpu1.json
```

The probe rejects architecture overrides when a manifest is supplied. It uses repeated random tokens, a fixed learning rate, 5 warmup updates, and 20 timed updates. It excludes the data pipeline and evaluation and does not follow the full trainer's learning-rate schedule.

Once separate training and evaluation text files exist, inspect the effective run configuration:

```sh
python -m scripts.train_research \
  --config research/configs/ctm_reference_v1.json \
  --data_path data/train.txt --eval_data_path data/validation.txt \
  --checkpoint_dir checkpoints/ctm_reference_v1_seed17 \
  --device cuda:0 --seed 17 --dry_run
```

Remove `--dry_run` to train. This command requires existing data files; the paths above are placeholders. Device, seed, file paths, output directory, and total step budget are explicit run settings. The launcher records the manifest identity and complete effective configuration in `research_run.json`, and requires a fresh output directory to prevent the legacy trainer from resuming unrelated weights. A step-budget override must exceed the 100 warmup steps. The shared baseline runner now records data content hashes, tokenizer rules, and source hashes; the complete experiment registry remains planned. See [BASELINES.md](BASELINES.md) for its data and scheduling policy.

## Validation

The 11 canonical checks cover exact final-tick loss and masked targets, absence of auxiliary readout gradients, sequential layer routing, checkpoint recomputation gradients with and without dropout, strict checkpoint reload, execution at unseen thought depth, legacy checkpoint defaults, and the launcher's dry run. The 23 existing model correctness checks also pass on GPU after the architecture changes.

```sh
CTM_TEST_DEVICE=cuda:0 python -m pytest tests/test_canonical_config.py -q
CTM_TEST_DEVICE=cuda:1 python -m pytest tests/test_research_correctness.py -q
```

The finalized GPU 1 probe measured **1,627 tokens/s**, a **303 ms median update**, and **1.861 GiB peak allocated memory**, with 20 timed updates after 5 warmups. See `results/canonical_profile_gpu1.json` for the full timing configuration, manifest digest, loss sequence, and GPU memory readings. Measurements are preliminary and can be affected by work on the other GPU through shared host resources.

The actual launcher also completed 101 updates on GPU 0 using temporary repeated-text fixtures (51,712 input-token positions processed). Its saved checkpoint reloaded strictly; every model parameter and AdamW moment was finite FP32. One separate 128-token evaluation window produced finite loss. `results/canonical_training_smoke.json` records the configuration, data hashes, and checks. This validates integration, not generalization or a meaningful validation score. The smoke log predates a correction to its startup loss banner and parameter-count display; the saved configuration and objective checks are authoritative.

The final manifest SHA-256 is `210247dc93ccf3cbaac9c18b52cf30115d732f3ca2c9c180d74249bec38f3861`.

## Controlled variants and next work

Standard and recurrent-depth Transformer baselines are now implemented behind one experiment interface; see [BASELINES.md](BASELINES.md). Disjoint generated-task datasets and the first pilots are also complete; see [ALGORITHMIC_TASKS.md](ALGORITHMIC_TASKS.md) for the outcomes and remaining learning gates. Align tokenizer, position encoding, training data, masking, optimizer, and measurement conventions. Report embedding/readout parameters separately from the recurrent core, and measure both parameter-matched and compute-matched comparisons.

Then create separately named variants for attention residuals on/off, auxiliary temporal objective on/off, shared versus independent temporal MLPs, history length, and synchronization. Attention residuals and auxiliary supervision must have their own ablations before attributing improvements to CTM temporal history. A fidelity comparison to a published recurrent-depth architecture may require its own scaffold; document those differences rather than describing every weight-tied Transformer as that published model.
