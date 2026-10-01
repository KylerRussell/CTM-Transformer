# Language-model cost profile — 2026-10-01

This is a planning measurement, not a paper endpoint (`scripts/profile_lm_costs.py`).

**Setup:**
- One RTX 3090 (24 GiB), random tokens, vocabulary 32,768, sequence length 1,024.
- BF16 autocast, AdamW, gradient clipping. RoPE in every model.
- Throughput is the median over timed updates at the largest power-of-two batch that fits.
- Two GPUs with data parallelism should give close to twice these rates.

| Family | Size (non-embedding) | Training depth | Largest batch | Tokens/s (one GPU) | Peak GiB |
|---|---|---|---:|---:|---:|
| Transformer | 10.6M | — | 32 | 158,610 | 16.6 |
| Transformer | 25.7M | — | 32 | 102,794 | 20.2 |
| Transformer | 49.6M | — | 16 | 62,580 | 12.9 |
| RDT | 10.9M | T = 16 | 8 | 10,061 | 17.7 |
| RDT | 10.9M | T = 48 (sampler cap) | 2 | 921 | 12.0 |
| RDT | 10.9M | T = 48, activation checkpointing | 16 | 4,549 | 11.7 |
| RDT | 26.2M | T = 16 | 4 | 5,411 | 12.4 |
| RDT | 26.2M | T = 48 | 2 | 919 | 16.0 |
| RDT | 26.2M | T = 48, checkpointing | 16 | 3,579 | 13.7 |
| RDT | 50.4M | T = 16 | 4 | 3,432 | 20.8 |
| RDT | 50.4M | T = 48 | 1 | 315 | 15.0 |
| RDT | 50.4M | T = 48, checkpointing | 16 | 2,018 | 19.6 |
| CTM-LM | 9.4M | 16 ticks | 1 | 2,790 | 14.1 |
| CTM-LM | 21.9M | 16 ticks | 1 | 3,142 | 18.1 |
| CTM-LM | 48.7M | 16 ticks | 1 | 2,220 | 22.3 |

RDT layouts are prelude/core/coda 1/4/1 at 10M, 2/4/2 at 25M and 2/6/2 at 50M. CTM-LM backbone, latent and pairs are 4 layers, 512 and 512 at 10M; 4, 1,024 and 1,024 at 25M; and 6, 1,536 and 1,536 at 50M. Untied 32k embeddings add 25–70M parameters to every model.

## Findings

1. **Recurrent models cost 10–30× more per token than the Transformer of the same parameter count.** At T = 16 they apply their core layers 16 times. At 50M, the time for 1B tokens on both GPUs is:

   | Model | Hours per 1B tokens |
   |---|---:|
   | Transformer | about 2 |
   | RDT at T = 16 | about 40 |
   | CTM-LM | about 60 |

2. **CTM-LM is limited by its loss, not its model.** It fits only one sequence per batch at every size, and its throughput barely changes with size. CTM's loss needs logits at every tick, so it materializes a [batch × 1,024 × 16 × 32,768] tensor, plus FP32 log-softmax copies.

   That tensor is avoidable without changing the loss. Only two ticks per token (the minimum-loss tick and the most certain one) receive gradient, and choosing them needs no gradient. Computing per-tick cross-entropy and certainty in chunks without gradient, then recomputing logits with gradient only for the two selected ticks, gives **identical loss and gradients** with about T/2 = 8× less logit memory.
3. **RDT's randomized depth is limited by memory in the tail.** The sampler's cap of 48 drops the feasible batch to 1–2. Options:
   - activation checkpointing: exact, batch 16 at T = 48, but about 2× slower;
   - a lower depth cap;
   - truncated backpropagation through the last k steps, as in Geiping et al. This is a recipe change: the S₃ results used full backpropagation, and Parcae reports that truncation hurts extrapolation.
4. **Every model wastes memory on full-vocabulary FP32 logits.** A chunked cross-entropy would free memory for larger batches.

## Implications for pretraining

- **25M non-embedding parameters at 0.5–1B tokens is feasible for all families** after the CTM-LM loss fix: roughly 1–2 days per recurrent arm on both GPUs. 50M doubles that.
- A sequence length of 512 instead of 1,024 roughly halves attention and logit cost. Tied embeddings or a smaller vocabulary would cut the embedding share.
- Comparisons should state their matching rule. Matching parameters gives recurrent models 10–30× more compute; matching compute requires fewer tokens or smaller recurrent models. Both are reported in practice (Geiping et al.).

## Update: language-model-scale training paths and 500M — 2026-10-01

`ctm_transformer/lm_scale.py` gives exact training paths. Loss and gradients equal the reference models (`tests/test_lm_scale.py`):
- a chunked cross-entropy that never stores full logits;
- CTM's loss with gradient logits only for the two selected ticks;
- activation checkpointing per block (Transformer and RDT) or per tick (CTM-LM).

One RTX 3090, sequence length 1,024, AdamW state on the GPU, all with checkpointing:

| Family | Non-embedding | Training depth | Largest batch | Tokens/s | Peak GiB |
|---|---:|---|---:|---:|---:|
| Transformer (50M, scaled) | 49.6M | — | 64 | 54,167 | 5.4 |
| RDT (50M, scaled) | 50.4M | T = 16 | 32 | 6,007 | 11.6 |
| CTM-LM (50M, scaled) | 48.7M | 16 ticks | 4 | 2,645 | 11.8 |
| **Transformer 500M** (d 1,280, 24 layers) | 476M | — | 32 | **10,536** | 12.9 |
| **RDT 500M** (d 2,304, 2/4/2) | 520M | T = 16 | 8 | **1,269** | 15.4 |
| RDT 500M | 520M | T = 48 | 4 | 432 | 17.5 |
| **CTM-LM 500M** (d 1,536, 12-layer backbone, D 4,096) | 446M | 16 ticks | 1 | **772** | 15.9 |

**Reading at 500M:**
- **Memory fits for every model, but CTM-LM fits only one sequence.** Its AdamW and master weights take about 10 GiB, and its per-tick state (history and synchronization accumulators) about 3.4 GiB. Offloading optimizer state to system RAM (629 GiB) would free most of the 10 GiB.
- **Compute is the binding constraint.** Per token, RDT costs about 8× the Transformer and CTM-LM about 14×.
  - CTM-LM runs near 19 TFLOPS, which is good use of the GPU. Its cost is inherent: a 147M-parameter synapse is applied at every one of 16 ticks, plus CTM's loss needs full logits at every tick (about 4 GFLOP per token without gradient).
  - RDT's cost comes from its 4 core layers at width 2,304, applied 16 times.
- **How parameters are placed matters more than how many there are.** Parameters in once-applied layers (RDT prelude and coda, the CTM-LM backbone) cost 1× per token. Parameters in the recurrent block (RDT core, CTM-LM synapse, NLMs and synchronization) cost about T×. A model of 500M total with a small recurrent block would cost only 2–3× the Transformer.


## Update: the five 500M arms on two GPUs — 2026-10-01

The arms are `scripts/make_pretrain_configs.py` and the trainer is `scripts/pretrain.py`. Both arms of each recurrent family were profiled: one where most parameters recur ("heavy", A) and one where most parameters are applied once ("compute-aware", B).

**Setup:**
- Real FineWeb-Edu tokens, sequence length 1,024.
- The true global batch: 256 sequences = 262,144 tokens per optimizer step.
- DDP over both RTX 3090s, `torch.compile`, activation checkpointing.
- AdamW in system RAM for the arms marked "offloaded".
- Each figure is the mean of optimizer steps 2–3, after compilation.

| Arm | Total / non-embedding | Micro-batch × accumulation per GPU | Optimizer | Tokens/s (2 GPUs) | Peak GiB | Hours per 1B tokens |
|---|---:|---|---|---:|---:|---:|
| Transformer (d 1,280, 24 layers) | 560M / 476M | 16 × 8 | GPU | 23,200 | 14.4 | 12 |
| RDT compute-aware (d 1,280, 11/2/11) | 563M / 479M | 8 × 16 | GPU | 9,500 (depths 19, 17) | 15.2 | 29 |
| CTM-LM compute-aware (d 1,280, 24-layer backbone, D 2,048) | 550M / 441M | 2 × 64 | offloaded | 4,040 | 15.7 | 69 |
| RDT heavy (d 2,304, 2/4/2) | 671M / 520M | 2 × 64 | offloaded | 2,270 (depths 19, 17) | 10.4 | 122 |
| CTM-LM heavy (d 1,536, 12-layer backbone, D 4,096) | 631M / 446M | 1 × 128 | offloaded | 2,080 | 14.8 | 134 |

Single-GPU profiles of the compute-aware designs, at their largest batch, were:
- RDT B: 4,180 tok/s eager and 5,380 compiled.
- CTM-LM B: 1,660 eager and 2,310 compiled (batch 2).

On two GPUs at the real batch, every arm reaches 1.8–2.3× its single-GPU rate. The offloaded optimizer step costs a few seconds per 262k-token step, which accumulation amortizes.

**Fixes found by these runs:**
- **DDP + compile + checkpointing.** DDP's graph splitting in `torch.compile` saved different tensors in the forward pass and in the checkpoint recomputation (a `CheckpointError`). The trainer now sets `torch._dynamo.config.optimize_ddp = False`.
- **RDT-heavy gradient explosion at initialization.** With the small-scale recipe's fixed init std of 0.02, the d 2,304 core's backward pass grows about 1.5× per recurrence. Gradient norms at initialization, on two real sequences:

  | Init | Depth 1 | Depth 4 | Depth 16 | Depth 32 |
  |---|---:|---:|---:|---:|
  | std 0.02, RDT heavy | 20.9 | 60.1 | 7,222 | 5,342,472 |
  | std 0.02, RDT aware | 20.4 | 21.8 | 22.4 | 22.4 |
  | std √(2/5d), RDT heavy | 9.2 | 10.8 | 11.3 | 11.3 |
  | std √(2/5d), RDT aware | 11.8 | 12.5 | 12.7 | 12.7 |

  Clipping would hide this, but the clipped direction would be dominated by the exploding path. All Transformer and RDT arms now use the width-scaled std √(2/5d) (as in Huginn). CTM-LM keeps the reference CTM initialization; its initial gradient norm is small (about 0.03), not large, because its synchronization readout starts near zero.

**Budget.** One pass of 1B tokens through all five arms takes about 366 GPU-pair hours, or 15 days of continuous use of both GPUs.
