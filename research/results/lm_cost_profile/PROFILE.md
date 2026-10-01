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
