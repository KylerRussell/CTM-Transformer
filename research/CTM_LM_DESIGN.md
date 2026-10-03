# CTM-LM: a CTM as close to the published design as language-model scale allows

**Status: design adopted 2026-09-28. The tiny implementation is in `ctm_transformer/ctm_lm.py`.**

## Intent

The project's CTM should be **CTM-inspired, but as faithful to the published Continuous Thought Machine (Darlow et al., [arXiv 2505.05522](https://arxiv.org/abs/2505.05522)) as scaling to language models allows**. Some features cannot scale, just as full attention does not scale for ordinary Transformers at LLM size. Those are adapted *in kind* and stated explicitly.

The [deep-research review](DEEP_RESEARCH_REPORT.md) showed that the project's earlier "reference CTM" departed from the published design in ways that plausibly explain its failures. From now on it is called the **CTM-inspired reference**, and it is dropped from the language-model phase.

## Component mapping

The published details are as the review verified them from the CTM paper.

| Published CTM | CTM-inspired reference (earlier) | **CTM-LM (this design)** | Scaling note |
|---|---|---|---|
| Cross-attention to features from a backbone (e.g. a ResNet over the input) | Static token + position embeddings as keys and values | **A causal Transformer backbone** whose outputs are the keys and values: the language-model analogue of CTM's feature backbone. This supplies contextual features, and so induction-style lookup, faithfully. | Standard |
| Internal ticks, decoupled from the input | 16 fixed | Ticks per token: tiny scale 16 in training; larger T and randomized depth under test; certainty-based halting at inference | The paper's 50–100 ticks are not affordable per token |
| Neuron-level models: a private MLP per neuron over its pre-activation history | Per-neuron MLP with a repo-specific gate blending the latest input | **Private per-neuron MLPs** over the pre-activation history, with GLU nonlinearities; post-activation = NLM output, with no blending gate | Linear in width |
| Synapse: a U-Net-style MLP over the post-activations and the attention output | Plain MLP (or a repo-specific convolutional U-Net) | **A U-Net-style MLP** (down to a bottleneck and back up, with skip connections and LayerNorm) over [attention output ; post-activations] | Standard |
| Synchronization: decayed pairwise products of post-activation histories over randomly sampled pairs, normalized by √(sum of decays), learnable decay per pair | Recomputed over an 8-step buffer each tick; one pair set | **Recursive** decayed products (α ← e^{−r}α + z_i z_j, β ← e^{−r}β + 1, S = α/√β), with r ≥ 0 learned per pair; no D×D matrix and no buffer | O(pairs) per tick |
| Two pair sets: S_action gives the attention queries, S_out gives the outputs | One set; queries only; output from latent + token embedding | **Both sets.** Queries q = W_in·S_action; logits = W_out·S_out, with no token-embedding shortcut in the readout | A normal output head |
| Loss: mean of the minimum-loss tick and the maximum-certainty tick (certainty = 1 − normalized entropy) | Uniform across ticks | **The published loss, per supervised token** | Needs per-tick logits; a fused cross-entropy is planned for large vocabularies |
| Learnable initial post-activation state and initial pre-activation history | Learnable start state only | **Both learnable** | — |
| Adaptive compute by certainty | Not used | Certainty-based tick selection and halting at inference | — |
| Each input processed by one CTM; no latent–latent communication | Same (per position) | **Same:** each position thinks independently over the causal backbone's features. Training parallelizes over positions, and generation needs only a KV cache of backbone features. | Very scalable |

**Language-model-specific additions, each stated as a departure:**
- **Rotary position embeddings on the CTM cross-attention queries and keys.** The first query comes from the learned initial state, so it is identical at every position. With absolute positions it cannot target "my own token"; with RoPE it can prefer nearby or self positions. This is the second limitation the retrieval probes found, and its effect is a hypothesis for the capability checks to test.
- **Causal masking:** position p attends to backbone features at positions ≤ p.

**Explicitly not matched:** 50–100 ticks per token, and the paper's large pair counts and widths. Only the kind is matched, not the amount.

## Tiny configuration (capability checks)

The tiny configuration is built from the CTM recipe config (vocabulary, width 128, 4 heads, sequence window, recipe learning rate), with the loss recorded as `dynamic_aggregate`: the min-loss plus max-certainty rule, which is CTM's loss.
- **Backbone:** 2 causal Transformer blocks (width 128, learned absolute positions, as the baselines).
- **CTM:** 128 neurons, memory length 8, per-neuron hidden width 16, 128 action pairs, 128 output pairs, 4 attention heads with RoPE, and a U-Net synapse of widths 256 → 128 → 64 → 128 → 128.
- **Ticks:** 16 in training. Parameter count and throughput are measured by the tests and the first probes.

## Adopted pre-LM plan (revised after the review)

The language-model phase starts only if the synchronization effect survives steps 2–4. Each step gets a frozen protocol with a **continuous primary metric** (mean accuracy over positions or an area measure) alongside the correct prefix, **exact paired p-values** with Holm correction across declared comparisons, and threshold-sensitivity reporting.

0. **Build the tiny CTM-LM** with tests: exact recursive synchronization, the published loss, RoPE, causality, gradients, round trips and runner compatibility. *(This change.)*
1. **CTM-LM capability checks (development).** One-hop all-keys MQAR retrieval, and S₃ running products, with tick use at T = 4 to 32. These test whether the earlier CTM failures came from the reference's departures. Later, if affordable, one longer-tick S₃ run nearer the paper's regime.
2. **Confound sweep for Sync-RDT.**
   - A per-cell learning-rate sweep {1e-4, 2e-4, 3e-4, 6e-4, 1e-3} × 3 seeds for RDT, width-matched RDT and `current`.
   - At each cell's best learning rate, the scale-control ladder: RMS-normalized sync, QK-norm, norm-matched linear features, random quadratic features, a learned temperature only, a low-rank bilinear query, and gated attention.
   - Log the maximum attention logit, query norm and attention entropy.
3. **Randomized depth.** Log-normal-Poisson depth with truncated backpropagation, against fixed-T twins; evaluation at T up to 64 and at T = sequence length; frontier speed (prefix / T). *Done on S₃ ([interpretation](results/randdepth_s3_v1/INTERPRETATION.md)), with full rather than truncated backpropagation. RDT: favourable but not significant. CTM-LM: loses tick use. Extrapolation was unmeasurable because of learned absolute positions. With RoPE and tuned learning rates, the [recipe confirmation](results/recipe_s3_v1/INTERPRETATION.md) confirmed randomized depth for RDT: 24/30 escapes against 0/30. CTM-LM was 22.7 points below fixed-depth RDT, so it leaves the serial-task track.*
4. *(Revised 2026-10-01: done as development only. RDT solves A₅ at width 192 with randomized depth and 30,000 updates; see the [probes](results/recipe_probes/RECIPE_PROBES.md). A frozen A₅ study is not planned; see the [scope update](../RESEARCH_PLAN.md).)* **A₅ as the primary hard task:** a length curriculum, T ≥ n and longer training, with RDT, Sync-RDT and CTM-LM. Also an equal-compute comparison (RDT at more steps and wider, against the synchronization variants).
5. **Mini language-model stability and signal gate:** 10–20M parameters, 200–300M tokens of FineWeb-Edu with synthetic probes mixed in, RoPE with QK-norm, and 2 learning rates. It proceeds only if nothing diverges and the synthetic-probe gaps have confidence intervals that exclude 0.
6. **The language-model phase**, designed as in the review: about 40–60M parameters, 1–2B tokens, 32k BPE, 5–10% synthetic data, Parcae-style stable injection, randomized depth, per-arm learning-rate tuning, a gated-attention arm, and measured compute accounting. The CTM-LM joins only if step 1 shows it learns retrieval and uses its ticks. *(Revised 2026-10-01: CTM-LM joins pretraining regardless, as the paper's subject. At most one CTM-augmented RDT arm may join after screening; see the [scope update](../RESEARCH_PLAN.md).)*

**Dropped:** the bio-inspired homeostasis and structural-plasticity factor. The review found no Transformer evidence for it, and its source used SGD on one-hidden-layer MLPs. It is deferred to a separate side study, if pursued at all.

## Language-model scale: the faithful CTM-LM does not use its input (2026-10-02)

In the [learning-rate sweep](LR_SCALING.md), the CTM-heavy arm at width 448 (12-layer backbone, 1,216 neurons and pairs, 16 ticks) was trained on 100M FineWeb-Edu tokens at 5e-4, 1e-3 and 2e-3. Every run ended at the unigram level:
- held-out loss 7.66–7.73 nats, against 7.62 for a frequency-only model;
- the RDT-heavy arm at the same width reached 5.13.

A step-250 checkpoint (lr 2e-3) shows why.

**The prediction ignores the input.**
- The final-tick logits are the same at every position: their spread across positions is 0.0006, against 1.45 across the vocabulary.
- Replacing every input token with a random token leaves the loss unchanged (7.738 against 7.738).

**The input pathway never trained.** The CTM reads the sequence only through its tick attention, with queries from action synchronization.
- At initialization the tick-0 query has rms 0.012, so the attention is close to uniform: entropy 5.94 against 6.24 for uniform.
- The query and key projections receive gradients of about 1e-6, against about 2e-2 for the backbone and the output head.
- Uniform causal attention over 1,024 tokens gives each position the mean of its prefix, which carries almost no information about the current token.
- The model learns token frequencies through the output head. The backbone then receives almost no useful gradient (2e-4 at step 250, against 1.8 for the head) and drifts. Its residual stream comes to be dominated by one token-independent vector, whose norm grows from 2 after block 0 to 170 after block 11, while the token-dependent part stays near 0.24. After the final norm, token information is 0.1% of the features. This is not an attention sink: attention mass on position 0 is 0.2%.

The synthetic studies used sequences of 16–32 tokens, where prefix averaging dilutes the current token far less. This is the likely reason CTM-LM learned some state tracking there but never one-hop retrieval.

**This failure is a reported result of the paper** (user decision, 2026-10-02). Two fixes are probed against the faithful model (`ctm_transformer/ctm_lm_adapt.py`, `scripts/ctm_lm_probes.py`):
- **A:** rescale the query projection so the tick-0 query has unit rms. This changes initialization only; the architecture is unchanged.
- **B:** give each position's CTM its own backbone feature in the synapse input. This is an adaptation for sequence models, departing from the CTM's input-only-through-attention design.

The smallest change that works is adopted, and B is reported as an adaptation if it is needed. [Probe results](results/ctm_lm_probes/PROBES.md).

### Probe results (2026-10-02)

A per-tick analysis of the first probes found a **second failure, in CTM's loss**. The loss averages two cross-entropies: one for the tick with the lowest loss (selected with the label) and one for the most certain tick. At 32k vocabulary the ticks learn to make different confident guesses. Early ticks score 12–14 nats, worse than uniform (10.4). The label-selected tick scores 5.3–5.8 nats on held-out data, but every label-free readout scores 7.3–7.8, even in the faithful model. Fix **C** was added: train on the mean cross-entropy over all ticks.

**Setup:** 300 steps × 32 × 1,024 tokens (9.8M), d 448 CTM-heavy, lr 5e-4. The held-out loss is the final tick for CTM-LM, measured on 64 windows. Context use is the loss with random input tokens minus the real loss. The rule, fixed before the runs: a probe works if its loss is ≤ 7.0 and its context use is ≥ 0.5.

| Probe | Change | Held-out loss | Context use | Works |
|---|---|---:|---:|---|
| faithful | — | 7.797 | +0.001 | no |
| faithful, lr 1.25e-4 | — | 7.689 | −0.008 | no |
| unit embedding | initialization | 7.727 | −0.001 | no |
| A (unit query) | initialization | 7.394 | +0.917 | no |
| A + unit embedding | initialization | 7.736 | −0.000 | no |
| C (mean-tick loss) | loss | 7.168 | +0.682 | no |
| A + C | initialization + loss | 7.046 | +1.140 | no (0.05 above the bar) |
| B (token observation) | architecture | 7.406 | +0.919 | no |
| **B + C** | architecture + loss | **6.731** | **+1.597** | **yes** |
| Transformer reference (d 384) | — | 6.311 | +1.812 | yes |
| RDT-aware reference with unit embeddings (d 384) | — | 5.979 | +2.767 | yes |

**What this shows:**
- **The faithful CTM-LM fails on both counts.** It ignores its input, and its loss rewards hedging.
- **Each fix contributes.** A and B each restore context use (+0.9 nats). C improves every variant, by 0.36–0.68 nats.
- **Only B + C clears the bar.** It is an architecture adaptation plus a loss change. A + C, which keeps the architecture, comes within 0.05 nats.
- **Even B + C trails the references** at this budget, by 0.4 nats against the Transformer and 0.75 against the RDT.

The choice between B + C and A + C is the user's decision, because C changes the CTM's loss. [Full table](results/ctm_lm_probes/PROBES.md).

### Exploration beyond A, B and C (2026-10-03)

B + C was adopted first. The user then asked for adaptations beyond A, B and C. Each was judged against B + C under a replacement rule fixed before the runs: an alternative replaces B + C if it works and either is at least 0.05 nats better, or is within 0.05 nats while being more faithful or cheaper. Same setup as above. Throughput is single-GPU training throughput at d 448.

| Probe | Change | Held-out loss | Context use | Throughput |
|---|---|---:|---:|---:|
| B + C (adopted first) | token observation in the synapse; mean-tick loss | 6.731 | +1.60 | 3,850 tok/s |
| **D + C** | token-conditioned start state z₀ = z_init + W·f_i; mean-tick loss | **6.392** | +1.84 | 3,880 |
| **D + final** | token-conditioned start state; loss on the final tick only | **6.313** | +2.21 | **7,500** |
| B + final | token observation; final-tick loss | 6.750 | +1.81 | 7,330 |
| B + certainty | token observation; CTM's label-free half alone (most certain tick) | 6.802 | +1.92 | 5,440 |
| A + B + C | B + C with unit query initialization | 6.744 | +2.04 | — |
| B + C + state readout | output also read from the neuron state | 6.839 | +1.68 | — |
| E + C | the position's feature added to its tick-attention query | 7.115 | +1.07 | 3,880 |
| Transformer reference | | 6.311 | +1.81 | |
| RDT-aware reference | | 5.979 | +2.77 | |

**What this shows:**
- **The token-conditioned start state (D) is the best input fix.** It also uses an existing CTM mechanism, the learned start state, rather than a new synapse input.
- **Feeding the feature through the attention query (E) is the weakest.** The attention pathway remains hard to train.
- **Reading out from the neuron state does not help,** so the synchronization readout is not the bottleneck.
- **The final-tick loss matches or beats the mean-tick loss at about twice the throughput.** But its earlier ticks are untrained: D + final's most-certain-tick readout scores 10.35 nats, worse than uniform. Training on the final tick gives up the CTM's per-tick predictions and its certainty-based adaptive computation, while the mean-tick loss keeps every tick usable.

By the rule, D + C and D + final both replace B + C. The choice between them trades the CTM's per-tick behaviour against cost, so it is the user's decision.

### Sparse-tick loss (2026-10-03)

**D-sparse** is the token-conditioned start state trained on the mean cross-entropy over ticks 4, 8, 12 and 16. It reaches a held-out loss of 6.415 with +1.81 context use, against 6.392 for D + mean-tick and 6.313 for D + final. Its throughput is 6,370 tok/s, against 3,880 and 7,500.

Per-tick held-out loss, on 8 windows:

| Tick | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-sparse | 7.29 | 6.65 | 6.42 | 6.36 | 6.34 | 6.36 | 6.37 |
| D + mean-tick | 6.39 | 6.32 | 6.30 | 6.29 | 6.30 | 6.32 | 6.34 |

- **Unsupervised ticks stay usable.** Every D-sparse tick from 3 on is a good prediction, including those not in the loss. Its all-tick certainty readout (7.36) is spoiled only by untrained tick 1, which is the most certain tick for every token. Restricted to the trained ticks, the certainty readout scores 6.357.
- **Tick use is small.** Even with every tick trained (D + mean-tick), tick 1 is within 0.10 nats of the best tick, and later ticks are slightly worse. At this scale and budget the adapted CTM gains little from additional ticks. This is a finding to test at 500M.

### Can a combination beat the Transformer? (2026-10-03)

Probes at 9.8M tokens (held-out loss on 64 windows). The rule was fixed before the runs: compare each model at its best probed learning rate, with a margin larger than the seed-to-seed spread.

| Model | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 | Second seed at 5e-4 |
|---|---:|---:|---:|---:|---:|
| Transformer (d 384) | 6.572 | 6.311 | 6.172 | **6.079** | 6.313 |
| CTM D + sparse (adopted) | 6.626 | 6.415 | **6.227** | — | 6.434 |
| CTM D + H + sparse (H: token-conditioned initial history) | — | 6.355 | 6.205 | **6.183** | 6.904 |
| CTM D + B + sparse | — | 6.422 | — | — | — |
| CTM D + A + sparse | — | 6.433 | — | — | — |

**No CTM variant reaches the Transformer.**
- At their best probed rates, the CTM trails by 0.055 (D + sparse at 1e-3) and 0.104 (D + H at 2e-3, against the Transformer at 2e-3).
- The Transformer's optimum lies at or above 2e-3, the edge of the grid, and the gap widened as the rate rose. A bracketed comparison at the real batch size comes from the [learning-rate sweep](LR_SCALING.md).

**D + H looked better on one seed (6.355) but not on the second (6.904).**
- The second seed left the unigram plateau about 50 steps later and never caught up.
- Its two-seed mean (6.63) is worse than D + sparse's (6.42).
- The replacement rule had compared single seeds, so D + H was **not adopted**, and the recipe stays D + sparse.
- The delayed, seed-dependent escape from the unigram plateau is itself a CTM-LM trait worth reporting. D + sparse escaped at the same step on both seeds.
