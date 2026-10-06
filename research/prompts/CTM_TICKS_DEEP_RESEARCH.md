# Deep research prompt: why does our CTM language model not scale or use its thinking ticks?

## Role and goal

You are helping analyse a research result for a paper on the feasibility and benefits of Continuous Thought Machines (CTM; Darlow et al., Sakana AI, 2025) as language models, compared with standard Transformers and recurrent-depth Transformers (RDT; e.g. Geiping et al. 2025, "Huginn").

Our language-model adaptation of the CTM trains, but it has three problems:
1. it **does not improve with model width**;
2. it **barely uses its 16 internal "thinking" ticks**;
3. it trails both baselines at matched parameters, and by far more at matched compute.

Explain why, and propose cheap, decisive diagnostics and literature-backed fixes.

Use the literature and first-principles reasoning. For every claim, cite sources (with links), separate established results from your inferences, and state your confidence. If a premise below looks wrong, say so.

## The model

**Data and training (identical for all arms):**
- FineWeb-Edu (sample-10BT), 32k byte-level BPE, sequence length 1,024.
- Global batch 256 × 1,024 tokens.
- AdamW (β 0.9/0.95, weight decay 0.1 on all parameters), warmup then cosine decay to 10%.
- bf16, RoPE, 2× RTX 3090.

**CTM-LM.** A causal pre-norm Transformer backbone (12 layers for "CTM-heavy", 24 for "CTM-aware") produces per-token features f_i. At every position i, an independent CTM instance runs T = 16 ticks over D neurons, with D ≈ 2.7·d (heavy) or 1.6·d (aware). One tick:
- **Query:** action synchronization S_action is mapped by a linear layer to a query. The query plus RoPE attends causally over keys and values projected from the backbone features.
- **Synapse:** a U-Net MLP maps [attention output, z_t] to pre-activations, which are appended to a per-neuron FIFO history of length M = 8.
- **Neuron-level models:** private per-neuron GLU MLPs (hidden size 32) map each neuron's history to its new activation z_{t+1}.
- **Synchronization:** S = α/√β over P = D random neuron pairs, with α_t = e^{-r}·α_{t-1} + z_i z_j and β_t = e^{-r}·β_{t-1} + 1. The decay r is learned per pair, clamped to [0, 15] and initialized at 0, as in the CTM reference.
- **Output:** logits_t = W_out · S_out,t.

**What the faithful CTM did (it failed):**
- **It never saw its input.** The tick-0 query had rms ≈ 0.01, so the tick attention was nearly uniform. Query and key gradients were ≈ 1e-6, against ≈ 2e-2 for the backbone. Over 1,024 tokens, uniform causal attention gives each position its prefix mean. The model learned unigram statistics, and its backbone collapsed to a token-independent vector.
- **CTM's loss rewarded hedging.** The loss averages two cross-entropies: one at the minimum-loss tick (selected with the label) and one at the most certain tick. At 32k vocabulary the ticks learned different confident guesses. Held-out, the label-selected tick scored 5.3–5.8 nats while every label-free readout scored 7.3–7.8, and early ticks scored 12–14, worse than uniform.

**Adapted CTM (current recipe):**
- **D:** a token-conditioned start state, z_0 = z_init + W·f_i.
- **Sparse-tick loss:** the mean cross-entropy over ticks 4, 8, 12 and 16.

**Other adaptations tried** (held-out loss at 9.8M tokens, d 448; Transformer 6.31, RDT 5.98):

| Adaptation | Result |
|---|---|
| Unit-rms query initialization | Restores input use, insufficient alone |
| Backbone feature in the synapse input | 6.73 with mean-tick loss |
| Feature added to the query | 7.12 |
| Token-conditioned history | Seed-unstable |
| Readout from z as well as sync | No gain |
| Final-tick-only loss | 6.31, but earlier ticks become garbage |
| Certainty-only loss | 6.80 |
| Mean-tick loss | 6.39 |
| Current recipe (D + sparse-tick) | 6.42 |

## Observations to explain

1. **Little tick use.** Per-tick held-out cross-entropy of D + mean-tick loss: 6.39 at tick 1, 6.29 at tick 4, 6.34 at tick 16. The most certain tick is tick 1 for 87% of tokens. With the sparse-tick loss, every tick from 3 on is about 6.35 (tick 1 is untrained at 7.29). Thinking adds about 0.1 nats.
2. **No width scaling.** At 100M tokens and each width's best learning rate:

   | Arm | Narrower | Wider |
   |---|---:|---:|
   | CTM-heavy | 5.157 (40M non-embedding) | 5.159 (80M) |
   | Transformer | 5.01 (43M) | 4.98 (77M) |
   | RDT-heavy | 4.72 (48M) | 4.60 (78M) |
3. **High optimal learning rates.** The CTM prefers 2e-3 to 4e-3, against 1e-3 to 2e-3 for the RDT at similar size.
4. **Training-length trend.** At 400M tokens (~43M parameters), the Transformer scores 3.735, RDT-aware 3.998 and CTM-aware 4.025. The CTM learns fastest late in training.
5. **Compute.** Per token, CTM-aware costs about 6× the Transformer and CTM-heavy about 10×; the RDTs cost about 2.3–4.7×.
6. **Dead decays.** In every trained CTM we inspected, 97–99% of the decay parameters r sit slightly below 0 (−0.005 to 0); in the faithful model, 83–87% do. The clamp passes zero gradient below 0, so these decays are frozen at "no decay": S is an undecayed running sum normalized by √(t+1). Only AdamW's weight decay slowly pulls them back toward 0. The cause appears to be initialization exactly on the clamp boundary.
7. **Mechanisms don't transfer.** CTM mechanisms added to an RDT (synchronization-derived attention queries, synchronization readout, learned start state) gave no gain at 78M parameters and 100M tokens: all within ±0.02 nats, against a seed spread of 0.015.
8. **Earlier synthetic results.** At tiny scale (0.6M parameters) the CTM trailed the RDT on S₃ state tracking. Under randomized tick counts it collapsed its tick use, and it never learned one-hop associative retrieval.

## Questions

1. **Ranked hypotheses for the lack of tick use.** Consider at least the following:
   - **Synchronization inertia from the frozen decays (observation 6):** each new tick changes S by roughly 1/t of the accumulated sum. How large is this effect analytically, and does the CTM paper's own setup avoid it? Check the reference code's initialization and clamping of the decay parameters.
   - **The token-conditioned start state solves most of the problem at tick 0,** leaving nothing to refine.
   - **Next-token prediction at this scale rarely needs iterative computation.** Most tokens are easy, so the benefit should concentrate on hard tokens.
   - **The loss gives no incentive for later ticks to be better than earlier ones.** The sparse-tick and mean losses train every tick to predict independently.
   - **Per-tick input bandwidth is low:** one attention read per tick with a single sync-derived query.
   - **The neuron-level models are small and private,** so per-tick computation is weak.
   - **Optimization:** gradients across 16 ticks, and conditioning (consider observation 3).
2. **Ranked hypotheses for the lack of width scaling.** Consider at least:
   - the backbone, not the CTM, being the effective model;
   - the CTM acting as a quadratic-feature readout head with a fixed bottleneck;
   - parameter placement: synapse and NLM parameters that scale with D but are applied per tick;
   - fixed tick count and pair count relative to D;
   - learning-rate and initialization issues for synchronization at larger D;
   - whether the result at 100M tokens is simply undertraining. Compare with the 400M-token trend.
3. **What does the literature say?** Cover:
   - **The CTM paper and any follow-ups:** tick use on hard vs easy inputs, any language-modelling results, and known failure modes.
   - **Adaptive computation:** ACT (Graves 2016), PonderNet, Universal Transformers, looped Transformers, CALM / early exit, "overthinking".
   - **Recurrent-depth language models:** Huginn and its training recipe (randomized depth, truncated backpropagation, sandwich norms), and other work on recurrent depth vs scale and compute.
   - **Losses that make later iterations better:** progressive or improvement losses, deep supervision with increasing weights, ponder costs, and multiple-choice learning pathologies.
   - **Second-order and bilinear (synchronization-like) representations:** their normalization and trainability.
4. **Cheap decisive diagnostics.** Design 5–8 experiments that run at d 448, 10–100M tokens, on one RTX 3090 in under an hour each. For each, give the predicted outcome under each hypothesis. Include at least:
   - fixing the decay parameterization (e.g. softplus, or initialization inside the interval) and re-measuring per-tick loss;
   - a tick-ablation and test-time tick-extrapolation curve;
   - per-token tick benefit against token difficulty (e.g. unigram surprisal, or loss under a small model);
   - freezing the backbone vs freezing the CTM;
   - scaling D at fixed backbone, and the backbone at fixed D.
5. **Fixes, ranked by expected effect and by fidelity to the CTM.** For each: the mechanism, the literature support, the cost, and how to test it. Consider:
   - the decay fix;
   - a loss that rewards improvement across ticks;
   - token-difficulty-aware or randomized tick counts;
   - a residual state update;
   - more attention reads per tick;
   - larger or shared neuron-level models;
   - normalization of S;
   - adjusting the CTM-to-backbone parameter ratio.
6. **What to expect at 500M parameters and 1–2B tokens.** Given the evidence, what should we expect, and which measurements in those runs would most change the paper's conclusions?
7. **Framing for the paper:** how to present the faithful failure, the adaptations, and these limitations honestly and usefully. Which claims are publishable, and which need more evidence?

## Output format

- An executive summary of at most 10 bullets.
- A ranked hypothesis table: hypothesis, the evidence for and against from our data, the literature, a diagnostic, and the predicted result.
- The experiment plan, in order, with go/no-go criteria.
- Recommended fixes, with fidelity and cost.
- An annotated bibliography with links. Mark anything you could not verify.
