# Deep research prompt: why do our recurrent models trail the Transformer, what are we missing on the CTM side, and how should an RDT + CTM hybrid be built?

## Role and goal

You are advising a research project on whether **Continuous Thought Machines** (CTM; Darlow et al., Sakana AI, 2025) are feasible and useful as language models, compared with standard Transformers and **recurrent-depth Transformers** (RDT; e.g. Geiping et al. 2025, "Huginn").

We have run many controlled experiments. Our working expectations:
1. RDTs will outperform CTMs.
2. An RDT with CTM internals will beat both.
3. **The plain Transformer should not be beating both recurrent families.** Our Transformer wins, so we suspect we are doing something wrong.

Treat all three as hypotheses. Tell us where the evidence and literature support them and where they do not, including whether expectation 3 is even correct, and under which comparison (matched parameters, matched FLOPs, or matched tokens).

Specifically:
- (A) Diagnose why our RDTs and CTMs trail the Transformer.
- (B) Find what we are missing from the CTM paper and reference code.
- (C) Find what else to take from Sakana's new **Continuous Memory Machines** paper.
- (D) Design RDT + CTM hybrids.
- (E) Give a concrete, budget-aware experiment plan.

**Ground rules:**
- Cite sources with links.
- Separate established results from your inferences, and give a confidence for each claim.
- Mark anything you could not verify.
- If a premise below looks wrong, say so.

## Setup (identical for every language-model arm)

| Setting | Value |
|---|---|
| Data | FineWeb-Edu sample-10BT, 32k byte-level BPE, sequence length 1,024 |
| Batch | 256 × 1,024 = 262,144 tokens per step (100M tokens = 381 steps; 400M = 1,526) |
| Optimizer | AdamW (β 0.9/0.95, weight decay 0.1 on all parameters), grad clip 1.0, warmup 100 steps, cosine to 10% |
| Precision and hardware | bf16 autocast; one RTX 3090 per run (two GPUs total) |
| Positions and embeddings | RoPE everywhere; untied 32k embeddings |
| Evaluation | held-out cross-entropy (nats per token) on 1,024 windows; "non-embedding" excludes the embeddings and the output head |
| Learning rates | every arm at the best rate from a learning-rate sweep (lattice 1e-3 · 2^k) |

## Architectures

**Transformer:**
- pre-norm RMSNorm, SwiGLU FFN, init std √(2/(5d));
- d 384 with 24 layers and FFN 1,024 (42.5M non-embedding parameters);
- best lr 4e-3 at both 100M and 400M tokens.

**RDT** (our implementation of Huginn's design):
- prelude, recurrent core and coda, all built from the same decoder block;
- "sandwich" norm in every block, prelude and coda included: x = Norm(x + Attn(Norm(x))), then the same for the FFN;
- input embeddings initialized at std 1.0 (needed to avoid collapse with sandwich norm);
- the core input at each step is Linear([state; prelude output]), with the state starting at zero;
- coda after the last step; depth per optimizer step drawn from a log-normal Poisson distribution (mean 15, σ 0.5, at most 48);
- backpropagation through every step (no truncation).

| RDT variant | Width | Layout (prelude/core/coda) | Non-embedding parameters | Best lr |
|---|---:|---|---:|---|
| RDT-aware | 384 | 11/2/11 | 42.8M | 2e-3 at 100M tokens, 1e-3 at 400M; 4e-3 diverges |
| RDT-heavy | 704 | 2/4/2 | 48.2M | 1e-3; 2e-3 diverges |

**CTM-LM** (our language-model adaptation of the CTM):
- A causal pre-norm Transformer backbone produces per-token features f_i. The backbone uses init std 0.02 and FFN width 2d.
- At every position, a CTM runs T ticks over D neurons:
  - **Query:** a linear map of the action synchronization, with RoPE, attends causally over keys and values projected from the backbone features.
  - **Synapse:** a U-Net MLP maps [attention output, z_t] to pre-activations, which enter a FIFO history of length 8.
  - **Neuron-level models (NLMs):** private per-neuron GLU MLPs (hidden size 32) map each neuron's history to its new activation.
  - **Synchronization:** S = α/√β, with α and β decayed sums over D random neuron pairs; the decay r is learned, clamped to [0, 15] and initialized at 0.
  - **Output:** logits = W·S_out.
- **Required adaptations** (the faithful CTM fails at LM scale):
  - a token-conditioned start state, z_0 = z_init + W·f_i;
  - a sparse-tick loss: the mean cross-entropy at ticks 4, 8, 12 and 16, in place of CTM's minimum-loss plus maximum-certainty loss, which made ticks hedge at a 32k vocabulary.
- **"Cross-position" ticks (our addition):** at each tick, the keys and values also include a zero-initialized projection of every causal position's current state z. Refinement at one position can then reach later positions, which is recurrent depth with CTM dynamics.
- **Variants:**

| Variant | Backbone layers | Width | D | Other |
|---|---:|---:|---:|---|
| CTM-aware | 24 | 384 | 640 | |
| Shallow CTM ("S2") | 2 | 384 | 640 | cross-position ticks |
| Wide shallow CTM | 2 | 384 | 640 | S2 with the synapse width and NLM hidden size doubled |

## Results

### 1. Learning-rate sweep at 100M tokens: best held-out loss per width

| Arm | Narrow | Middle | Wide |
|---|---|---|---|
| Transformer | 5.012 (43M) | 4.981 (77M) | 4.880 (170M) |
| RDT-aware | 4.724 | 4.791 | 4.613 |
| RDT-heavy | 4.721 (48M) | 4.600 (78M) | — |
| CTM-heavy (12-layer backbone) | 5.157 (40M) | 5.159 (80M) | — |
| CTM-aware | 5.526 (40M) | — | — |

**At 100M tokens both RDTs beat the Transformer. At 400M tokens the ranking flips** (Result 2).

### 2. Iteration against a 1-iteration twin trained identically (same parameters), 400M tokens

| Model | Non-embedding parameters | Iterated | 1 iteration | Gain | Seconds per step (iterated / 1) |
|---|---:|---:|---:|---:|---:|
| Transformer d 384 | 42.5M | — | **3.735** | — | 4.2 |
| RDT-aware d 384 (11/2/11) | 42.8M | 3.998 | 3.997 | 0.000 | 10 / 6.4 |
| RDT-heavy d 704 (2/4/2) | 48.2M | 3.783 | 3.828 | 0.045 | 26 / 6 |
| CTM-aware (24-layer backbone) | 40.5M | 4.025 | not run | — | 27 |
| Shallow CTM (S2) | 8.6M | 4.138 | 4.295 | 0.157 | 27 / 4 |
| Wide shallow CTM | 16.0M | 4.086 | 4.282 | 0.196 | 34 / 4 |

Timings are on shared GPUs and indicative only.

**Key clue:**
- RDT-aware at depth 1 has the same 24 unique layers as the Transformer, plus an injection layer.
- It scores 3.997 against 3.735, so 0.26 nats of the RDT's deficit comes from its recipe, not from recurrence. Candidate causes:
  - sandwich norm in every block;
  - unit-scale embeddings;
  - the injection layer;
  - a lower usable learning rate (1e-3 against 4e-3);
  - randomized-depth training.
- The training curves cross:

| Tokens | 100M | 200M | 300M | 400M |
|---|---:|---:|---:|---:|
| Transformer | 4.862 | 4.062 | 3.803 | 3.735 |
| RDT-aware (randomized depth) | 4.744 | 4.238 | 4.037 | 3.998 |

  These are evaluations within the 400M-token runs.

### 3. At 100M tokens no iterative model benefits from iteration

| Model | Iterated | 1 iteration |
|---|---:|---:|
| Shallow CTM | 5.303 | 5.246 |
| Wide shallow CTM | 5.230 | 5.127 |
| RDT-heavy | 4.721 | 4.681 |

- **Per-iteration curves within one model are misleading.** An RDT trained at random depth and evaluated at depth 1 looks much worse than depth 16, but a model trained at depth 1 is as good.

### 4. CTM tick behaviour (diagnostics on trained models)

- **The start state carries the prediction.** In the 24-layer CTM, removing the token-conditioned start state puts tick 1 at the uniform loss, and 16 ticks only reach 8.7 nats.
- **Later ticks read their input only to hold their state.** Switching attention off after tick 3 costs 0.31 nats.
- **Tick-to-tick differences look like noise.** Spread across ticks matches MC-dropout noise on the Transformer.
- **Overthinking:** the CTM is best near tick 8, then drifts. Wide CTM at 400M tokens: tick 4 4.069, tick 8 4.042, tick 16 4.055, tick 32 4.146.
- **The RDT converges instead.** RDT-heavy is flat from depth 8 to 32 (3.756).
- **Randomized tick counts hurt the CTM** (+0.44 nats at 100M tokens). An improvement-rewarding hinge loss failed to train.
- **Dead decays.**
  - The synchronization decays sit at r = 0, where the clamp passes no gradient.
  - The official code clamps on `.data` instead; ours clamps in the graph.
  - The gradients mostly push r toward 0 anyway.
  - Probes with softplus decays and with spread initialization were not decisive: Adam moves a parameter about one learning rate per step, about 0.1 over 300 steps.

### 5. A task that needs serial computation: the S₃ word problem

- **Setup:** running products over S₃, tiny models (~0.5M parameters), 10 seeds; mean accuracy over positions 1–16.
- **References:** all models use RoPE.

| Model | Iterated | 1 iteration | Median paired gain |
|---|---:|---:|---:|
| CTM, cross-position ticks | 0.856 | 0.685 | +18.6 points |
| + mean-normalized sync and attention sink (from CMM) | 0.882 | 0.682 | +20.5 |
| + CMM tick memory | 0.850 | 0.691 | +15.9 |
| CTM without cross-position | 0.726 | 0.679 | +3.2 |
| RDT (1/2/1) | 0.821 | 0.677 | +17.5 |
| Transformer (2 layers) | — | 0.614 | — |

**The CTM and the RDT are tied** (CTM higher on 5 of 10 seeds, median −3.0 points). Cross-position recurrence is what makes the CTM's ticks work here.

### 6. CTM mechanisms added to an RDT: an earlier screen, now suspect

- **Setup:** RDT-aware at d 512, 100M tokens.
- **Mechanisms:** each zero-initialized, so the model equals the RDT at initialization:
  - synchronization-derived attention queries;
  - a synchronization readout;
  - a learned start state.
- **Result:** all within ±0.02 nats of the RDT, against a seed spread of 0.015.
- **Why the screen is suspect:** it ran where recurrence itself was worthless (Result 2: RDT-aware gains nothing from depth), so it could not detect a benefit from better recurrence.

### 7. Ideas from Continuous Memory Machines (Regan et al., Sakana AI, arXiv 2610.07907)

**What the paper does:**
- a Transformer block over [sink; long-term memory slots; short-term memory, one token per recent tick];
- a zero-value sink;
- slot embeddings on queries and keys only, and residual gates initialized at 0;
- synchronization read as α/β in place of α/√β, which the authors say generalizes to longer thinking horizons;
- lr 1e-4 and no weight decay.

**What we found:**
- On the LM (shallow CTM, 100M tokens), mean-normalized sync plus the sink scored 5.629 against 5.303, and drifted more.
- On S₃ neither change improved on the CTM with cross-position ticks.
- We did not separate the two changes.

## Questions

### A. What are we doing wrong? Why does the Transformer win?

1. **Is expectation 3 correct?** Under what matching should an RDT or CTM beat a Transformer: parameters, FLOPs, tokens, or tokens per parameter? Summarize what the literature reports, with numbers, for:
   - Huginn;
   - Saunshi et al. 2025 ("Reasoning with Latent Thoughts", looped Transformers);
   - Ouro / LoopLM;
   - Mixture-of-Recursions;
   - Relaxed Recursive Transformers;
   - Parcae (arXiv 2604.12946);
   - Universal Transformers;
   - anything newer.

   In particular: do looped models match a non-looped model with the same unique parameters in perplexity, or mainly in reasoning?
2. **Diagnose the 0.26-nat recipe deficit** (RDT-aware at depth 1 against the Transformer). Rank the causes and design ablations, considering:
   - sandwich (post-residual) norm in every block, against Huginn's actual placement;
   - embedding scale 1.0;
   - the injection layer and zero initial state;
   - init scales;
   - the lower stable learning rate and divergence at 4e-3;
   - weight decay on norms and embeddings;
   - warmup length;
   - untruncated backpropagation through up to 48 steps;
   - the depth distribution.

   What does Huginn's recipe actually use: norm placement, learning rate, initialization, truncated backpropagation, embedding scale? Were later fixes reported? Is there a known way to get both stable recurrence and Transformer-level learning-rate tolerance (for example pre-norm with a normalized state, μP-style scaling, QK-norm, or a gated update)?
3. **Why do the RDTs win at 100M tokens but lose at 400M?** Is this a learning-rate schedule artefact (for example the per-arm optimum shifting with horizon), or an optimization ceiling of sandwich norm?
4. **The CTMs' gap.** How much is the RDT-style recipe problem (our CTMs do not use sandwich norm), and how much is CTM-specific? Possible CTM-specific causes:
   - the output head reads synchronization of D pairs rather than the residual stream;
   - private NLMs;
   - the per-position recurrent state;
   - the cost per tick.

### B. The CTM side: what are we missing?

5. **Compare our CTM-LM with the CTM paper and the official repository** (SakanaAI/continuous-thought-machines) and list every difference that could matter. Include:
   - synchronization pairing (random pairs, self-pairs, "first-last" selections, numbers of action and output pairs);
   - decay initialization and the `.data` clamp;
   - NLM depth and memory length;
   - synapse depth;
   - how inputs are tokenized and attended (backbone features, keys and values per tick);
   - the number of attention heads;
   - the learning rate (1e-4 to 5e-4), no weight decay, grad clip 20;
   - tick counts, and the loss with its certainty definition.

   Which of these plausibly affect tick use or LM quality?
6. **Why does the CTM drift past its best tick while the RDT converges to a fixed point?** Consider:
   - synchronization dynamics: an undecayed running sum, α/√β, magnitude drift;
   - the NLM FIFO window;
   - the absence of an input re-injection (the RDT re-injects the prelude output every step);
   - randomized depth hurting the CTM but helping the RDT.

   What would make CTM dynamics contractive or fixed-point convergent? Options include re-injecting f_i every tick, a gated or residual state update, normalizing z, or a learned per-pair decay with its own learning rate.
7. **Any CTM follow-up work** on sequences, language, or making ticks useful. This includes Sakana's own and independent work after May 2025.

### C. Continuous Memory Machines: what else to take

8. **Which CMM components plausibly transfer to an LM setting where positions are processed in parallel?** CMM carries state across input timesteps like an RNN; our CTM processes positions in parallel.
9. **Why might α/β plus the sink have hurt our LM CTM by 0.33 nats while helping or neutral on S₃?**
   - **Scale interaction:** α/β has magnitude about 1/√t of α/√β, so the readout and query inputs shrink.
   - **Learning rate:** the change may need a different rate.
   - **Sink usage:** the sink may absorb attention.

   Design a decisive separation test.
10. **Did CMM report anything about tick use?** That is, tick ablations, extrapolation, or computation by difficulty.

### D. RDT + CTM hybrids

11. **Propose 3–5 hybrid designs.** Each should:
    - keep the RDT's strengths (re-injection, fixed-point convergence under randomized depth, shared core weights);
    - add CTM internals where they could help.

    Candidates:
    - NLMs (per-neuron history MLPs) inside the recurrent core, as an FFN replacement or addition;
    - synchronization (decayed pairwise products over iterations) as a readout or as attention queries;
    - tick-history attention (CMM-style) over the core's iteration states;
    - certainty-based adaptive exit;
    - a learned start state;
    - CTM-style "attend to input each tick" in place of injection.

    For each design, give:
    - the mechanism and its literature support;
    - an initialization that makes it exactly the RDT at initialization;
    - the cost per iteration;
    - what result would show it helps.
12. **Where should the hybrid sit in the parameter and compute trade-off,** given that the shallow-outer RDT (2/4/2) is the only RDT whose recurrence pays at our scale?

### E. Experiment plan

Our budget is two RTX 3090s. Representative times:
- a 100M-token run at ~40–50M parameters takes 0.5–3 hours;
- 400M tokens takes 2–12 hours;
- there are ~2 GPU-days per round.

Our previous 500M-parameter plan (1–2B tokens) was challenged as undertrained.

13. **Give an ordered plan of 6–10 experiments** with go/no-go rules, covering:
    - fixing the RDT recipe so that RDT at depth 1 matches the Transformer;
    - then showing that recurrence adds to it, at matched unique parameters and also at matched FLOPs;
    - then the hybrids;
    - then the right final-scale comparison (model size and tokens) for a paper.

    Include the controls we must always run (1-iteration twins, seeds, matched-parameter and matched-FLOP baselines).
14. **What final comparison would be most convincing to reviewers on our budget,** and what should we expect it to show?

### F. Paper framing

15. **Given the results so far** (faithful CTM fails at LM; adapted CTM ticks contribute only with a shallow backbone and cross-position recurrence, with enough data or a serial task; RDT ties or beats the CTM; both currently trail the Transformer because of what looks like a recipe problem):
    - What is publishable now?
    - What would make a strong paper?
    - Which claims need more evidence?

## Output format

1. An executive summary of at most 12 bullets.
2. Diagnosis of the Transformer-wins problem: a ranked table of causes, with evidence, literature, a test and the expected result.
3. CTM differences from the reference: a table of difference, likely effect and test.
4. CMM takeaways.
5. Hybrid designs: one table, then details.
6. The experiment plan, in order, with go/no-go rules and estimated GPU-hours.
7. Paper framing.
8. An annotated bibliography with links, with anything unverified marked.
