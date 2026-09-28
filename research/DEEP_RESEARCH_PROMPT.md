# Deep research prompt: independent check before the language-model phase

Copy everything below the line into a deep-research tool.

---

## Role and goal

You are an independent reviewer with expertise in recurrent and looped Transformers, the Continuous Thought Machine (CTM), state tracking and expressivity, multiplicative interactions in neural networks, and small-scale language-model training. A small research project is about to commit several weeks of GPU time to a language-modeling phase. Before it does, check whether its direction is sound, and find anything it is missing: prior work, mechanisms, confounds or design choices that could invalidate, explain away or materially change its findings and future runs.

Be skeptical and specific. Prefer primary sources (papers, official code) and cite them with links. Clearly separate **verified facts** (with a citation) from **your inferences**. Do not invent citations. If you cannot verify something, say so.

## Project summary

**Research question.** At comparable budgets, do CTM-style mechanisms (per-neuron activation histories, neuron-level temporal MLPs, and "synchronization", i.e. decayed pairwise products of neuron activations over time) improve recurrent computation in language models beyond what recurrent depth alone provides?

**Hardware and scale so far.** Two RTX 3090 GPUs (24 GB each), about 629 GB of RAM. All studies so far use tiny models of about 0.5M parameters (width about 96–140), a 71-token character vocabulary, AdamW, BF16 autocast with FP32 parameters, peak LR 3e-4, cosine schedule, 10,000 updates with batch 32, and fresh generated data. All training runs, and all analyses, are preregistered and hash-frozen. Decisions use paired seeds with declared rules: for example, a cell "exceeds" another if it is better in at least 4 of 5 seeds with a median gap of at least 2.

**Model families.**

1. **CTM-Transformer (reference CTM adaptation):**
   - Per-position latent state, starting from one learned vector z0 shared by all positions.
   - 16 internal "thought ticks".
   - Each tick, synchronization of the latent's post-activation history (sparse decayed pairwise products, 128 pairs, 8-step history) produces a query that cross-attends (causally) to **static** token-plus-position embeddings. Keys and values are computed once and are never contextual.
   - A gated synapse MLP and per-neuron temporal MLPs update the latent.
   - The readout concatenates latent and token embedding. It does *not* read out from synchronization, unlike the original CTM.
   - The objective is uniform supervision across ticks; a confidence readout selects the minimum-entropy tick per token.
2. **RDT (recurrent depth, adapted from Geiping et al. 2025, "Scaling up Test-Time Compute with Latent Reasoning"):**
   - Prelude (1 block), shared core (2 blocks) applied T = 16 times with input injection (a linear map of the concatenated state and embedding), coda (1 block).
   - Zero initial state, sandwich norms, **fixed T = 16 in training** (no randomized depth, no truncated backprop), final-step cross-entropy, confidence readout.
3. **Sync-RDT (our hybrid):** RDT with CTM mechanisms inside the core, each as a switch.
   - (a) "history": CTM's per-neuron temporal MLPs over each position's last 8 core outputs.
   - (b) "sync": CTM's sparse-decay synchronization of the state history, added as a **zero-initialized additive term to every core self-attention query**, q = W_q·norm(u) + W_sync·sync.
   - With both switches off it reproduces RDT exactly.
4. **Transformer:** a 2-layer decoder of fixed depth.

## Findings so far (development, then frozen studies)

1. **In-context retrieval (MQAR-style pointer lookup).**
   - With 1 or 6 queries per map, every model got stuck in shortcut basins (exclusion or frequency leaks). Only querying all 12 keys per map made one-hop lookup learnable.
   - Learning is an abrupt, seed-dependent "escape", and it is learning-rate sensitive: the Transformer and RDT learn at 3e-4 but not at 1e-3.
   - **The reference CTM and four repair variants never learned retrieval** within 10,000 updates. The variants were: Kimi-style attention residuals (mixing across layers per position); contextual K/V via one causal self-attention block; token-conditioned initial state; per-tick input injection.
   - Our diagnosis: static K/V prevent induction heads; the shared start state means queries cannot express "my token"; and CTM latents never attend to each other's evolving states.
2. **Retrieval sample efficiency (Sync-RDT, 6 cells × 5 seeds, frozen):** no mechanism effect. A width-matched RDT was the only cell to learn at every seed.
3. **Serial state tracking: the S₃ word problem.**
   - Running products of S₃ elements, labeled at every position with no teacher-forcing leak. Trained on lengths 1–16, evaluated on held-out length-32 words.
   - Primary metric: the "correct prefix" (number of leading positions at ≥ 90% accuracy).
   - Frozen study, 7 cells × 5 seeds:
     - the **Sync-RDT "sync" cell raised the median correct prefix from 5 to 10** against both RDT and a width-matched RDT, at every seed;
     - accuracy at positions 9–16 went from 36–43% to 78%;
     - its gain grew with inference steps (median 3, 7, 10, 11 at T = 4, 8, 16, 32);
     - plain RDT used its steps (T = 16 beat T = 4) but did not beat the fixed-depth Transformer (median 5 vs 5);
     - **the reference CTM did not use its ticks at all** (identical at T = 4 through T = 32) and matched the Transformer;
     - no model extrapolated beyond length 16;
     - A₅ was at the floor for everyone.
4. **Mechanism ablations of the sync cell (frozen, same data and seeds):**
   - **No decay** and **current state only (no history)** *preserved* the gain.
   - **Linear history features instead of pairwise products** *removed* it (worse than RDT).
   - **Adding the same term to the state instead of the queries diverged** at every seed (non-finite gradients).
   - **Self-pairs only** (per-channel energy) was *partial*: sync-level at 2 of 5 seeds, RDT-level at 3.
   - Learned decay rates stayed about 0.
   - Current interpretation: **pairwise multiplicative features of the current recurrent state, used to form attention queries, carry the gain. CTM's temporal machinery does not.**
5. **In progress:** a locked confirmation of "sync" and "current" against the RDT controls. It uses 7 never-used seeds, a test set evaluated once, and a 6-of-7 rule.

**Planned next:** gated secondary settings (width 128; training lengths 1–24), a second task family, compute-matched controls, a bio-inspired factor (homeostasis and structural plasticity from Leon 2026, arXiv 2608.28184), and then a multi-week **language-model pilot at about 30–100M parameters and 100–500M tokens** comparing Transformer, RDT, Sync-RDT (sync/current) and reference CTM.

## Questions to answer

### A. Prior work and novelty

1. What published work already shows that **multiplicative or second-order (bilinear, quadratic, product) features in attention queries, keys or gating** improve state tracking, algorithmic reasoning or length generalization? Consider, for example:
   - Jayakumar et al., *Multiplicative Interactions and Where to Find Them* (ICLR 2020);
   - gated and GLU-style attention, including recent work on gating attention outputs in LLMs;
   - bilinear and polynomial networks;
   - hypernetwork-style query generation.

   Is our finding a special case of a known result? What would a reviewer say is closest?
2. What does theory say about which architectures can solve **group word problems** such as S₃, A₅ and S₅ with bounded depth or recurrence? Look at:
   - Liu et al. 2023, *Transformers Learn Shortcuts to Automata*;
   - Merrill, Petty and Sabharwal 2024, *The Illusion of State in State-Space Models*;
   - Grazzi et al. 2024 on negative eigenvalues in linear RNNs;
   - DeltaNet and Gated DeltaNet state tracking;
   - work on looped Transformers and length generalization.

   Does any of it predict or explain that multiplicative query features help recurrent depth on S₃? Is S₃, a solvable group, a weak test, and what task would be stronger?
3. What does the original CTM paper (arXiv 2505.05522) and its code actually do that our adaptation omits? For example:
   - synchronization used for the **output readout** as well as attention;
   - separate synchronization sets;
   - its specific loss (the minimum-loss tick plus the maximum-certainty tick);
   - how the start state is initialized;
   - how inputs are attended.

   Could any omission explain why our reference CTM fails at retrieval and does not use its ticks for state tracking? Is there independent work that has reproduced or extended CTM for language or sequence modeling?
4. What should we know from the recurrent-depth and looped-model literature? This includes:
   - Geiping et al. 2025 and their code (randomized recurrence counts, truncated backprop, initialization and normalization choices);
   - Universal Transformers and ACT/PonderNet;
   - looped Transformers (algorithm-learning and length-generalization papers);
   - Mixture-of-Recursions and relaxed recursive Transformers;
   - LoopFormer (arXiv 2602.11451), Parcae (arXiv 2604.12946) and SpiralFormer (arXiv 2602.11698).

   In particular: **does training with a fixed T = 16, instead of randomized depth, bias our results**, for example our finding that no model gains beyond the trained T or extrapolates beyond the trained length?

### B. Threats to validity in what we have done

5. We used **one learning rate (3e-4) for every cell**. Could a sync-specific advantage be an artifact of the learning rate or optimization, for example multiplicative features effectively raising the attention temperature or learning rate? What controls would a careful reviewer require: per-cell learning-rate sweeps, query-norm or temperature controls, a "random fixed quadratic features" control?
6. The synchronization features are **unnormalized pairwise products**. Could the gain come from query *scale* or *sharpening* rather than from second-order information? Suggest a clean control. (One idea: a matched-norm linear or random-feature term.)
7. Are **parameter-matched and wall-clock-compared controls** sufficient? How should FLOPs be matched for recurrent models with different per-step costs? Is "same T" the right comparison, or should RDT get more steps to match compute?
8. Assess our evaluation choices:
   - the "correct prefix at ≥ 90%" metric;
   - the confidence (minimum-entropy tick) readout;
   - teacher-forced per-position scoring;
   - training lengths 1–16 with evaluation at 32;
   - five seeds with declared paired rules.

   Anything misleading or non-standard? What would strengthen the statistics?

### C. Mechanisms or risks we may be missing for the language-model phase

9. What will **break or change at 30–100M parameters on natural text**? Consider:
   - stability of multiplicative query terms (our state-placement variant diverged);
   - normalization needs;
   - BF16 overflow of products;
   - interaction with RoPE versus learned absolute positions;
   - interaction with KV caching and with inference cost for recurrent models;
   - the need for in-context retrieval, which the reference CTM structurally lacks.
10. Which **language-model evaluations** are sensitive to state tracking or recurrent computation at small scale? Examples: entity tracking, code execution or variable tracking, bAbI-style tasks, LAMBADA, synthetic in-context tasks mixed into pretraining. Which common benchmarks are uninformative at 30–100M parameters?
11. What is a **realistic budget** on 2× RTX 3090 to see recurrent-depth effects in language modeling, according to the literature? What model sizes, token counts and baselines have small-scale looped or recurrent-depth language-model papers used? Is our plan (30–100M parameters, 100–500M tokens) plausible, or should the pilot be designed differently? For example: fine-tuning or retrofitting an existing small model, mixing synthetic state-tracking data into text, or a smaller "tiny language" corpus.
12. Are there **other mechanisms** we should add as controls or alternatives because they are cheaper or better established? For example:
    - gated linear attention or DeltaNet-style state updates (known to help state tracking);
    - differential attention;
    - hyper-connections;
    - Mixture-of-Depths;
    - adaptive halting.
13. Is the **bio-inspired homeostasis and structural-plasticity factor** (Leon 2026) worth including, given that the paper's grokking results used SGD on one-hidden-layer MLPs and reports that Adam behaved differently? Is there evidence for homeostatic regulation in Transformers or LLMs?

## Deliverables

Produce a structured report with:

1. **Verdict:** is the current direction sound, in one paragraph?
2. **Ranked list of missed items**: prior work, mechanisms, confounds and design choices, most important first. For each: why it matters, the evidence (with citations), and the concrete action it implies.
3. **Closest prior art** to our main finding, and an honest novelty assessment, including how to frame the contribution.
4. **Recommended experiments before the language-model phase:** at most five, prioritized. Each needs a control and a decision rule, and should be feasible at tiny scale on 2× RTX 3090 within days.
5. **Recommended language-model-phase design:** scale, data, tokenizer, baselines, training recipe (including randomized depth or not), evaluation suite, compute accounting, and a budget estimate in GPU-hours on RTX 3090s.
6. **Open questions** that literature cannot answer and experiments must.
7. A **references list** with links, marking which claims you verified directly in the source and which you are inferring.
