# Deep research report: pre-LM review (received 2026-09-28)

Provenance: an external deep-research tool produced this report in response to [`DEEP_RESEARCH_PROMPT.md`](DEEP_RESEARCH_PROMPT.md). It is reproduced verbatim below. Its claims, citations and estimates are the tool's own. They have not been independently verified here unless noted in [`CTM_LM_DESIGN.md`](CTM_LM_DESIGN.md) or later study documents. Several cited sources are 2026 preprints.

---

# Pre-LM Review: CTM-Style Mechanisms in Recurrent-Depth Transformers

The direction is only partly sound, and the multi-week language-model phase should not start yet. The one positive result, that pairwise products of the current recurrent state improve serial S₃ tracking when added to attention queries, is interesting. But it is not specific to CTM: the project's own ablations show CTM's temporal machinery is not needed. It has not been separated from learning-rate, query-scale and fixed-depth confounds. And it was measured on a solvable group that fixed-depth Transformers can in principle shortcut. The planned LM pilot is also under-tokened relative to every looped-LM paper found, and it includes a reference CTM that was tested in a way that departs from the original on several points that plausibly explain its failures.

## TL;DR

- **Keep the sync→query finding, but reframe it and de-confound it before scaling.** It is best described as "second-order, state-conditioned attention queries in a looped Transformer", a special case of multiplicative interactions (Jayakumar et al., ICLR 2020). It may be an optimization or query-scale effect rather than extra expressivity. A per-cell LR sweep, normalized and random-quadratic controls, randomized-depth training and a non-solvable group (A₅) are needed first. All of these are cheap at 0.5M parameters.
- **The CTM comparisons so far do not test CTM as published.** The original CTM reads outputs from synchronization, uses separate output and action synchronization sets, trains on the average of the minimum-loss tick and the maximum-certainty tick, and needed 75–100 ticks and 200,000 updates at LR 1e-4 on parity. Measured against that, the adaptation's 16 ticks, uniform tick supervision and 10,000 updates make "never learned retrieval" and "did not use its ticks" weak evidence. Drop the reference CTM from the LM phase unless a faithful re-run changes this.
- **Redesign the LM pilot around stability and signal, not scale.** Use about 40–60M unique parameters, 1–2B tokens of FineWeb-Edu plus a 32k BPE tokenizer, and 5–10% mixed-in synthetic state-tracking and recall data. Use Parcae-style stable injection with per-sequence randomized depth (µ_rec=8, µ_bwd=4), and QK-norm with RoPE applied after the sync term is added. Include a gated-attention control. This fits in about 300–450 RTX 3090 GPU-hours (my estimate).

## 1. Verdict

The project's method is unusually strong for a small effort: preregistration, hash-frozen analyses, paired seeds and a locked confirmation. Its substantive direction needs correcting on four points.

First, the research question has already been answered in the negative for CTM's distinctive parts. "No decay" and "current state only" preserve the gain, and learned decay stays at about 0. So per-neuron histories, temporal MLPs and decayed synchronization are not what helps. What survives is a generic multiplicative query feature. That makes the relevant prior art the literature on multiplicative interactions and gating, not CTM.

Second, the gain is not yet separated from scale, temperature and learning-rate effects. Query-side quadratic terms are exactly the kind of change that alters logit growth and LR sensitivity.

Third, S₃ is solvable. Known theory says a constant-depth Transformer can represent it, so it is a weak test of "recurrent computation beyond depth".

Fourth, the LM plan runs 30–100M parameters on only 100–500M tokens, compares against a CTM that lacks induction-style retrieval, and trains at fixed T. That design would most likely produce noisy perplexity differences that cannot be interpreted.

Spend about one to two weeks on the five tiny-scale experiments below, and commit LM compute only if the sync effect survives them.

## 2. Ranked list of missed items

### 1. Fixed T=16 training plus the "budget law" may explain the no-extrapolation and prefix-scaling results

- **Why it matters.** A July 2026 preprint studies weight-tied looped Transformers on group word problems (Zhang, Hu, Peng & Xie, arXiv 2607.20594; not peer-reviewed). It reports that free training installs a "linear computation frontier": a mechanism that solves about v positions per loop, with "v≈n_tr/T_tr". It adds that "granting more test-time loops than ever trained rescues late positions at fixed input length."
  - Under this account, training on lengths ≤16 at T=16 needs v≈1. A frontier that moves at most about 1 position per loop cannot reach position 32 at T=16. That alone would produce "no model extrapolated beyond length 16".
  - The sync cell's correct prefix of 3, 7, 10 and 11 at T = 4, 8, 16 and 32 fits a serial frontier of roughly 0.6–0.9 positions per loop that saturates at the trained budget. Plain RDT's prefix of 5 at T=16 would mean a slower frontier of about 0.3 positions per loop. This is my inference from the project's numbers.
- **Evidence on randomized depth.**
  - Geiping et al. (arXiv 2502.05171) state: "To ensure that the model can function when we scale up recurrent iterations at test-time, we randomly sample iteration counts during training". They use a log-normal Poisson distribution and backpropagate "through only the last k iterations".
  - Parcae (arXiv 2604.12946) reports that test-time gains "plateau near µrec". It also warns that "truncating based on µbwd significantly hurts extrapolation", because Geiping's sampling "truncat[es] and compress[es] the distribution of recurrence actually observed during pre-training."
  - Fan et al. (arXiv 2409.15647) obtain length generalization only with "an adaptive number of steps" tied to input length.
- **Action.**
  - Retrain RDT and sync with per-sequence log-normal-Poisson depth, backpropagating through about half the sampled depth.
  - At evaluation, scale T with length (T ≥ n).
  - Report correct prefix divided by T as an estimate of frontier speed.
  - Treat "gain grows with T" as consistent with a faster frontier, not as proof of more expressivity.

### 2. Learning rate and query scale could produce the sync advantage

- **Why it matters.**
  - The sync term makes query logits a higher-degree function of the state and parameters.
  - Wortsman et al. (arXiv 2309.14322) show that "attention logit growth instability" appears "in small models at high learning rates". They hypothesize that it comes from "the quadratic dependence of attention logits on parameter norms", and they find qk-layernorm mitigates it.
  - Gated attention (Qiu et al., arXiv 2505.06708, NeurIPS 2025) reports that a multiplicative sigmoid gate after SDPA "enhances training stability, tolerates larger learning rates".
  - The project's own retrieval results are LR-sensitive: Transformer and RDT learn at 3e-4 but not at 1e-3.
  - A single shared LR of 3e-4 can therefore favor whichever cell's effective step size or temperature happens to suit it.
- **Action.**
  - Run a per-cell LR sweep over {1e-4, 2e-4, 3e-4, 6e-4, 1e-3} with 3 seeds each, and compare each cell at its own best LR.
  - Log the maximum attention logit, the query norm and the attention entropy for every run.

### 3. There is no clean control for "second-order information" versus "a bigger or sharper query"

- **Why it matters.** The products are unnormalized, and the linear-feature ablation (which removed the gain) was not norm-matched. So "linear features fail" could just mean "a small-norm term fails". Note also that the original CTM does normalize: its code divides the decayed sum of products by the square root of the summed decay weights ("normalise by sqrt of AUC of decays"). The adaptation's current-only variant has no such normalization.
- **Action: a control ladder, all inside the same zero-initialized additive query slot.**
  - (a) RMS-normalize the sync vector before W_sync.
  - (b) Apply QK-norm after the sum, so only direction matters.
  - (c) Rescale the linear features per batch to the same norm as the sync features.
  - (d) Use frozen random pair products with a trainable W_sync (random quadratic features).
  - (e) Use a learned per-head temperature only.
  - (f) Use a low-rank bilinear query, W_2(W_a u ⊙ W_b u), which generalizes the sampled pairs.
  - If (a), (b) and (c) keep the gain and (e) does not, the second-order reading holds.

### 4. S₃ is solvable, so it is a weak test, and A₅ at the floor is probably a recipe failure rather than a hard limit

- **Why it matters.**
  - Liu et al. (arXiv 2210.10749) show that "Constant-depth shortcuts exist for solvable semiautomata", while none exist for non-solvable ones "unless TC0=NC1". Merrill, Petty & Sabharwal (arXiv 2404.08819) place S₅ and A₅ word problems as NC¹-complete.
  - The 2-layer Transformer matching RDT at a median of 5 fits shortcut-type solutions, whose out-of-distribution performance Liu et al. describe as brittle.
  - Zhang et al. (2607.20594) report that for looped Transformers, "NC1-completeness costs nothing (A5 generalizes fully)", while S₅'s larger operator "deadlocks joint learning", and that "an operator-first curriculum dissolves the wall". A₅ at the floor for every model therefore points to a training-recipe problem: 10,000 updates, fixed T and no curriculum.
  - DeltaProduct (arXiv 2502.10297) needed batch size 2048 for reliable S₃ results and reports S₃, S₄, A₅ and S₅ separately.
- **Action.** Make A₅ the primary hard task. Use a length curriculum and T ≥ n, and train for more updates. Keep S₃ as a sanity check.

### 5. The reference CTM departs from the original in ways that plausibly explain both failures

Verified from the CTM paper (arXiv 2505.05522v4):

- **Synchronization drives both outputs and queries.** Outputs are "y^t = W_out · S^t_out" and queries are "q^t = W_in · S^t_action". These are two separately sampled pair sets.
- **The loss averages two selected ticks:** the minimum-loss tick and the maximum-certainty tick, "L = (L^t1 + L^t2)/2".
- **Initial state:** "Initial pre-activation history and z^{t=1} are learnable parameters."
- **The authors found snapshot readouts limiting:** "'snapshot' representations were too constraining".
- **Parity compute:** the parity CTMs used 75–100 ticks and "200000 iterations… A learning rate of 1e-4". Results "vary significantly between seeds", with one of three runs converging "to a suboptimal solution".

The adaptation departs from this as follows:

- **Readout.** It reads from the latent concatenated with the token embedding, which is a shortcut that bypasses the ticks.
- **Supervision.** It uses uniform supervision across all ticks. That rewards every tick for being right, which plausibly flattens tick use and would explain identical results from T=4 to T=32 (my inference).
- **Training budget.** It trains for 16 ticks and 10,000 updates at 3e-4.

On retrieval, the static non-contextual K/V and the absence of latent-to-latent attention remain real structural limits. Induction-style lookup needs keys that encode previous-token context.

- **Action.**
  - Stop presenting CTM results as a test of "CTM mechanisms".
  - Either run one faithful CTM on S₃ with sync readout, separate pair sets, the CTM loss, T≥50 and about 50,000 or more updates at 1e-4, or state clearly that the reference is a "CTM-inspired cross-attention recurrent model".
  - Exclude it from the LM phase. Its retrieval deficit is structural: Zoology (Arora et al., arXiv 2312.04927) found that 82% of the up-to-2.1-perplexity gap between gated-convolution and attention LMs on the Pile is explained by in-context recall.

### 6. Closest prior art for the mechanism is missing: multiplicative interactions and gated attention

- **Evidence.**
  - Jayakumar et al. describe "gating, attention layers, hypernetworks, and dynamic convolutions" as multiplicative interactions that "strictly enrich the representable function classes". They conjecture a strong inductive bias "when conditional computation is required".
  - Qiu et al. attribute gated attention's gains to "non-linearity upon the low-rank mapping" and "query-dependent sparse gating scores".
- **Action.** Add gated attention (a state-conditioned sigmoid gate on the SDPA output) and the bilinear query from item 3(f) as competitor arms.

### 7. Instability in the state-placement variant is predictable and matters for the LM phase

- **Why it matters.** Parcae recasts looping as a dynamical system. It finds that "divergent runs learn a spectral radius of ρ(A) ≥ 1", and that baseline RDMs diverge at LR ≥ 4e-4 where Parcae does not (Table 2). Feeding a quadratic term back into the recurrent state adds a positive-feedback path, so divergence is expected (my inference). Keeping the term in the queries is safer, especially behind softmax with QK-norm.
- **Action.** Adopt Parcae's constrained injection and its input normalization ("e ← LN(P(x))"), and keep sync out of the state path.

### 8. Weak statistical rules

- **Why it matters.** Under a null of no difference, where each seed is a fair coin flip, "better in ≥4 of 5 seeds" happens with probability 6/32 ≈ 0.19. The median-gap clause lowers this, but by an unknown amount. The locked 6-of-7 rule gives 8/128 ≈ 0.06, and 5 of 5 gives about 0.03.
  - Many cells and ablations were compared, so the family-wise error rate is high.
  - Correct prefix at ≥90% is a thresholded, discontinuous statistic. Small per-position accuracy changes can move it by several positions.
- **Action.**
  - Pre-register a continuous primary metric (mean accuracy over positions 1–32, or area under the per-position curve) alongside correct prefix.
  - Report seed-level bootstrap confidence intervals and exact sign or permutation p-values with Holm correction.
  - Show the prefix metric's sensitivity at 80%, 90% and 95% thresholds.
  - Report fixed-final-step readout alongside the minimum-entropy readout, because RDT is trained on the final step but scored by minimum entropy.

### 9. Compute accounting

- **Why it matters.** "Same T" matches depth but not FLOPs or wall-clock time. The sync gain rises with T, and plain RDT also "used its steps". An iso-FLOP comparison should therefore give RDT more steps or more width.
- **Action.** Report FLOPs per token (forward plus backward, counting the depth actually applied), tokens per second and peak memory. Compare curves across T ∈ {16, 20, 24, 32} rather than single points.

### 10. The bio-inspired homeostasis factor is weakly motivated for this project

- **Verified** (Leon, arXiv 2608.28184; the paper exists). It uses "one hidden layer of 1000 neurons", sparse parity and noisy XOR, and "SGD with a learning rate of 0.1". It reports that "Homeostasis provides the strongest and most consistent benefit, while structural sparsification emerges as the second major mechanism."
- **The Adam caveat.** Tests with Adam "produced substantially different results… with generally worse performance", and the author says "further experiments would be needed". The paper's LLM relevance is framed speculatively: the mechanisms "may accelerate" generalization.
- **Not found.** I did not find, within this review's search budget, primary evidence that homeostatic regulation helps Transformers or LLMs.
- **Action.** Drop the factor, or defer it to a separate SGD-regime side study. It adds a large new factor whose evidence comes from a different optimizer and architecture class.

## 3. Closest prior art, novelty and framing

**Closest prior art.**

- **General mechanism.** Jayakumar et al. 2020 on multiplicative interactions: gating, hypernetworks and bilinear layers.
- **Attention-specific multiplicative conditioning.** Gated attention (Qiu et al. 2025), which applies query-dependent multiplicative modulation of attention and improves stability and LR tolerance at LLM scale.
- **Synchronization as a query source.** CTM itself, which projects attention queries from pairwise products of neuron activity (S_action). The project's "current state only" sync is essentially CTM's action synchronization with its history removed, placed inside RDT's self-attention.
- **State-tracking expressivity through multiplicative state transitions.** Grazzi et al. (arXiv 2411.12537) show that negative eigenvalues let linear RNNs solve parity. DeltaProduct uses products of Householder matrices and shows that more factors per token improve S₃–S₅ extrapolation. These work on the state transition, not on the query. That is a useful contrast.
- **Algorithm selection in looped Transformers.** Zhang et al. 2607.20594 (frontier speed, budget law) and Fan et al. 2024 (adaptive steps).

**Honest novelty assessment.** A reviewer would most likely say this is a special case of multiplicative or bilinear query conditioning, shown on one solvable-group task at tiny scale. The mechanism is not new. What may be new, if it survives controls, is narrower:

- Inside a weight-tied looped Transformer, second-order state features in the queries (not in the state) raise the speed of the serial computation frontier.
- A rigorous preregistered ablation shows that CTM's temporal machinery (histories, neuron-level MLPs, decay) contributes nothing here.

**Framing.** Present it as "Dissecting CTM inside recurrent-depth Transformers: what transfers is multiplicative query conditioning, not neural timing." Report the negative CTM results as a contribution. Report frontier speed (positions per loop) as the mechanistic readout. Do not frame it as "CTM mechanisms improve recurrent LMs."

## 4. Recommended experiments before the LM phase (at most five, in priority order)

All run at the existing 0.5M-parameter scale on 2× RTX 3090. Each is a few GPU-days at most (my estimate, based on current run sizes).

1. **Confound sweep: learning rate × scale.**
   - Cells: RDT, width-matched RDT, sync-current, and controls 3(a)–(f).
   - Each cell at 5 LRs × 3 seeds, compared at its own best LR.
   - Control: the best-LR RDT and the norm-matched linear cell.
   - Decision rule: the sync claim survives only if sync-current at its best LR beats best-LR RDT and the norm-matched linear cell by at least 2 positions of median correct prefix and at least 5 points of mean accuracy, and if RMS-normalized sync (a) keeps at least half the gain. Otherwise, reclassify the effect as optimization or temperature.
2. **Randomized depth plus T-scaled evaluation.**
   - Train RDT and sync with per-sequence log-normal-Poisson depth (µ_rec=16, backprop through the last 8) against fixed T=16.
   - Evaluate at T ∈ {4, 8, 16, 32, 64} and at T=n for n up to 64.
   - Control: fixed-T twins.
   - Decision rule: if randomized-depth RDT closes more than 50% of the sync gap, the "mechanism" is partly a training-recipe artifact. If sync keeps a higher frontier speed (prefix/T) under both recipes, keep it.
3. **Hard-task transfer: A₅, with S₅ as a stretch goal.**
   - Length curriculum (1→8→16→24) and T ≥ n.
   - Control: RDT under the same recipe, plus a small DeltaProduct-style or negative-eigenvalue linear-RNN reference as a state-tracking yardstick.
   - Decision rule: if sync does not beat RDT on A₅ (paired sign test with p ≤ 0.06 on 7 seeds and a positive mean-accuracy CI), limit all claims to solvable groups and down-weight the LM phase.
4. **Iso-FLOP and known-multiplicative competitors.**
   - Arms: RDT at T ∈ {16, 20, 24, 32}, wider RDT, a gated-attention RDT (sigmoid gate after SDPA), and the bilinear query 3(f), all against sync-current.
   - Decision rule: sync must lie on or above the iso-FLOP Pareto front and beat gated attention. If gated attention matches it, use gated attention in the LM phase, since it is established and has an LLM-scale track record.
5. **Mini-LM stability and signal gate.**
   - About 10–20M parameters, 200–300M tokens of FineWeb-Edu with about 5% synthetic data (A₅/S₃ words in text form, MQAR and variable tracking), BF16.
   - Arms: RoPE with QK-norm, sync on versus off, at 2 LRs.
   - Monitor maximum attention logit, recurrent state norm and loss spikes.
   - Decision rule: proceed to the LM phase only if (i) no arm diverges at either LR, and (ii) sync shows a positive paired gap on the synthetic probes with a CI excluding 0 across 3 seeds. A difference in validation loss alone is not enough.

## 5. Recommended LM-phase design

- **Scale.**
  - About 40–60M unique parameters, for example d=640 with prelude, core and coda of 2 layers each and a 32k tied vocabulary.
  - Compare parameter-matched, not only FLOP-matched.
  - For context, the smallest looped-LM setups found are much larger and longer-trained than the current plan:
    - Parcae: 100M in the RDM setup, and 140M on "Training Tokens 11.2B" in the Transformer setup.
    - LoopFormer: about 1B parameters on about 25B tokens.
    - SpiralFormer: 160M–1.4B parameters.
    - Mixture-of-Recursions: 135M–1.7B parameters.
- **Tokens.** Use 1–2B tokens, not 100–500M. At 100–500M tokens a 50M model sees only 2–10 tokens per parameter, so differences mostly reflect early optimization speed (my inference).
- **Data and tokenizer.**
  - FineWeb-Edu, following Parcae's small setup, with a 32k BPE trained on the corpus. Parcae trained its tokenizer "on 2 billion characters from the FineWeb-Edu training set".
  - Add 5–10% interleaved synthetic data: group word problems, MQAR, boxes-style entity tracking and code-variable tracking.
  - Include some code. Kim, Schuster & Toshniwal (arXiv 2405.21068) found that models additionally trained on large amounts of code, such as Code Llama and DeepSeek-Coder, outperform their base models on entity tracking.
- **Baselines.**
  1. A fixed-depth Transformer, parameter-matched.
  2. A fixed-depth Transformer, FLOP-matched (deeper).
  3. RDT with Parcae-style stable injection.
  4. RDT + sync-current (the best normalized variant from experiment 1).
  5. RDT + gated attention.
  6. Optional: a Gated DeltaNet or DeltaProduct hybrid as a state-tracking reference.
  - Drop the reference CTM.
- **Training recipe.**
  - AdamW with warmup. LoopFormer found warmup "important for stability".
  - QK-norm, and RoPE applied after the sync term is added to the query so relative-position structure is preserved (my inference).
  - Keep sync features in FP32 inside the autocast region. NVIDIA's GA102 whitepaper describes BF16 as having an 8-bit exponent (the same as FP32) but only a 7-bit mantissa, so overflow is unlikely; the real risk is precision loss in products and in the accumulated decayed sums (my inference).
  - Randomize depth per sequence with µ_rec=8 and µ_bwd=⌈µ_rec/2⌉=4, following Parcae ("we choose µbwd = ⌈µrec/2⌉ throughout").
  - Use prelude-output normalization.
  - Tune LR per arm with a small sweep.
- **Evaluation suite.**
  - Validation loss and bits per byte on held-out text.
  - Loss as a function of test-time T, from 1 to 4×µ_rec.
  - In-distribution and length-extrapolated synthetic probes: A₅/S₃ in text, MQAR at several key counts, variable tracking.
  - The boxes entity-tracking task (Kim & Schuster, ACL 2023), expected near floor at this scale. They found "pure-text models up to 175 billion parameters failed", although a 2026 preprint reports that implicit, naturalistic entity tracking reaches human level "at 410M parameters" (arXiv 2608.18083).
  - LAMBADA as a secondary check.
  - Treat knowledge and commonsense benchmarks (HellaSwag, ARC, MMLU) as uninformative at 30–100M. This is my inference; I did not verify chance-level numbers here.
- **Compute accounting.**
  - Report FLOPs per token for forward passes over all applied layers plus backward passes over the truncated depth.
  - Report tokens per second, peak memory and KV-cache size at generation. KV memory grows with recursion steps unless KV is shared across recursions, as in Mixture-of-Recursions' KV-sharing variant.
- **Budget estimate (my inference).**
  - Assumes the RTX 3090's dense BF16 tensor peak of 71 TFLOPS (142 with sparsity), per NVIDIA's Ampere GA102 whitepaper, and about 30% MFU, so about 21 TFLOPS effective.
  - A 50M-parameter looped model at µ_rec=8, µ_bwd=4 costs about 0.7 GFLOP per token, so about 9–10 GPU-hours per 1B tokens. A 6-layer Transformer costs about half that.
  - 6 arms × 3 seeds × 1.5B tokens comes to about 250–300 GPU-hours. Adding LR sweeps at 300M tokens and evaluation gives about 300–450 GPU-hours, or about 7–10 days on 2× 3090.
- **Cheaper alternative worth running first.** Retrofit a small pretrained model into recurrent depth and add a zero-initialized sync query adapter. McLeish et al. (arXiv 2511.07384) retrofit pretrained models, initializing from Llama layers, and show that "scheduling the mean of the depth distribution is efficient in terms of both data and compute."

## 6. Open questions that experiments must answer

- Does sync raise frontier speed (positions per loop), or only in-distribution sample efficiency? The sync cell's 78% at positions 9–16 shows that even in-length positions are not fully solved.
- Is the gain second-order information or conditioning? Only the normalized, random-feature and temperature controls can decide this.
- Do off-diagonal pairs matter? The self-pairs-only result was split, 2 of 5 at sync level and 3 of 5 at RDT level. Does the gain scale with the number of pairs or their rank?
- Does the effect survive on non-solvable groups and under randomized depth, and does it help at all on natural text, where gated attention already captures some multiplicative benefit?
- Could a faithful CTM, with sync readout, its own loss and a long enough schedule, use its ticks on S₃? No independent reproduction of CTM for autoregressive language modeling was found. The official repository covers image classification, mazes, sorting, parity, QA-MNIST and RL, and the paper lists language modeling only as future work.

## 7. Evidence status of key sources

| Source (arXiv ID or venue) | What was verified directly | Status |
|---|---|---|
| CTM, Darlow et al. (2505.05522v4) | Sync used for outputs and queries, separate pair sets, min-loss + max-certainty loss, learnable z init, normalization by sqrt of decay sum, parity schedule, seed variance, no LM experiments | Verified in PDF |
| Geiping et al. (2502.05171) | Log-normal Poisson depth, truncated backprop through last k | Verified; Huginn's mean 32 and k=8 seen only in a secondary blog |
| Parcae (2604.12946) | 100M/140M setups, 11.2B tokens, µ_bwd=⌈µ_rec/2⌉, LR-divergence table, spectral-radius finding, prelude LN | Verified (subagent read PDF); LR value inconsistency inside the paper |
| LoopFormer (2602.11451) | ~1B params, ~25B tokens, shortcut-consistency training, warmup for stability | Verified; tokenizer and small-model parameter counts not stated |
| SpiralFormer (2602.11698) | Multi-resolution recursion, 160M–1.4B Pythia-suite scales | Verified from abstract and HTML snippets |
| Leon (2608.28184) | Exists; 1-hidden-layer MLP, SGD, homeostasis strongest, Adam worse | Verified in HTML |
| Zhang et al. (2607.20594) | Budget law v≈n_tr/T_tr, A₅ generalizes, S₅ wall, curriculum | Verified; preprint, not peer-reviewed |
| Liu et al. (2210.10749); Merrill et al. (2404.08819) | Solvable shortcuts; TC⁰ limits of SSMs, NC¹-complete S₅/A₅ | Verified from abstracts and snippets |
| Grazzi et al. (2411.12537); DeltaProduct (2502.10297) | Negative eigenvalues enable parity; Householder products for S₃–S₅ | Verified from abstracts and snippets |
| Jayakumar et al. (ICLR 2020); Qiu et al. (2505.06708) | Multiplicative interactions framework; gated attention stability and LR tolerance | Verified from abstracts |
| Wortsman et al. (2309.14322) | Attention logit growth at high LR in small models; qk-layernorm fix | Verified in PDF snippet |
| Zoology (2312.04927); MoR (2507.10524); Fan et al. (2409.15647); Kim & Schuster (ACL 2023) | MQAR gap; 135M–1.7B recursion; adaptive-step length generalization; boxes task | Verified from abstracts and snippets |
| Universal Transformers, ACT/PonderNet, Mixture-of-Depths, differential attention, hyper-connections, relaxed recursive Transformers | Not retrieved in this review | Mentioned from general knowledge only; verify before citing |

## Caveats

- Several key sources are 2026 preprints (Zhang et al., Leon, the entity-tracking preprint) and have not been peer-reviewed.
- The mapping from the project's prefix numbers to "frontier speed" is my interpretation, not the authors' claim.
- All GPU-hour and FLOP figures are back-of-envelope estimates assuming about 30% MFU on RTX 3090s. Measure throughput in the stability pilot before locking the budget.
- The search budget ran out before I could verify Mixture-of-Depths, differential attention, hyper-connections or evidence for homeostasis in Transformers. Treat those items as unverified.
