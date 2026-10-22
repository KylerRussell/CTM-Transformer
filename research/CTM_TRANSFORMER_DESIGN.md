# The CTM Transformer: design (2026-10-14)

## Goal

The user's words: "take the CTM arch and apply it so that we can merge it to work properly with language modeling in an LLM, which is frankly stupid to move away from the transformer. In this way we are basically just slightly adapting a transformer using the theory proposed in the CTM and CMM papers."

The paper compares the CTM Transformer's best with the RDT's best, with the plain Transformer as the baseline.

## Why the earlier CTM-LM is not this

The earlier CTM-LM, and the "matched CTM" of 2026-10-13, were a Transformer backbone feeding a separate per-position CTM head. The Transformer ran once, and the head read it through one cross-attention per tick. Diagnostics on 2026-10-14 (`research/RDT_RECIPE.md`) showed the head's ticks contributed nothing:
- Every gain came from pairwise products of a token-conditioned start state.
- The tick states were about 30 times smaller than that start state, so the multiplicative readout gave them almost no gradient.
- The synchronization decays were stuck at 0.

## Principle

Start from the exact Transformer and adapt its own components with CTM and CMM mechanisms. Every mechanism is zero-initialized, so that:
- the untrained CTM Transformer at 1 tick is exactly the 8-layer Transformer at d 704 (47.2M non-embedding parameters);
- at any tick count it is exactly Dans, the RDT's best (`ctm_transformer/rdt_recipe.py`).

Any difference from those two references therefore comes from the CTM/CMM mechanisms.

## The model (`ctm_transformer/ctm_transformer.py`)

Layers 1–2 and 7–8 run once. Layers 3–6 form the thinking group and run for T internal ticks. The residual stream at the group's output is the state, and it is normalized and added to the input every tick, as in Dans.

| CTM / CMM idea | Where it lives in the Transformer |
|---|---|
| Internal ticks, decoupled from tokens (CTM) | The thinking group runs T ticks. Training draws T per step (log-normal Poisson, mean 15, at most 48), as for Dans. |
| Neuron-level models: each neuron processes the history of its own pre-activations with private weights (CTM) | In each thinking layer's feed-forward network. Each of the 1,856 hidden neurons keeps the history of its gate pre-activation over the last 4 ticks. A private per-neuron MLP (hidden width 4) maps that history to an activation, which is added to the SwiGLU activation with a per-neuron scale initialized to zero. |
| Synchronization: neurons' co-activation over internal time is the representation (CTM), normalized as a proper average whose scale does not depend on the tick count (CMM's α/β) | The state's neurons, normalized, are paired at random: 704 pairs for action and 704 for output. Each pair keeps a decayed running average of the products of its two neurons over ticks. The learned decay is softplus-parametrized so it can move away from 0, unlike the CTM's clamp. |
| Synchronization drives attention (CTM: the query comes from action synchronization) | In each thinking layer, the attention query is the Transformer's query plus a zero-initialized projection of the action synchronization from earlier ticks. |
| Synchronization is the output (CTM) | The output synchronization is added to the state before the last two layers through a zero-initialized projection. It reaches the logits through the residual stream, never only through the multiplicative path that trapped the CTM-LM. |

The extra parameters are the neuron-level models, 4 query projections (2.0M) and the output projection (0.5M). The total is about 50.4M non-embedding parameters, against 47.2M.

**Implementation decisions for speed and memory (2026-10-14):**
- **Neuron-level models at hidden width 4, not 8.** Per-token, per-neuron MLPs over 1,856 neurons are memory-bound. At hidden width 8 they cost 24 ms per layer per tick, against 4 ms for the whole feed-forward network. They run as one fused `torch.compile` kernel, which matches the reference computation to within 4e-7.
- **Pair products as two one-hot matrix multiplies**, twice as fast as gathering by index.
- **Synchronization recomputed in the backward pass**, like the layers: only its running sums are stored, and the attention query reads them inside the checkpointed layer.
- **Micro-batch 8 with 32 accumulation steps** (Dans uses 16 × 16). Tokens per step and the averaged gradient are the same.


**Options for later tests:**
- the CMM attention sink;
- the CTM loss over several ticks with a most-certain-tick readout.

The first build keeps Dans's training (final-tick loss, randomized depth), so that only the architecture differs.

## Testing

1. **Correctness (unit tests):**
   - At initialization the model equals Dans at any depth, and the 8-layer Transformer at depth 1.
   - Every mechanism receives a gradient, and changes the output once its zero-initialized parameter is nonzero.
   - Checkpointed and plain execution agree.
2. **Liveness (diagnostics on training checkpoints).** Remove each mechanism at evaluation time (neuron-level models, synchronization in the queries, synchronization at the output) and measure the loss increase. A mechanism the model does not use is reported as dead. This is the check that caught the CTM-LM's dead ticks.
3. **Language modelling:**
   - the full CTM Transformer at 400M tokens, lr 2e-3, seeds 1234 and 2, the same data, schedule and evaluation as Dans;
   - references: Dans (3.569, 3.579), the 8-layer Transformer (3.618, 3.638), and the 20-layer Transformer at equal compute to 4 ticks (3.562).
   - **Rule, fixed before the runs:** the CTM mechanisms add value if the two-seed mean is at least 0.02 below Dans's (3.574), about one seed spread. A smaller difference is a tie.
4. **Then:**
   - ablations of each mechanism;
   - the CMM sink and the CTM loss options;
   - a learning-rate check;
   - the scaling ladder with the Transformer, the RDT's best and the CTM Transformer's best.

## Progress (2026-10-14)

Two seeds of CTfull (all three mechanisms) are training at 400M tokens. The step-381 snapshots were not written: snapshots are saved only on checkpoint steps (every 50), so the snapshot interval is now 400. The liveness check therefore used the rolling checkpoint at step 500, on CPU, with 4 held-out windows (`research/results/ctm_transformer/diagnose_*`).

| | Seed 1234 | Seed 2 |
|---|---:|---:|
| Held-out loss at step 381 (Dans / 8-layer Transformer) | 4.470 (4.443 / 4.422) | 4.463 (4.465 / 4.455) |
| Loss at 1 / 4 / 8 / 16 / 32 ticks, step 500 | 4.293 / 4.196 / 4.193 / 4.193 / 4.193 | 4.297 / 4.173 / 4.169 / 4.169 / 4.169 |
| Loss increase when sync_query is removed | **0.050** | **0.058** |
| Loss increase when sync_output is removed | 0.009 | 0.004 |
| Loss increase when neuron_models are removed | −0.000 (unused) | 0.001 (unused) |

- **Synchronization drives the attention.** The model relies on the synchronization-derived queries, the CTM's central mechanism.
- **The output synchronization is used weakly.**
- **The neuron-level models are unused** although their scales grew. The thinking group converges by 4–8 ticks and stays flat to 32, so each neuron's pre-activation history is nearly constant at the fixed point. A per-neuron model of a constant history adds nothing that the activation function does not already provide.
- **Using a mechanism is not the same as gaining from it.** Against Dans on the same tokens, training loss over the last 80 steps is +0.023 (seed 1234) and +0.004 (seed 2).

## Result at 400M tokens (2026-10-15)

| | Seed 1234 | Seed 2 | Mean |
|---|---:|---:|---:|
| **CTM Transformer (CTfull)** | **3.586** | **3.590** | **3.588** |
| Dans (RDT's best) | 3.569 | 3.579 | 3.574 |
| 8-layer Transformer (baseline) | 3.618 | 3.638 | 3.628 |

- **Against the baseline:** the CTM Transformer beats the 8-layer Transformer on both seeds (−0.032, −0.048; mean −0.040).
- **Against Dans:** +0.014. Under the pre-registered rule (at least 0.02 below Dans) this is a **tie, on the worse side**. The CTM mechanisms do not add value beyond the RDT's best at this scale.
- **Compute:** the CTM Transformer costs about 2.9 times Dans per tick.

**Liveness at the final checkpoint** (32 held-out windows, loss increase when removed at 16 ticks): all three mechanisms are used.

| Mechanism | Seed 1234 | Seed 2 |
|---|---:|---:|
| sync_query | 0.063 | 0.070 |
| sync_output | 0.011 | 0.011 |
| neuron_models | 0.010 | 0.005 |

The neuron-level models, unused at step 500, came into use late. The model relies on the mechanisms but does not end up better than Dans, which suggests they replace capacity Dans gets elsewhere rather than adding to it. The per-mechanism ablations (CTsq, CTso, CTnlm) test this directly.

On these 32 windows, loss converges by 8 ticks and is flat to 32 (seed 1234: 3.835 at 1 tick, 3.580 at 4, 3.569 at 8, 3.568 at 16 and 32), with no drift.

## Ablations (2026-10-15, running)

- **Runs:** CTsq (synchronization in the queries only), CTso (synchronization at the output only) and CTnlm (neuron-level models only), each on seeds 1234 and 2, at 400M tokens. They run in that order on each GPU.
- **Rules, fixed before the runs** (in `scripts/ctm_transformer_study.py`):
  - a variant beats Dans if its two-seed mean is at least 0.02 below 3.574;
  - against CTfull (3.588), a difference under 0.02 is a tie.

**CTsq result** (synchronization in the attention queries only, 400M tokens): **3.584 (seed 1234), 3.579 (seed 2), mean 3.582.**
- Against Dans (3.574): +0.008, a tie.
- Against CTfull (3.588): −0.006, a tie.
- Against the 8-layer Transformer (3.628): −0.046.

The synchronization-driven query carries the full model's effect at about half its cost per tick (about 1.5 times Dans, against 2.9). The neuron-level models and output synchronization add cost without measurable gain.

**CTso result** (synchronization at the output only): **3.571 (seed 1234), 3.580 (seed 2), mean 3.576.**
- Against Dans (3.574): +0.002, a tie.
- Against CTfull (3.588): −0.012, a tie.
- Against the 8-layer Transformer (3.628): −0.052.

Seed 2 had the only gradient spikes in any CTM Transformer or Dans run: six steps between 1128 and 1141 with gradient norm above 5, peaking at 10,774 at step 1135. Clipping bounded the update. The training loss rose about 0.04 above Dans for about 40 steps and then fully recovered; the step-1143 evaluation (3.692) fell inside that window. A likely source is the output synchronization itself (products of normalized neurons), which no other variant uses without the query path.

**CTnlm result** (neuron-level models only): **3.570 (seed 1234), 3.574 (seed 2), mean 3.572.**
- Against Dans (3.574): −0.002, a tie.
- Against CTfull (3.588): −0.016, a tie.
- No gradient spikes.
- About 2.7–3 times Dans per tick: the per-neuron MLPs are where the full model's cost comes from.

## Ablation summary (400M tokens, two seeds each)

| Model | Seed 1234 | Seed 2 | Mean | vs Dans | Cost per tick vs Dans |
|---|---:|---:|---:|---:|---:|
| 8-layer Transformer (baseline) | 3.618 | 3.638 | 3.628 | +0.054 | (one pass) |
| Dans (RDT's best) | 3.569 | 3.579 | 3.574 | — | 1× |
| CTfull (all three mechanisms) | 3.586 | 3.590 | 3.588 | +0.014 | about 2.9× |
| CTsq (synchronization queries) | 3.584 | 3.579 | 3.582 | +0.008 | about 1.5× |
| CTso (synchronization output) | 3.571 | 3.580 | 3.576 | +0.002 | about 1.45× |
| CTnlm (neuron-level models) | 3.570 | 3.574 | 3.572 | −0.002 | about 2.8× |

- **Every CTM Transformer variant beats the baseline Transformer**, by 0.040–0.056.
- **Every one ties Dans** under the pre-registered rule; none is 0.02 below it. The CTM mechanisms, alone or combined, do not improve on the RDT's best at this scale.
- **Combining them is slightly worse than any one alone.** CTfull is last among the CTM variants, though all are within 0.016 of each other, about one seed spread.
- **Untested so far:**
  - the CTM's loss across ticks with a most-certain-tick readout (its adaptive-compute mechanism);
  - the CMM attention sink and memory;
  - a learning-rate check for the CTM Transformer, which reused Dans's 2e-3;
  - scale.

## Adaptive compute from certainty (2026-10-18; `scripts/adaptive_compute.py`, `research/results/adaptive_compute/`)

Trained checkpoints only, 64 held-out windows (65,536 tokens). Each token's readout is computed at every tick from 1 to 32. CE at fixed tick counts, and CE under certainty-threshold exit interpolated to a mean of 2, 3 and 4 ticks:

| Model (seed 1234) | T = 1 | T = 2 | T = 4 | T = 16 | Exit at a mean of 2 / 3 / 4 ticks | Most certain of 1–16 (mean tick, CE) |
|---|---:|---:|---:|---:|---|---|
| Dans | 3.796 | 3.596 | 3.530 | 3.519 | 3.780 / 3.765 / 3.750 | (5.2, 3.605) |
| CTfull | 3.813 | 3.617 | 3.547 | 3.535 | 3.797 / 3.782 / 3.767 | (5.3, 3.618) |
| CTsq | 3.826 | 3.616 | 3.544 | 3.533 | 3.809 / 3.793 / 3.776 | (5.6, 3.615) |
| CTso | 3.796 | 3.598 | 3.532 | 3.519 | 3.780 / 3.765 / 3.750 | (5.7, 3.600) |
| CTnlm | 3.798 | 3.597 | 3.528 | 3.516 | 3.781 / 3.766 / 3.751 | (5.1, 3.603) |

Seed 2 gives the same pattern (`research/results/adaptive_compute/*_s2.json`).

- **Certainty is a poor stopping signal for every model.** At the same mean tick count, certainty-threshold exit is about 0.22 worse than running every token for that many ticks (3.75 against 3.53 at 4 ticks). The CTM's label-free most-certain readout is 0.08 worse than the final tick. Early-tick readouts are confidently wrong on many tokens.
- **The CTM Transformer uses its ticks no more efficiently than Dans.** The fixed-tick curves have the same shape. Every model is within about 0.012 of its 16-tick loss at 4 ticks.
- **Caveat:** these models were trained with Dans's final-tick loss. The CTM's own loss across ticks (with certainty selection) is the mechanism meant to make certainty meaningful, and it is the next test (item 3).

## Learning-rate check (2026-10-18)

Correction: Dans was trained only at lr 2e-3, taken from its depth-1 twin (the 8-layer Transformer's best). Only the 8-layer Transformer had its rate tuned at 400M tokens. The check therefore covers both:
- Dans and CTsq at lr 1e-3 and 4e-3, seed 1234.
- CTsq stands in for the CTM Transformer: it is the cheapest stable variant (1.5 times Dans per tick, no gradient spikes), and all variants tie.

## Item 3: the CMM attention sink and the CTM loss (2026-10-18, queued after the learning-rate check)

- **CTsqS:** CTsq plus the CMM attention sink. Each thinking layer's attention also sees a learned per-head key with a zero value, so a tick can attend to nothing. It is mixed in through a per-head gate initialized to zero.
- **CTsqL:** CTsq trained with the CTM loss: per token, the mean of the cross-entropy at the minimum-loss tick and at the most certain tick, over every tick's readout.

Two seeds each, at 400M tokens. Rules (in `scripts/ctm_transformer_study.py`):
- a variant improves on CTsq if its two-seed mean is at least 0.02 below 3.582, and beats Dans if at least 0.02 below 3.574;
- for CTsqL, certainty counts as a usable stopping signal if threshold exit at a mean of 4 ticks is within 0.02 of its fixed 4-tick loss.

Unit tests:
- the sink equals attention over the sink and the causal prefix when its gate is open;
- the loss equals the CTM loss computed from all ticks' logits;
- with the gate closed, the untrained model is still Dans.

**Learning-rate check result** (seed 1234, 400M tokens):

| lr | CTsq | Dans |
|---|---:|---:|
| 1e-3 | 3.666 | 3.659 |
| **2e-3** | **3.584** | **3.569** |
| 4e-3 | 3.714 | 3.657 |

2e-3 is the best of the three for both, by 0.08–0.13. The main comparison was made at each family's best tried rate, and the tie stands.

**CTsqS result** (2026-10-19, 400M tokens): 3.579 / 3.593, mean 3.586. It ties CTsq (3.582) and Dans (3.574), and the rule's improvement threshold is not met.
- The gates opened: 0.4–0.8 per layer on average, with sink keys of norm 2–3. However, removing the sink at evaluation (gate set to zero) costs only 0.004 / 0.006 at 16 ticks on 8 windows, at the 0.005 liveness threshold. Attention puts little weight on the sink, so the gated mix stays close to plain attention.
- Synchronization queries remain the mechanism the model relies on: removing them costs 0.057 / 0.058.
- Seed 2 had one clipped gradient spike (norm 195 at step 803) with no effect on the loss. It was already 0.02 behind CTsq seed 2 at step 762.
- Loss by tick count (8 windows): 1 tick 3.731 / 3.806, 4 ticks 3.467 / 3.487, 16 ticks 3.454 / 3.476. Nearly all the gain from thinking arrives by 4 ticks, as for the other variants.

**CTsqL result** (2026-10-20): failed on both seeds and stopped at steps 1,108 / 1,142 at the user's instruction.
- Training loss tracked CTsq until about step 40, then stalled at 7.0–7.6; CTsq reached 5.07 by step 256. Validation at 16 ticks: 8.77 → 8.14 (seed 1234) and 7.57 → 10.31 (seed 2).
- From step about 90, gradient norms ran at 1e2–1e9 on most steps, against about 0.3 for CTsq. Clipping kept the norms finite, so the trainer's divergence check (non-finite norm) never fired.

Diagnosis (`scripts/ctm_loss_diagnosis.py`; 2 held-out windows at 16 ticks; results in `research/results/ctm_transformer/ctm_loss_diagnosis_*.json`):
- **The loss and its gradient are correct.** At depth 1, the CTM loss reproduces the final-tick gradient to 1e-6. On trained CTsq and at initialization, its gradient norm is close to the final-tick loss's (1.38 vs 1.44, and 8.2 vs 9.0). On a healthy model the selected ticks spread over all 16 ticks.
- **The model collapsed.** At step 400, every layer's internal activations are 10–300 times larger than in trained CTsq, while weight norms are not larger (37 vs 47), so the weights have aligned into a few dominant directions:
  - FFN down-projection inputs: RMS 10–108, against about 0.4.
  - The recurrent state's RMS: about 2,000 after the first tick, against 4.7.
  - Predictions are close to the unigram distribution: CE 7.9 at tick 1 and 8.0 at every tick from 3 on, with the same value at each.
  - The most certain tick is tick 1 for 95% of tokens.
- **The gradient through ticks 2 and later is pathological.** It is 30–200 times larger than in a healthy model: 60–67 at step 400 and up to 332 at step 1,100, against 1.4. It goes mostly into the token embedding and the first prelude layer, and dominates the clipped update.
- **Measurement note:** a first version of the diagnosis ran its no-grad pass and gradient passes inside one bf16 autocast block. Autocast reused weight casts from the no-grad pass and cut the gradient to core, prelude and coda weights. Each pass now has its own autocast block. Training is unaffected: every micro-step enters autocast fresh.
- **Not yet known:** what starts the collapse around steps 40–90. Two candidates:
  - the selection trains mostly tick 1 early on, so deeper ticks drift untrained, and their occasional selection sends large gradients through an untrained recurrence;
  - an optimization instability that the per-token selection amplifies.

**CTsqL pilots** (2026-10-21; 200 steps on the full schedule, seed 1234, per-step diagnostics; rule in `scripts/ctm_transformer_study.py`; table in `research/results/ctm_transformer/PILOTS.md`). CTsq's training loss over steps 191–200 is 5.394.

| Pilot | Loss, steps 191–200 | vs CTsq | Max gradient norm after step 100 | Passes |
|---|---:|---:|---:|---|
| CTsqL reproduction (lr 2e-3) | 6.542 | +1.147 | 6.2e7 | no |
| Fix (a): CTM loss + final-tick loss | 5.934 | +0.540 | 1.5e3 | no |

- **The reproduction is not bit-exact.** It matches CTsqL to four decimals for the first steps, then drifts (GPU nondeterminism). It tracks CTsq until about step 125, later than CTsqL's step 40, then falls behind and spikes (gradient norm 64 at step 200). The failure is robust; its onset is not.
- **Signature, in both pilots: the state grows.** The state's RMS (the residual stream leaving the thinking layers) climbs steadily:
  - reproduction: 0.9 at step 1, 28 at step 100, 60–76 at step 200;
  - fix (a): 43 at step 200;
  - healthy CTsqS at step 400: 6–7, falling to 4–5 by the end.

  The gradient spikes go into the token embedding and the first prelude layer, as in the failed runs.
- **The selected ticks are unstable.** In the reproduction, the share of tokens whose most certain tick is tick 1 swings between 0 and 1 within 5–10 steps (step 5: 0.91, step 10: 0.02, steps 15–20: 1.00, step 30: 0.01). From step 70 on it is about 0, and tick 1's cross-entropy rises to 11, above its value at initialization (10.6): tick 1 is abandoned.
- **Fix (a) only delays the failure.** With the final-tick term the selection is steadier and the loss matches or beats CTsq until step 100 (6.20 vs 6.25). The state still grows (12 at step 50, 43 at step 200), and the run falls behind from step 125.
- **Fix (b) fails too:** the CTM loss alone at lr 1e-3 scored 6.563 (+1.169) over steps 191–200, with a maximum gradient norm of 8.5e11. It tracked CTsq until about step 80, then the state grew from 6 at step 60 to 42–62 at step 200. A lower learning rate postpones the failure; it does not prevent it.
- **Conclusion:** the state grows whenever the CTM loss is used, whatever the learning rate and with or without the final-tick term. The working explanation is that the CTM loss's per-tick gradient into the recurrence, which asks every tick's state to be directly readable, drives the growth, and Dans has nothing that bounds the state the coda reads.

**Second CTsqL pilots** (2026-10-21, the same rule):
- **Fix (c), CTsqLD:** the mean of two terms. The first is the CTM loss on readouts of the detached state, which trains only the coda and the output head. The second is the full-gradient final-tick cross-entropy, which trains the recurrence as in Dans. A test checks that the recurrence's gradient is exactly half the final-tick loss's.
- **Fix (d), CTsqLN:** the CTM loss alone, with the shared norm on the coda input (`coda_norm: shared`). In round 1 this norm cost about 0.59 nats.

**Second pilot results** (2026-10-21):

| Pilot | Logged loss, steps 191–200 | vs CTsq | Max gradient norm after step 100 | State RMS, step 200 | Rule as written |
|---|---:|---:|---:|---:|---|
| (c) CTsqLD, detached readouts | 5.468 | +0.074 | 0.75 | 12.6 (peak 20 at step 75, falling) | passes |
| (d) CTsqLN, coda-input norm | 8.02 | +2.63 | 2.4e9 | 1,147 | fails |

- **(c) is stable, which confirms the explanation.** Stopping the CTM loss's gradient at every tick's state ends the state growth and the gradient spikes. The per-tick gradient into the recurrence was the cause.
- **(d) makes the failure worse.** Normalizing the coda input does not bound the recurrence, and the state grows to over 1,000.
- **The rule compared mixed losses, and I missed this when setting it.** The logged loss of a CTM-loss variant includes the CTM terms; CTsq's is the final-tick loss alone. Like for like, (c)'s final-tick cross-entropy (last micro-batch of each step) is:
  - 5.517 over steps 191–200, against CTsq's 5.394 (+0.12);
  - 5.635 over steps 151–200, against 5.55 (+0.085).

  On the fair measure, (c) sits at the 0.1 threshold. Its tick choices are healthy: at step 200 the best tick is the final tick, and tick 1 is the most certain for 45% of tokens.

**CTsqLD result** (fix (c) at 400M tokens, 2026-10-22): 3.832 / 3.848, mean 3.840.
- It fails all three item-3 rules. It is 0.26 worse than CTsq (3.582) and 0.27 worse than Dans (3.574).
- It was stable throughout: no gradient spikes, and only 3 steps above norm 2 after step 100.
- The gap to CTsq widened over training. On validation it was +0.21 / +0.27 at step 381 and +0.25 / +0.24 at step 1,143; on the final-tick training loss, +0.1 at step 400 and +0.22 / +0.26 at the end.
- The state shrank to about 10 by step 300, then grew again to 34–36 by step 1,350 (CTsqS: 4–5). Detaching the readouts slows the growth but does not stop it. The likely route is the coda and output head, which the CTM loss now adapts to every tick's state; this is not tested.

Adaptive compute (`scripts/adaptive_compute.py`, 64 windows). The gap is the loss with threshold exit at a mean of 4 ticks (interpolated) minus the fixed 4-tick loss:

| Model | Fixed 4 ticks | Threshold exit at mean 4 | Gap | Most certain tick (1–16) vs fixed 16 |
|---|---:|---:|---:|---:|
| Dans | 3.530 / 3.542 | 3.750 / 3.780 | +0.22 / +0.24 | +0.086 / +0.066 |
| CTsq | 3.544 / 3.540 | 3.776 / 3.831 | +0.23 / +0.29 | +0.082 / +0.059 |
| CTsqLD | 3.784 / 3.798 | 4.614 / 4.454 | **+0.83 / +0.66** | +0.056 / +0.034 |

- The certainty rule (within 0.02) fails. Threshold exit is much worse for CTsqLD than for any other model, because its 1-tick readout is poor (5.00 / 4.92) and certainty sends many tokens out at tick 1.
- Reading each token at its most certain tick costs CTsqLD less (+0.03 to +0.06) than the other models (+0.06 to +0.09). The CTM loss does calibrate the choice among ticks somewhat, but at a cost of about 0.26 in loss.

**Item 3 conclusion.**
- **The CMM attention sink:** ties CTsq and is barely used (removing it costs 0.004–0.006).
- **The CTM loss:** in four forms it either collapses the model (alone at lr 2e-3 or 1e-3, with the final-tick term, with a coda-input norm) or trains stably but much worse (detached readouts, +0.26). It does not make certainty a usable stopping signal.
- **Neither mechanism improves the CTM Transformer.** Its best remains the single-mechanism variants (CTnlm 3.572, CTso 3.576, CTsq 3.582), all tied with Dans.
