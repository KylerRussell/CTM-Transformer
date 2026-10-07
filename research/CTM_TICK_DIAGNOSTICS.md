# CTM-LM tick-use diagnostics — protocol (2026-10-06)

## Why

The adapted CTM-LM gains about 0.1 nats from its 16 ticks, and its per-tick loss is flat from tick 4 on ([CTM_LM_DESIGN.md](CTM_LM_DESIGN.md)). The [deep-research review](prompts/CTM_TICKS_DEEP_RESEARCH.md) ranks the explanations as follows:

1. **No cross-position refinement.** Each position's ticks re-read the same static backbone features, so later ticks get no new information.
2. **The start state already holds the answer.** The token-conditioned start state z_0 = z_init + W·f_i does most of the work.
3. **The loss gives no reward for improvement.** It trains every tick independently.
4. **Next-token prediction rarely needs iteration** at this scale.

This study runs the review's cheap evaluation-only tests (E1–E3) on trained checkpoints, then the training tests (E4c, E7, E8) under rules fixed here.

## Evaluation-only tests (`scripts/ctm_tick_diagnostics.py`)

**Models:**
- **CTM-aware, d 384, 400M tokens, lr 2e-3:** the best-trained CTM (D + sparse; trained ticks 4/8/12/16).
- **CTM-heavy, d 448, 100M tokens, lr 4e-3:** D + sparse.
- **D + mean-tick loss, d 448, 9.8M tokens:** every tick trained.
- **Transformer, d 384, 400M tokens, lr 4e-3:** the difficulty reference and the null.

**Data:** 32 held-out windows from offset 40,000 (32,768 tokens). The re-implemented tick loop reproduces the model's own per-tick losses exactly.

| Test | What it measures | Reading |
|---|---|---|
| **E1a:** tick attention switched off after tick k (k = 1, 3, 4, 8) | Whether later ticks use new input | If switching attention off after tick 3 costs < 0.02 nats at tick 16, later ticks do not use their attention reads. That supports hypothesis 1 or 2 and puts the training fixes before anything else. |
| **E1b:** token start state removed (z_0 = z_init) | How much of the prediction the start state carries | A large cost (≫ 0.1) supports hypothesis 2. This is out of distribution, so it is read only together with E1a. |
| **E1c:** ticks 17–32 (trained with 16) | Overthinking | If loss rises beyond 16, later ticks are not converging toward a better answer. |
| **E2:** benefit Δ = CE(tick 4) − CE(tick 16), binned by unigram surprisal, by the Transformer's loss, and by token type | Whether ticks help hard tokens | Hypothesis 4 predicts that more than 70% of the benefit comes from the top two difficulty deciles. A uniform or noisy benefit means the ticks act as an ensemble, not as computation. |
| **E3:** oracle gap (mean CE − per-token minimum CE) for ticks 4–16, against 13 MC-dropout passes of the Transformer (dropout on each block's residual update at rates 0.02–0.2) | Whether the 0.4-nat oracle gap is useful diversity or just noise | Compare the gaps at matched per-token spread. If the Transformer's noise reaches a similar gap, the tick "diversity" is noise, and losses that select among ticks are ruled out. |

## Training tests (fixed before the evaluation results)

**Setup** (amended before any training run: 100M tokens instead of 50M, so that an existing sweep run is the control):
- Every test is the CTM-aware arm at d 384, trained with the learning-rate sweep's exact settings: 381 steps of 262,144 tokens (100M tokens), warmup 100, cosine to 10%, one GPU, final evaluation on 1,024 held-out windows.
- **Control:** the sweep's best run at this width, `ctm_aware_d384_h1_k+2` (D + sparse, lr 4e-3, held-out loss 5.526). E7 and E8 use lr 4e-3.
- **Frozen-backbone tests (E4):** they use the backbone of the 400M-token CTM-aware model (lr 2e-3) and train at 2e-3.
- **Runner:** `scripts/tick_use_tests.py`. Seed spread is taken as 0.02.

- **E4c: plain head on the trained backbone.** Freeze the backbone of the 400M-token CTM-aware model and train a linear readout from f_i (fresh output weights).
- **E4a: fresh CTM on the same frozen backbone.** Trained for the same tokens and rate as E4c.
  - If E4c is within 0.04 nats of E4a, the CTM adds no computation beyond a readout of the backbone's features.
- **E7: improvement-rewarding loss with randomized tick count.**
  - Tick weights rise linearly over ticks.
  - A hinge term max(0, CE_t − sg(CE_{t−1})) penalizes any tick that is worse than the one before.
  - The tick count is drawn per step from the RDT's log-normal Poisson distribution (mean 16, maximum 32).
  - **Success:** tick 16 beats tick 4 by more than 0.04 nats, loss does not rise at tick 32, and final-tick loss is no worse than the control's.
- **E8: cross-position tick recurrence.** At each tick, the keys and values come from all causal positions' current CTM states (a projection of z_t) added to the backbone keys and values. Refinement at one position can then reach later positions.
  - **Success:** per-tick gains of more than 0.1 nats beyond tick 4, and final-tick loss better than the control's by more than 0.04.
  - Its compute cost is reported alongside.

E7 and E8 run whatever the evaluation results, because the review ranks them as the top fixes. A fix that passes on one seed gets a second seed before adoption.

## Results: evaluation-only tests (2026-10-06)

Full tables: [results/ctm_tick_diagnostics/DIAGNOSTICS.md](results/ctm_tick_diagnostics/DIAGNOSTICS.md). The figures below are for CTM-aware at 400M tokens; the other two models agree.

**E1a: later ticks do read their input, but only to stay where they are.**
- Switching the tick attention off after tick 3 costs 0.31 nats at tick 16 (4.302 against 3.992). The rule's "< 0.02" outcome did not occur.
- With attention on, the loss is lowest around tick 8 (3.967) and is worse by tick 16.
- Switching attention off after tick 8 still costs 0.07 at tick 16. The later reads keep the state from drifting; they do not improve it.

**E1b: the start state carries almost the whole prediction.**
- Without the token-conditioned start state, tick 1 is at the uniform loss: 10.33 against ln 32,768 = 10.40.
- The ticks recover only to 8.71 by tick 16, so the ticks cannot assemble a prediction through their attention reads.
- The model is out of distribution here, but the size of the effect leaves no doubt about where the prediction comes from.

**E1c: overthinking.** Every model is best at ticks 4–8 and gets worse with more ticks:

| Model | Tick 8 | Tick 16 | Tick 32 |
|---|---:|---:|---:|
| CTM-aware, 400M tokens | 3.967 | 3.992 | 4.082 |
| CTM-heavy, 100M tokens | 5.045 | 5.093 | 5.313 |

Mean predictive entropy rises steadily with ticks: 3.79 at tick 4 and 4.14 at tick 16 for CTM-aware.

**E3: the ticks differ no more than dropout noise does.**
- For ticks 4–16, the gap between the mean loss and the per-token best tick (an oracle that sees the label) is 0.209, at a per-token spread of 0.134.
- 13 MC-dropout passes of the Transformer at rate 0.02 give a gap of 0.286 at a spread of 0.171.
  - Gap per unit of spread: 1.56 for the ticks against 1.67 for the dropout passes.
  - Averaging the predictions gains 0.014 for the ticks and 0.018 for the dropout passes.
- The oracle gap is therefore noise. Losses that select among ticks (CTM's minimum-loss term, multiple-choice-learning variants) have nothing useful to select.

**E2: no difficulty-specific benefit once confidence is controlled.**
- Raw, the tick-4-to-16 change looks concentrated on hard tokens: +0.28 nats in the Transformer's hardest loss decile, −0.08 to −0.12 in the easiest, +0.003 overall.
- That pattern is what a tick does when it only becomes less confident.
- After fitting one softmax temperature per tick, the pattern shrinks to between −0.04 and +0.04 per decile, with a net of +0.001 (CTM-heavy: −0.013).
- By unigram surprisal and by token type, the benefit is noisy and has no consistent concentration.

**Reading.** The adapted CTM-LM's ticks:
- act on a prediction that the token-conditioned start state already contains;
- keep reading the static backbone features, but only to hold their state;
- grow less confident after about tick 8;
- differ from one another by no more than noise.

This supports hypotheses 2 (the start state holds the answer) and 3 (the loss gives no reward for improvement), and is consistent with hypothesis 1 (the reads return no new information). It gives no support to hypothesis 4: no hard-token benefit was found to explain away. E7 tests hypothesis 3, and E8 tests hypothesis 1.

## Results: E4c and E8 (2026-10-06)

- **E4c, plain head on the frozen backbone:** held-out 4.273, against 4.025 for the full CTM on the same backbone. The CTM adds 0.25 nats over a linear readout of f_i. E4a will show whether that comes from the ticks or from a larger nonlinear head.
- **E8, cross-position tick recurrence:** held-out 5.538, against 5.526 for the control.
  - The tick key and value projections grew from zero to about the size of the backbone's key projection, so the pathway was used.
  - The per-tick curve is unchanged: 5.461 at tick 4, 5.421 at tick 8, 5.474 at tick 16 and 5.686 at tick 32.
  - Fails the rule: it neither improves the final tick nor makes later ticks better.

## Recurrent depth in the RDT baselines (2026-10-06)

The same question for the RDT: held-out loss on the same 32 windows, evaluated at each recurrence depth. All three models were trained with randomized depth (log-normal Poisson, mean 15).

| Model (layout prelude/core/coda) | Depth 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| RDT-aware d 384 (11/2/11), 100M tokens | 4.706 | 4.671 | 4.666 | 4.665 | 4.665 | 4.665 |
| RDT-aware d 384 (11/2/11), 400M tokens | 4.031 | 3.979 | 3.971 | 3.971 | 3.971 | 3.971 |
| RDT-heavy d 704 (2/4/2), 100M tokens | 4.878 | 4.691 | 4.661 | 4.660 | 4.660 | 4.660 |

**The RDTs also stop gaining after about four iterations.**
- With a deep non-recurrent stack (11/2/11), recurrence is worth only 0.04–0.06 nats.
- With a shallow outer stack (2/4/2), it is worth 0.22 nats, all of it by depth 4, which is 20 effective layers.
- Unlike the CTM, the RDTs do not get worse with more iterations. Training with randomized depth makes them converge to a fixed point.

**Reading.** At this scale (40–80M parameters, 100–400M tokens), next-token prediction does not use more than about 20 effective layers. The CTM-aware model already has 24 backbone layers, so its ticks have no depth left to supply, and its loss (fixed 16 ticks, no randomization) lets the state drift after the answer is reached.

To make the ticks contribute, the ticks must supply depth the model needs. That is the next test:
- **S1:** the CTM-aware model with a 2-layer backbone.
- **S2:** S1 with cross-position recurrence, which makes the ticks a recurrent-depth core with CTM dynamics.

Both run at lr 2e-3. Their references are the CTM-aware control at the same rate (5.591), the RDT-aware model (4.724) and the Transformer (5.012 at its best rate).

**Success rule (fixed before the runs):**
- **Ticks contribute:** the tick 16 loss is at least 0.2 nats below tick 1, and ticks 8–16 are no worse than tick 4.
- **Worth adopting:** the final tick is also no worse than the 24-layer control at the same rate (5.591).

## Results: E7 and S2 (2026-10-07)

**E7 (improvement loss, randomized ticks, lr 4e-3) failed and was stopped at step 165.**
- It stalled above the unigram loss: held-out 9.62 at step 95, against 7.16 for the control and 7.62 for a unigram model.
- Its training loss oscillated between 8.2 and 9.7, with gradient norms up to 50.
- The hinge is summed over up to 31 tick pairs and dominates the loss. That formulation is not retried. Randomized tick counts are tested on their own in S3.
- E4a was deprioritized after S2 and has not run.

**S2 (2-layer backbone, cross-position ticks, lr 2e-3) is the first CTM-LM whose ticks do the work.** Held-out loss on 1,024 windows; per-tick loss on 32 windows:

| Model | Non-embedding parameters | Seconds per step | Held-out | Tick 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTM-aware control (24-layer backbone, lr 2e-3) | 40.5M | 26.6 | 5.591 | 6.332 | 5.787 | 5.517 | 5.491 | 5.526 | 5.670 |
| **S2** | **8.6M** | **20.9** | **5.303** | 5.906 | 5.528 | 5.253 | 5.214 | 5.237 | 5.338 |
| Transformer d 384 (lr 2e-3 / best 4e-3) | 42.5M | 4.2 | 5.133 / 5.012 | | | | | | |
| RDT-aware d 384 (lr 2e-3) | 42.8M | 10.0 | 4.724 | | | | | | |

- S2 beats the 24-layer CTM by 0.29 nats with a fifth of the non-embedding parameters, and costs 21% less per step.
- It passes the rule fixed before the runs:
  - tick 16 is 0.67 nats below tick 1;
  - ticks 8–16 are no worse than tick 4 (5.214–5.237 against 5.253);
  - the final tick beats the control.
- **The rule's first criterion turned out to be weak.** Under the sparse loss tick 1 is not trained, and the 24-layer control also shows a 0.81-nat drop from tick 1 to tick 16.
- **The decisive test is two copies of S2 trained with a fixed 1 or 4 ticks (S2_T1, S2_T4)** at the same parameter count. If S2 with 16 ticks is clearly better than S2_T1, the ticks contribute.
- **Drift remains.** S2 is best at tick 8, and its loss rises by tick 32 (5.338). S3 adds randomized tick counts against it.
- **It still trails the baselines.** S2 is 0.17 nats behind the Transformer at the same rate and 0.58 behind the RDT, but with a fifth of their non-embedding parameters. A version with matched parameters (a wider CTM state) is the next step if the tick-count controls confirm that the ticks contribute.

## Results: S1, S3 and the fixed-iteration controls (2026-10-07) — correction to S2

Full table: [results/tick_use/TESTS.md](results/tick_use/TESTS.md). Held-out loss on 1,024 windows; all at 100M tokens.

| Model | Iterations in training | Held-out | Seconds per step |
|---|---|---:|---:|
| S2: 2-layer backbone, cross-position ticks | 16 | 5.303 | 20.9 |
| **S2_T1:** the same model trained with 1 tick | 1 | **5.246** | 7.1 (GPU shared with S3) |
| S2_T4: trained with 4 ticks | 4 | 5.303 | — |
| S1: 2-layer backbone, no cross-position | 16 | 5.321 | — |
| S3: S2 with randomized tick count | 1–32, mean ~16 | 5.744 | 35.8 (GPU shared) |
| RDT-heavy d 704, randomized depth (sweep) | 1–48, mean ~16 | 4.721 | 22.1 |
| **R1:** the same RDT trained at depth 1 | 1 | **4.681** | 4.3 |

**Correction.** S2's per-tick curve does not show that its ticks contribute. The same model trained with a single tick is better (5.246 against 5.303) at a third of the cost. As with the earlier models, the drop from tick 1 to tick 8 inside S2 only reflects training: an untrained tick 1 is bad. It does not show that 16 ticks beat 1. S2's gain over the 24-layer CTM comes from the shallower backbone, not from the ticks.

**The same holds for the RDT.**
- Trained at a fixed depth of 1, RDT-heavy scores 4.681, against 4.721 with randomized depth (mean 15), at a fifth of the cost.
- Its depth curve (4.878 at depth 1, 4.660 at depth 16) was the same artefact: depth 1 was rarely sampled in training.

**Other results:**
- **Cross-position recurrence adds nothing:** S1 scores 5.321, S2 5.303.
- **Randomized tick counts hurt the CTM** (S3: 5.744 against 5.303) and make training slower.

**Conclusion.** In this regime (8–80M non-embedding parameters, 100M FineWeb-Edu tokens), neither the CTM's ticks nor the RDT's recurrence improves next-token prediction over the same model trained with a single iteration. The CTM-LM's failure to use its ticks is therefore not specific to the CTM. In this regime, extra iterative computation has no measured value, and the 1-tick and depth-1 models are both better and cheaper.

**Implications:**
- The paper's comparisons of iterative models must include a single-iteration control trained the same way. Within-model per-iteration curves are not evidence of tick use.
- Whether ticks help with more data, or on tasks that need iteration, is open.

## Next: settings where iteration could pay (2026-10-07, user decision: A and B, plus wider CTM internals)

Each test pairs a 16-tick model with the same model trained with a single tick. Rules are fixed before the runs. The LM seed-to-seed spread is taken as 0.02 nats.

| Test | Models | Ticks contribute if |
|---|---|---|
| **A: longer training** | A_S2_400M and A_S2_T1_400M: S2 and S2_T1 for 400M tokens (1,526 steps, the sweep's 4× schedule) | the 16-tick model's held-out loss is at least 0.04 below the 1-tick model's |
| **W: wider CTM internals** (user suggestion) | W_S2_wide and W_S2_wide_T1: S2 with the synapse U-Net width doubled (1,280 to 2,560) and the neuron-level models' hidden width doubled (32 to 64), at 16 and 1 ticks, 100M tokens | the same 0.04 margin; W_S2_wide against S2 shows the effect of width itself |
| **B: a task that needs serial computation** | `scripts/tick_s3.py`: the S₃ word problem on the reliability study's first 10 seeds and data. Tiny adapted CTM-LM, plain, cross-position and wide, each at T = 16 and T = 1. The reliability study's Transformer, RDT and faithful CTM-LM runs are paired references. | mean accuracy over positions 1–16 beats the T = 1 control on at least 8 of 10 seeds, with a median gain of at least 5 points |

The sweep is held on both GPUs while these run. Its two interrupted runs resume from their checkpoints.

## Results: B (S₃) and W (wide), with A in progress (2026-10-07)

**B: on the S₃ word problem, the ticks contribute.** Full table: [results/tick_s3/TICK_S3.md](results/tick_s3/TICK_S3.md). Ten seeds; mean held-out accuracy over positions 1–16.

| Cell | T = 16 | T = 1 control | Seeds higher than the control | Median paired gain | Escapes | Verdict under the rule |
|---|---:|---:|---:|---:|---:|---|
| Adapted CTM, cross-position ticks | **0.856** | 0.685 | 9/10 | **+18.6 points** | 4/10 | ticks contribute |
| Adapted CTM, wide internals | 0.716 | 0.662 | 8/10 | +7.0 | 0/10 | ticks contribute |
| Adapted CTM | 0.726 | 0.679 | 9/10 | +3.2 | 0/10 | below the 5-point margin |
| *References (reliability study, same seeds and data):* Transformer / RDT / faithful CTM-LM | 0.614 / 0.715 / 0.579 | | | | 0 / 1 / 0 | |

- **With cross-position ticks, the adapted CTM is the strongest model tested on S₃.**
  - It gains 18.6 points from its ticks, and its accuracy rises with ticks up to 16: 0.463, 0.764, 0.844, 0.856 at T = 1, 4, 8, 16.
  - It escapes to the serial solution on 4 of 10 seeds, against 1 of 10 for the RDT.
- **Caveat on the references:** they are not like-for-like. They use learned absolute positions and their own losses; the adapted CTMs use RoPE and the sparse-tick loss. A Transformer and an RDT with the same RoPE setup are needed before an architecture claim.
- **Cross-position recurrence is what matters here.** It adds nothing on FineWeb at 100M tokens, but on S₃, where each answer must be carried forward position by position, it turns the ticks into serial computation.

**W: wider internals help the LM, but the ticks still do not.** Held-out loss at 100M tokens:

| Model | 16 ticks | 1 tick |
|---|---:|---:|
| S2 | 5.303 | 5.246 |
| S2, wide internals | 5.230 | **5.127** |

- Width is worth 0.07–0.12 nats.
- The 1-tick model is better at both widths: by 0.10 with the wide internals, which fails the rule.

**A: at 400M tokens the ticks begin to pay (provisional; the 16-tick run is at step 1,275 of 1,526).** Held-out loss at the shared evaluation steps (128 windows; the final is on 1,024):

| Step (tokens) | 381 (100M) | 762 (200M) | 1,143 (300M) | 1,526 (400M) |
|---|---:|---:|---:|---:|
| S2, 16 ticks | 5.119 | 4.442 | 4.194 | running |
| S2, 1 tick | 5.050 | 4.566 | 4.351 | 4.295 (final) |
| References: Transformer / RDT-aware / 24-layer CTM-aware | 4.862 / 4.744 / 5.415 | 4.062 / 4.238 / 4.479 | 3.803 / 4.037 / 4.117 | 3.735 / 3.998 / 4.025 |

The 1-tick model leads early. The 16-tick model overtakes it by 200M tokens and leads by 0.157 at 300M.

## Next round (2026-10-07, user approved): is it general, is it fair, does it combine?

Rules fixed before the runs:

| Question | Runs | Rule |
|---|---|---|
| **Does recurrence also pay for the RDT at 400M tokens?** | `R_aware_d1_400M`: the RDT-aware 400M sweep run (3.998) retrained at depth 1. `R_heavy_d1_400M` and `R_heavy_rand_400M`: RDT-heavy d 704, 2/4/2, at depth 1 and at randomized depth, 400M tokens, lr 1e-3. | Recurrence pays if the randomized-depth model is at least 0.04 below its depth-1 twin. |
| **Is the S₃ comparison fair?** | `transformer_rope`, `rdt_rope` and `rdt_rope_t1` (`scripts/tick_s3.py`): the reliability study's Transformer and RDT retrained with RoPE, the position scheme of the adapted CTMs, on the same 10 seeds. | adapt_cross exceeds rdt_rope if it is higher on at least 8 of 10 seeds with a median gain of at least 5 points. The RDT's own tick use is judged by the same rule against rdt_rope_t1. |
| **Do width, cross-position ticks and longer training combine?** | `C_wide_400M` and `C_wide_T1_400M`: the wide CTM (W_S2_wide) at 16 and 1 ticks for 400M tokens. | Ticks contribute if the 16-tick model is at least 0.04 below the 1-tick model. Compared with A_S2_400M, this gives the effect of width at 400M. |
