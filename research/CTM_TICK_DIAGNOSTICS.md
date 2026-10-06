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
