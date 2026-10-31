# Per-token ticks: making extra thinking pay in a language model (proposal, 2026-10-31)

**Status:** proposal for item 4. Nothing here is implemented or run.

## Why

- **Parity found what makes extra ticks pay** (`scripts/parity_ticks.py`, `_lent`). Give each training input a tick count that grows with its difficulty: there, ceil(0.75 × length) + U{0..8}.
  - With it, Dans solved inputs up to 3–4 times the trained length by thinking longer: 3 of the first 4 seeds, reaching 80–95% on 81–160 bits at 128 ticks.
  - With the usual sampler (mean 15 for every input), no CTsq or Dans seed did (0 of 70 runs).
  - The working explanation: a serial solution only gets gradient when the training ticks are enough to finish it. With too few ticks, the fixed-reach solution wins.
- **In the language model, every token of a sequence runs the same tick count**, drawn from the same sampler. Loss is flat after about 8 ticks for every model; the state reaches a fixed point.
- **The language-model analogue of "longer input, more ticks" is "harder token, more ticks".** That needs two things the model does not have:
  1. a tick count per position;
  2. a measure of each token's difficulty available during training.

## The mechanism: a tick budget per position

- Each position i has a budget T_i. The thinking group runs max_i T_i ticks for the sequence.
- **Freezing:** after its T_i-th tick, position i's state is frozen, the state copied unchanged into later ticks. The same holds for its neuron-model history and its synchronization running sums.
  - Today the synchronization's beta is one value per pair, shared across positions. It becomes one per position and pair.
- **Attention:** at every tick, active positions attend to all earlier positions' current states. A frozen position offers its final state. This is the per-position halting of the Universal Transformer and of ACT, applied to Dans and the CTM Transformer alike.
- **Readout:** the coda reads each position's final (possibly frozen) state, as now.
- **Equivalence check:** when every T_i equals T, the model is exactly today's model. A unit test checks this against the current forward pass.
- **Cost:** training still runs max T_i ticks per sequence, so it saves nothing in training. The cost comes from the budget map below.

## The budget: harder tokens get more ticks

- **Difficulty:** each micro-batch first runs a no-gradient pass at 2 ticks. A token's difficulty is its cross-entropy there. It uses the labels, which is fine in training only, as the parity length did.
  - The pass costs about 5% of a training step.
  - Difficulty is the model's own, now: what is hard for this model at this point in training.
- **Budget map:** rank the micro-batch's tokens by difficulty (quantile q_i in [0, 1]). Then T_i = 1 + round(24 · q_i) + j_i, with j_i ~ U{0..6} per token.
  - The mean is about 16 ticks per token, matched to the sampler's 15. Dans and the per-token model spend the same ticks per token on average.
  - The maximum is 31, so a step costs about 31 ticks for the sequence, about 2.1 times today's mean 15.
- **At evaluation, no labels are available.** The model runs:
  - a fixed T for every token (1–96, as in the capability evaluation);
  - the convergence exit, which stops a token once its prediction changes little between ticks. On the existing checkpoints it reaches the 16-tick loss at about 3.8 mean ticks (`research/results/exit_calibration`).

  No difficulty predictor is needed at inference.

**Risks:**
- **High loss is not the same as "helped by thinking".** Some tokens are unpredictable (a name, a number), and extra ticks cannot help them.
  - An alternative difficulty measure is the gain from thinking: CE at 2 ticks minus CE at 8 ticks, from one no-gradient pass at 8 ticks (about 15% of a step).
  - I'd add it as a second arm only if the first arm shows nothing.
- **Easy tokens never see deep ticks in training.** At fixed-T evaluation they might drift, the "overthinking" problem. The jitter j_i and the convergence exit address this. The fixed-T curve shows whether it happens.

## Stage A: the bridge test, parity inside text (about 5 hours, both GPUs)

**Purpose:** before the language model, check that the per-position mechanism reproduces the `_lent` result inside a language-model sequence, where difficulty is known.

- **Data:** FineWeb-Edu windows of 256 tokens, our 32k tokenizer. Each window has one parity problem inserted at a random position: `<sep> bits <sep> answer`, with 1–40 bits drawn as two reserved tokens.
  - The loss covers every token: text and answer.
- **Models:** CTsq and Dans at width 128, as in the parity runs. The 32k embedding adds 4M parameters.
  - 5,000 updates of batch 128, learning rate 1e-3, as in `scripts/parity_ticks.py`.
- **Arms** (10 seeds each, CTsq and Dans):
  - **uniform:** every position gets the sampler's tick count (mean 15, at most 48), as now;
  - **per-token, known difficulty:** the answer position gets ceil(0.75 × bits) + U{0..8}; every other position gets the sampler's count, capped at the sequence's maximum.
- **Rules** (fixed before any run):
  - Tick-scaling extrapolation is counted as in `parity_ticks.py`: on 41–80 bits inside text, 64-tick accuracy at least 80% and at least 20 points above 16-tick accuracy.
  - The mechanism works if a per-token arm has at least 3 of 10 such seeds and at least 3 more than its uniform arm.
  - Text loss must stay within 0.02 of the uniform arm, on two-seed means of the held-out text tokens.
  - The synchronization queries matter if CTsq's count exceeds Dans's by at least 3.
- **Decision:** if the mechanism does not work here, stop before Stage B.

## Stage B: the language-model pilot (about 21 hours, both GPUs)

**Purpose:** do per-token ticks, with difficulty measured by the model itself, make extra ticks pay on web text?

- **Setup:** d704, the 2 + 4 + 2 layout, 100M tokens (381 updates of 262,144 tokens; warmup 50; cosine to 10%; learning rate 2e-3). Two seeds.
- **Arms:**
  - Dans and CTsq with the usual sampler, at 100M tokens: a new baseline, since the 400M-token runs' schedule differs;
  - Dans and CTsq with per-token ticks.
- **Cost per seed:**

  | Arm | Hours |
  |---|---:|
  | Dans, sampler | 2.5 |
  | CTsq, sampler | 3.9 |
  | Dans, per-token | about 5.5 |
  | CTsq, per-token | about 8.6 |
  | **Total** | **about 20.5** |

  Two seeds: about 41 GPU-hours, about 21 hours on both GPUs.
- **Measures:**
  - held-out loss at fixed 1, 2, 4, 8, 16, 32, 64 and 96 ticks;
  - loss and mean ticks under the convergence exit.
- **Rules** (fixed before any run; two-seed means; both seeds must agree):
  - **Ticks pay:** loss at 32 ticks and at 64 ticks each at least 0.01 below loss at 16 ticks. No existing model meets this; every one is flat from about 8 ticks.
  - **No cost:** the per-token model's 16-tick loss within 0.01 of its sampler baseline at 100M tokens.
  - **Better use of compute:** at the convergence exit's mean ticks, at least 0.02 lower loss than the sampler baseline at the same mean ticks.
  - **The CTM matters:** CTsq's per-token gain (sampler minus per-token loss, at 32 ticks and under the exit) exceeds Dans's by at least 0.02.
- **Decision:** if ticks pay and cost nothing, the next step is 400M tokens with two seeds, the standard of every other comparison, for Dans and CTsq.

## Implementation

1. **Per-position budgets:** freezing masks in `ctm_transformer/rdt_recipe.py` (Dans) and `ctm_transformer/ctm_transformer.py` (state, neuron-model history, synchronization sums and a per-position beta). Unit tests:
   - equal budgets reproduce today's forward pass exactly;
   - a frozen position's state and readout do not change after its budget;
   - gradients flow only through a position's active ticks.
2. **Training:** in `scripts/pretrain.py`, a `per_token_ticks` option: the no-gradient difficulty pass, the budget map, and the budgets passed to the model. Bridge data and its evaluation go in a new `scripts/parity_in_text.py`.
3. **Order:** smoke tests, the rules committed, Stage A, then Stage B only after its decision rule.
