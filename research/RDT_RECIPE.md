# RDT recipe study and the path to the final comparison — protocol (2026-10-08)

## Why

The second deep-research review ([prompt](prompts/CTM_RDT_HYBRID_DEEP_RESEARCH.md)) and our own data point to a recipe problem, not a recurrence problem:

- **The deficit at depth 1.** RDT-aware at depth 1 has the Transformer's 24 unique layers but scores 3.997 against 3.735 at 400M tokens.
  - Its usable learning rate is 1e-3, against 4e-3 for the Transformer.
  - It needs unit-scale embeddings to avoid collapse.
- **Huginn's reasons for its recipe** (Geiping et al. 2025, checked in the paper):
  - Sandwich norm in every block was a fix for training at 3.5B parameters. "At small scales, most normalization strategies, e.g. pre-norm, post-norm and others, work almost equally well."
  - Huginn's initial state is random (variance 2/5); ours is zero.
  - Its embeddings are scaled to an effective std of about 0.63; ours use 1.0.
  - It truncates backpropagation to the last 8 iterations and warms up for 4,096 steps.
- **The caution.** Huginn's pre-norm attempt "learned early to ignore the incoming state": loss was the same at 1 and 32 iterations. A recipe fix must therefore keep the recurrence useful, not only close the depth-1 gap. Every variant gets a randomized-depth run and its depth-1 twin.
- **The literature** (review, partly verified): at matched unique parameters a looped model should beat its single-pass twin. At matched FLOPs, a deeper Transformer should win on perplexity. Loops pay mainly on reasoning tasks and when tokens per parameter are high.

## Round 1 (`scripts/rdt_recipe.py`, `ctm_transformer/rdt_recipe.py`)

**Base:**
- RDT-heavy at d 704, prelude/core/coda 2/4/2, 48.2M non-embedding parameters. It is the only RDT layout whose recurrence paid at 400M tokens: 3.783 with randomized depth against 3.828 at depth 1.
- 400M tokens; the sweep's schedule (1,526 steps, warmup 100, cosine to 10%).

**Variants** (both drop the unit-scale embeddings):

| Variant | Normalization | State and injection |
|---|---|---|
| B | pre-norm prelude and coda; sandwich norm only in the core | zero initial state |
| C | pre-norm everywhere | RMSNorm on the injected prelude output and at the core exit; random initial state (Huginn) |

**Phase 1** (depth-1 twins pick the learning rate cheaply):
- each variant at depth 1, lr 1e-3, 2e-3 and 4e-3;
- a matched-unique-parameter Transformer: 8 layers at d 704, lr 2e-3 and 4e-3.

**Phase 2:**
- each variant with randomized depth at the rate of its best depth-1 run;
- at half that rate if that run diverges or collapses.

**Rules, fixed before the runs:**
- **Adopted:** randomized-depth loss below 3.783, and at least 0.04 below its own depth-1 twin at the same rate (recurrence still pays).
- **Recipe gap closed:** depth-1 loss within 0.03 of the 8-layer Transformer's best.

**Round 2 (planned):**
- the winner plus weight decay excluded from norms and embeddings, and a 300-step warmup;
- truncated backpropagation (k = 8) against full;
- then the hybrids (RDT core plus CTM internals, each zero-initialized so that the model equals the RDT at initialization), on S₃ and on language modelling at matched FLOPs.

## CTM drift fixes on S₃ (`scripts/tick_s3.py`)

| Cell | Change from adapt_cross |
|---|---|
| `cross_anchor` (and its T = 1 twin) | the backbone feature is re-injected every tick (`observe_token`) and the state update is gated (`gated_state`) |
| `cross_clamp` | the official code's in-place decay clamp (`decay_data_clamp`) |

**Rule:** improves on adapt_cross if it is higher on at least 8 of 10 seeds with a median gain of at least 3 points. Drift is read from accuracy at T = 32 against T = 16.

## The final comparison: a scaling ladder (user direction, 2026-10-08)

The user wants the final runs larger, so that performance at higher parameter counts can be estimated. A single large run would be one point, and at our budget an undertrained one: 500M parameters at 1–2B tokens is 2–4 tokens per parameter. The plan is therefore a ladder:
- roughly 50M, 150M and 500M unique parameters;
- every arm at the same tokens-per-parameter ratio, so the fitted curve is a consistent slice of the scaling law;
- the top rung limited to the Transformer, the best RDT and the best hybrid.

The ladder's sizes, ratio and arms are decided after round 1 fixes the recipe.
