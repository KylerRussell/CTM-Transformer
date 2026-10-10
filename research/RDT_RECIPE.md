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

## Round 1 results (interim, 2026-10-09)

Summaries: `research/results/rdt_recipe/RDT_RECIPE.md`, `research/results/tick_s3/TICK_S3.md`.

**Phase 1, held-out loss at 400M tokens:**

| Run | lr 1e-3 | lr 2e-3 | lr 4e-3 |
|---|---:|---:|---:|
| B at depth 1 | collapsed (7.62) | collapsed (7.62) | collapsed |
| C at depth 1 | 4.464 | 5.783 | diverged or collapsed |
| 8-layer Transformer, d 704 | — | **3.618** | 3.674 |

- **The 8-layer Transformer beats every model in the study.** It scores 3.618, against 3.735 for the 24-layer d 384 Transformer and 3.783 for RDT-heavy with randomized depth. It has the same unique layers as RDT-heavy, so the current RDT recipe costs 0.21 nats at depth 1 (3.828), and 16 recurrences recover only 0.045 of that.
- **B's collapse was a design error, not a result.** B kept sandwich norm in the core and dropped the unit-scale embeddings. Those embeddings were added (commit 60b5efd) because the sandwich-normed RDT collapsed to the unigram with the small embedding std. B reproduced that collapse at every rate: the loss stayed at 7.61–7.62 from step 25 onwards. Measured at initialization, the token signal entering B's core has RMS 0.26, against 0.63 in the current recipe, while each sandwich block renormalizes the stream.
- **C learns, but slowly.** It sat at the unigram until step 50–100 (the Transformer left it by step 50) and ends 0.85 nats behind the Transformer at depth 1. Its recurrence pays heavily: with randomized depth it was 4.476 at step 762, against 4.953 for its depth-1 twin. Suspects for the depth-1 deficit are the random initial state, which is pure noise at depth 1, and the normalization.

**Phase 2:**
- **C with randomized depth at 1e-3: 4.078.** That is 0.386 below its depth-1 twin (4.464), so the recurrence pays, but it is far above 3.783. **C is not adopted.**
- B with randomized depth at 1e-3 is at the unigram (7.61 at step 610). It is a casualty of the design error, so its result carries no information.

**Round 1b (added 2026-10-09, after the results above), on GPU 1** (`scripts/rdt_recipe.py`):
- **D** at depth 1, lr 2e-3: pre-norm everywhere, zero state, no extra norms. At depth 1 this is the 8-layer Transformer plus the injection adapter and a norm between core and coda. If it matches 3.618, the recipe (sandwich norm and unit embeddings) explains the whole depth-1 gap.
- **E** at depth 1, lr 2e-3: C with a zero initial state. It tests whether the random state's noise causes C's slow start.
- **D with randomized depth**, lr 2e-3. It tests whether a plain pre-norm recurrence still pays, or ignores its state as Huginn's pre-norm run did.

**S₃ drift cells** (pre-registered rule: higher than adapt_cross on at least 8 of 10 seeds, median gain at least 3 points):

| Cell | Accuracy 1–16 | Accuracy at T = 1 / 4 / 8 / 16 / 32 | vs adapt_cross | Verdict |
|---|---:|---|---|---|
| adapt_cross | 0.856 | 0.463 / 0.764 / 0.844 / 0.856 / 0.848 | — | — |
| cross_anchor | **0.923** | 0.508 / 0.799 / 0.905 / 0.923 / 0.910 | 7/10, median +5.1 | does not improve (one seed short) |
| cross_clamp | 0.808 | 0.469 / 0.720 / 0.787 / 0.808 / 0.782 | 4/10, median −3.8 | does not improve |

- cross_anchor has the study's highest accuracy and most escapes (7 of 10 seeds with positions 9–16 at least 0.9). Its ticks contribute strongly: +28.4 points median over its T = 1 twin, on 9 of 10 seeds.
- Exploratory, not pre-registered: against rdt_rope, cross_anchor is higher on 7 of 10 seeds, median +6.8 points.
- Drift from T = 16 to T = 32 is about one point for every CTM cell. The anchor does not remove it.

**Round 1b result: D at depth 1 scores 4.123** (lr 2e-3), against 3.618 for the 8-layer Transformer. The two configurations are identical apart from D's two RDT components: the concatenation adapter, through which the whole residual stream passes, and the output norm reused before the coda. The Transformer is ahead from step 100 onwards and leads by 0.85 at step 381 (4.422 against 5.277). The RDT variants are also strongly rate-sensitive (C at depth 1: 4.464 at 1e-3, 5.783 at 2e-3), so 2e-3 may be the wrong rate for D.

**Round 1c (2026-10-09): 100M-token probes** (the first 381 steps of the same schedule, evaluated at step 381):
- **Da:** additive injection (state + input, no adapter), at lr 2e-3 and 1e-3;
- **Dn:** no norm before the coda, at lr 2e-3;
- **D** at lr 1e-3.

At depth 1, additive injection without the coda norm is exactly the Transformer (tested in `tests/test_rdt_recipe.py`), so Da and Dn each isolate one component. D's randomized-depth run and B's half-rate fallback were stopped. B's failure is explained, and D's rate and injection are in question.

**Round 1b/1c results.** E (C with a zero state) at depth 1: 5.353, against 5.783 for C at the same rate. Probes at step 381 (100M tokens) of the same schedule:

| Depth 1 | lr 2e-3 | lr 1e-3 |
|---|---:|---:|
| D: adapter + shared coda norm | 5.277 | 4.994 |
| adapter, no coda norm (Dn) | 4.712 | |
| additive injection + shared coda norm (Da) | 5.035 | 4.773 |
| additive, no coda norm (= the Transformer) | **4.422** | |

- **The shared coda norm costs about 0.59 nats and the adapter about 0.27, roughly additively.** Lowering the rate recovers about 0.26 but does not close the gap.
- The common pattern: every norm on the main path between the embeddings and the output hurts here. That is sandwich norm in every block (which needed unit-scale embeddings to train at all), C and E's injection and exit norms, and the shared coda norm. A likely mechanism is that the residual stream is small early on (embedding std 0.024), so a norm in the middle amplifies the gradient to everything before it. This is not tested.

**Round 1d (2026-10-09):** Dan (additive injection, no coda norm) and Dans (Dan plus an RMSNorm on the recurrent state only) with randomized depth at lr 2e-3, 400M tokens.
- Both are exactly the 8-layer Transformer at depth 1 (tested), so their depth-1 twin is T8_lr0.002 (3.618). They are adopted if they score at least 0.04 below it.
- **Expected risk for Dan:** at initialization its state grows linearly with depth (RMS 0.9 after one iteration, 29 after 32), and by iteration 32 each iteration changes it by 3%. That is Huginn's "ignore the state" failure in the making. Dans bounds the state, and RMSNorm maps the zero initial state to zero, so depth 1 is unchanged.

**Round 1d result: Dans is adopted.** Held-out loss at 400M tokens, randomized depth evaluated at 16:

| Run | Loss | vs depth-1 twin (T8_lr0.002, 3.618) |
|---|---:|---:|
| Dan (additive, no norms) | 3.617 | −0.001 |
| **Dans** (Dan + state norm) | **3.569** | **−0.049** |
| previous RDT-heavy recipe | 3.783 | |
| C (pre-norm, normalized injection, random state) | 4.078 | |

Dans meets both rules: below 3.783, and at least 0.04 below its depth-1 twin. It is one seed and passes the 0.04 margin by 0.009.

**Loss against test-time depth** (`scripts/rdt_depth_curve.py`; final evaluation windows; results in `research/results/rdt_recipe/depth_curve_*.json`):

| Run | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dans | 3.840 | 3.643 | 3.580 | **3.569** | 3.569 | 3.569 | 3.569 |
| Dan | 4.445 | 3.848 | 3.677 | 3.624 | 3.617 | 3.634 | 3.650 |
| C | 4.469 | 4.134 | 4.081 | 4.078 | 4.078 | 4.078 | 4.078 |

- **Dans converges by 8 iterations and does not drift out to 48**, three times the mean training depth. Every CTM variant drifts past its trained tick count. At 4 iterations it already beats the Transformer (3.580).
- **Dan uses its recurrence** (4.445 at one pass, 3.617 at 16) but gains nothing over the Transformer, and it drifts past 16 as its unnormalized state grows.
- **Compute:** Dans at 8 iterations applies 36 layers per token, against 8 for the Transformer (4.5 times the FLOPs); at 4 iterations, 20 layers (2.5 times). The comparison so far is at matched unique parameters, not matched FLOPs.
