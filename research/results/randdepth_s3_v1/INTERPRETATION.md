# Interpretation of the S3 randomized-depth study

## Finding under the declared rules

Forty runs completed and passed the freeze audit, including the exact check of every run's per-update depth sequence against the seeded sampler. `rdt_rand` and `ctm_lm_rand` were trained with log-normal-Poisson depth (mean 15, σ 0.5, cap 48; expected depth about 16). Each run was paired by seed and data with the fixed-T = 16 runs of the [reliability study](../reliability_s3_v1/INTERPRETATION.md). The evaluation was read once, after the freeze.

**None of the six declared paired tests is significant after Holm correction.** The protocol's claims therefore stand as not shown: randomized depth is not established to raise escape, mean accuracy or extrapolation for either family.

| Comparison | Effect | p | Holm p |
|---|---|---:|---:|
| RDT: escape at T = 16 | 7/20 against 2/20 (6 against 1 discordant) | 0.125 | 0.47 |
| RDT: positions 1–16 at T = 16 | +9.9 points | 0.062 | 0.31 |
| RDT: positions 17–32 at T = 32 | +0.3 points | 0.116 | 0.47 |
| CTM-LM: escape at T = 16 | 0/20 against 0/20 | 1.0 | 1.0 |
| CTM-LM: positions 1–16 at T = 16 | −4.0 points | 0.042 | 0.25 |
| CTM-LM: positions 17–32 at T = 32 | −0.1 points | 0.67 | 1.0 |

## Descriptive reading (not declared tests)

1. **For RDT, the direction is favourable and the effect size is large, but the study is underpowered.**
   - The escape probability rose from 10% (CI 1.2–31.7%) to 35% (CI 15.4–59.2%).
   - Six seeds escaped only with randomized depth, and one escaped only with fixed depth.
   - 13 of 20 seeds improved on positions 1–16 (median +8.6 points).

   With 20 pairs, exact McNemar needs at least 6 against 0 discordant pairs to reach p < 0.05, before multiplicity correction. The effect is worth a larger confirmation, not a claim.
2. **Randomized depth makes RDT faster per step and stable beyond its training depth.** Mean accuracy over positions 1–16 by inference depth:

   | Cell | T = 4 | T = 8 | T = 16 | T = 32 | T = 48 | T = 64 |
   |---|---:|---:|---:|---:|---:|---:|
   | Randomized depth | 65.2% | 78.2% | 80.4% | 80.5% | 80.5% | 80.5% |
   | Fixed depth | 46.4% | 62.0% | 70.5% | 70.8% | — | — |

   Median correct prefix at T = 4 rose from 2 to 6. This matches the recurrent-depth literature. Training on short sampled depths forces progress per step, and training on long ones makes extra steps harmless. But the model also plateaus at the training mean instead of improving with more steps, as Parcae reports.
3. **For CTM-LM, randomized depth removed tick use.**
   - Randomized-depth CTM-LM scores the same at every inference depth: 51.5% at T = 4 and 52.6% at T = 16–64, with median prefix 4 throughout.
   - Fixed-T CTM-LM rose from 19.9% at T = 4 to 56.6% at T = 16.

   CTM's loss averages the minimum-loss tick and the most certain tick. Under randomized depth, many updates have few ticks, so the loss rewards answering at the earliest ticks. The model then settles on a tick-independent solution. More seeds sat at chance on positions 9–16: 7 against 2.
4. **Nothing extrapolates, and in this study that result is uninformative by construction.** Every run, including runs perfect on positions 1–16, is at chance (1/6) on positions 17–32 at every T. Both families use **learned absolute position embeddings** (`use_positional_encoding = True`, `max_seq_len = 128`), and training words have lengths 1–16. So rows 17–32 of the position table never receive a gradient and stay at their random initialization.

   Declared test 3 therefore could not detect length generalization in either cell. This is a protocol design error, missed when the protocol was declared. It also accounts for the "nothing extrapolates past the trained length" finding in every earlier S₃ study. That finding tells us nothing about the architectures, only that positions 17 and later were never trained. The recorded test results stand as declared.

## Consequences

- **The next S₃ protocol must fix positional encoding before any extrapolation endpoint**:
  - NoPE or RoPE, with no learned absolute table;
  - or random position offsets during training, so every position row is trained.

  Serial state tracking on S₃ needs no absolute position, so NoPE or RoPE is the cleaner choice.
- **For RDT, randomized depth is the best training recipe so far** (35% escape, faster per-step progress, stable at large T). Keep it as the default recipe for the next step, which adds a length curriculum and per-cell learning rates. Its effect on escape needs a larger or combined confirmation.
- **For CTM-LM, randomized depth with CTM's loss is counterproductive.** It should stay on fixed ticks, or randomized depth should be paired with a loss that rewards later ticks. CTM-LM remains significantly below RDT. Its inclusion in the language-model phase stays conditional on a later recipe making its ticks useful.

See [results](RESULTS.md) and the [frozen protocol](PLAN_BEFORE_RUNS.md).
