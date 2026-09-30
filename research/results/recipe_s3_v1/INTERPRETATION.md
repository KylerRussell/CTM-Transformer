# Interpretation of the S3 recipe confirmation

## Findings under the declared rules

All 120 runs completed (4 cells × 30 new seeds) and passed the freeze audit: 1,200,000 updates, including every run's depth sequence, learning rate, the absence of the position table, and the factory name. The evaluation was read once, after the freeze. Every model uses RoPE, and each cell uses its development-tuned learning rate.

**All four declared tests are significant after Holm correction**:

| Test | Comparison | Effect | Holm p |
|---|---|---|---:|
| T1 | randomized-depth vs fixed-depth RDT, positions 1–16 at T = 16 | **+14.3 points** (95.8% against 81.5%) | < 2 × 10⁻⁶ |
| T2 | the same pair, escape at T = 16 | **24/30 against 0/30** (24 against 0 discordant) | 4.8 × 10⁻⁷ |
| T3 | randomized-depth RDT vs Transformer, positions 17–24 | **+26.0 points** (43.2% against 17.2%; chance is 16.7%) | < 2 × 10⁻⁶ |
| T4 | CTM-LM vs fixed-depth RDT, positions 1–16 at T = 16 | **−22.7 points** (58.9% against 81.5%) | < 2 × 10⁻⁶ |

The sign-flip p-values sit at the Monte Carlo floor: no draw out of 2,000,000 was as extreme as the observed statistic.

Under the protocol's decision rules:
- **randomized depth becomes the default RDT recipe** for A₅ and the language-model phase;
- **the extrapolation claim holds in its declared, limited form**: randomized-depth RDT stays above the fixed-depth Transformer on positions 17–24;
- **CTM-LM leaves the next steps as a candidate for serial tasks.**

## Reading

1. **The recipe turned serial learning from rare into routine.**
   - Escape probability for randomized-depth RDT is 80% (CI 61–92%). Under the old recipe (learned absolute positions, learning rate 3e-4) it was 10% ([reliability study](../reliability_s3_v1/INTERPRETATION.md)).
   - Fixed-depth RDT with RoPE at its own tuned rate never escaped (0/30), although it improved on average.
   - Randomized depth also converges in fewer steps. It reaches 94.9% on positions 1–16 at T = 8 and median correct prefix 15 from T = 8 on. Fixed depth needs T = 16 for 81.5%.
2. **Recurrent depth beats fixed depth again.** Fixed-depth RDT is +15.7 points above the Transformer on positions 1–16 (secondary, uncorrected p = 2 × 10⁻⁶). This replicates the reliability study's +10.1 under a different position scheme and different learning rates.
3. **Extrapolation is real but short.** Mean accuracy by position for randomized-depth RDT at T = 32 is:

   | Position | 16 | 17 | 18 | 19–26 |
   |---|---:|---:|---:|---:|
   | Accuracy | 0.83 | 0.75 | 0.55 | 0.25–0.38 |

   The Transformer is at chance from position 18. The models carry the serial solution one or two positions past the trained length, then degrade, though they stay above chance. More steps do not help: accuracy is flat from T = 16 to T = 64. So this is **not length generalization**. RDT keeps a partial state estimate a little past its training range. Extending it probably needs longer training words or more training depth.
4. **CTM-LM falls further behind under the tuned recipe.** It is 22.7 points below fixed-depth RDT, and numerically below the Transformer too (58.9% against 65.8%; not a declared test). It uses its ticks (19.2% at T = 4, 58.9% at T = 16). But it gets worse past its trained tick count (46.8% at T = 64), and it never escaped. The faithful CTM design does not compete with recurrent depth on serial state tracking at this scale, at its best development learning rate.

## Limits

- **Scheme and learning rate are not separated.** Each RDT cell ran at its own tuned rate: randomized depth at 1e-3, fixed depth at 6e-4. In development, fixed depth at 1e-3 scored 0.873 with 1 of 3 escapes, against 0.986 and 3 of 3 for randomized depth at the same rate. That suggests most of the effect belongs to randomized depth, but only three seeds support it. A learning-rate-matched control (fixed depth at 1e-3 on these seeds) would settle it.
- One task (S₃), one model size and one sampler setting. Evaluation extends only to twice the trained length.
- The escape threshold (0.9 on positions 9–16) is the one declared in earlier studies. The continuous endpoint T1 agrees with it.

## Consequences for the plan

- **Adopted recipe for recurrent depth:**
  - RoPE with no absolute position table;
  - log-normal-Poisson randomized depth (mean about 16);
  - learning rate tuned per cell, which here was well above the earlier default.
- **Next:**
  - optionally, the learning-rate-matched fixed-depth control;
  - A₅ with the adopted recipe, the hard task on which everything has so far been at the floor;
  - then the mini language-model stability gate.
- **CTM-LM:** by the declared rule, it no longer goes forward as a serial-task candidate. Its remaining open question is retrieval, whether it can learn one-hop lookup at all. That decides whether any CTM-LM arm joins the language-model phase.

See [results](RESULTS.md), the [frozen protocol](PLAN_BEFORE_RUNS.md) and the [development probes](../recipe_probes/RECIPE_PROBES.md).
