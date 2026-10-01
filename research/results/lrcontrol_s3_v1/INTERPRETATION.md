# Interpretation of the S3 learning-rate control

## Finding under the declared rules

Thirty fixed-depth RoPE RDT runs at learning rate 1e-3 completed and passed the freeze audit. They used the recipe confirmation's seeds and data, and were paired with that study's randomized-depth runs (also at 1e-3). **Both declared tests are significant after Holm correction:**

| Test | Effect | Holm p |
|---|---|---:|
| C1: mean accuracy over positions 1–16 at T = 16 | randomized depth **+11.7 points** (95.8% against 84.1%) | 5.0 × 10⁻⁷ |
| C2: escape at T = 16 | **24/30 against 1/30** (23 against 0 discordant) | 4.8 × 10⁻⁷ |

Under the declared decision rule, **the recipe confirmation's effect is attributed to randomized depth at a matched learning rate**. It is not an artifact of the higher learning rate.

## Reading

- **The learning rate helps fixed depth only a little.** Raising it from 6e-4 to 1e-3 moved fixed-depth RDT from 81.5% to 84.1% (+2.6 points, uncorrected p = 0.24) and from 0/30 to 1/30 escapes. Its median correct prefix stayed at 8 at every T ≥ 16.
- **Randomized depth changes the solution itself.**
  - Fixed-depth runs at either learning rate plateau at about 70–80% on positions 9–16. Only one of sixty runs crossed 0.9.
  - Randomized-depth runs reach the serial solution at 80% of seeds, and get there in fewer steps: 83% at T = 4 against 49% for fixed depth.
- **Short extrapolation also favours randomized depth at the matched rate:** +13.6 points on positions 17–24 (secondary, uncorrected p = 0.006).

## Consequence

The adopted recurrent-depth recipe stands on a controlled comparison: RoPE, randomized depth, and a tuned learning rate. **Randomized-depth training is the component that makes serial state tracking reliable on S₃.** This matches the development observation on A₅: at width 192, fixed depth stayed stuck where randomized depth learned.

See [results](RESULTS.md) and the [frozen protocol](PLAN_BEFORE_RUNS.md).
