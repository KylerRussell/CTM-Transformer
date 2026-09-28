# Interpretation of the S3 reliability study

## Finding under the declared rules

Sixty runs completed and passed the freeze audit: Transformer, RDT and CTM-LM × 20 new seeds, each seed with fresh S₃ data. The evaluation was read once, after the freeze. A run *escapes* if its mean accuracy over positions 9–16 at the trained budget is at least 0.9.

| Cell | Escaped | Escape probability (95% Clopper–Pearson) | Mean accuracy, positions 1–16 | Median correct prefix |
|---|---:|---|---:|---:|
| Transformer | 0/20 | 0% (0–16.8%) | 60.4% | 5 |
| RDT | 2/20 | 10% (1.2–31.7%) | 70.5% | 5.5 |
| CTM-LM | 0/20 | 0% (0–16.8%) | 56.6% | 4 |

Declared paired tests, Holm-corrected over all four:

- **RDT vs Transformer, mean accuracy over positions 1–16:** +10.1 points, p = 0.0073, **Holm p = 0.022**.
- **CTM-LM vs RDT, mean accuracy over positions 1–16:** −13.9 points, p = 0.0003, **Holm p = 0.0013**.
- The escape comparisons (exact McNemar) are not significant: 2 vs 0 and 0 vs 2 discordant pairs, Holm p = 1.0.

## Reading

1. **Recurrent depth helps on average, modestly.** This is the project's first result supported by an exact paired test with multiple-comparison control and 20 seeds: RDT beats the fixed-depth Transformer by about 10 points of accuracy over positions 1–16. The gain comes from partial improvements across seeds, not from frequent full solutions.
2. **Full serial solutions are rare at this budget.** Only 2 of 60 runs escaped. Earlier development studies that showed frequent escapes (for example, the S₃ study's `sync` cell) were lucky draws. This is consistent with the locked confirmation's non-replication.
3. **The faithful CTM-LM uses its ticks but does not benefit from them relative to recurrent depth.** Its median correct prefix rises from 0 at T = 4 to 4 at T = 16, so its computation depends on ticks, unlike the CTM-inspired reference. But on average it is significantly worse than RDT, and it is numerically slightly below the Transformer (not a declared test).

## Consequences

- This study is the **baseline** for the training changes: randomized depth (running now, [protocol](../../RANDDEPTH_S3.md)), then length curriculum and learning rate. The target is to raise escape probability and mean accuracy above these values.
- Any later architecture claim on S₃ should use at least 20 seeds and a continuous metric with exact paired tests, as here.
- For CTM-LM, the open questions are whether its ticks can be made useful: through randomized ticks, longer training closer to the paper's regime, or a curriculum. Otherwise the faithful CTM design does not add value over recurrent depth on serial state tracking at this scale.

See [results](RESULTS.md) and the [frozen protocol](PLAN_BEFORE_RUNS.md).
