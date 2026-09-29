# S3 recipe confirmation with RoPE

## Protocol declared before new scientific training — 2026-09-29

### Why

The [randomized-depth study](results/randdepth_s3_v1/INTERPRETATION.md) found two things:
- For RDT, randomized depth pointed the right way but was not significant: escape 7/20 against 2/20 and +9.9 points, with Holm p = 0.31–0.47.
- **No S₃ study could measure length extrapolation.** Every model used a learned absolute position table, and rows 17 and later were never trained.

[Development probes](results/recipe_probes/RECIPE_PROBES.md) (development seeds 307–337, disjoint from this study) chose the recipe:
- **RoPE** in every attention layer, with no absolute table. With RoPE, RDT reached the serial solution at 2 of 3 development seeds, against 0 of 1 with the learned table and 0 of 2 without any position encoding. The best seed stayed accurate for a few positions past length 16 (0.87, 0.77, 0.68 at positions 17–19).
- **No length curriculum.** At three paired seeds it moved positions 9–16 by +0.17, 0.00 and −0.10.
- **Per-cell learning rates.** Each cell's learning rate is the one with the best mean accuracy over positions 1–16 at its trained budget, over development seeds 307, 311 and 313. The learning rate mattered more than anything else probed:

  | Cell | Grid | Selected | Mean positions 1–16 at the selected LR |
  |---|---|---:|---:|
  | Transformer | 1e-4, 3e-4, 1e-3, 3e-3 | 1e-3 | 0.642 (all four within 0.54–0.64) |
  | RDT, fixed depth | 3e-4, 6e-4, 1e-3 | 6e-4 | 0.884 |
  | RDT, randomized depth | 2e-4, 3e-4, 6e-4, 1e-3 | 1e-3 | 0.986; 3/3 seeds escaped |
  | CTM-LM | 3e-4, 6e-4, 1e-3 (1e-3 at one seed) | 3e-4 | 0.597 |

  The randomized-depth optimum lies at the edge of its grid. A higher rate could only help the randomized cell, so the tuned comparison is, if anything, conservative for T1 and T2.

### Question

With position encoding that permits extrapolation, each cell at its own tuned learning rate, and on 30 new seeds:
1. does randomized-depth training make RDT more accurate and more reliable than fixed-depth training?
2. does RDT with randomized depth extrapolate past the trained length beyond what a fixed-depth Transformer does?
3. does CTM-LM reach RDT's accuracy at the same training depth?

### Design

- **Cells.** All four use RoPE on queries and keys of every self-attention layer (`ctm_transformer/positions.py`) and no absolute table. Parameters and initialization are otherwise identical to the frozen models.

  | Cell | Model | Training depth | Readout | LR |
  |---|---|---|---|---|
  | `transformer_rope` | 2-layer Transformer | — | final | 1e-3 |
  | `rdt_rope_fixed` | RDT | T = 16 | confidence | 6e-4 |
  | `rdt_rope_rand` | RDT | log-normal Poisson (mean 15, σ 0.5, cap 48; expected about 16), as in the randomized-depth study | confidence | 1e-3 |
  | `ctm_lm_rope` | CTM-LM with CTM's loss (RoPE also in its causal backbone) | T = 16 | confidence | 3e-4 |

- **Seeds and data.** 30 new seeds: 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509 and 521. Each seed has fresh S₃ data (`research/data/recipe_s3_v1`):
  - 320,000 training words of lengths 1–16;
  - 512 validation words;
  - 1,024 held-out evaluation words of length 32.

  All four cells share a seed's data.
- **Training.** 10,000 updates, batch 32, warmup 100, cosine schedule, runner v5 without a presentation-order sampler (identical to runner v4). The final checkpoint is the primary endpoint.
- **Evaluation.** Read once, after the freeze audit. Recurrent cells are evaluated at T = 4, 8, 16, 32, 48 and 64; the Transformer at T = 1.

### Endpoints and tests (fixed now)

- **Escape:** mean accuracy over positions 9–16 is at least 0.9 at the trained budget (T = 16; the Transformer at T = 1).
- **Extrapolation:** mean accuracy over positions 17–24 of the length-32 words. Recurrent cells are read at T = 32 (T ≥ the evaluation length), the Transformer at T = 1. Positions 17–24 are where the development probes showed extrapolation; positions 25–32 are reported as secondary.

**Declared paired tests, Holm-corrected together:**

| Test | Comparison | Metric | Test statistic |
|---|---|---|---|
| T1 | `rdt_rope_rand` vs `rdt_rope_fixed` | mean accuracy, positions 1–16, T = 16 | two-sided paired sign-flip |
| T2 | `rdt_rope_rand` vs `rdt_rope_fixed` | escape at T = 16 | two-sided exact McNemar |
| T3 | `rdt_rope_rand` vs `transformer_rope` | extrapolation, positions 17–24 | two-sided paired sign-flip |
| T4 | `ctm_lm_rope` vs `rdt_rope_fixed` | mean accuracy, positions 1–16, T = 16 | two-sided paired sign-flip |

With 30 pairs, exact enumeration of 2³⁰ sign assignments is impractical. Sign-flip p-values therefore use **2,000,000 seeded Monte Carlo sign assignments** (seed 20260929) with the +1 correction. The tests check this against exact enumeration at n = 14.

**Secondary** (reported, not corrected):
- `rdt_rope_fixed` against `transformer_rope` on positions 1–16 and on positions 17–24;
- escape probabilities with Clopper–Pearson intervals;
- correct prefix at every T;
- accuracy by position and by T;
- positions 25–32;
- training time.

**Decision rules.**
- Randomized depth becomes the default RDT recipe for the next steps (A₅, and the language-model phase) if T1 or T2 is significant after Holm, in its favour. Otherwise it stays an option, not a default.
- An extrapolation claim requires T3 to be significant after Holm. That claim is limited to "RDT with randomized depth stays above the Transformer on positions 17–24". It is not a claim of length generalization.
- T4 decides whether CTM-LM stays in the next steps as a candidate on serial tasks.

### Verification and execution

Before freezing, these checks must pass on both GPUs:
- the RoPE checks: exact reduction to NoPE at position 0, causality, relative-position invariance, unchanged parameters and initialization, and training;
- the curriculum and runner-v5 checks, including exact parity with runner v4;
- the depth-sampling, CTM-LM and S₃ checks;
- the study wiring and statistics checks.

A dry run of the full pipeline (prepare, train, freeze, evaluate and summarize) on a miniature suite in scratch space must also complete.

The freeze audit checks each run's per-update depth sequence (sampled or fixed), its learning rate, the absence of the position table, the factory name and the presentation order. The study then runs under the restart-safe supervisor with autostart. There are 120 runs, estimated at about 16 hours on six worker slots. Failures are recorded and never retried.

### Interpretation limits

- One task (S₃), one model size and one sampler setting.
- Learning rates were chosen on three development seeds per cell. T1 and T2 compare the two RDT training schemes, each at its own tuned learning rate; they do not separate the scheme from its interaction with the learning rate.
- The evaluation length is 32, so extrapolation is measured only up to twice the trained length.
- RoPE makes the models differ from the earlier studies, so this study's numbers are not directly comparable with the reliability or randomized-depth studies.
