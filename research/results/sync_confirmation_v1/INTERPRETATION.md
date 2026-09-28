# Interpretation of the locked confirmation

## Finding under the declared rules

**Neither primary claim is confirmed.**
- **C1 (`sync` exceeds both RDT controls): no.** Against `rdt`: larger at 2 of 7 seeds, median +0. Against `rdt_wide`: 2 of 7, median −1.
- **C2 (`current` exceeds both): no.** Against `rdt`: 3 of 7, median +0. Against `rdt_wide`: 3 of 7, median +0.

This study used 35 runs: five cells × seven never-used seeds (97, 101, 103, 107, 109, 113, 127), on a fresh S₃ suite. The locked test was evaluated once, after the checkpoint freeze. Median correct prefix at T = 16:

| Transformer | RDT | RDT wide | sync | current |
|---:|---:|---:|---:|---:|
| 5 | 5 | 5 | 5 | 6 |

The only declared effect that holds is that **every recurrent cell uses its steps**: T = 16 beats T = 4 at 7 of 7 seeds for `rdt`, `rdt_wide`, `sync` and `current`.

The configurations are identical to the development S₃ study apart from output paths. The same factories and recipe were used, so the non-replication is not a pipeline difference. Learned synchronization decay again stays near zero (−0.006 to −0.048).

## Why the development result did not replicate

Outcomes are **bimodal by seed**. A run either *escapes* and solves positions 9–16 (above 90% accuracy, correct prefix 13–16), or stays near the fixed-depth profile (about 30%, prefix 4–7):

| Cell | Seeds escaping (positions 9–16 > 0.9), development S₃ | Seeds escaping, confirmation |
|---|---:|---:|
| `rdt` | 0 of 5 | 1 of 7 |
| `rdt_wide` | 0 of 5 | 1 of 7 |
| `sync` | 1 of 5 (plus three at 0.74–0.85) | 2 of 7 |
| `current` (A3) | 2 of 5 (plus 0.69 and 0.81) | 1 of 7 |

In the development study, `sync` happened to escape, or nearly escape, at 4 of 5 seeds while RDT mostly did not. That produced a paired 5-of-5 pattern that met the declared rule. With a per-run escape probability that seems to be well below one half for *every* recurrent cell, 5–7 seeds cannot reliably order the cells. The development "effect", and the ablation classifications that were measured against it, were most plausibly a chance fluctuation, not a mechanism. The [deep-research review](../../DEEP_RESEARCH_REPORT.md) had warned about exactly this: a 4-of-5 sign rule gives about a 19% false-positive rate under the null, before multiple comparisons.

## Consequences

- **The claim that second-order state features in the queries extend serial state tracking is withdrawn.** The S₃ study and the A1–A7 ablation interpretations remain as records of development results, but they are superseded here. Their mechanism conclusions are not supported.
- **Robust findings so far:**
  - recurrent cells use extra steps;
  - no cell generalizes beyond the trained length;
  - the CTM-inspired reference does not use its ticks on S₃;
  - the retrieval findings (static-K/V and shared-start-state limits; the retrieval study's null result).
- **The real bottleneck is the reliability of escaping to the serial solution, not a mechanism gap.** Every recurrent cell *can* solve positions up to 16 at some seeds. Future comparisons should estimate **escape probability** directly, with many more seeds (for example, 20 or more per cell) and a continuous metric. They should also try training changes aimed at reliable escape (randomized depth, length curriculum, per-cell learning rates) before comparing architectures.
- **The adopted pre-LM plan is re-prioritized** (see `research/CTM_LM_DESIGN.md`). The synchronization confound sweep loses its premise, because there is no confirmed effect to de-confound. Randomized depth, curriculum and learning rate, as means to make serial learning reliable, move first. They are followed by CTM-LM checks and A₅, with escape-probability endpoints.

See [results](RESULTS.md) and the [frozen protocol](PLAN_BEFORE_RUNS.md).
