# Amendment 1 — 2026-09-27, before any further training

## What happened

The first attempt (22:37 to 03:09 UTC) stopped when **all five A5 (`state`) runs diverged**. Gradient norms became non-finite, and the runner's `clip_grad_norm_(error_if_nonfinite=True)` raised an error. The last completed updates were:

| Seed | Last completed update | Largest gradient norm, last 50 updates |
|---:|---:|---:|
| 47 | 421 | 57.96 |
| 53 | 459 | 29.30 |
| 59 | 220 | 1.69 |
| 61 | 197 | 2.63 |
| 67 | 257 | 7.25 |

Each worker stops its queue at its first failure. Six declared A1–A3 cells queued after an A5 cell therefore never started: `shuffled` at seeds 59, 61 and 67, `nodecay` at 61 and 67, and `current` at 67. Fourteen A1–A4 runs completed normally.

## Declared handling

- **A5 is recorded as "diverged at every seed" and is never retried.** The failure records are kept unchanged in `recorded_failures/`, and the partial run directories stay in `research/runs/sync_ablation_v1/state/`. A5 is removed from the run queues, the freeze audit and the evaluation, and the summary reports it as diverged. No learning rate, clipping or initialization change is applied, so no favorable replacement is attempted.
- **The six unrun A1–A3 cells run unchanged:** the same frozen configs, worker, factory, data and seeds. Running them completes the declared design; it does not replace any result.
- The declared classification rules for A1–A4 are unchanged.

## Interpretation recorded now

In the `sync` cell, synchronization enters attention queries and is normalized by the softmax. In A5, the same term is a decayed sum of **products of state values**, added back into the state. That creates a positive feedback loop: a larger state gives larger products, which give a larger state. It diverges early at every seed. So *where* synchronization enters is essential at least for stability: as a query term it is stable and helpful, and as an additive state term, without further normalization, it is unstable. Whether a normalized state placement would help is not tested here.

## Files changed by this amendment

- `registry.json`: A5 entries moved to `diverged`, and queues and evaluation queues updated. The original is kept as `registry.original.json`.
- `scripts/summarize_sync_ablation.py`: reports diverged ablations.
- `pre_run_source.json`: re-hashed for the files above and this document. The original is kept as `pre_run_source.original.json`, and the pre-run archive is unchanged.

The original freeze is commit `dcb6f53`. This amendment is committed before the remaining cells start.
