# Sync-RDT mechanism ablations (part 1 of the confirmation plan)

## Protocol declared before new scientific training — 2026-09-26

### Question

In the [S₃ study](results/group_s3_v1/INTERPRETATION.md), the `sync` cell raised the median correct running-product prefix from 5 (both RDT controls) to 10, at every seed. The cell adds CTM's sparse-decay synchronization of each position's 8-step recurrent-state history as a zero-initialized term on the core attention queries. What property of synchronization carries the effect? This is part 1 of the [confirmation and ablation plan](SYNC_CONFIRMATION_PLAN.md), with the review's choices: ablations first, and A1–A5 first.

### Observation recorded before training

In the frozen `sync` checkpoints, the learned decay rates stay close to zero (|r| < 0.14, mostly slightly negative). An exponential weight of exp(−r·τ) with r ≈ 0 makes synchronization a nearly flat, order-free sum of pairwise products over the last 8 states. So **A1 and A2 are expected to preserve the effect**. They remain declared checks, because they would detect any order dependence this reading misses. A3–A5 are the informative ablations.

### Ablations

Each changes one property of `sync` (`ctm_transformer/sync_ablations.py`). The frozen `sync_rdt.py` is unchanged, and an `identity` kind is tested to reproduce `sync` exactly.

| Ablation | Change | Parameters |
|---|---|---:|
| A1 `shuffled` | history randomly permuted in time before synchronization, independently per position and step | 550,688 |
| A2 `nodecay` | decay rates frozen at r = 0 | 550,688 |
| A3 `current` | synchronization of the newest state only (history length 1) | 550,688 |
| A4 `linear` | a linear projection of the flattened 8-step history replaces the pairwise products (same 128 features, same placement) | 648,992 |
| A5 `state` | the synchronization term is added to the recurrent state after the core, not to the queries | 538,400 |

### Data, seeds and controls

The data are the S₃ study's exact data and seeds (47, 53, 59, 61, 67): `research/data/group_s3_v1`, with the hashes verified. The recipe is the same as the `sync` cell (LR 0.0003, 10,000 updates, T = 16, confidence readout, runner v3).

The controls are **not retrained**. They are the frozen `sync`, `rdt` and `rdt_wide` runs and evaluations from `group_s3_v1`; their evaluation files and checkpoint freeze are verified by hash.

### Endpoints and declared classification

The primary endpoint is the correct prefix at T = 16 on the length-32 held-out words: leading positions with at least 90% accuracy. The rule is the S₃ study's: one cell exceeds another if it is larger in at least 4 of 5 seeds and the median paired difference is at least 2 positions. Each ablation is classified as:
- **removes the effect** if `sync` exceeds it;
- **preserves the effect** if it exceeds both `rdt` and `rdt_wide`;
- **partial** otherwise.

Secondary: whether it uses its steps (T = 16 against T = 4, same rule), mean accuracy at positions 9–16, step scaling at T = 4, 8, 16 and 32, and training time.

### Verification and execution

Before freezing:
- the ablation checks: exact identity parity with `sync`, declared parameter changes, causality, reproducible evaluation including the shuffles, gradients, round trips, the shuffle's dependence on decay, and the classification rule;
- the Sync-RDT checks;
- the S₃ study checks;
- the archived-trainer replay, on both GPUs.

The freeze audit uses the S₃ study's audit, with this study's factory and worker. `research/launch_sync_ablation.sh` starts the restart-safe supervisor, with autostart after restarts. Twenty-five runs are spread over six worker slots; the estimate is about 4.5 hours.

### Interpretation limits

- Development data that the S₃ study already used, so these are not new confirmatory evidence.
- The A4 and A5 parameter counts differ from `sync` (by +98,304 and −12,288).
- Five seeds.
- A6–A8 (replacing the queries, self-pairs, and a low-gate combination) are declared later, if needed.

## Outcome — 2026-09-27, 06:52 UTC

Twenty A1–A4 runs completed and passed the freeze audit. A5 diverged at every seed ([amendment 1](results/sync_ablation_v1/AMENDMENT_1.md)).
- **A2 (no decay) and A3 (current state only) preserve the effect** (median 11 each).
- **A4 (linear history features) removes it** (median 3, below RDT).
- A1 (shuffled history) is classified *removes* (sync is larger in 4 of 5 seeds, median +3), although it still exceeds both RDT controls.

The benefit comes from **pairwise products of the current recurrent state feeding the attention queries**, not from temporal history or decay. See [results](results/sync_ablation_v1/RESULTS.md) and [interpretation](results/sync_ablation_v1/INTERPRETATION.md).

**Superseded (2026-09-28):** the [locked confirmation](SYNC_CONFIRMATION.md#outcome--2026-09-28-0551-utc) did not replicate the synchronization effect on fresh seeds and data. The mechanism conclusions drawn from this development study are withdrawn. See its [interpretation](results/sync_confirmation_v1/INTERPRETATION.md).
