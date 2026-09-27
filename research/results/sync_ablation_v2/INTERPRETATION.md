# Interpretation of ablation A7 (self-pairs only)

## Finding under the declared rule

A7 replaces the `sync` cell's 128 random channel pairs with self-pairs (i, i), so every feature is one channel's decayed energy and no feature couples two channels. It is classified **partial**:
- `sync` is not larger at the declared margin (3 of 5 seeds, median +1);
- A7 does not exceed both RDT controls (vs `rdt` 3 of 5, median +1; vs `rdt_wide` 3 of 5, median +2).

| Cell | Correct prefix per seed (47, 53, 59, 61, 67) | Median | Mean accuracy, positions 9–16 |
|---|---|---:|---:|
| `sync` | 12, 10, 7, 16, 10 | 10 | 77.6% |
| **`self`** | **15, 3, 6, 5, 11** | **6** | **54.4%** |
| `rdt` | 4, 3, 5, 8, 8 | 5 | 43.3% |
| `rdt_wide` | 4, 6, 4, 5, 8 | 5 | 36.4% |

## Reading

Per-channel second-order features (squared magnitudes) are enough **at some seeds**: 15 and 11 at seeds 47 and 67, matching `sync`. At the other three seeds they stay near plain RDT. Cross-channel products give the benefit **reliably** (every seed at least 7, median 10); per-channel energy gives it only sometimes. Five seeds cannot separate "coupling is necessary" from "coupling makes the benefit more reliable to learn". The declared outcome is partial, and the claim below is worded accordingly.

## Consequence

The mechanism claim, from A1–A5 and A7, is: **second-order features of the current recurrent state in the attention queries extend serial state tracking. Cross-channel products are the form that works reliably; per-channel energy is partial.** The locked confirmation will therefore test `sync` (cross-channel, with history) and `current` (cross-channel, current state only). It does not include `self`.

See [results](RESULTS.md) and the [frozen protocol](PLAN_BEFORE_RUNS.md).
