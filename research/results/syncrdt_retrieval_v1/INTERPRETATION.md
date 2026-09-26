# Interpretation of the Sync-RDT retrieval sample-efficiency study

## Finding under the declared rules

Thirty runs completed and passed the freeze audit: six cells × five new seeds (71, 73, 79, 83, 89), identical data per seed, 10,000 updates at peak LR 0.0003. **None of the four declared mechanism effects was established** at the 90% position-1 threshold:

| Declared decision | Result | Evidence (earlier seeds, first/second; median paired difference) |
|---|---|---|
| Sync speeds retrieval (over `rdt` *and* `rdt_wide`) | **no** | vs `rdt`: 1/2, +0; vs `rdt_wide`: 1/4, −3,500 (favors `rdt_wide`) |
| History slows retrieval | no | vs `rdt`: 0/3, −1,250 |
| History speeds retrieval | no | as above |
| Gate initialization explains a history slowdown | no | no slowdown established; `history_lowgate` vs `history`: 3/0, +1,250 |

The development probe (round 12), in which sync learned fastest, did not replicate: one seed was not representative.

## What the runs show descriptively

These patterns do not meet a declared rule, so they are observations, not claims.

**Reliability varies more than speed.** Runs either escaped the plateau and reached near 100%, or failed to escape within the budget:

| Cell | Parameters | Seeds reaching 90% | Seeds reaching 50% | Final held-out position 1 (per seed) |
|---|---:|---:|---:|---|
| `rdt` | 525,984 | 3/5 | 3/5 | 100, 100, 26.6, 100, 19.1% |
| `rdt_wide` | 565,900 | **5/5** | **5/5** | 100% at every seed |
| `history` | 541,536 | 2/5 | 4/5 | 26.4, 100, 87.3, 100, 48.2% |
| `history_lowgate` | 541,536 | 3/5 | 3/5 | 100, 100, 33.6, 100, 22.7% |
| `sync` | 550,688 | 2/5 | 4/5 | 41.8, 100, 59.0, 100, 46.3% |
| `sync_rdt` | 566,240 | 3/5 | **5/5** | 69.5, 100, 99.8, 100, 50.0% |

- **The width-matched RDT was the only cell to learn at every seed** (median 5,000 updates to 90%). It was built as a parameter control for the largest mechanism cell, and it beat the mechanism cells. Its advantage over the base RDT (3/2 seeds, median +2,250) does not meet the rule. So, at this scale, a 4% increase in width has at least as much practical effect on retrieval learning as either CTM mechanism. Whether width itself or incidental differences between initializations drives this is not identified.
- **`history_lowgate` behaves like `rdt`:** the same seeds fail, with nearly the same timing and final accuracy. With the gate near zero, the temporal MLP contributes little, so this is expected.
- **The mechanism cells more often reach 50% without reaching 90%.** `sync_rdt` reached 50% at every seed, while `rdt` reached it at 3. This suggests partial learning where the base model shows none. It is not what the primary endpoint measures, and it is not established.
- **Seeds 79 and 89 were hard for almost every cell.** Seed-level difficulty (data order and initialization) is a larger source of variation than architecture in this study.
- **Cost:** the mechanisms add 20–39% training time per run (52 minutes for `rdt`, 62–63 for single mechanisms, 73 for both), measured with three workers sharing each GPU.

## What this does and does not answer

This is a bounded negative result. In the recurrent-depth scaffold, at this scale and learning rate, neither CTM's neuron-level temporal MLPs nor its synchronization-derived query terms speed up or reliably stabilize learning of in-context retrieval.

It does not test the capability CTM's mechanisms are meant to provide: extended, multi-step computation. One-hop retrieval needs one composition of attention heads, not iteration. It also does not show that the mechanisms cannot help under other initializations, gate settings, learning rates or scales. The additive query form was chosen to preserve RDT's retrieval, and a replacement form was not tested.

## Consequence for the paper plan

- Report this study as a controlled negative result for the mechanisms on retrieval sample efficiency. Include per-seed escape failures and costs, and record that the parameter control outperformed the mechanism cells.
- **Treat reliability as a first-class concern in the next study.** At this model size, whether a run escapes the plateau depends strongly on seed. Future comparisons should use at least five seeds and report escape counts, and they should consider the width-matched scaffold as the baseline.
- **Next: the serial-depth task.** It should need iteration and should not depend on first learning in-context retrieval, so the mechanisms are tested where they are supposed to matter. See [the draft design](../../SERIAL_DEPTH_DESIGN.md).

See [results](RESULTS.md), [frozen protocol](PLAN_BEFORE_RUNS.md), `summary.json` and `checkpoints.json`.
