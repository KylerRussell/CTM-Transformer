# Interpretation of the S3 word-problem study

## Finding under the declared rules

Thirty-five runs completed and passed the freeze audit: seven cells × five seeds (47, 53, 59, 61, 67), identical data per seed, 10,000 updates at peak LR 0.0003. The primary endpoint is the **correct prefix**: the number of leading positions of a held-out length-32 word whose running product is at least 90% accurate, at the trained step budget.

| Declared decision | Result | Evidence (larger seeds, first/second; median paired difference) |
|---|---|---|
| Recurrence beats fixed depth (`rdt` > `transformer`) | no | 2/2, +0 |
| RDT beats the reference CTM (`rdt` > `ctm`) | no | 2/2, +0 |
| **Sync extends state** (`sync` > `rdt` and `rdt_wide`) | **yes** | vs `rdt` 5/0, **+7**; vs `rdt_wide` 5/0, **+4** |
| History extends state | no | vs `rdt` 3/1, +2; vs `rdt_wide` 2/1, +0 |
| **Both mechanisms extend state** (`sync_rdt` > `rdt` and `rdt_wide`) | **yes** | 4/1, +4 against each |
| Steps are used (T=16 > T=4) | **yes** for `rdt`, `rdt_wide`, `history`, `sync`, `sync_rdt`; **no** for `ctm` | recurrent cells 5/0 (+3 to +7); `ctm` 1/0, +0 |

Correct prefix per seed at the trained budget, with the median:

| Cell | Per seed | Median | Mean accuracy, positions 9–16 |
|---|---|---:|---:|
| `transformer` | 5, 5, 5, 4, 5 | 5 | 33.3% |
| `ctm` | 5, 5, 5, 4, 5 | 5 | 33.7% |
| `rdt` | 4, 3, 5, 8, 8 | 5 | 43.3% |
| `rdt_wide` | 4, 6, 4, 5, 8 | 5 | 36.4% |
| `history` | 4, 5, 7, 5, 14 | 5 | 46.7% |
| **`sync`** | **12, 10, 7, 16, 10** | **10** | **77.6%** |
| `sync_rdt` | 8, 5, 12, 6, 12 | 8 | 59.5% |

## What the result shows

**Synchronization-derived query terms roughly double how far recurrent depth carries serial state, at the same step budget.** In the recurrent-depth scaffold, adding CTM's sparse-decay synchronization of each position's state history as a zero-initialized term on the core attention queries does the following:
- raises the median correct prefix from 5 (RDT and width-matched RDT) to 10;
- is larger at every seed against both controls;
- raises mean accuracy over positions 9–16 from 36–43% to 78%.

The width-matched control (565,900 parameters, more than the sync cell's 550,688) rules out parameter count as the explanation.

**The gain comes through the use of steps.** Median correct prefix by inference budget (T = 4, 8, 16, 32):
- `sync`: 3, 7, 10, 11. It keeps gaining through T = 16 and slightly beyond.
- `rdt`: 2, 5, 5, 5.
- `sync_rdt`: 3, 6, 8, 8.

Per seed, sync's T = 4 → 16 gain ranges from 4 to 12 positions; RDT's ranges from 2 to 5. Synchronization makes additional recurrence steps productive for state tracking, where plain recurrence saturates at about 5 positions.

**Temporal MLPs do not help on their own, and dilute the sync effect when combined.** `history` is not established either way. `sync_rdt` exceeds both RDT controls but sits between `rdt` and `sync` (median 8), consistent with the history mechanism's CTM gate initialization halving the state update. Only one mechanism of CTM's design carries the benefit here.

**The reference CTM does not use its ticks.** Its correct prefix is 5 at every budget from T = 4 to T = 32, identical per seed to the fixed-depth Transformer. Every position can read all earlier input tokens directly, so this is not the retrieval limitation. In the standalone architecture, synchronization drives attention over *static* inputs, and its latent states never exchange information across positions. The mechanism that helps inside the recurrent-depth scaffold does not make the standalone CTM track state.

## What did not replicate or does not hold

- **Plain RDT does not beat the Transformer at the declared margin.** The development probe's RDT (a correct prefix of 12–14 at one seed) was an optimistic draw. Across these seeds plain RDT ranges from 3 to 8, and only two seeds exceed the Transformer. Recurrence alone *uses* its steps (5/5 seeds gain from T = 4 to T = 16), but that is not enough to beat fixed depth reliably at T = 16.
- **No cell extrapolates beyond the trained lengths.** Positions 17–32 sit at chance (15.8–16.8%) for every cell. Neither the mechanisms nor extra inference steps (T = 32) produce length generalization at this scale.

## Limits

- One group (S₃), one width, one learning rate and one training-length range.
- Development data: the evaluation words are held out, but no locked test set was declared, and the design followed development probes on the same task.
- Five seeds, with the replication rule declared in advance. There is no significance test.
- The additive query form was chosen to preserve RDT's retrieval. Why synchronization helps is not identified: temporal structure, pairwise products, learned decay, or simply an extra history-derived signal on the queries.
- **Cost:** sync adds about 17% training time per run (61 vs 52 minutes with three workers per GPU).

## Consequence for the paper plan

This is the first declared positive result for a CTM mechanism. It is specific: **synchronization of recurrent state histories, used to steer attention, makes recurrent depth more effective at serial state tracking.** It is not a result for the CTM architecture as a whole, which fails here, as it did on retrieval. Before it becomes a paper claim, it needs:

1. **Confirmation on locked data:** fresh seeds, a pre-registered test set, and a second setting (width, group or length range).
2. **Mechanism ablations** that identify what in synchronization matters.

See the [confirmation and ablation plan](../../SYNC_CONFIRMATION_PLAN.md), [results](RESULTS.md) and the [frozen protocol](PLAN_BEFORE_RUNS.md).
