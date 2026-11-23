# Recall memory: phase 7 controls (seeds 5-24)
Protocol and rules: `scripts/recall_memory.py`.

| Cell | Eval ticks | Bound | ID | Updates to 90% (bound runs) |
|---|---:|---:|---:|---:|
| T8 | - | 19/20 | 96.6 | 3789 |
| Dans | 16 | 2/20 | 41.1 | 7000 |
| CTsqS | 16 | 15/20 | 82.9 | 4200 |
| DansS | 16 | 1/20 | 36.6 | 5000 |
| DansT4 | 4 | 5/20 | 48.9 | 4600 |
| CTsq1S | 16 | 3/20 | 42.0 | 4333 |

## Rule outcomes (one-sided Fisher; Holm-adjusted over the primary tests run so far)

- **S1: CTsqS binds more reliably than DansS:** yes (15/20 vs 1/20, p = 0.0000, Holm 0.0000; log-rank z = 4.63, p = 0.0000)
- **B1: DansT4 binds more reliably than Dans:** no (5/20 vs 2/20, p = 0.2037, Holm 0.2037; log-rank z = 1.28, p = 0.2015)
- **S2: CTsqS binds more reliably than CTsq1S:** yes (15/20 vs 3/20, p = 0.0002, Holm 0.0003; log-rank z = 3.86, p = 0.0001)
- Reported: DansS against Dans: 1/20 vs 2/20, p = 0.8846
- Reported: DansT4 ID at 16 ticks, mean 48.8
