# Recall memory: phase 3 confirmation (seeds 5-14)
Recipe cur; protocol and rules: `scripts/recall_memory.py`.

| Cell | Bound | ID | OOD at 16 | OOD of bound runs | Mann-Whitney p (OOD > Dans) |
|---|---:|---:|---:|---:|---:|
| T8 | 10/10 | 100.0 | 83.0 | 83.0 | 0.000 |
| Dans | 1/10 | 38.2 | 18.4 | 35.3 | - |
| CTsq | 6/10 | 72.5 | 54.3 | 79.3 | 0.010 |
| CTsqS | 7/10 | 79.6 | 62.9 | 82.9 | 0.016 |
| CTfull | 5/10 | 65.8 | 48.2 | 80.1 | 0.099 |

## Rule outcomes

- **CTsq OOD win over Dans:** confirmed (mean 54.3 vs 18.4, higher on 8 of 10)
- **CTfull OOD win over Dans:** not confirmed (mean 48.2 vs 18.4, higher on 7 of 10)
- T8 binds more reliably than Dans: yes (10/10 vs 1/10, p = 0.000)
- CTsq binds more reliably than Dans: yes (6/10 vs 1/10, p = 0.029)
- CTsqS binds more reliably than Dans: yes (7/10 vs 1/10, p = 0.010)
- CTfull binds more reliably than Dans: no (5/10 vs 1/10, p = 0.070)
- **Synchronization helps binding (sync-query cells pooled vs Dans):** yes (18/30 vs 1/10, p = 0.007)
- Dans against T8: T8 beats it (OOD 18.4 vs 83.0; bound 1 vs 10)
- CTsq against T8: T8 beats it (OOD 54.3 vs 83.0; bound 6 vs 10)
- CTsqS against T8: neither (OOD 62.9 vs 83.0; bound 7 vs 10)
- CTfull against T8: T8 beats it (OOD 48.2 vs 83.0; bound 5 vs 10)
