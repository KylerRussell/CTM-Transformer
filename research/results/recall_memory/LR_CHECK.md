# Recall memory: phase 4 learning-rate check
Protocol and rules: `scripts/recall_memory.py`.

## Tuning (seeds 0-4)

| Cell | lr 0.0003: bound / OOD | lr 0.001: bound / OOD | lr 0.002: bound / OOD | Best |
|---|---:|---:|---:|---|
| T8 | 4/5 / 67.4 | 4/5 / 71.4 | 3/5 / 46.6 | 0.001 |
| Dans | 1/5 / 31.7 | 2/5 / 40.7 | 1/5 / 31.9 | 0.001 |
| CTsq | 2/5 / 37.1 | 3/5 / 54.0 | 3/5 / 56.7 | 0.002 |
| CTsqS | 2/5 / 36.2 | 4/5 / 68.2 | 2/5 / 40.7 | 0.001 |

Best lr: {'T8': 0.001, 'Dans': 0.001, 'CTsq': 0.002, 'CTsqS': 0.001}

## At tuned learning rates (seeds 5-14)

| Cell | lr | Bound | OOD at 16 |
|---|---:|---:|---:|
| T8 | 0.001 | 10/10 | 83.0 |
| Dans | 0.001 | 1/10 | 18.4 |
| CTsq | 0.002 | 2/10 | 29.8 |
| CTsqS | 0.001 | 7/10 | 62.9 |

## Rule outcomes

- CTsq OOD win over Dans: does not hold (higher on 6 of 10)
- T8 binds more reliably than Dans: yes (p = 0.000)
- CTsq binds more reliably than Dans: no (p = 0.500)
- CTsqS binds more reliably than Dans: yes (p = 0.010)
- Synchronization helps binding (CTsq + CTsqS vs Dans): no (p = 0.062)
- **The synchronization result does not survive tuning.**
