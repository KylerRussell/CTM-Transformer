# Recall memory: phase 2 comparison
Recipe cur; protocol and rules: `scripts/recall_memory.py`.

| Cell | ID | OOD at 16 | OOD at 64 | Updates to 90% (per seed) |
|---|---:|---:|---:|---|
| T8 | 86.3 | 71.4 | - | 3000, 4000, -, 3000, 4000 |
| Dans | 58.9 | 40.7 | 40.7 | -, 5000, 5000, -, - |
| CTsq | 72.7 | 54.0 | 54.0 | 5000, -, 5000, -, 4000 |
| CTsqS | 86.2 | 68.2 | 68.1 | 5000, 5000, 4000, 5000, - |
| CTfull | 72.5 | 53.4 | 53.4 | 5000, -, 4000, 5000, - |
| CTnlmA | 59.0 | 40.5 | 40.6 | -, 4000, 4000, -, - |

## Rule outcomes

- **T8 beats Dans:** ood, speed
- **CTsq beats Dans:** ood
- **CTfull beats Dans:** ood
- Dans: OOD 16 -> 64 ticks 40.7 -> 40.7
- CTsq: OOD 16 -> 64 ticks 54.0 -> 54.0
- CTsqS: OOD 16 -> 64 ticks 68.2 -> 68.1
- CTfull: OOD 16 -> 64 ticks 53.4 -> 53.4
- CTnlmA: OOD 16 -> 64 ticks 40.5 -> 40.6
