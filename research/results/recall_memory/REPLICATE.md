# Recall memory: phase 5 replication (seeds 15-24)
Protocol and rules: `scripts/recall_memory.py`.

| Cell | Bound | ID | OOD at 16 | OOD of bound runs | Updates to 90% (bound runs) |
|---|---:|---:|---:|---:|---:|
| T8 | 9/10 | 93.2 | 75.0 | 81.5 | 3778 |
| Dans | 1/10 | 44.1 | 24.5 | 79.4 | 5000 |
| CTsqS | 8/10 | 86.1 | 68.4 | 81.3 | 4250 |

## Rule outcomes

- **Primary, CTsqS binds more reliably than Dans:** yes, replicated (8/10 vs 1/10, p = 0.003)
- Secondary, CTsqS OOD win over Dans: yes (mean 68.4 vs 24.5, higher on 9 of 10)
- Reported, seeds 5-24 pooled: CTsqS 15/20 vs Dans 2/20, p = 0.0000
- Reported, T8: 9/10 bound; T8 vs CTsqS p = 0.500, T8 vs Dans p = 0.0005
