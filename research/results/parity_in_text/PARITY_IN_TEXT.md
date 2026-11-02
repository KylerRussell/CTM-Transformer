# Parity inside text: per-position tick budgets

Protocol and rules: `scripts/parity_in_text.py` (docstring).

| Cell | Seeds | Extrapolating | Fitting 1-40 (>=95% at 16 ticks) | Text loss, seeds 0-1 | 41-80 at 16 / 64 ticks, per seed |
|---|---:|---:|---:|---:|---|
| CTsq_uniform | 10 | 0 | 5 | 3.957 | 99/99, 80/80, 49/49, 52/52, 100/100, 83/87, 52/52, 48/48, 48/48, 100/100 |
| Dans_uniform | 10 | 0 | 3 | 3.954 | 97/97, 48/48, 49/48, 51/51, 50/51, 99/99, 52/52, 48/48, 48/48, 99/99 |
| CTsq_pertoken | 10 | 4 | 3 | 3.955 | 52/93*, 54/100*, 48/48, 50/93*, 53/53, 48/47, 48/48, 52/52, 48/48, 50/100* |
| Dans_pertoken | 10 | 2 | 2 | 3.962 | 54/99*, 57/100*, 48/48, 50/50, 52/52, 49/48, 48/48, 52/52, 48/48, 48/49 |
| CTsq_sequence | 5 | 2 | 0 | 3.948 | 53/99*, 52/98*, 52/52, 52/52, 50/52 |
| Dans_sequence | 5 | 2 | 1 | 3.948 | 54/99*, 51/100*, 48/48, 51/51, 52/53 |

## Rule outcomes

- **CTsq:** pertoken 4 vs uniform 0; text loss 3.9549825433351407 vs 3.9572362319331633; per-position budgets work: yes
- **Dans:** pertoken 2 vs uniform 0; text loss 3.9622654857371913 vs 3.9544498271992614; per-position budgets work: no
- **Synchronization queries matter (CTsq pertoken count exceeds Dans by at least 3):** no
- **Decision:** Stage B may run
