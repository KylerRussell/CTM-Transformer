# Parity follow-up: tick-scaling length extrapolation

Protocol and rules: `scripts/parity_ticks.py` (docstring).

| Model | Seeds | Extrapolating seeds | Accuracy on 41-80 at 16 / 64 ticks, per seed |
|---|---:|---:|---|
| CTsqS | 10 | 1 | 52/52, 55/99*, 53/53, 77/87, 50/50, 92/92, 50/50, 51/51, 48/48, 52/52 |
| CTsq | 10 | 0 | 49/49, 52/52, 51/51, 79/76, 48/48, 48/48, 48/48, 50/51, 53/53, 50/50 |
| Dans | 10 | 0 | 49/49, 51/50, 52/52, 76/76, 48/48, 48/48, 48/48, 50/50, 48/49, 48/48 |
| CTsq_m32 | 10 | 0 | 48/48, 59/61, 51/51, 49/49, 52/52, 48/48, 53/53, 49/49, 52/52, 52/52 |
| Dans_m32 | 10 | 0 | 53/53, 52/52, 52/52, 48/48, 51/51, 48/48, 49/49, 50/50, 52/52, 53/53 |
| CTsq_cur | 10 | 0 | 48/49, 48/48, 92/92, 52/52, 52/52, 48/48, 48/47, 93/95, 84/84, 51/51 |
| Dans_cur | 10 | 0 | 72/74, 96/96, 52/52, 52/52, 90/91, 49/49, 50/50, 47/47, 50/50, 50/50 |
| CTsqH | 10 | 0 | 53/53, 84/84, 49/49, 82/82, 50/50, 48/48, 52/52, 53/53, 48/48, 81/95 |
| CTsqHS | 10 | 0 | 53/53, 96/96, 96/98, 54/54, 52/52, 52/52, 71/71, 49/49, 48/48, 46/46 |
| DansH | 10 | 0 | 50/50, 52/52, 51/52, 80/80, 52/52, 49/49, 50/50, 53/53, 49/49, 49/49 |
| CTsq_prog | 10 | 0 | 49/49, 48/47, 72/72, 48/48, 48/48, 51/51, 49/49, 51/51, 54/54, 53/53 |
| Dans_prog | 10 | 0 | 49/49, 48/48, 51/51, 52/52, 50/50, 49/49, 48/48, 52/52, 53/53, 51/51 |
| CTsq_mono | 10 | 0 | 53/53, 50/50, 53/53, 48/48, 53/53, 51/51, 48/48, 50/50, 48/48, 48/48 |
| Dans_mono | 10 | 0 | 84/91, 48/48, 49/49, 51/51, 47/47, 49/49, 49/48, 47/47, 51/51, 48/48 |
| CTsq_lent | 10 | 2 | 53/53, 54/91*, 51/51, 52/52, 56/84*, 48/48, 52/57, 48/48, 52/52, 51/51 |
| Dans_lent | 10 | 4 | 61/99*, 51/52, 50/99*, 51/99*, 48/48, 48/48, 52/52, 52/52, 48/48, 54/100* |
| CTsq_w256 | 10 | 0 | 52/52, 48/48, 52/52, 50/50, 52/52, 48/48, 50/50, 49/50, 51/51, 50/49 |
| Dans_w256 | 10 | 0 | 52/52, 48/48, 51/52, 50/50, 76/80, 49/49, 48/48, 50/49, 52/52, 48/48 |
| CTsq_w256lr | 10 | 0 | 48/48, 53/53, 48/48, 52/52, 50/50, 52/52, 48/48, 52/52, 51/51, 48/48 |
| Dans_w256lr | 10 | 0 | 52/52, 52/52, 48/48, 48/48, 49/49, 48/48, 52/52, 49/49, 54/54, 79/82 |

Reliable (at least 3 of 10): CTsqS no, CTsq no, Dans no, CTsq_m32 no, Dans_m32 no, CTsq_cur no, Dans_cur no, CTsqH no, CTsqHS no, DansH no, CTsq_prog no, Dans_prog no, CTsq_mono no, Dans_mono no, CTsq_lent no, Dans_lent yes, CTsq_w256 no, Dans_w256 no, CTsq_w256lr no, Dans_w256lr no
Secondary (128 ticks in place of 64): CTsqS 1, CTsq 0, Dans 0, CTsq_m32 0, Dans_m32 0, CTsq_cur 0, Dans_cur 0, CTsqH 0, CTsqHS 0, DansH 0, CTsq_prog 0, Dans_prog 0, CTsq_mono 0, Dans_mono 0, CTsq_lent 2, Dans_lent 4, CTsq_w256 0, Dans_w256 0, CTsq_w256lr 0, Dans_w256lr 0
- Dans_lent has the property more than CTsqS (4 vs 1)
- Dans_lent has the property more than CTsq (4 vs 0)
- Dans_lent has the property more than Dans (4 vs 0)
- Dans_lent has the property more than CTsq_m32 (4 vs 0)
- Dans_lent has the property more than Dans_m32 (4 vs 0)
- Dans_lent has the property more than CTsq_cur (4 vs 0)
- Dans_lent has the property more than Dans_cur (4 vs 0)
- Dans_lent has the property more than CTsqH (4 vs 0)
- Dans_lent has the property more than CTsqHS (4 vs 0)
- Dans_lent has the property more than DansH (4 vs 0)
- Dans_lent has the property more than CTsq_prog (4 vs 0)
- Dans_lent has the property more than Dans_prog (4 vs 0)
- Dans_lent has the property more than CTsq_mono (4 vs 0)
- Dans_lent has the property more than Dans_mono (4 vs 0)
- Dans_lent has the property more than CTsq_w256 (4 vs 0)
- Dans_lent has the property more than Dans_w256 (4 vs 0)
- Dans_lent has the property more than CTsq_w256lr (4 vs 0)
- Dans_lent has the property more than Dans_w256lr (4 vs 0)
Extra ticks pay (41-80 bits, +10 points from 16 to 128 ticks): CTsqS 2, CTsq 0, Dans 0, CTsq_m32 0, Dans_m32 0, CTsq_cur 0, Dans_cur 0, CTsqH 1, CTsqHS 0, DansH 0, CTsq_prog 0, Dans_prog 0, CTsq_mono 0, Dans_mono 0, CTsq_lent 2, Dans_lent 4, CTsq_w256 0, Dans_w256 0, CTsq_w256lr 0, Dans_w256lr 0
Fitting the trained lengths (1-40 bits at 16 ticks, at least 95%): CTsqS 4, CTsq 5, Dans 3, CTsq_m32 2, Dans_m32 5, CTsq_cur 9, Dans_cur 10, CTsqH 4, CTsqHS 7, DansH 7, CTsq_prog 4, Dans_prog 5, CTsq_mono 2, Dans_mono 4, CTsq_lent 2, Dans_lent 2, CTsq_w256 1, Dans_w256 1, CTsq_w256lr 1, Dans_w256lr 5

## Extrapolating runs: smallest tick count reaching 90% per length band

- CTsqS seed 1: 1-40: 16, 41-60: 32, 61-80: 64, 81-120: 64, 121-160: -, 161-250: -
- CTsq_lent seed 1: 1-40: 16, 41-60: 32, 61-80: -, 81-120: -, 121-160: -, 161-250: -
- CTsq_lent seed 4: 1-40: 16, 41-60: 32, 61-80: -, 81-120: -, 121-160: -, 161-250: -
- Dans_lent seed 0: 1-40: 16, 41-60: 32, 61-80: 32, 81-120: 64, 121-160: -, 161-250: -
- Dans_lent seed 2: 1-40: 16, 41-60: 32, 61-80: 64, 81-120: -, 121-160: -, 161-250: -
- Dans_lent seed 3: 1-40: 16, 41-60: 64, 61-80: 64, 81-120: 128, 121-160: -, 161-250: -
- Dans_lent seed 9: 1-40: 16, 41-60: 32, 61-80: 64, 81-120: 64, 121-160: -, 161-250: -
