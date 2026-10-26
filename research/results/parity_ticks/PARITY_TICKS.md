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

Reliable (at least 3 of 10): CTsqS no, CTsq no, Dans no, CTsq_m32 no, Dans_m32 no, CTsq_cur no, Dans_cur no
Secondary (128 ticks in place of 64): CTsqS 1, CTsq 0, Dans 0, CTsq_m32 0, Dans_m32 0, CTsq_cur 0, Dans_cur 0

## Extrapolating runs: smallest tick count reaching 90% per length band

- CTsqS seed 1: 1-40: 16, 41-60: 32, 61-80: 64, 81-120: 64, 121-160: -, 161-250: -
