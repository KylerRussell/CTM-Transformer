# Parity follow-up: tick-scaling length extrapolation

Protocol and rules: `scripts/parity_ticks.py` (docstring).

| Model | Seeds | Extrapolating seeds | Accuracy on 41-80 at 16 / 64 ticks, per seed |
|---|---:|---:|---|
| CTsqS | 10 | 1 | 52/52, 55/99*, 53/53, 77/87, 50/50, 92/92, 50/50, 51/51, 48/48, 52/52 |
| CTsq | 10 | 0 | 49/49, 52/52, 51/51, 79/76, 48/48, 48/48, 48/48, 50/51, 53/53, 50/50 |
| Dans | 10 | 0 | 49/49, 51/50, 52/52, 76/76, 48/48, 48/48, 48/48, 50/50, 48/49, 48/48 |

Reliable (at least 3 of 10): CTsqS no, CTsq no, Dans no

## Extrapolating runs: smallest tick count reaching 90% per length band

- CTsqS seed 1: 1-40: 16, 41-60: 32, 61-80: 64, 81-120: 64, 121-160: -, 161-250: -
