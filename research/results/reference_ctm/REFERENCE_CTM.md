# Reference CTM (official code)
Protocol and rules: `scripts/reference_ctm.py`.

## R1, recall learning rate (seeds 0-4)

| lr | Bound | ID (most certain tick) | ID (final tick) |
|---:|---:|---:|---:|
| 0.0001 | 0/5 | 25.6 | 24.7 |
| 0.0003 | 0/5 | 25.3 | 25.2 |
| 0.001 | 0/5 | 24.7 | 23.9 |

Selected lr: 0.0001

## R2, recall test (seeds 5-24)

Reference CTM: 0/20 bound; ID 26.3; OOD 14.0

- Reference CTM 0/20 against CTsqS 15/20: two-sided Fisher p = 0.0000 (differs)
- Reference CTM 0/20 against Dans 2/20: two-sided Fisher p = 0.4872
- Reference CTM 0/20 against T8 19/20: two-sided Fisher p = 0.0000 (differs)
- **Outcome:** the CTM binds less reliably than the CTM Transformer

## R3, other grid tasks (seeds 0-2, sequence accuracy, most certain tick)

| Task | ID | OOD |
|---|---:|---:|
| copy | 11.8 | 0.0 |
| sort | 0.1 | 0.0 |
| parity | 49.1 | 49.0 |
