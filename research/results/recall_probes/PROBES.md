# Probes of the synchronization binding result

Protocol and rules: `scripts/recall_probes.py`.

## Reproduction

- CTsqS: 15/20 bound on retraining; outcome as originally in 20 of 20 seeds
- Dans: 2/20 bound on retraining; outcome as originally in 20 of 20 seeds

## P1, use

- Bound CTsqS runs (15): ID 100.0 intact, 99.2 without the synchronization query (mean drop 0.7; at least 10 points in 0%)
- **Outcome:** the bound solution does not need it

## P2, precursor (update 1,000)

- Retrieval score: CTsqS 0.015, Dans 0.009 (one-sided Mann-Whitney p = 0.0042)
- **Outcome:** synchronization makes retrieval attention appear before the behaviour
- Within CTsqS, AUROC of the update-1,000 score for later binding: 0.32

## Retrieval score by snapshot (mean; bound / unbound runs)

| Update | CTsqS bound | CTsqS unbound | Dans bound | Dans unbound |
|---:|---:|---:|---:|---:|
| 0 | 0.000 (15) | 0.000 (5) | 0.000 (2) | 0.000 (18) |
| 250 | 0.006 (15) | 0.006 (5) | 0.002 (2) | 0.004 (18) |
| 500 | 0.010 (15) | 0.014 (5) | 0.003 (2) | 0.006 (18) |
| 750 | 0.012 (15) | 0.016 (5) | 0.008 (2) | 0.009 (18) |
| 1000 | 0.015 (15) | 0.016 (5) | 0.005 (2) | 0.010 (18) |
| 1250 | 0.037 (15) | 0.014 (5) | 0.008 (2) | 0.010 (18) |
| 1500 | 0.074 (15) | 0.017 (5) | 0.028 (2) | 0.011 (18) |
| 1750 | 0.132 (15) | 0.020 (5) | 0.052 (2) | 0.014 (18) |
| 2000 | 0.199 (15) | 0.020 (5) | 0.113 (2) | 0.016 (18) |
| 2250 | 0.283 (15) | 0.022 (5) | 0.160 (2) | 0.017 (18) |
| 2500 | 0.357 (15) | 0.020 (5) | 0.212 (2) | 0.017 (18) |
| 2750 | 0.406 (15) | 0.020 (5) | 0.264 (2) | 0.017 (18) |
| 3000 | 0.475 (15) | 0.019 (5) | 0.313 (2) | 0.017 (18) |
| 4000 | 0.567 (15) | 0.019 (5) | 0.383 (2) | 0.016 (18) |
| 5000 | 0.569 (15) | 0.018 (5) | 0.424 (2) | 0.016 (18) |
| 6000 | 0.546 (15) | 0.019 (5) | 0.470 (2) | 0.017 (18) |
| 7000 | 0.534 (15) | 0.018 (5) | 0.567 (2) | 0.022 (18) |
| 8000 | 0.523 (15) | 0.018 (5) | 0.644 (2) | 0.027 (18) |
| 9000 | 0.514 (15) | 0.018 (5) | 0.678 (2) | 0.029 (18) |
| 10000 | 0.514 (15) | 0.018 (5) | 0.687 (2) | 0.031 (18) |

## Final weights, every cell

| Cell | Bound runs: retrieval | Unbound runs: retrieval | Key retrieval (bound) |
|---|---:|---:|---:|
| CTsqS | 0.514 (15) | 0.018 (5) | 0.069 |
| Dans | 0.687 (2) | 0.031 (18) | 0.255 |
| DansS | 0.920 (1) | 0.014 (19) | 0.220 |
| CTsq1S | 0.864 (3) | 0.019 (17) | 0.167 |
| DansT4 | 0.456 (5) | 0.018 (15) | 0.109 |
- CTsqS, update 1000: logit advantage at the retrieval head, ordinary 0.09, synchronization 0.07; retrieval 0.014, without the synchronization query 0.011
- CTsqS, update 10000: logit advantage at the retrieval head, ordinary 22.23, synchronization 14.10; retrieval 0.507, without the synchronization query 0.469
- CTsq1S, update 10000: logit advantage at the retrieval head, ordinary 4.92, synchronization 5.13; retrieval 0.146, without the synchronization query 0.121
