# Language-model revision
Protocol and rules: `scripts/lm_revision.py`. Held-out loss, 1,024 windows, 16 ticks (Transformers: one pass).

## Phase A, learning-rate grid (seed 1234)

| Arm | 0.001 | 0.0014 | 0.002 | 0.0028 | 0.004 | Selected |
|---|---:|---:|---:|---:|---:|---|
| Dans | 3.6595 | 3.5974 | 3.5689 | 3.5854 | 3.6567 | 0.002 |
| CTsq | 3.6658 | 3.6026 | 3.5844 | 3.6088 | 3.7143 | 0.002 |
| CTsqS | - | 3.6062 | 3.5791 | 3.6034 | - | 0.002 |
| T8 | - | 3.6553 | 3.6183 | 3.6360 | 3.6735 | 0.002 |
| T20 | 3.6493 | 3.5886 | 3.5624 | 3.5818 | 3.7180 | 0.002 |

## Phase B and C, seeds

| Arm | lr | Seeds | Mean | 95% interval |
|---|---:|---:|---:|---|
| Dans | 0.002 | 5 | 3.5747 | 3.5690 to 3.5803 |
| CTsq | 0.002 | 5 | 3.5809 | 3.5745 to 3.5874 |
| CTsqS | 0.002 | 5 | 3.5837 | 3.5763 to 3.5912 |
| T8 | 0.002 | 5 | 3.6302 | 3.6202 to 3.6402 |
| T20 | 0.002 | 5 | 3.5724 | 3.5628 to 3.5821 |
| T8x | 0.002 | 3 | 3.1669 | 3.1524 to 3.1815 |

## Rule outcomes

- **CTsq against Dans:** worse than Dans (difference +0.0062, 90% CI +0.0005 to +0.0120; Holm p worse 0.039, p beats-by-0.02 1.000)
- **CTsqS against Dans:** worse than Dans (difference +0.0091, 90% CI +0.0027 to +0.0154; Holm p worse 0.030, p beats-by-0.02 1.000)
- **Dans against T8:** Dans beats it (difference -0.0555, 95% CI -0.0655 to -0.0455, p = 0.000)
- **Dans against T8x:** Dans does not beat it by 0.02 (difference +0.4078, 95% CI +0.3961 to +0.4194, p = 1.000)
- Reported, Dans at 4 ticks (3.5859) against T20 (3.5724), equal compute: difference +0.0135, 95% CI +0.0038 to +0.0232
