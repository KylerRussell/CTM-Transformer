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
| Dans | 0.002 | 2 | 3.5740 | 3.5093 to 3.6387 |
| CTsq | 0.002 | 2 | 3.5818 | 3.5495 to 3.6141 |
| CTsqS | 0.002 | 2 | 3.5858 | 3.5009 to 3.6708 |
| T8 | 0.002 | 2 | 3.6279 | 3.5053 to 3.7506 |
| T20 | 0.002 | 1 | 3.5624 | nan to nan |
| T8x | - | 0 | - | - |
