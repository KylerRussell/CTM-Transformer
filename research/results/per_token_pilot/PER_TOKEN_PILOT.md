# Per-token ticks: language-model pilot (100M tokens)

Protocol and rules: `scripts/per_token_pilot.py` (docstring). Loss in nats; two-seed means (per seed in brackets).

| Measure | Dans_sampler | CTsq_sampler | Dans_pertoken | CTsq_pertoken | Dans_entropy | CTsq_entropy |
|---|---:|---:|---:|---:|---:|---:|
| loss at 1 | 4.794 (4.782/4.806) | 4.837 (4.818/4.857) | 4.813 (4.804/4.823) | 10.635 (9.865/11.406) | 4.813 (4.799/4.826) | 4.844 (4.829/4.859) |
| loss at 2 | 4.764 (4.751/4.778) | 4.768 (4.740/4.796) | 4.779 (4.774/4.785) | 13.413 (12.792/14.034) | 4.787 (4.777/4.797) | 4.775 (4.762/4.788) |
| loss at 4 | 4.761 (4.747/4.775) | 4.757 (4.727/4.788) | 4.776 (4.771/4.781) | 11.187 (10.666/11.708) | 4.784 (4.775/4.794) | 4.764 (4.750/4.778) |
| loss at 8 | 4.761 (4.747/4.775) | 4.756 (4.725/4.786) | 4.775 (4.771/4.780) | 8.260 (7.970/8.551) | 4.784 (4.775/4.794) | 4.762 (4.748/4.776) |
| loss at 16 | 4.761 (4.747/4.775) | 4.755 (4.725/4.786) | 4.775 (4.770/4.780) | 6.860 (6.834/6.885) | 4.784 (4.775/4.794) | 4.762 (4.748/4.776) |
| loss at 32 | 4.761 (4.747/4.775) | 4.755 (4.725/4.786) | 4.775 (4.770/4.780) | 11.157 (10.556/11.758) | 4.784 (4.775/4.794) | 4.762 (4.748/4.776) |
| loss at 64 | 4.761 (4.747/4.775) | 4.756 (4.725/4.786) | 4.775 (4.771/4.780) | 9.406 (7.096/11.715) | 4.784 (4.775/4.794) | 4.762 (4.748/4.776) |
| loss at 96 | 4.761 (4.747/4.775) | 4.756 (4.725/4.786) | 4.775 (4.770/4.780) | 7.772 (6.713/8.832) | 4.784 (4.775/4.794) | 4.762 (4.748/4.776) |
| exit, mean 4 | 4.761 (4.747/4.775) | 4.757 (4.726/4.787) | 4.776 (4.771/4.780) | 11.611 (10.745/12.477) | 4.784 (4.775/4.794) | 4.763 (4.749/4.777) |
| exit, mean 8 | inf (inf/inf) | inf (inf/inf) | inf (inf/inf) | 11.147 (9.840/12.454) | inf (inf/inf) | inf (inf/inf) |

## Rule outcomes

- **Dans_pertoken:** ticks pay (per-token arm): no (sampler arm: no); 16-tick cost +0.014 (fails); exit gain at mean 4 / 8 ticks -0.014 / +nan (better use of compute: no); per-token ticks work: no
- **CTsq_pertoken:** ticks pay (per-token arm): no (sampler arm: no); 16-tick cost +2.104 (fails); exit gain at mean 4 / 8 ticks -6.855 / +inf (better use of compute: no); per-token ticks work: no
- **Dans_entropy:** ticks pay (per-token arm): no (sampler arm: no); 16-tick cost +0.023 (fails); exit gain at mean 4 / 8 ticks -0.023 / +nan (better use of compute: no); per-token ticks work: no
- **CTsq_entropy:** ticks pay (per-token arm): no (sampler arm: no); 16-tick cost +0.007 (ok); exit gain at mean 4 / 8 ticks -0.007 / +nan (better use of compute: no); per-token ticks work: no

## Per-token inference (added before its results; label-free entropy budgets)

| Arm | s = 24 | s = 48 | s = 96 | cross-entropy budgets, s = 24 (reported) | mean ticks at s = 24 / 48 / 96 |
|---|---:|---:|---:|---:|---|
| Dans_sampler | 4.761 | 4.761 | 4.761 | 4.761 | 16.0 / 28.0 / 52.0 |
| CTsq_sampler | 4.756 | 4.756 | 4.756 | 4.755 | 16.0 / 28.0 / 52.0 |
| Dans_pertoken | 4.775 | 4.775 | 4.775 | 4.775 | 16.0 / 28.0 / 52.0 |
| CTsq_pertoken | 8.839 | 9.803 | 9.191 | 4.090 | 16.0 / 28.0 / 52.0 |
| Dans_entropy | 4.784 | 4.784 | 4.784 | 4.784 | 16.0 / 28.0 / 52.0 |
| CTsq_entropy | 4.762 | 4.762 | 4.762 | 4.762 | 16.0 / 28.0 / 52.0 |
- **Dans_pertoken (per-token inference):** cost vs sampler at 16 ticks +0.014 (fails); ticks pay: no; better use of compute: no; per-token ticks work: no
- **CTsq_pertoken (per-token inference):** cost vs sampler at 16 ticks +4.083 (fails); ticks pay: no; better use of compute: no; per-token ticks work: no
- **Dans_entropy (per-token inference):** cost vs sampler at 16 ticks +0.023 (fails); ticks pay: no; better use of compute: no; per-token ticks work: no
- **CTsq_entropy (per-token inference):** cost vs sampler at 16 ticks +0.007 (ok); ticks pay: no; better use of compute: no; per-token ticks work: no
- **The CTM matters (pertoken arms, per-token inference):** no
- **The CTM matters (entropy arms, per-token inference):** no
- **The CTM matters (CTsq per-token gain exceeds Dans by at least 0.02):** no
