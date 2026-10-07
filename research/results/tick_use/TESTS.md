# CTM-LM tick-use training tests

Protocol: [CTM_TICK_DIAGNOSTICS.md](../../CTM_TICK_DIAGNOSTICS.md). Held-out loss is the final tick on 1,024 windows; per-tick loss is on 32 windows (ticks beyond 16 extrapolate).

| Model | Held-out loss | tick 1 | tick 2 | tick 3 | tick 4 | tick 8 | tick 12 | tick 16 | tick 24 | tick 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (D + sparse, lr 4e-3) | 5.526 | 6.470 | 5.821 | 5.553 | 5.464 | 5.421 | 5.436 | 5.464 | 5.542 | 5.644 |
| E4 backbone source (400M tokens) | 4.025 | 4.774 | 4.298 | 4.073 | 3.995 | 3.967 | 3.976 | 3.992 | 4.033 | 4.082 |
| E8_cross_position | 5.538 | 6.442 | 5.808 | 5.551 | 5.461 | 5.421 | 5.442 | 5.474 | 5.565 | 5.686 |
| E4c_feature_head | 4.273 |  |  |  |  |  |  |  |  |  |
| S1_shallow | 5.321 | 5.928 | 5.495 | 5.326 | 5.263 | 5.229 | 5.239 | 5.257 | 5.301 | 5.356 |
| S2_shallow_cross | 5.303 | 5.906 | 5.528 | 5.343 | 5.253 | 5.214 | 5.221 | 5.237 | 5.280 | 5.338 |
| S3_shallow_cross_random | 5.744 | 6.179 | 5.823 | 5.699 | 5.646 | 5.612 | 5.643 | 5.681 | 5.747 | 5.806 |
| R1_rdt_heavy_depth1 | 4.681 |  |  |  |  |  |  |  |  |  |
| S2_T1 | 5.246 | 5.187 | 5.336 | 5.566 | 5.792 | 8.973 | 13.535 | 18.072 | 25.569 | 31.561 |
| S2_T4 | 5.303 | 5.247 | 5.217 | 5.224 | 5.242 | 5.673 | 6.132 | 6.622 | 7.724 | 8.688 |
