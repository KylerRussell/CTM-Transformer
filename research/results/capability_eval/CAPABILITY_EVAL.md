# Capability breakdown of held-out loss

Protocol and rules: `scripts/capability_eval.py` (docstring). Two-seed means; T20 has one seed.

| Measure | T8 | T20 | Dans | CTfull | CTnlm | CTsq | CTso | CTsqS | CTsqLD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 3.630 | 3.562 | 3.575 | 3.590 | 3.572 | 3.583 | 3.577 | 3.587 | 3.842 |
| repeat | 1.441 | 1.395 | 1.414 | 1.426 | 1.409 | 1.414 | 1.417 | 1.416 | 1.637 |
| novel | 5.139 | 5.061 | 5.076 | 5.089 | 5.071 | 5.083 | 5.076 | 5.093 | 5.367 |
| other | 2.714 | 2.650 | 2.658 | 2.675 | 2.657 | 2.669 | 2.662 | 2.668 | 2.914 |
| pos_0-63 | 4.042 | 3.987 | 3.996 | 4.008 | 3.999 | 4.010 | 4.005 | 4.007 | 4.226 |
| dist_1-32 | 1.065 | 1.032 | 1.043 | 1.045 | 1.040 | 1.041 | 1.043 | 1.038 | 1.271 |
| dist_33-128 | 1.295 | 1.253 | 1.268 | 1.280 | 1.265 | 1.266 | 1.270 | 1.268 | 1.480 |
| pos_64-255 | 3.661 | 3.591 | 3.607 | 3.621 | 3.607 | 3.613 | 3.609 | 3.618 | 3.872 |
| dist_129-512 | 1.782 | 1.727 | 1.754 | 1.770 | 1.746 | 1.754 | 1.758 | 1.764 | 1.984 |
| pos_256-511 | 3.598 | 3.532 | 3.545 | 3.559 | 3.538 | 3.551 | 3.545 | 3.556 | 3.813 |
| pos_512-1023 | 3.582 | 3.514 | 3.526 | 3.542 | 3.523 | 3.534 | 3.529 | 3.538 | 3.798 |
| dist_513-1023 | 2.386 | 2.302 | 2.342 | 2.368 | 2.335 | 2.348 | 2.343 | 2.349 | 2.569 |
| recall8_loss | 7.473 | 7.825 | 8.247 | 7.858 | 7.801 | 7.631 | 8.012 | 8.055 | 9.092 |
| recall8_acc | 7.8 | 7.4 | 4.4 | 7.2 | 7.4 | 9.3 | 6.5 | 6.5 | 1.4 |
| recall32_loss | 9.476 | 9.570 | 9.762 | 9.476 | 9.428 | 9.425 | 9.696 | 9.811 | 10.209 |
| recall32_acc | 1.0 | 0.5 | 0.5 | 1.8 | 1.4 | 2.3 | 0.8 | 0.8 | 0.3 |
| recall96_loss | 10.258 | 10.401 | 10.308 | 10.301 | 10.340 | 10.410 | 10.481 | 10.417 | 10.596 |
| recall96_acc | 0.1 | 0.0 | 0.1 | 0.2 | 0.1 | 0.2 | 0.1 | 0.0 | 0.0 |
| copy16_loss | 5.379 | 5.618 | 5.401 | 5.029 | 4.961 | 4.932 | 5.279 | 5.104 | 7.585 |
| copy16_acc | 24.6 | 23.2 | 25.8 | 28.5 | 31.7 | 31.5 | 27.9 | 29.6 | 7.1 |
| copy64_loss | 6.283 | 6.278 | 6.770 | 6.272 | 6.122 | 6.281 | 6.794 | 6.630 | 8.305 |
| copy64_acc | 18.0 | 18.3 | 12.8 | 17.8 | 20.9 | 17.1 | 14.1 | 14.8 | 4.0 |
| copy256_loss | 8.138 | 7.569 | 8.409 | 8.365 | 8.451 | 8.385 | 8.819 | 8.742 | 10.040 |
| copy256_acc | 4.3 | 7.8 | 2.8 | 4.2 | 3.2 | 3.7 | 2.1 | 2.4 | 0.4 |
| drift32 |  |  | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| drift48 |  |  | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 |
| drift64 |  |  | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| drift96 |  |  | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 |

## Rule outcomes against Dans

- **T8:** overall: worse (+0.054); repeat: worse (+0.027); repeat: specific advantage (excess -0.027 / -0.028); novel: worse (+0.063); other: worse (+0.056); pos_0-63: worse (+0.046); dist_1-32: worse (+0.022); dist_1-32: specific advantage (excess -0.035 / -0.029); dist_33-128: worse (+0.027); dist_33-128: specific advantage (excess -0.030 / -0.025); pos_64-255: worse (+0.054); dist_129-512: worse (+0.028); dist_129-512: specific advantage (excess -0.021 / -0.032); pos_256-511: worse (+0.053); pos_512-1023: worse (+0.056); dist_513-1023: worse (+0.044); recall8_loss: better (-0.774); recall32_loss: better (-0.285); copy64_loss: better (-0.488); copy64_acc: better (+5.159); copy256_loss: better (-0.271)
- **CTfull:** dist_513-1023: worse (+0.026); copy64_loss: better (-0.498); copy64_acc: better (+4.973)
- **CTnlm:** copy64_acc: better (+8.073)
- **CTsq:** no measure differs from Dans by the rule
- **CTso:** recall96_loss: worse (+0.173); copy256_loss: worse (+0.410)
- **CTsqS:** no measure differs from Dans by the rule
- **CTsqLD:** overall: worse (+0.267); repeat: worse (+0.223); repeat: specific advantage (excess -0.028 / -0.060); novel: worse (+0.291); other: worse (+0.257); pos_0-63: worse (+0.231); pos_0-63: specific advantage (excess -0.036 / -0.037); dist_1-32: worse (+0.229); dist_1-32: specific advantage (excess -0.021 / -0.056); dist_33-128: worse (+0.212); dist_33-128: specific advantage (excess -0.032 / -0.077); pos_64-255: worse (+0.265); dist_129-512: worse (+0.230); dist_129-512: specific advantage (excess -0.026 / -0.047); pos_256-511: worse (+0.268); pos_512-1023: worse (+0.272); dist_513-1023: worse (+0.227); dist_513-1023: specific advantage (excess -0.045 / -0.035); recall8_loss: worse (+0.845); recall8_acc: worse (-3.027); recall96_loss: worse (+0.288); copy16_loss: worse (+2.185); copy16_acc: worse (-18.698); copy64_loss: worse (+1.535); copy64_acc: worse (-8.780); copy256_loss: worse (+1.631); copy256_acc: worse (-2.448)

## Ticks beyond training (M4)

- **Dans:** drift at 32/48/64/96 ticks vs 16: +0.000 / +0.000 / +0.000 / +0.000 (per seed at 96: +0.000, +0.000); generalizes: yes
- **CTfull:** drift at 32/48/64/96 ticks vs 16: -0.000 / -0.000 / -0.000 / -0.000 (per seed at 96: +0.000, -0.000); generalizes: yes
- **CTnlm:** drift at 32/48/64/96 ticks vs 16: +0.000 / +0.000 / +0.000 / +0.000 (per seed at 96: +0.000, +0.000); generalizes: yes
- **CTsq:** drift at 32/48/64/96 ticks vs 16: +0.000 / +0.000 / +0.000 / +0.000 (per seed at 96: +0.000, +0.000); generalizes: yes
- **CTso:** drift at 32/48/64/96 ticks vs 16: +0.000 / +0.000 / +0.000 / +0.000 (per seed at 96: +0.000, +0.000); generalizes: yes
- **CTsqS:** drift at 32/48/64/96 ticks vs 16: +0.000 / +0.000 / +0.000 / +0.000 (per seed at 96: +0.001, +0.000); generalizes: yes
- **CTsqLD:** drift at 32/48/64/96 ticks vs 16: +0.000 / -0.000 / +0.000 / -0.000 (per seed at 96: -0.000, -0.000); generalizes: yes
- **CMM alpha/beta claim (CTsq and CTso drift at 96 ticks at least 0.01 below both Dans seeds):** not supported
