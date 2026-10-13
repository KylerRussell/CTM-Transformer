# Matched CTM arm

Protocol: `scripts/ctm_matched.py` (docstring). Held-out loss at 400M tokens on 1,024 windows, final tick.

References: Dans 3.569 / 3.579 (mean 3.574); 8-layer Transformer 3.618 / 3.638.

| Run | Held-out loss |
|---|---:|
| M_T16_lr0.004 | running |
| M_T1_lr0.001 | 3.828 |
| M_T1_lr0.002 | 3.738 |
| M_T1_lr0.004 | 3.732 |
| M_T1_lr0.004_s2 | running |
