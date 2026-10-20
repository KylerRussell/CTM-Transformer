# CTM Transformer on language modelling

Protocol: `scripts/ctm_transformer_study.py` (docstring). Held-out loss at 400M tokens on 1,024 windows, 16 ticks.

References: Dans 3.569 / 3.579 (mean 3.574); 8-layer Transformer 3.618 / 3.638; 20-layer Transformer 3.562.

| Run | Held-out loss |
|---|---:|
| CTfull_lr0.002 | 3.586 |
| CTfull_lr0.002_s2 | 3.590 |
| CTnlm_lr0.002 | 3.570 |
| CTnlm_lr0.002_s2 | 3.574 |
| CTso_lr0.002 | 3.571 |
| CTso_lr0.002_s2 | 3.580 |
| CTsqL_lr0.002 | running |
| CTsqL_lr0.002_s2 | running |
| CTsqS_lr0.002 | 3.579 |
| CTsqS_lr0.002_s2 | 3.593 |
| CTsq_lr0.001 | 3.666 |
| CTsq_lr0.002 | 3.584 |
| CTsq_lr0.002_s2 | 3.579 |
| CTsq_lr0.004 | 3.714 |
