# Stage 0C: synthetic arithmetic and program tracing

Protocol and rules: `scripts/code_math_tasks.py` (docstring). Sequence accuracy (%), mean of 3 seeds (per seed in brackets).

| Task-split | T8 | T20 | Dans | CTsq | CTnlm | CTsqH |
|---|---:|---:|---:|---:|---:|---:|
| addition id | 10.6 (10/11/11) | 11.1 (10/11/12) | 10.9 (11/11/11) | 11.1 (11/11/12) | 10.9 (11/11/11) | 10.9 (11/11/11) |
| addition ood | 0.0 (0/0/0) | 0.0 (0/0/0) | 0.0 (0/0/0) | 0.0 (0/0/0) | 0.0 (0/0/0) | 0.0 (0/0/0) |
| expression id | 74.9 (78/73/74) | 72.3 (73/71/73) | 72.7 (72/74/72) | 73.3 (71/76/73) | 69.8 (65/70/74) | 75.2 (77/76/73) |
| expression ood | 17.6 (18/17/18) | 17.2 (15/18/19) | 16.4 (14/17/18) | 16.7 (16/17/17) | 17.6 (18/18/17) | 17.4 (17/19/16) |
| tracking id | 38.2 (40/36/39) | 38.7 (39/38/39) | 38.6 (39/40/37) | 38.0 (39/38/37) | 38.4 (39/37/39) | 38.8 (38/40/39) |
| tracking ood | 16.1 (15/14/19) | 17.1 (16/17/19) | 16.2 (14/17/17) | 15.9 (17/17/14) | 16.4 (17/15/18) | 16.3 (15/18/16) |

## Rule outcomes


## OOD sequence accuracy at more ticks (recurrent models; 16 / 32 / 64)

- **Dans:** addition 0.0 / 0.0 / 0.0; expression 16.4 / 16.5 / 16.4; tracking 16.2 / 16.2 / 16.2
- **CTsq:** addition 0.0 / 0.0 / 0.0; expression 16.7 / 16.8 / 16.9; tracking 15.9 / 15.9 / 15.9
- **CTnlm:** addition 0.0 / 0.0 / 0.0; expression 17.6 / 17.6 / 17.6; tracking 16.4 / 16.3 / 16.4
- **CTsqH:** addition 0.0 / 0.0 / 0.0; expression 17.4 / 17.4 / 17.5; tracking 16.3 / 16.4 / 16.5
