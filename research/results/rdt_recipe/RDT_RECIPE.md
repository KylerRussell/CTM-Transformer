# RDT recipe study, round 1

Protocol: `scripts/rdt_recipe.py` (docstring). Held-out loss at 400M tokens on 1,024 windows (randomized-depth runs evaluated at depth 16).

Current recipe (sandwich norm everywhere, unit embeddings): randomized depth 3.783, depth 1 3.828 (lr 1e-3).

Probes (`p381_`) stop after 381 steps of the same schedule and report the step-381 evaluation; references at that step: T8_lr0.002 4.422, D_d1_lr0.002 5.277.

| Run | Held-out loss |
|---|---:|
| B_d1_lr0.0005 | running |
| B_d1_lr0.001 | diverged or collapsed |
| B_d1_lr0.002 | diverged or collapsed |
| B_d1_lr0.004 | diverged or collapsed |
| B_rand_lr0.001 | diverged or collapsed |
| C_d1_lr0.001 | 4.464 |
| C_d1_lr0.002 | 5.783 |
| C_d1_lr0.004 | diverged or collapsed |
| C_rand_lr0.001 | 4.078 |
| D_d1_lr0.002 | 4.123 |
| Dan_rand_lr0.002 | 3.617 |
| Dans_rand_lr0.002 | 3.569 |
| Dans_rand_lr0.002_s2 | 3.579 |
| E_d1_lr0.002 | 5.353 |
| T20_lr0.001 | 3.649 |
| T20_lr0.002 | 3.562 |
| T8_lr0.002 | 3.618 |
| T8_lr0.002_s2 | 3.638 |
| T8_lr0.004 | 3.674 |
| p381_D_d1_lr0.001 | 4.994 |
| p381_Da_d1_lr0.001 | 4.773 |
| p381_Da_d1_lr0.002 | 5.035 |
| p381_Dn_d1_lr0.002 | 4.712 |
