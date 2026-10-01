# S3 training-recipe development probes — 2026-09-29

These are exploratory development probes. They choose the recipe for the [S₃ recipe confirmation](../../RECIPE_S3.md) and support no claim.

**Setup:**
- Development suite `research/data/recipe_dev_s3`: seeds 307–337, disjoint from every study. Each seed has 320,000 training words of lengths 1–16, and held-out words of length 32.
- 10,000 updates; runner v5 (`scripts/run_recipe_probe.py`).
- Positions 17–32 measure extrapolation. Recurrent models are read at T = 32 for them and at T = 16 for positions 1–16.
- A run escapes when its mean accuracy over positions 9–16 is at least 0.9.

## Table

| Group | Model | Positions | Depth | Curriculum | LR | Seeds | Positions 9–16 per seed | Mean positions 1–16 | Mean positions 17–24 | Escaped |
|---|---|---|---|---|---:|---|---|---:|---:|---:|
| A5 | rdt | rope | fixed | none | 0.0003 | 307 | 0.02 | 0.085 | 0.017 | 0/1 |
| A5 | rdt | rope | fixed | none | 0.0006 | 307 | 0.02 | 0.084 | 0.016 | 0/1 |
| A5 | rdt | rope | fixed | none | 0.001 | 307 | 0.01 | 0.079 | 0.016 | 0/1 |
| A5 | rdt | rope | rand | linear_half | 0.001 | 307, 311, 313 | 0.02, 0.02, 0.02 | 0.085 | 0.016 | 0/3 |
| A5 | rdt | rope | rand | none | 0.0001 | 307 | 0.01 | 0.077 | 0.015 | 0/1 |
| A5 | rdt | rope | rand | none | 0.0003 | 307 | 0.02 | 0.083 | 0.017 | 0/1 |
| A5 | rdt | rope | rand | none | 0.001 | 307, 311, 313 | 0.12, 0.01, 0.02 | 0.211 | 0.041 | 0/3 |
| A5 | transformer | rope | — | none | 0.001 | 307, 311, 313 | 0.01, 0.02, 0.02 | 0.214 | 0.017 | 0/3 |
| S3 | ctm_lm | rope | fixed | none | 0.0003 | 307, 311, 313 | 0.35, 0.33, 0.36 | 0.597 | 0.183 | 0/3 |
| S3 | ctm_lm | rope | fixed | none | 0.0006 | 307, 311, 313 | 0.20, 0.33, 0.33 | 0.547 | 0.162 | 0/3 |
| S3 | ctm_lm | rope | fixed | none | 0.001 | 307 | 0.18 | 0.477 | 0.170 | 0/1 |
| S3 | rdt | learned | rand | none | 0.0003 | 307 | 0.38 | 0.661 | 0.175 | 0/1 |
| S3 | rdt | nope | rand | none | 0.0003 | 307, 311 | 0.37, 0.44 | 0.667 | 0.220 | 0/2 |
| S3 | rdt | rope | fixed | none | 0.0003 | 307, 311, 313 | 0.63, 0.54, 0.79 | 0.812 | 0.218 | 0/3 |
| S3 | rdt | rope | fixed | none | 0.0006 | 307, 311, 313 | 0.82, 0.78, 0.78 | 0.884 | 0.264 | 0/3 |
| S3 | rdt | rope | fixed | none | 0.001 | 307, 311, 313 | 0.97, 0.53, 0.80 | 0.873 | 0.462 | 1/3 |
| S3 | rdt | rope | rand | linear_half | 0.0003 | 307, 311, 313 | 0.99, 0.27, 0.87 | 0.826 | 0.268 | 1/3 |
| S3 | rdt | rope | rand | none | 0.0002 | 307, 311, 313 | 0.91, 0.30, 0.31 | 0.690 | 0.250 | 1/3 |
| S3 | rdt | rope | rand | none | 0.0003 | 307, 311, 313, 317, 331, 337 | 0.82, 0.29, 0.97, 0.34, 0.51, 0.38 | 0.749 | 0.249 | 1/6 |
| S3 | rdt | rope | rand | none | 0.0006 | 307, 311, 313 | 0.98, 0.87, 0.92 | 0.960 | 0.331 | 2/3 |
| S3 | rdt | rope | rand | none | 0.001 | 307, 311, 313 | 1.00, 0.96, 0.97 | 0.986 | 0.480 | 3/3 |
| S3 | transformer | nope | — | none | 0.0003 | 307 | 0.34 | 0.561 | 0.231 | 0/1 |
| S3 | transformer | rope | — | none | 0.0001 | 307, 311, 313 | 0.33, 0.32, 0.25 | 0.543 | 0.166 | 0/3 |
| S3 | transformer | rope | — | none | 0.0003 | 307, 311, 313 | 0.37, 0.37, 0.35 | 0.632 | 0.178 | 0/3 |
| S3 | transformer | rope | — | none | 0.001 | 307, 311, 313 | 0.35, 0.41, 0.35 | 0.642 | 0.178 | 0/3 |
| S3 | transformer | rope | — | none | 0.003 | 307, 311, 313 | 0.42, 0.34, 0.35 | 0.634 | 0.173 | 0/3 |

## Findings (development; 1–6 seeds per row)

1. **RoPE enables extrapolation.** With RoPE, RDT keeps some accuracy past the trained length. Most escaped runs stay above 0.7 for 1–3 positions past 16 and then decay. One run, fixed depth at 1e-3 with seed 307, stays at 0.88 or above through position 26. With the learned table, or with no position encoding (NoPE), positions 17 and later stay near chance. RDT with NoPE also never escaped. RoPE is adopted for every model.
2. **The learning rate is the largest factor found so far.**
   - RDT with randomized depth escapes at 1 of 6 seeds at 3e-4, 2 of 3 at 6e-4, and **3 of 3 at 1e-3**, with mean accuracy over positions 1–16 of 0.986.
   - This includes seed 311, which failed under every setting at 3e-4.
   - Fixed-depth RDT improves too, from 0.81 at 3e-4 to 0.88 at 6e-4, but escapes at only 1 of 3 seeds even at its best rate.
   - The earlier finding that "RDT learns at 3e-4, not 1e-3" came from learned absolute positions and fixed depth, and does not transfer.
3. **Randomized depth and a higher learning rate may interact.** Each at its best rate, randomized depth reaches 0.986 against 0.884 for fixed depth. It also extrapolates further: 0.48 against 0.26 on positions 17–24. But fixed depth at 1e-3 also reaches 0.46 there, so extrapolation may depend on the learning rate as much as on depth sampling.
4. **A length curriculum adds nothing clear.** At 3e-4 and three paired seeds, its differences are +0.17, 0.00 and −0.10 on positions 9–16. It is not used on S₃.
5. **CTM-LM is best at 3e-4 and still far below RDT** (0.60 on positions 1–16). Its accuracy still depends on ticks: at T = 4 it is at chance.
6. **The Transformer is insensitive to the learning rate** between 3e-4 and 3e-3 (0.63–0.64 on positions 1–16).

Per-run records are in this directory (`*.json`), and `probe_table.md` is regenerated by `scripts/summarize_recipe_probes.py`.

## A₅ probes — 2026-09-30 to 2026-10-01

The A₅ development suite is `research/data/recipe_dev_a5`: seeds 307, 311 and 313, 320,000 training words of lengths 1–16, and held-out words of length 32. Chance is 1/60. Runs of 30,000 updates present each training word three times. Width and training depth were varied with `--width` (d_model; the FFN is 3× wider) and `--train-depth`.

1. **At the default width (96), RDT never learns A₅.** It stays stuck at loss about 3.2: position 1 is copied, position 2 is near chance, and everything after is at chance. This held across learning rates 1e-4 to 1e-3, fixed and randomized depth, training depth 2, 4 and 16, and with or without the curriculum (14 runs). The 2-layer Transformer (width 140) learns 3–4 positions.
2. **Width removes the barrier, for some seeds.**
   - At width 192 (2.02M parameters) with randomized depth and learning rate 1e-3, seed 307 reached 63% on positions 1–16 in 10,000 updates and was still improving.
   - Seeds 311 and 313 stayed stuck at 10,000 updates.
   - Fixed depth at width 192 (learning rate 6e-4) stayed stuck.
3. **With 30,000 updates, width-192 randomized-depth RDT solves A₅ on 2 of 3 seeds.**
   - Seeds 311 and 313 reach **99.5% and 99.7% on positions 1–16**. They stay above 0.9 through about position 20.
   - The solution is serial: about two positions per step (about 8 positions at T = 4, about 16 at T = 8).
   - Seed 307 at 30,000 updates (a different learning-rate schedule from its 10,000-update run) learned positions 2–4 and then decayed, reaching 44%.
4. **A parameter-matched Transformer (width 280, 2.08M parameters) handles only 2–4 positions** (16–26% on positions 1–16), as expected for fixed depth.

These are development observations from three seeds. A frozen A₅ study (fresh seeds, one presentation of each word, matched controls) is needed before any claim.

