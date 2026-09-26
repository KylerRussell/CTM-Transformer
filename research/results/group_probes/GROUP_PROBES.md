# Group word problem probes (round g1) — 2026-09-26

**Development only**, not preregistered: one run per cell, seed 41 (a calibration seed), LR 0.0003, 10,000 updates. Training uses fresh random words of lengths 1–16. Evaluation uses 128 held-out words at each of lengths 4, 8, 12, 16, 24, 32, 48 and 64; words of length 8 or more are guaranteed unseen. All families share data within a group. Code: `ctm_transformer/group_word.py`, `scripts/run_group_probe.py`. Chance: S₃ 1/6 = 16.7%; A₅ 1/60 = 1.7%.

## Accuracy by position (running product correct at position i), held-out

| Group | Model, inference steps | p4 | p6 | p8 | p10 | p12 | p14 | p16 | p20–p64 (untrained) | Train min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| S₃ | Transformer (fixed depth) | 100 | 81 | 47 | 37 | 35 | 31 | 30 | 13–21 | 4 |
| S₃ | RDT, T=4 | 100 | 80 | 45 | 29 | 26 | 23 | 22 | 15–18 | 55 |
| S₃ | RDT, T=8 | 100 | 100 | 96 | 75 | 48 | 32 | 27 | 15–19 | |
| S₃ | **RDT, T=16 (trained)** | 100 | 100 | **98** | **92** | **86** | **75** | **60** | 14–20 | |
| S₃ | RDT, T=32 | 100 | 100 | 98 | 92 | 86 | 76 | 63 | 15–20 | |
| S₃ | CTM, T=16 (T=4 to 32 identical within 2 points) | 100 | 64 | 37 | 34 | 36 | 34 | 34 | 15–26 | 113 |
| A₅ | Transformer | 7 | 2 | 1 | 2 | 2 | 2 | 2 | 0–2 | 4 |
| A₅ | RDT, T=16 (all T identical) | 2 | 2 | 2 | 1 | 2 | 2 | 1 | 2 | 55 |
| A₅ | CTM, T=16 (all T identical) | at chance from position 3 | | | | | | | | 112 |

(A₅ Transformer is correct only at positions 1–3; A₅ RDT at position 1, and 25% at position 2.)

## Observations

- **RDT uses its recurrence steps as serial computation.** On S₃ the correct prefix grows with the inference step budget: about 4 positions at T=4, about 8 at T=8, and about 12–14 at T=16. That is roughly one position of state per step, consistent with a sequential composition algorithm. T=32 adds nothing beyond T=16 (the trained budget), so the steps do not extrapolate past training, and no model generalizes to unseen lengths (positions 20 and beyond are at chance).
- **The fixed-depth Transformer** is exact through position 4 and then degrades, consistent with its depth limit.
- **The reference CTM does not use its ticks for this task.** Accuracy is the same at T=4 and T=32, and matches the Transformer's profile (exact through position 4, about 35% after). Its validation curve plateaus near 70% (RDT: 96%). Every position can read all earlier input tokens directly, so this is not the retrieval limitation. The ticks simply do not become serial state tracking here.
- **A₅ is at the floor for every family** at this model size and budget. Only the first two or three positions are learned.

## Implication for the next step

S₃ meets the fixed operating-point rule: across evaluated lengths 24–64, the mean of CTM and RDT accuracy lies between 20% and 80%, and no family is at 95% or higher everywhere. It also separates the families in an interpretable way (step budget against correct prefix length). A₅ is too hard at this scale; it could be revisited with a length curriculum or larger models. The natural primary endpoints for a frozen study on S₃:
- the correct-prefix length (last position with at least 90% accuracy) at the trained budget;
- its growth with the inference step budget;
- accuracy at positions 9–16 (trained lengths that need depth).
