# S3 reliability: escape probability of Transformer, RDT and CTM-LM

## Protocol declared before new scientific training — 2026-09-28

### Question

The [locked confirmation](results/sync_confirmation_v1/INTERPRETATION.md) showed that S₃ outcomes are **bimodal by seed**. A recurrent run either *escapes* to a serial solution of positions 9–16 (above 90%) or stays near the fixed-depth profile (about 30%). Each recurrent cell escaped at only 1–2 of 7 seeds, so 5–7 seeds cannot order architectures. This study measures the **escape probability** itself, with enough seeds to estimate it. It is step 1 of the re-prioritized pre-LM plan, and it sets the baseline that later training changes (randomized depth, curriculum, learning rate) and architecture claims must beat.

### Cells

| Cell | Model | Parameters | Readout | Loss |
|---|---|---:|---|---|
| `transformer` | 2-layer decoder (fixed depth) | 548,660 | final | final CE |
| `rdt` | recurrent depth (T16) | 525,984 | confidence | final CE |
| `ctm_lm` | CTM-LM ([design](CTM_LM_DESIGN.md)), T16 | 636,928 | confidence | CTM min-loss plus max-certainty |

The shared recipe is the S₃ recipe: LR 0.0003, 10,000 updates, batch 32, warmup 100, cosine to 10%, AdamW, BF16 autocast, and runner v3 with named factories. Each cell keeps its natural loss; CTM's own loss is part of the faithful design.

### Data and seeds

There are **20 new seeds**, disjoint from every earlier seed: 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229 and 233. Each seed has its own fresh S₃ data (`research/data/reliability_s3_v1`): 320,000 training words of lengths 1–16, 512 validation words, and 1,024 evaluation words of length 32. The evaluation split is read once, by the frozen evaluator, after the checkpoint freeze. All three cells share the data for a given seed.

### Endpoints and tests (fixed now)

**Escape:** a run escapes if its mean held-out accuracy over positions 9–16 at the trained budget (Transformer T = 1, others T = 16) is **at least 0.9**.

**Primary:** the escape probability per cell, with a 95% Clopper–Pearson interval.

**Declared comparisons.** The tests are paired over seeds, because cells share data within a seed. They cover two comparisons, `rdt` against `transformer` and `ctm_lm` against `rdt`, each on two metrics:
- a **continuous metric**, mean accuracy over positions 1–16, tested with an exact two-sided sign-flip permutation test;
- **escape**, tested with an exact two-sided McNemar test.

All four p-values are **Holm-corrected** together. Results are reported with effect sizes whatever the outcome; no single threshold defines a "win".

**Secondary:** the correct prefix at T = 4, 8, 16 and 32 (tick use), accuracy by position, training time, and the distribution of positions 9–16 accuracy per cell.

### Verification and execution

Before freezing:
- the study checks: Clopper–Pearson against known values, the exact McNemar and sign-flip tests, Holm adjustment, and the factory mapping;
- the CTM-LM checks;
- the group-word and S₃ checks;
- the Sync-RDT checks, including exact RDT parity;
- the archived-trainer replay, on both GPUs.

The freeze audit is the S₃ audit with the CTM-LM factory added. `research/launch_reliability_s3.sh` starts the restart-safe supervisor, with autostart. Sixty runs are spread over six worker slots by longest-first scheduling, taking an estimated 6–7 hours. Failures are recorded and never retried.

### Interpretation limits

- One task (S₃), one learning rate, one budget and one width. Escape probability may depend on all of them; the next studies vary depth sampling, curriculum and learning rate.
- The escape threshold (0.9 over positions 9–16) is declared here, informed by the bimodality seen in the confirmation. Accuracy by position is reported so other thresholds can be checked.
