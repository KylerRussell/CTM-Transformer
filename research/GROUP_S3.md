# S3 word problem: serial state across architectures

## Protocol declared before new scientific training — 2026-09-26

### Question

Do recurrent models turn extra computation steps into longer correct state tracking? And do CTM's mechanisms (neuron-level temporal MLPs and synchronization-derived queries, inside the recurrent-depth scaffold) or the standalone reference CTM do this better than recurrent depth alone?

The [development probes](results/group_probes/GROUP_PROBES.md) (one seed) motivate the study:
- on the S₃ word problem, RDT's correct running-product prefix grew with its inference step budget (about 4, 8 and 12–14 positions at T = 4, 8 and 16);
- the fixed-depth Transformer was exact only through position 4;
- the reference CTM did not use its ticks: its accuracy was the same from T = 4 to T = 32.

S₃ meets the operating-point rule fixed in the [calibration protocol](POINTER_CALIBRATION.md#operating-point-rules-for-stages-1b-and-2-fixed-now). A₅ was at the floor for every family and is not used.

### Task and data (`research/data/group_s3_v1`)

- **Task:** running products over S₃ as sequence labeling (`ctm_transformer/group_word.py`). The label at gᵢ's position is g₁·…·gᵢ, composed left to right. No label ever appears in the input. Chance is 1/6.
- **Seeds:** 47, 53 and 59 (reserved for this confirmation stage) and 61, 67 (new). All are disjoint from the calibration seeds 41 and 43, from the retrieval study's seeds, and from the probe seed.
- **Per seed:** 320,000 fresh random training words of lengths 1–16, balanced, each presented once. 512 validation words of lengths 1–16. 1,024 evaluation words of length 32.
- **Held-out words:** validation and evaluation words of length 8 or more never occur in training. Shorter words cannot be held out, because there are too few of them.
- **What the evaluation measures:** all models are causal, so positions 1–16 of an evaluation word measure trained-length behavior and positions 17–32 measure extrapolation.
- The five training files are regenerable from the manifest and are hashed, not archived.

### Cells (identical data per seed across cells)

| Cell | Model | Parameters | Readout | Inference step budgets evaluated |
|---|---|---:|---|---|
| `transformer` | 2-layer decoder (fixed depth) | 548,660 | final | 1 |
| `ctm` | reference CTM (static K/V, uniform temporal objective, T16) | 544,839 | confidence | 4, 8, 16, 32 |
| `rdt` | recurrent depth (T16) | 525,984 | confidence | 4, 8, 16, 32 |
| `rdt_wide` | recurrent depth, width 100, FFN 297 | 565,900 | confidence | 4, 8, 16, 32 |
| `history` | Sync-RDT, temporal MLP | 541,536 | confidence | 4, 8, 16, 32 |
| `sync` | Sync-RDT, sync queries | 550,688 | confidence | 4, 8, 16, 32 |
| `sync_rdt` | Sync-RDT, both | 566,240 | confidence | 4, 8, 16, 32 |

**Shared recipe:**
- peak LR 0.0003 (the rate at which RDT learns; also the CTM and Transformer recipe rate);
- 10,000 updates, batch 32, warmup 100, cosine to 10% of peak;
- AdamW, FP32 parameters, BF16 autocast;
- runner v3 with named factories, validation every 250 updates, and only the final and best checkpoints kept.

Every family keeps its recipe objective: final CE, except CTM's uniform temporal supervision.

### Endpoints

**Primary: correct prefix** at the trained step budget (Transformer T = 1, all others T = 16). This is the number of leading positions whose held-out accuracy is at least 90%, on the length-32 evaluation words.

**Decision rule, fixed now.** Cell A *exceeds* cell B if both hold:
- A's correct prefix is larger in at least 4 of 5 seeds;
- the median paired difference is at least 2 positions.

The declared decisions:

1. **Recurrence beats fixed depth:** `rdt` exceeds `transformer`.
2. **RDT beats the reference CTM:** `rdt` exceeds `ctm`.
3. **Sync extends state:** `sync` exceeds **both** `rdt` and `rdt_wide`.
4. **History extends state:** `history` exceeds both `rdt` and `rdt_wide`.
5. **Both mechanisms extend state:** `sync_rdt` exceeds both `rdt` and `rdt_wide`.
6. **Steps are used,** for each recurrent cell and CTM: the correct prefix at T = 16 exceeds the prefix at T = 4, by the same rule.

`ctm` against `transformer` and `rdt_wide` against `rdt` are descriptive. Five seeds support no significance claim, so the rule defines what is reported as a replicated effect.

**Secondary:**
- mean accuracy over positions 9–16 (trained lengths that need depth) and over positions 17–32 (extrapolation);
- the correct prefix at T = 8 and T = 32;
- validation learning curves;
- training time.

### Verification and execution

Before freezing:
- the study checks: the suite is deterministic, long held-out words never occur in training, the correct-prefix and decision logic behave as declared, and the factory mapping is correct;
- the group-word checks;
- the Sync-RDT checks, including exact RDT parity;
- the archived-trainer replay, on both GPUs.

The same checks run before every supervisor attempt.

The freeze audit checks, for every run:
- all 10,000 metric rows and the learning-rate schedule;
- per-update counts;
- all 40 validation evaluations (512 words, 4,864 labels);
- the factory and family, data hashes, config overrides and the final checkpoint.

`research/launch_group_s3.sh` starts the restart-safe supervisor, and an autostart entry resumes it after a restart. Thirty-five runs are spread over six worker slots by longest-first scheduling on estimated cost. A genuine worker error is recorded and not retried.

**Estimated wall time:** about 7 hours.

### Interpretation limits

- One group (S₃), one model scale and one learning rate.
- Training lengths are at most 16, so extrapolation beyond 16 is not expected. It is measured, not targeted.
- The reference CTM keeps its recipe objective (uniform temporal supervision). Other families use final CE, so CTM-versus-RDT differences include the objective.
- Parameter counts differ by up to 7.7%. `rdt_wide` controls the largest difference.
- Development data, not a locked test.
