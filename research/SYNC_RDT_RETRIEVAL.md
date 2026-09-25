# Sync-RDT retrieval sample efficiency

## Protocol declared before new scientific training — 2026-09-25

### Question

Does adding CTM mechanisms to the recurrent-depth scaffold change how quickly in-context retrieval is learned? The mechanisms are neuron-level temporal MLPs over each position's state history (**history**) and synchronization-derived attention-query terms (**sync**). In a single-seed development check ([round 12](results/pointer_probes/PROBES.md)), all four cells learned retrieval. Sync escaped earliest and reached 99.5% held-out; RDT reached 91.4%, the full cell 88.2%, and history 56.8% and still rising. Escape timing varies strongly between runs, so this study replicates across seeds and adds two controls. It asks about **learning speed**, a measure that stays informative even when every cell eventually reaches 100%.

### Cells

All cells are defined in `ctm_transformer/sync_rdt.py` and [the design](SYNC_RDT_DESIGN.md). The `rdt` cell reproduces the baseline exactly (`tests/test_sync_rdt.py`).

| Cell | Mechanisms | Parameters | Purpose |
|---|---|---:|---|
| `rdt` | none | 525,984 | control |
| `history` | temporal MLP (CTM gate init, σ(0) = 0.5) | 541,536 | mechanism A |
| `sync` | sync query terms | 550,688 | mechanism B |
| `sync_rdt` | both | 566,240 | A × B |
| `rdt_wide` | none; width 100, FFN 297 | 565,900 | parameter control for the largest cell |
| `history_lowgate` | temporal MLP, gate init −4 (σ ≈ 0.018) | 541,536 | tests whether the halved initial update explains a history slowdown |

### Fixed recipe

The RDT recipe applies to every cell:
- T = 16 recurrence steps, prelude/core/coda of 1/2/1 blocks, final CE, confidence readout;
- peak LR 0.0003 (the rate at which RDT learns; 0.001 fails, probes r9b and r10b);
- 10,000 updates, batch 32, warmup 100, cosine to 10% of peak;
- AdamW, FP32 parameters, BF16 autocast.

Training runs through runner v3 with the cell's named factory, which is recorded in each run manifest. Validation runs every 250 updates, and only the final and best checkpoints are kept.

### Data (`research/data/syncrdt_retrieval_v1`)

Five seeds: **71, 73, 79, 83, 89**. They are new and disjoint from the calibration seeds (41, 43) and from the later confirmation seeds (47, 53, 59).

- **Task:** all-keys one-hop MQAR (`map Al;Gf;…;hops 1;ask Kd;Ia;…;`). The 12 keys are queried in random order, and 12 answers plus EOS are supervised.
- **Per seed:** 320,000 training maps (so every map is presented once), 256 validation maps and 512 evaluation maps, all disjoint within the seed.
- **Identical data across cells:** every cell trains on the same data for a given seed. The seed also sets initialization (seed), data order (seed + 1) and depth sampling (seed + 2).
- The five training files are regenerable from the manifest and are hashed, not archived.
- **Position-1 accuracy** (chance 1/11) is the leak-free retrieval measure. Later positions can use exclusion of earlier gold answers.

### Endpoints

**Primary.** Updates until the **validation position-1 accuracy** (confidence readout, 256 maps) first reaches **50%** and **90%**, measured every 250 updates. A run that never reaches a threshold is censored and ranked after every run that does.

**Decision rule, fixed now.** For a paired comparison between two cells (same seed and same data), the first cell is *favored* at the 90% threshold if both hold:
- it reaches the threshold earlier in at least 4 of 5 seeds;
- the median paired difference is at least 1,000 updates.

The declared decisions:
- **Sync speeds retrieval** if sync is favored over **both** `rdt` and `rdt_wide`.
- **History slows retrieval** if `rdt` is favored over `history`. **History speeds retrieval** if `history` is favored over `rdt`.
- **Gate initialization explains a history slowdown** if history slows retrieval *and* `history_lowgate` is favored over `history`.

Other comparisons are descriptive: `sync_rdt` against `rdt` and `sync`, `rdt_wide` against `rdt`, and the 50% threshold. Five seeds support no significance claim, so the thresholds define what will be reported as a replicated effect.

**Secondary.**
- Final held-out accuracy on 512 evaluation maps per seed: position 1, all answers, and all 12 correct. The confidence readout is primary; the final readout is also reported.
- Mean validation position-1 accuracy across the curve (area).
- Training time and memory.

### Verification and execution

Before freezing:
- the study checks: the suite is deterministic, all-keys and disjoint; wrong answers are detected despite a refreshed hash; the threshold and censoring logic behaves as declared;
- the Sync-RDT checks: exact RDT parity in initialization, outputs and training; design parameter counts; causality; per-step decoding versus truncation; gradients; round trips; the low-gate control;
- the archived-trainer replay, on both GPUs. The same checks run before every supervisor attempt.

The freeze audit checks, for every run:
- all 10,000 metric rows and the learning-rate schedule;
- per-step example, token and label counts;
- all 40 validation evaluations;
- the factory name, data hashes, config overrides and the final checkpoint.

`research/launch_syncrdt_retrieval.sh` starts the restart-safe supervisor, and an autostart entry resumes it after a container restart. Thirty runs are spread over six worker slots, three per GPU; these small models leave the GPUs mostly idle. A genuine worker error is recorded and not retried; nothing is added or replaced adaptively.

**Estimated wall time:** 5 rounds of about 75–90 minutes, roughly 7 hours.

### Interpretation limits

- One task: one-hop in-context retrieval, the capability check, not serial depth.
- One model scale and one learning rate.
- The cells differ in parameters by up to 7.7%. The `rdt_wide` control covers the largest difference, not every pair.
- Validation curves are measured on data that never selects checkpoints. It is still development data, not a locked test.
- A positive sync result would show faster retrieval learning in this scaffold. It would not show improved multi-step computation, which the serial-depth study must test.
