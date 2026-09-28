# Locked confirmation: second-order state features in the attention queries

**Status: protocol declared before new scientific training — 2026-09-28.** Approved as drafted. The suite's `evaluation` split is the locked test set: it is read only by the frozen evaluator, once, after the checkpoint freeze.

## Claim under test

The development studies concluded the following:
- In the recurrent-depth scaffold, **pairwise products of the recurrent state, added to the attention queries**, roughly double how far recurrent depth tracks state on the S₃ word problem ([S₃ study](results/group_s3_v1/INTERPRETATION.md)).
- They do not need history, decay or temporal order ([A1–A5](results/sync_ablation_v1/INTERPRETATION.md)).
- Cross-channel products are the reliable form ([A7](results/sync_ablation_v2/INTERPRETATION.md)).

This study tests the claim on data and seeds that shaped no decision, with a stricter rule. It is part 2 of the [plan](SYNC_CONFIRMATION_PLAN.md), with the review's choices: seven seeds and a 6-of-7 rule. The replication setting is primary; the secondary settings are gated.

## Primary setting: exact replication

- **Task:** S₃ running products (`ctm_transformer/group_word.py`), with the same format, training lengths 1–16 and evaluation length 32 as the S₃ study.
- **Seeds:** **97, 101, 103, 107, 109, 113, 127**. None has been used before.
- **Data:** a fresh suite (`research/data/sync_confirmation_v1`) from a new generator name, with 320,000 training, 512 validation and 1,024 **test** words per seed. Test words of length 8 or more never occur in training.
- **The test split is locked.** No development run, probe or analysis touches it. It is evaluated **exactly once**, after all checkpoints are frozen, by the frozen evaluator. Validation is used only for training-time curves and the secondary best-checkpoint report.

| Cell | Model | Parameters | Role |
|---|---|---:|---|
| `sync` | Sync-RDT sync cell (cross-channel pairs over the 8-step history) | 550,688 | tested |
| `current` | the same, with synchronization of the current state only (ablation A3) | 550,688 | tested: the minimal form |
| `rdt` | recurrent depth | 525,984 | control |
| `rdt_wide` | recurrent depth, width 100, FFN 297 | 565,900 | parameter control |
| `transformer` | 2-layer decoder | 548,660 | fixed-depth reference |

The recipe is identical to the S₃ study: LR 0.0003, 10,000 updates, batch 32, warmup 100, cosine to 10%, T = 16, confidence readout (final readout for the Transformer), runner v3. There are 35 runs, about 5 hours.

## Endpoint and declared decisions

**Primary endpoint:** the correct prefix at T = 16 on the locked test words, meaning the number of leading positions at least 90% accurate (Transformer at T = 1).

**Rule:** A *exceeds* B if A's correct prefix is larger in **at least 6 of 7 seeds** and the median paired difference is **at least 2 positions**.

**Primary claims.** Each is confirmed only if both parts hold:
1. **C1:** `sync` exceeds `rdt` **and** `sync` exceeds `rdt_wide`.
2. **C2:** `current` exceeds `rdt` **and** `current` exceeds `rdt_wide`.

**Secondary, declared but not part of the primary claim:**
- `sync` against `current`, reported both ways;
- each recurrent cell uses its steps (T = 16 against T = 4);
- `rdt` against `transformer`.

**Other secondary measures:** step scaling at T = 4, 8, 16 and 32; mean accuracy at positions 9–16 and 17–32; training time; learned decay rates in `sync`.

## Secondary settings (gated, declared separately)

Each setting is run only if a short development probe (`rdt` and `sync`, one calibration seed) shows that the setting is learnable and meets the operating-point rule:

- **Width 128** for every recurrent cell, as a scale check.
- **Training lengths 1–24**, evaluated at length 48, as a harder state-tracking range.

They reuse this protocol's cells, rule and endpoint, and are reported whatever their outcome.

## Verification and execution

The pre-freeze checks are all the relevant checks used so far:
- group-word and group-suite checks;
- Sync-RDT checks, including exact RDT parity;
- ablation checks, including identity parity with `sync`;
- the archived-trainer replay, on both GPUs.

The freeze audit is the S₃ audit. The study runs under the restart-safe supervisor with autostart. Failures are recorded and never retried. If a worker queue stops, only the declared unrun cells are completed, as in amendment 1 of the ablation study.

## Interpretation limits

- One task family and one learning rate.
- The primary setting deliberately replicates the development setting, so the confirmation tests *reproducibility on fresh data and seeds*, not generality. The secondary settings address generality.
- Related work on multiplicative interactions and bilinear or gated query formation must be reviewed before any novelty claim.

## Outcome — 2026-09-28, 05:51 UTC

All 35 runs completed and passed the freeze audit, and the locked test was evaluated once. **Neither C1 nor C2 is confirmed**: median correct prefix is 5, 5, 5, 5 and 6 for the Transformer, RDT, RDT wide, sync and current. Every recurrent cell uses its steps (7 of 7 seeds). Outcomes are bimodal by seed: a run escapes to the serial solution at 1–2 of 7 seeds for every recurrent cell. The development S₃ effect was most plausibly a chance fluctuation, and **the synchronization claim is withdrawn**. See [results](results/sync_confirmation_v1/RESULTS.md) and [interpretation](results/sync_confirmation_v1/INTERPRETATION.md).
