# Sync-RDT ablation A7: self-pairs only

## Protocol declared before new scientific training — 2026-09-27

### Question

Ablations A1–A5 ([results](results/sync_ablation_v1/INTERPRETATION.md)) showed that the `sync` cell's gain comes from **pairwise products of the recurrent state in the attention queries**, not from temporal history or decay. The linear-feature ablation removed the effect, and current-state-only synchronization preserved it. A7 asks whether the products must **couple different channels**, or whether each channel's own square (energy) is enough.

### Ablation

`self` (`ctm_transformer/sync_ablation_self.py`) is identical to `sync` except that each of the 128 synchronization pairs uses the same channel twice (i, i). Every feature is then one channel's decayed energy over the 8-step state history, and no feature couples two channels. The pairs' first indices, the decay parameters, the zero-initialized query projections, the placement and the parameter count (550,688) are unchanged. The frozen `sync_ablations.py` is not modified.

### Data, seeds, recipe and controls

These are the S₃ study's exact data and seeds (47, 53, 59, 61, 67), with the `sync` recipe (LR 0.0003, 10,000 updates, T = 16, confidence readout, runner v3). The controls are not retrained: they are the frozen `sync`, `rdt` and `rdt_wide` runs and evaluations from `group_s3_v1`, verified by hash.

### Endpoint and declared classification

As for A1–A5, the endpoint is the correct prefix at T = 16 on held-out length-32 words. The exceed rule is at least 4 of 5 seeds and a median difference of at least 2 positions. A7 is classified as:
- **removes** if `sync` exceeds `self`: cross-channel coupling is needed;
- **preserves** if `self` exceeds both `rdt` and `rdt_wide`: per-channel energy suffices;
- **partial** otherwise.

Secondary: whether it uses its steps (T = 16 against T = 4), mean accuracy at positions 9–16, and step scaling.

### Verification and execution

Before freezing:
- the A7 checks: only the pair indices differ from `sync`; each feature equals the channel's energy; causality, gradients and round trips;
- the A1–A5 ablation checks;
- the Sync-RDT checks;
- the S₃ study checks;
- the archived-trainer replay, on both GPUs.

The freeze audit is the A1–A5 audit with the A7 factory and worker. `research/launch_sync_ablation_v2.sh` starts the restart-safe supervisor, with autostart after restarts. There are five runs, one per worker slot, taking about 1 hour.

### Interpretation limits

Development data and seeds already used by the S₃ study; five seeds. Self-pairs are computed over the 8-step history, as in `sync`. A3 showed that history is not needed, so a current-state self-pair variant would give the same answer to this question, and it is not run.

## Outcome — 2026-09-27, 12:27 UTC

All five runs completed and passed the freeze audit. **A7 is partial**: median correct prefix 6 (15, 3, 6, 5, 11). `sync` was not larger at the declared margin (3/5, +1), and A7 did not exceed both RDT controls. Per-channel energy matches `sync` at two seeds and stays near RDT at three; cross-channel products are the reliable form. See [results](results/sync_ablation_v2/RESULTS.md) and [interpretation](results/sync_ablation_v2/INTERPRETATION.md).
