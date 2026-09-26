# Plan: mechanism ablations and locked confirmation of the synchronization result

**Status: DRAFT for review, 2026-09-26. Nothing here is implemented or frozen.**

## The result to explain and confirm

In the [S₃ study](results/group_s3_v1/INTERPRETATION.md), adding CTM's sparse-decay synchronization of each position's recurrent-state history as a zero-initialized term on the core attention queries (`sync`) had these effects:
- It raised the median correct running-product prefix from 5 to 10 at T = 16. This was larger at every seed against both RDT and a width-matched RDT.
- It kept gaining with inference steps: median 3, 7, 10 and 11 at T = 4, 8, 16 and 32.

Two things are needed before this becomes a paper claim. The **ablations** say *why* it works. The **locked confirmation** shows *that* it works on data that shaped no decision.

## Order

1. **Mechanism ablations (development)** on the frozen study's exact data and seeds (47, 53, 59, 61, 67). Every ablation is then paired with the existing `sync`, `rdt` and `rdt_wide` runs, so no control needs retraining.
2. **Locked confirmation**, declared after the ablations: `sync`, the RDT controls and the most informative ablation. It uses new seeds, fresh data and a test set evaluated exactly once.

Running ablations first means the confirmed claim can say which component carries the effect, while the confirmation stays untouched by any analysis.

## Part 1: mechanism ablations

Each ablation changes one property of the `sync` cell and keeps everything else fixed: scaffold, 128 pairs, zero-initialized query projection, additive placement, T = 16, recipe, data and seeds. They are implemented in a new module (`ctm_transformer/sync_ablations.py`) that subclasses `SyncRDT`, so the frozen `sync_rdt.py` stays unchanged.

| Ablation | Change | Question |
|---|---|---|
| **A1 `sync_shuffled`** | the post-update history is randomly permuted along the time axis before synchronization, independently at each step | Does the temporal order of the history matter, or only its contents? |
| **A2 `sync_nodecay`** | decay rates fixed at r = 0, not learned | Does learned exponential decay (weighting recent steps) matter? |
| **A3 `sync_current`** | history length 1: pairwise products of the current state only | Does history matter at all, or do pairwise features of the current state suffice? |
| **A4 `history_linear`** | the 128 synchronization features are replaced by a zero-initialized linear projection of the flattened history (8 × 96 → 128), same placement | Are the pairwise *products* essential, or is any history-derived query signal enough? |
| **A5 `sync_state`** | the same synchronization term is added to the recurrent state instead of the queries | Does synchronization help by *steering attention*, or by adding information to the state? |
| **A6 `sync_replace`** | queries come from synchronization only, as in CTM | Can synchronization replace content queries, or only augment them? (It is expected to hurt retrieval-like composition.) |
| **A7 `sync_self`** | only self-pairs (i = i): the decayed energy of each channel | Is coupling between channels needed, or only per-channel magnitude? |
| **A8 `sync_rdt_lowgate`** | `sync_rdt` with the temporal-MLP gate initialized at −4 | Does CTM's gate initialization explain why adding history dilutes the sync effect? |

**Priority.** A1–A5 identify the mechanism; A6–A8 refine it. Each costs 5 seeds × about 1 hour, with three workers per GPU, so A1–A5 take about 5 hours of wall time and A6–A8 about 3.

**Endpoints and rules.** These use the S₃ study's endpoint and rule: correct prefix at T = 16; paired over seeds; one cell *exceeds* another if it is larger in at least 4 of 5 seeds and the median difference is at least 2 positions. For each ablation:
- **it removes the effect** if `sync` exceeds it;
- **it preserves the effect** if it exceeds both `rdt` and `rdt_wide`, as `sync` did;
- otherwise it is **partial**.

Each ablation also reports its step scaling (T = 4, 8, 16, 32) and its accuracy at positions 9–16.

**How the outcomes read:**
- If A1 removes the effect and A3 removes it, synchronization works through *temporal structure*, the property that distinguishes CTM from a plain recurrent state.
- If A4 preserves it, the benefit is simply "extra history information for attention", and the products are incidental.
- If A5 preserves it, "steering attention" is not the explanation.

## Part 2: locked confirmation

**Declared after Part 1, before any confirmation training.** The protocol and test generator are committed first.

- **Cells:** `rdt`, `rdt_wide`, `sync`, the most informative ablation from Part 1, and the `transformer` as a cheap fixed-depth reference.
- **Seeds:** seven new seeds (97, 101, 103, 107, 109, 113, 127), never used before. Rule: exceeds in at least 6 of 7 seeds, with a median difference of at least 2 positions.
- **Data:** fresh training, validation and **locked test** words from a new generator seed. The test set is evaluated **once**, after every checkpoint is frozen. Validation shapes nothing except the secondary best-checkpoint report.
- **Settings.** The primary setting is the exact replication: S₃, width 96, training lengths 1–16. Two secondary settings test generality, each gated on a short development probe showing it is learnable:
  - **width 128** for every cell, as a scale check;
  - **training lengths 1–24**, a harder state-tracking range.
- **Primary claim:** in the replication setting, `sync` exceeds both `rdt` and `rdt_wide` on the locked test set. The secondary settings are reported whatever their outcome.

**Cost:** about 5 cells × 7 seeds per setting, roughly 5–6 hours per setting.

## After confirmation

- **Bio factor** (homeostasis, structural plasticity), applied to `rdt` and `sync`, including an SGD vs AdamW contrast.
- **A₅ or a longer-range curriculum,** to test whether synchronization helps where plain recurrence fails completely.
- **Language-model pilot** with `rdt`, `sync` and the reference CTM.

## Open questions for review

1. **Order:** ablations first (proposed), or confirmation first?
2. **Ablation set:** A1–A5 first (proposed), or all eight at once?
3. **Confirmation:** seven seeds with a 6-of-7 rule (proposed), or five seeds with 4-of-5 as before?
4. **Secondary settings:** include width 128 and training lengths 1–24 (proposed), or replication only?
