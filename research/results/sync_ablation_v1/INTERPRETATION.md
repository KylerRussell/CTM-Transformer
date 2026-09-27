# Interpretation of the Sync-RDT mechanism ablations (A1–A5)

## Finding under the declared rules

Each ablation changes one property of the `sync` cell. They were trained on the S₃ study's data and seeds (47, 53, 59, 61, 67) and paired with its frozen `sync`, `rdt` and `rdt_wide` runs. The primary endpoint is the correct running-product prefix at T = 16 on held-out length-32 words. The declared classification is:
- *removes*: `sync` exceeds the ablation;
- *preserves*: the ablation exceeds both RDT controls;
- *partial*: otherwise.

The exceed rule is at least 4 of 5 seeds and a median difference of at least 2 positions.

| Ablation | Change | Correct prefix per seed | Median | Classification |
|---|---|---|---:|---|
| (reference) `sync` | synchronization of the 8-step state history → query term | 12, 10, 7, 16, 10 | 10 | — |
| (controls) `rdt` / `rdt_wide` | no synchronization | — | 5 / 5 | — |
| A2 `nodecay` | decay frozen at 0 | 11, 5, 15, 16, 11 | 11 | **preserves** |
| A3 `current` | newest state only, no history | 3, 9, 14, 15, 11 | 11 | **preserves** |
| A1 `shuffled` | history order randomized every step | 11, 7, 11, 9, 7 | 9 | **removes** (by rule; see below) |
| A4 `linear` | linear history features, no pairwise products | 6, 3, 4, 3, 3 | 3 | **removes** |
| A5 `state` | same term added to the state, not the queries | — | — | **diverged at every seed** ([amendment 1](AMENDMENT_1.md)) |

## What carries the effect

1. **Pairwise products are essential.** Replacing them with a linear projection of the same 8-step history (A4) removes the effect: sync is larger at 5 of 5 seeds, median +7. It also falls below plain RDT (median 3 against 5), even though A4 adds 98,304 parameters. History information delivered linearly to the queries does not help, and here it hurts.
2. **History is not needed.** Synchronization computed from the **current state only** (A3: pairwise products of the newest state's channels, no temporal window) preserves the effect, with median 11 against sync's 10. The 8-step history, CTM's defining temporal ingredient, is not what helps.
3. **Learned decay is not needed** (A2 preserves, median 11). This is consistent with the near-zero decay rates learned by the `sync` cell, recorded before training.
4. **Placement is essential.** The same term added to the recurrent state diverges at every seed within 200–460 updates (A5). On the queries it is bounded by the attention softmax and helps. Added to the state, it feeds back quadratically.
5. **Randomly shuffling the history (A1) costs some of the effect but not most of it.** By the declared rule, A1 is classified as *removes* (sync is larger at 4 of 5 seeds, median +3). A1 also still exceeds both RDT controls (4 of 5 seeds, median +4), and the declared rule reports *removes* first. Given (2), the loss is more plausibly caused by the per-step random permutation injecting noise into the query features than by lost temporal order. This is an inference, not a tested explanation.

## Reframing the claim

The S₃ study found that "synchronization-derived query terms extend serial state tracking". These ablations narrow what that means: **second-order (pairwise multiplicative) features of the current recurrent state, used to form attention queries, roughly double how far recurrent depth tracks state at the same step budget.** CTM's synchronization is one way to compute such features. Its temporal machinery (history window, decay, ordering) does not carry the benefit in this setting.

This claim is more specific and more defensible than "CTM synchronization helps". It also changes the paper's framing. The contribution is not evidence that CTM's temporal dynamics aid recurrent computation. It is evidence that multiplicative state interactions in query formation make recurrent depth more effective, identified by starting from CTM and removing what does not matter.

## Limits

- Development data and seeds already used by the S₃ study, so this is not confirmatory evidence.
- One task (S₃), one scale, one learning rate, five seeds.
- The A4 and A5 parameter counts differ from `sync`.
- Not yet tested: whether cross-channel coupling is needed or per-channel squares suffice (A7, next); whether synchronization can *replace* content queries (A6); and the history dilution in `sync_rdt` (A8).
- Related work on multiplicative interactions and bilinear or gated query formation must be reviewed before any novelty claim.

## Consequence for the plan

1. **A7 (self-pairs only)**, as a declared development ablation, to test whether cross-channel products are needed.
2. **Locked confirmation (part 2)**, declared after A7: the cells are `sync`, `current` (the simplest variant that works) and whichever A7 form is relevant, with `rdt`, `rdt_wide` and the `transformer` as references. It uses seven new seeds and a test set evaluated once.

See [results](RESULTS.md), [frozen protocol](PLAN_BEFORE_RUNS.md) and [amendment 1](AMENDMENT_1.md).
