# Response to the external review (2026-11-21)

Tracks every concern in the review of `research/PAPER_DRAFT.md`, what settles it, and its status. Tests that can make
others unnecessary run first.

## Settled from existing data (no new runs)

- **Recall floor (minor).** Guessing uniformly among the distinct values present gives 19.3%; guessing by their frequency
  21.1% (N uniform on 2-16, 16 values). The draft's "about 32%, choosing among the values present" is wrong: unbound runs
  score about 50% at N = 2-4 and about 20% at N >= 10, so they partly solve small N. Fix the text.
- **Time to bind (concern 1).** Log-rank tests on updates to 90%, censored at 10,000: seeds 5-14 z = 2.86, p = 0.004;
  seeds 15-24 z = 3.25, p = 0.001; pooled confirmatory seeds 5-24 p = 1.2e-5. Binding happens at 4,000-5,000 updates or
  not at all (one Dans run bound at 9,000), so the fixed budget does not turn slow binders into failures.
- **Confirmatory p-value (concern 1).** Report the phase-5 replication alone (8/10 vs 1/10, p = 0.003) and seeds 5-24
  (15/20 vs 2/20), not the 25-seed total that includes the exploratory seeds 0-4.
- **CTsqS's learning rate (concern 1).** It went through the same phase-4 grid (3e-4, 1e-3, 2e-3) as Dans and CTsq.
- **Serial parity fit (concern 8).** With an intercept: slope 0.59, intercept -5.9; a quadratic term is significant
  (F = 41.4, p = 0.0007), so ticks per bit rise with length. On that curve 120 bits needs about 70 ticks; the 81-120 band's
  90% at 64 ticks is an average over the band. Report the curvature; drop "0.5 ticks per bit" as a constant.
- **Backprop through ticks (concern 4).** Not truncated in any model (`backprop_steps` None).
- **Lifelong task cue (minor).** Each task has its own start token (20-25), so forgetting is not the correct behaviour
  for an uncued input. State it, with the seed count (5).
- **The CMM's recall (concern 2).** The CMM paper's associative recall is a different task: N 2-6 items of three 6-bit
  vectors, about 200,000 updates (1,000 epochs of 6,400 episodes, batch 32), 500K-parameter models. Our CMM-task grid
  (N 2-16, 64 keys, 16 values, 5,000 updates) is not "the CMM budget"; fix the draft. The CMM paper itself reports the
  CTM behind the Transformer on recall (loss 3.2e-2 vs 2.7e-4) and copy (3.1e-1 vs near zero). The CMM code linked
  from the paper (github.com/SakanaAI/continuous-memory-machines) is not public as of 2026-11-21.

## Done (2026-11-23)

- **Recall controls (concern 1):** CTsqS 15/20 vs sink alone 1/20, first-order query 3/20 (both Holm p < 0.001); Dans with a
  small-tick sampler 5/20 vs Dans 2/20 (p = 0.20). The gain is second-order synchronization's. `recall_memory/CONTROL.md`.
- **LM learning-rate grid (concern 3):** 2e-3 best for Dans, CTsq, CTsqS, T8, T20 on a sqrt(2) grid.
- **One evaluation set (concern 5) and the state measure (concern 8):** the loss plateaus by 8 ticks, but the state keeps
  moving about 0.2% per tick to tick 96. `lm_unified/LM_UNIFIED.md`.
- **Writing:** revision draft with equations, variant legend, related work, corrected numbers, confirmatory statistics,
  figures 1-3 redrawn and figure 5 added (commit 6e48bb6).

## Done (2026-11-27)

- **Five LM seeds, TOST (concern 3):** CTsq +0.0062 and CTsqS +0.0091 against Dans; both meet the rule for "worse"
  (Holm p 0.039, 0.030) and both 90% CIs lie inside the 0.02 equivalence margin. The rules did not say which verdict
  takes precedence; the draft reports both. `lm_revision/LM_REVISION.md`.
- **Compute matching (concern 4):** T8x (Dans's parameters and training compute, 3.36B tokens, 3 seeds) 3.167, 0.408
  better than Dans; Dans does not beat it, so the T8 margin is per parameter only. T20 beats Dans at 4 ticks by 0.0135.
- **One evaluation set, five seeds (concern 5):** all 16 new checkpoints scored on the 1,024 windows; Figure 1 redrawn.
- **Reference CTM (concern 2):** the official code at our budgets binds recall in 0/15 (lr grid) and 0/20 (seeds 5-24),
  and learns none of copy, sort, parity. `reference_ctm/REFERENCE_CTM.md`.
- **Writing:** draft updated with all of the above (commit f3d0156).

## Running

| Concern | Test | Status |
|---|---|---|
| 2 | Reproduction of the CTM's published parity result (official script, 200,000 updates, seeds 0-2 in parallel) | started 2026-11-27, about 65 h |

## Not yet designed (marked Pending in the draft)

- Concern 1: attention maps and synchronization-query probes on saved phase-7 weights.
- Concern 2b: a warm-started (non-zero-init) variant, to test the basin bias.
- Concern 7: a second model size.
- Concern 8: selective-prediction AUROC for calibration.
- Concern 9: a harder recall-in-text variant (more pairs, interference).
- Concern 10: Appendix C, the commit that fixed each test's rules.
