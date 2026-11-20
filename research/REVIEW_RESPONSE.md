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

## Running

| Concern | Test | Script | Status |
|---|---|---|---|
| 1 | Recall controls, stage 1: Dans + sink (DansS), Dans with a small-tick sampler (DansT4), seeds 5-24 | `scripts/recall_memory.py` phase 7 | running |
| 1 | Stage 2 by the stage-1 rules: first-order query control (CTsq1S); CTsqS with the small sampler (CTsqST4) | same | launches automatically |
| 5, 8 | Every LM on the 1,024 final windows, ticks 1-96, state change per tick | `scripts/lm_unified_eval.py` | running |
| 3, 4 | Finer lr grid (1.4e-3, 2.8e-3), then 5 seeds, TOST; T8x (compute-matched) | `scripts/lm_revision.py` | queued (GPU 1 now, GPU 0 after recall stage 2) |

## To do

- Concern 1: attention maps and sync-query probes on saved phase-7 weights (target chosen by the controls' outcome).
- Concern 2: reference CTM (official code, github.com/SakanaAI/continuous-thought-machines) on our recall task and
  its published parity setup; claims reworded to "as merged here".
- Concern 7: a second model size.
- Writing: equations, related work (Fan et al.; Schwarzschild et al.; McLeish et al.; Dehghani et al.; Banino et al.;
  Saunshi et al.; Schlag et al.; Xiao et al.), variant legend, one evaluation set, scale qualifiers, Holm across test
  families, commit hashes for the pre-registration, figure fixes, restructure as a question-led paper.
