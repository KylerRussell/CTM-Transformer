# Design draft: a serial-depth task for recurrent computation

**Status: DRAFT for review, 2026-09-26. Nothing here is implemented or frozen.**

## What the task must do

The retrieval work showed that one-hop in-context lookup is a poor test of CTM's mechanisms:
- it needs one composition of attention heads, not iteration;
- learning it depends on a fragile, seed-dependent escape from a shortcut plateau;
- the reference CTM cannot do it for structural reasons that are unrelated to synchronization.

The main task should therefore:

1. **Require serial computation** whose depth grows with a difficulty knob, so a fixed-depth model must fail beyond some difficulty while an iterating model can keep going with more steps.
2. **Not require in-context retrieval first.** Inputs should be usable directly, so every family, including the standalone CTM, can attempt it on its own terms.
3. **Give dense supervision** (many labels per sequence) and no shortcut basin: no exclusion or frequency leak, and chance exactly 1/|outputs|.
4. **Have a clean theoretical reason to separate the families,** so a result is interpretable and not just a benchmark number.

## Proposed task: the group word problem (running products)

The input is a sequence of group elements g₁ … gₙ. At every position i the target is the running product pᵢ = g₁·g₂·…·gᵢ. This is **state tracking**: the model must carry an evolving state across the sequence.

| Group | Size | Structure | Why include it |
|---|---:|---|---|
| Z₂ | 2 | abelian | cumulative parity: the simplest state, and a known length-generalization failure for Transformers; the CTM paper also uses a parity task |
| S₃ | 6 | smallest non-abelian (solvable) | order matters; still within what constant-depth networks can express in principle |
| **A₅** | 60 | smallest non-solvable | its word problem is **NC¹-complete** (Barrington, 1989). Under standard complexity conjectures, fixed-depth log-precision Transformers (TC⁰) cannot solve it for all lengths, and the required depth grows about as log n. This is the theoretical case for recurrence. |

This is the setting studied in [Liu et al., *Transformers Learn Shortcuts to Automata* (ICLR 2023)](https://arxiv.org/abs/2210.10749) and [Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024)](https://arxiv.org/abs/2404.08819). Both relate the depth needed for non-solvable groups to sequence length.

**Why it fits each family:**
- **The fixed-depth Transformer** should succeed at short lengths and fail as n grows, especially for A₅.
- **RDT and Sync-RDT** can apply more recurrence steps. If they learn an iterative algorithm, such as parallel prefix composition (about log₂ n steps) or sequential composition, accuracy at long lengths should improve with more steps at inference.
- **The standalone CTM** is a fair participant here. Every position can read all earlier input tokens directly (static keys are enough), and its thought ticks could accumulate the product. So neither of its retrieval limitations applies.

## Format

Each group element maps to one character of the existing tokenizer: 62 alphanumerics, so A₅'s 60 elements fit. The product is always computed left to right with a fixed convention.

```
input : <BOS> g1 g2 … gn =
target:  —    p1 p2 … pn <EOS>
```

This is **sequence labeling**: the label at gᵢ's position is pᵢ, which is not the next input token. So no earlier product is ever visible, and there is no teacher-forcing leak. This needs a small labeling dataset with the existing batch interface; runner v3 and `evaluate_dense` then work unchanged. Scoring is per position and teacher-forced. It reports accuracy by position i, since the depth needed grows with i, and whole-sequence exact match.

**Difficulty knob.** Train on lengths 1 … L_train, with lengths balanced. Evaluate at lengths up to about 4 × L_train, which is unseen, within the models' 128-position window. Accuracy by position beyond L_train measures length and depth generalization. Accuracy against inference steps (4, 8, 16, 32) measures whether extra recurrence buys longer correct prefixes.

**Data.** Sequences are drawn uniformly at random, so each training sequence is fresh and memorization is impossible. Each seed gets disjoint validation and evaluation sets. Supervision is dense (n labels per sequence), which avoided plateaus in the retrieval probes.

## Families and cells

- **Transformer** (fixed depth, 2 layers).
- **RDT** and **width-matched RDT**. The latter was the most reliable baseline in the retrieval study.
- **Sync-RDT cells** (`history`, `sync`, `sync_rdt`), the mechanism tests.
- **Standalone reference CTM**, which this task does not structurally disadvantage.

All at LR 0.0003 (the rate at which RDT learns). Parameter counts and measured training and inference cost are reported with every result.

## Staged plan

1. **Implement** the labeling dataset and generator (Z₂, S₃, A₅), with tests: correct products under a fixed composition convention, determinism, disjoint splits, and no label visible in the input.
2. **Development probes (not endpoints).** Transformer, RDT and CTM on S₃ and A₅, one seed each, L_train = 16, evaluation up to 64, 10,000 updates. The purpose is to check learnability at trained lengths and find where the Transformer's accuracy falls off.
3. **Frozen calibration and operating point.** This applies the rules already fixed in the [calibration protocol](POINTER_CALIBRATION.md#operating-point-rules-for-stages-1b-and-2-fixed-now):
   - choose the group and length range so that the mean of the recurrent families' accuracy lies between 20% and 80% at two or more evaluated lengths;
   - no family may be at 95% or higher everywhere;
   - selection uses calibration seeds only.
4. **Frozen comparison.** All cells × at least 5 seeds (47, 53, 59 plus two new), reporting escape counts, given how strongly outcomes depend on seed.
   - **Primary endpoints:** accuracy by length beyond L_train; accuracy against inference steps; updates to reach thresholds.
   - **Secondary:** cost.
5. **Bio factor** (homeostasis, structural plasticity) on RDT and the best cell, then the language-model pilot.

## Alternatives considered

- **Multi-hop pointer chasing with a hop curriculum.** It is relevant to language models, but every hop requires in-context retrieval, which is fragile to learn here and structurally unavailable to the reference CTM.
- **Prefix sums or parity only.** Z₂ is included, but on its own it cannot separate fixed and recurrent depth cleanly, because abelian state is expressible in constant depth by counting.
- **Mazes or graph reachability.** Closer to the CTM paper's showcase, but it needs a 2-D or graph encoding and a larger implementation effort. It remains a later option if the word problem is not informative.

## Open questions for review

1. **Groups:** S₃ and A₅ (proposed), plus Z₂ as a sanity case, or A₅ only?
2. **Training lengths:** L_train = 16 with evaluation to 64 (proposed), or a smaller L_train, so that the failure point is visible earlier.
3. **Include the standalone CTM** (proposed: yes, since the task does not disadvantage it structurally), or Sync-RDT cells only?
4. **Label every position** (proposed; dense), or only the final product (sparse, closer to "answer only", but likely to hit the plateau problem again)?
