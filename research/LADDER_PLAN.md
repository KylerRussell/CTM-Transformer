# Scaling ladder plan, with a code-and-math probe first (draft, 2026-10-28)

**Goal:** compare the CTM Transformer's best against the RDT's best, with the plain Transformer as the baseline, across model sizes. Every arm gets:
- matched sizes;
- a learning rate tuned per size;
- a 1-tick twin;
- replicate seeds.

Nothing here has run. Costs are estimates from the measured 400M-token runs, and every rule below is to be fixed before its runs.

## Arms

| Arm | Model | Why |
|---|---|---|
| Transformer | the 8-layer Transformer at each width (T8-shaped) | the baseline, and the 1-tick twin of both recurrent arms |
| RDT | Dans (2 prelude + 4 thinking + 2 coda, randomized ticks) | the RDT's best |
| CTM Transformer | CTsq (Dans + synchronization queries) | the CTM Transformer's best |

Reasons for these arms:
- **CTsq:** its learning rate has been checked, it trains without gradient spikes, and it ties Dans and CTnlm. Its synchronization queries carry the one CTM-style capability found: tick-scaling parity extrapolation, `research/CTM_TRANSFORMER_DESIGN.md`.
- **One twin for both recurrent arms:** at one tick, CTsq's synchronization queries read nothing and CTsq equals Dans, which equals the 8-layer Transformer.
- **Optional:** a compute-matched Transformer (T20-shaped, about the compute of 4 ticks) at the lower rungs.

## Stage 0: a small-scale code and math probe (before the ladder)

**Motivation:** the language-model comparison so far uses FineWeb-Edu web text. The CTM Transformer ties Dans there, uniformly across the token categories tested (`research/results/capability_eval`). Code and math need multi-step structure, and differences between the additions might show there instead.

**Data:**
- every corpus below is public and reachable anonymously from the Hugging Face hub (checked 2026-10-28);
- The Stack is gated, so the code corpus is codeparrot-clean (Python);
- all text uses our 32k tokenizer (trained on FineWeb-Edu); code tokenizes less efficiently, equally for every model.

**Models:** T8, Dans, CTsq, CTnlm, CTso and CTsqH, each with its two existing 400M-token seeds (all checkpoints are on disk). T20 has one seed, so it is reported but not judged by the rules.

### 0A. Zero-shot domain loss on the existing checkpoints (about 3 hours)

Held-out loss on four sets:

| Set | What is scored |
|---|---|
| FineMath-4+ | math web text, 1M tokens |
| codeparrot-clean | Python, 1M tokens |
| GSM8K test | the solution tokens, given the question |
| MBPP and HumanEval | the reference solution's tokens, given the prompt |

These are out of distribution for FineWeb-Edu-trained models, so absolute losses will be high; the comparison is relative.

### 0B. Continued pretraining on math and code (about 1 day on both GPUs)

- **Training:** each checkpoint trains on 100M more tokens of a 50/50 FineMath-4+ / codeparrot-clean mix: 381 steps of 262,144 tokens, warmup 50, cosine to 10%, peak lr 1e-3 (half the pretraining peak), everything else as in pretraining.
- **Evaluation:** the 0A measures, plus FineWeb-Edu held-out loss to measure forgetting.
- **Question:** which models learn structured domains fastest from the same starting point and data.

### 0C. Synthetic arithmetic and program tracing, from scratch (about half a day)

Width 128, the `scripts/cmm_tasks.py` harness, 3 seeds, 5,000 updates. Each task has in-distribution (ID) and out-of-distribution (OOD) splits:

| Task | Train | OOD test |
|---|---|---|
| multi-digit addition (answer digits, least significant first) | up to 10 digits | 11–20 digits |
| nested arithmetic expressions mod 7 (`(3+(4*2))*5`) | nesting depth 1–3 | depth 4–6 |
| variable tracking, two interleaved chains (`a=3;b=a+4;c=b*2;?c` mod 7) | chains of 1–4 | chains of 5–8 |

Arithmetic is mod 7 with nonzero leaves: mod 10, products collapse to 0 too often and deep expressions become shallow.

Models: T8, T20, Dans, CTsq, CTnlm and CTsqH. Recurrent models are also evaluated at 32 and 64 ticks: depth extrapolation is where recurrence, and the CTM's ticks, should matter.

Implementation: `scripts/code_math_data.py` (data), `scripts/code_math_probe.py` (0A, 0B), `scripts/code_math_tasks.py` (0C) and `scripts/run_stage0.sh`. The script docstrings hold the exact rules.

### Rules for stage 0 (to be fixed in the scripts before any run)

- **0A / 0B:** a model differs from Dans on a measure if:
  - its two-seed mean differs by at least 0.02 nats;
  - both seeds lie on the same side of both Dans seeds.
- **0A / 0B, capability-specific:** an advantage if both seeds' excess (the measure's difference minus the FineWeb-Edu difference) is at least 0.02 in its favour. Dans against T8 is judged the same way.
- **0C:** as in the CMM tasks. A model beats another on a task split if:
  - its 3-seed mean sequence accuracy is at least 5 points higher;
  - it is higher on all 3 seeds.
- **0C, ticks:** a recurrent model extrapolates with ticks on a task if its OOD mean at 64 ticks is at least 10 points above its mean at 16.
- **Use of the outcome:** stage 0 can change which CTM Transformer variant represents the arm on the ladder:
  - a variant replaces CTsq if it beats CTsq on at least two of the 0B code / math measures and 0C task-splits together, and loses on none (0A is out of domain, so it is judged but not used here);
  - if several qualify, the one with the most wins;
  - if no variant differs, CTsq stays;
  - the ladder then adds the 0A measures to its evaluation.

## Stage 0.5: can training use fewer ticks? (about 15 hours on both GPUs)

- **Why:** the cost of the recurrent arms is in the thinking layers, run a mean of 15 ticks per update. Every recurrent model's loss is flat from about 8 ticks on (capability evaluation, M4).
- **Runs:** Dans and CTsq with the tick sampler at mean 8 (at most 24), two seeds each, 400M tokens.
- **Rule:** the ladder uses mean 8 if both arms' two-seed means are within 0.01 of their mean-15 values (Dans 3.574, CTsq 3.582); otherwise mean 15.
- **Saving:** mean 8 would cut recurrent cost to about 56% of mean 15: (4 + 4 × 8) / (4 + 4 × 15) block applications per token.

## The ladder

Widths keep the 2 + 4 + 2 layout, the FFN at about 2.64 × d, and heads of 64:

| Rung | d | Non-embedding parameters | Tokens | Status |
|---|---:|---:|---:|---|
| R1 | 704 | 47M | 400M | done (2 seeds per arm) |
| R2 | 1,024 | about 100M | 800M | |
| R3 | 1,408 | about 190M | 1B | |
| R4 | 2,304 | about 505M | 1B | the user's minimum size (2026-10-01) |

**Learning rate:** at R1 the best of {1e-3, 2e-3, 4e-3} was 2e-3 for the Transformer, Dans and CTsq. At each later rung:
- seed 1 runs at the previous rung's best rate and at half of it;
- seed 2 runs at the better of the two;
- R4 uses one rate per arm, extrapolated from R1–R3 (the fit of `research/LR_SCALING.md`, log₂ lr* linear in log₂ N);
- a rung's rate counts as resolved when the better of its two rates is the lower one, or when both are within 0.01. Otherwise a third rate (half again) runs.

**Evaluation at every rung:**
- the held-out FineWeb-Edu loss at 16 ticks (the primary measure, as at R1);
- loss by tick count (1–96);
- the capability breakdown;
- the stage-0A code and math measures;
- adaptive compute (certainty exit).

**Rules (as at R1):**
- on two-seed means, the CTM Transformer beats Dans at a rung if it is at least 0.02 lower;
- either recurrent arm beats the Transformer twin if at least 0.02 lower;
- across rungs, the gaps (CTsq − Dans and Dans − twin) are fitted against log N to say whether they grow, shrink or hold with scale.

## Cost (single-GPU hours; two GPUs run two at a time)

Estimated from the R1 runs, scaled by parameters × tokens. Tick mean 8 also scales the recurrent cost by 0.56. These are conservative, since wider models use the GPU more efficiently. At R4, AdamW state alone is about 9 GB and activations grow with width, so the recurrent arms need a smaller micro-batch with more gradient accumulation; the number of tokens per update stays the same.

| Rung | Twin | Dans (mean 15 / 8) | CTsq (mean 15 / 8) | One seed of all three, mean 8 |
|---|---:|---:|---:|---:|
| R2 (100M, 800M tokens) | 9 h | 43 / 24 h | 66 / 37 h | 70 h |
| R3 (190M, 1B) | 22 h | 102 / 57 h | 156 / 88 h | 167 h |
| R4 (505M, 1B) | 59 h | 273 / 154 h | 418 / 235 h | 448 h |

**Budget options** (both GPUs, tick mean 8; about 1.7× longer at mean 15):

| Option | Content | Wall-clock |
|---|---|---:|
| A: full | R2 and R3 with two rates + second seed; R4 with two seeds | about 33 days |
| B: one seed at the top | as A, but R4 one seed | about 24 days |
| C: no 500M rung | R2 and R3 only, with an R4 decision after the trend is seen | about 15 days |

Stage 0 and stage 0.5 together add about 2 days.

## Decisions needed

1. Stage 0's scope: all of 0A–0C, or 0A first and the rest depending on its result.
2. The budget option (A, B or C) and whether the 500M minimum still holds.
3. Whether to add the compute-matched Transformer at R2–R3.

## Stage 0 outcome (2026-10-29)

Results: `research/results/code_math/CODE_MATH.md` (0A, 0B, the selection) and `research/results/code_math_tasks/CODE_MATH_TASKS.md` (0C).

- **0A (zero-shot):**
  - Dans beats T8 on every measure.
  - CTsq, CTnlm and CTso trail Dans on HumanEval (by 0.045–0.063).
  - CTsqH is ahead of CTsq on MBPP and HumanEval, but behind Dans on GSM8K.
- **0B (after 100M tokens of math and code):**
  - CTsq, CTnlm and CTso tie Dans on every measure (within 0.01 on math and code); the 0A HumanEval gaps close.
  - CTsqH is behind Dans on MBPP (+0.027).
  - No variant differs from CTsq.
  - Dans beats T8 on every measure. T20 (one seed) is best on every measure.
  - Forgetting of web text is the same for all models (+0.37 to +0.40).
  - Loss stops improving by 8 ticks on math and code too.
- **0C (synthetic tasks):** no model learned these tasks, so 0C cannot separate the models and the tie is a floor, not a result:
  - OOD accuracy is at chance for every model: 0% on addition; 16–18% on expression and tracking, where chance is 1/7 = 14%.
  - In distribution, accuracy is the same floor for every model: about 11% on addition (consistent with solving only 1-digit sums) and about 38% on tracking.
  - No model gains with more ticks.
- **Selection:** no variant beats CTsq on any 0B measure or 0C task-split, so CTsq stays the ladder's CTM Transformer arm.
