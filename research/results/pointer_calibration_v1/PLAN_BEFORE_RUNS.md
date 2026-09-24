# Pointer calibration: learnable budget first, then a discriminating operating point

## Protocol declared before new scientific training — 2026-09-24

### Motivation

Two findings lead here. In the [presentation control](results/presentation_control_v1/INTERPRETATION.md), every family memorized or underfit a small repeated map set. The [stopped fresh-map study](ONLINE_POINTER.md#stopped-at-user-direction--2026-09-24-1900-utc) showed that with fresh 12-node maps the Transformer sits exactly on the chance plateau for all 3,000 updates: training CE is 1.205, and ln 11 / 2 = 1.199. Retrieval with one query per example and answer-only supervision is therefore not learned at the budgets used so far.

The eventual scientific target is **CTM versus recurrent depth (RDT)**. That comparison is only informative at an operating point where both families are off the ceiling and off the floor. If all families reach 100%, the result only shows that the curriculum is learnable. If all sit at chance, it shows nothing. This protocol therefore runs in stages:

- **Stage 1a (this launch).** A cheap Transformer optimizer/budget sweep establishes whether, and how quickly, this format is learnable with fresh data. It also shows how a fixed-depth reference degrades with hop count.
- **Stage 1b (declared separately after 1a).** A calibration of CTM and RDT, on calibration seeds only, at the budget from Stage 1a.
- **Stage 2 (declared separately after 1b).** The CTM-versus-RDT comparison, on new seeds, at the operating point selected by the rules below.

All data is development data. This protocol defines no test set.

### Stage 1a design

- **Model:** Transformer only, with the existing recipe (548,660 parameters, final CE, final readout, 2 layers).
- **Tasks:** fresh one-hop, and fresh multi-hop with hops 1–4 balanced.
- **Peak learning rate:** 0.0003, 0.001 or 0.003.
- **Calibration seeds:** 41 and 43. These are new and disjoint from any later Stage 2 seeds.
- **Runs:** 2 × 3 × 2 = 12.

Each run uses the frozen shared runner without code changes:
- 30,000 updates, batch 32, and 300 warmup updates (1% of the budget, raised from 30 because the budget is 10× longer and the grid includes 0.003);
- cosine decay to 10% of peak, the existing AdamW settings, FP32 parameters and moments, and BF16 autocast;
- validation every 1,000 updates, with every validation checkpoint kept.

Only learning rate, budget, warmup and validation interval differ from the recipe.

### Data (`research/data/pointer_calibration_v1`, generator seed 20260926)

Maps are single 12-cycles (chance 1/11 = 9.09%) with random noncanonical presentations. Every map appears in exactly one split and in none of the earlier 12-node data, including the stopped study. Each training file holds exactly 30,000 × 32 = 960,000 unique maps, **so every training map is presented exactly once**.

| Split | Maps | Hops |
|---|---:|---|
| `train_fresh_onehop` | 960,000 | 1 |
| `train_fresh_multihop` | 960,000 | 1–4 (240,000 each) |
| `validation` | 384 | 1–4 (96 each), shared by all runs |
| `eval_id` | 1,024 | 1–4 (256 each) |
| `eval_depth` | 1,024 | 5–8 (256 each), unseen depths |

The two training files are deterministic from the manifest seed. They are hashed in the pre-run record but not archived.

### Endpoints and decision rules (fixed now)

All endpoints use the final checkpoint at update 30,000, frozen before evaluation, with the final readout.

- **C1: learnability.** One-hop hop-1 accuracy on `eval_id`, per learning rate and seed, plus the first validation update at which hop-1 accuracy reaches 90%.
- **C2: fixed-depth profile.** For multi-hop training, accuracy by hop 1–8.
- **C3: learning curves.** Validation accuracy by hop against updates.

**D1: learning rate.** Choose the learning rate with the highest mean final one-hop hop-1 accuracy; ties go to the lower rate.

**D2: format gate.** The format is learnable if the D1 learning rate reaches a mean of at least 90% and every seed reaches at least 80%. If it is not learnable within 30,000 updates at any learning rate, stop using this format. The next step would be denser supervision (several queries answered per sequence), which needs a declared dataset/tokenizer extension.

### Operating-point rules for Stages 1b and 2 (fixed now)

These rules are fixed before any CTM or RDT calibration, so the operating point cannot be chosen to favor either family.

1. **Budget.** Stage 1b uses the D1 learning rate as the Transformer reference and 30,000 updates. CTM and RDT start from their existing learning rates. Stage 1b declares any learning-rate check for them.
2. **Difficulty band.** Stage 2 uses a training hop range and evaluation hops such that, in Stage 1b, the **mean of CTM and RDT** accuracy lies between 20% and 80% at two or more evaluated hop counts. Also, neither family may be at 95% or higher on every evaluated hop. The band is defined on the two-family mean, so no family-specific result selects it. C2 shows where the fixed-depth Transformer fails, which is where recurrence should matter.
3. **Non-saturating primary endpoints for Stage 2.** Stage 2 does not use a single accuracy at trained difficulty, which can saturate. Instead it uses:
   - accuracy by hop across the declared band, including unseen depths;
   - accuracy against inference tick budget;
   - sample efficiency, measured as the first validation update reaching fixed thresholds, from a common validation schedule.

   Accuracy against measured training and inference cost is reported alongside.
4. **Seeds.** Stage 2 uses seeds 47, 53 and 59, disjoint from the calibration seeds. Calibration outcomes are never pooled into Stage 2 results.

### Cost note

At the measured 12-node cost, 30,000 updates take about 17 minutes for the Transformer, 2.8 hours for RDT and 5.7 hours for CTM per run. Stage 1a is about 2 hours of wall time on both GPUs. A three-seed Stage 2 at 30,000 updates would be about 13 hours. Stage 1b will state its exact cost before launch.

### Verification and execution

Checks that must pass before freezing:
- the scaled-down suite is deterministic, hop-balanced, disjoint and excludes prior maps;
- a leaked training map is detected even after its manifest hash is updated;
- the helper for the first update reaching a threshold is correct.

The exact archived-trainer replay checks also run on both GPUs, before freezing and before every supervisor attempt.

The freeze audit checks, for every run:
- all 30,000 metric rows and the warmup/cosine learning-rate schedule;
- finite losses and gradients, and per-step example, token and label counts;
- all 30 validation checkpoints;
- data, configuration and code hashes, and that each map was presented once.

`research/launch_pointer_calibration.sh` starts the restart-safe supervisor (`scripts/study_supervisor.py`) detached, and an autostart entry resumes it after a container restart. GPU 0 runs seed 41 and GPU 1 runs seed 43. A genuine worker error is recorded and not retried; nothing is adaptively added or replaced.

### Interpretation limits

One query per map, answer-only supervision and the Transformer recipe's architecture are kept. Learning-rate transfer from the Transformer to CTM or RDT is not assumed. The sweep calibrates the task and budget, not the recurrent families' optimizers. Two seeds per cell suffice for calibration, but not for any architecture claim.
