# S3 randomized-depth training versus fixed depth

## Protocol declared before new scientific training — 2026-09-28

This protocol was declared while the [reliability study](RELIABILITY_S3.md) was still training, before any of its results were seen.

### Question

Every recurrent model trained at a fixed T = 16 used its steps. But none escaped to the serial solution reliably or improved beyond the trained step count, and none extrapolated past the trained length. The recurrent-depth literature trains with **randomized recurrence depth**:
- Geiping et al. (arXiv 2502.05171) sample it from a log-normal Poisson distribution.
- Parcae (arXiv 2604.12946) reports that test-time gains plateau near the training mean.

Does randomized-depth training, at the same expected compute, make escape to the serial solution more reliable, or improve accuracy beyond the trained length when inference uses T ≥ n? This is step 2 of the re-prioritized pre-LM plan.

### Design

- **Cells:**
  - `rdt_rand`: recurrent depth, factory `cell_factory[rdt]`, final CE, confidence readout;
  - `ctm_lm_rand`: CTM-LM, CTM's min-loss plus max-certainty loss, confidence readout.
- **Training depth:** one draw per update from **log-normal Poisson**, `τ ~ Normal(log 15 − σ²/2, σ)`, `depth = min(Poisson(e^τ) + 1, 48)` with σ = 0.5. The expected depth is about 16, matching the fixed T = 16 controls in expected compute, with the cap at 48 for memory. Draws come from the runner's seeded depth generator (seed + 2) and are audited against the sampler.
- **Backpropagation through every sampled step.** There is no truncation: Parcae reports that truncating hurts extrapolation, and CTM's loss selects ticks per token.
- **Runner v4** (`ctm_transformer/dense_experiment_v4.py`): runner v3 plus the injectable depth sampler. Without a sampler it reproduces v3 exactly (`tests/test_depth_sampling.py`).
- **Seeds and data:** the **same 20 seeds and data** as the reliability study (`research/data/reliability_s3_v1`). Everything else (LR 0.0003, 10,000 updates, batch 32, recipes) is identical.
- **Controls, not retrained:** the reliability study's fixed-T = 16 `rdt` and `ctm_lm` runs, paired by seed and data and verified by hash.
- **Evaluation:** held-out length-32 words, read once after the freeze, at T = 4, 8, 16, 32, 48 and 64.

### Endpoints and tests (fixed now)

**Escape:** mean accuracy over positions 9–16 is at least 0.9, as in the reliability study.

**Declared paired tests.** There are six: three for each pair (`rdt_rand` against `rdt`, and `ctm_lm_rand` against `ctm_lm`):
1. escape at T = 16, with an exact two-sided McNemar test;
2. mean accuracy over positions 1–16 at T = 16, with an exact two-sided sign-flip permutation test;
3. **extrapolation**: mean accuracy over positions 17–32 at T = 32 (T ≥ the evaluation length), with an exact sign-flip test against the control evaluated at T = 32.

All six p-values are **Holm-corrected** together, and effect sizes and confidence intervals are reported whatever the outcome.

**Secondary:** escape at T = 32; the correct prefix at every evaluated T; and the realized depth distribution and training time.

### Verification and execution

Before freezing:
- the study wiring checks;
- the depth-sampling and runner-v4 checks (exact v3 parity for RDT and CTM-LM, and sampled-and-recorded depths);
- the reliability-statistics checks;
- the CTM-LM, S₃ and Sync-RDT checks;
- the archived-trainer replay, on both GPUs.

The freeze audit adds an exact check of each run's per-update depth sequence against the seeded sampler.

The study launches automatically once the reliability study completes (they share the GPUs), under the restart-safe supervisor with autostart. There are 40 runs, estimated at about 7 hours. Failures are recorded and never retried.

### Interpretation limits

- One sampler setting (mean 15, σ 0.5, cap 48), one learning rate, one task.
- The controls were trained earlier under runner v3. Runner v4 without a sampler is tested to reproduce v3 exactly, so the only intended difference is the depth distribution.
- Randomized depth changes both the training signal and per-update cost variance. Wall time is reported, and expected compute is matched only in expectation.
