# S3 learning-rate control for randomized depth

## Protocol declared before new scientific training — 2026-09-30

### Question

In the [recipe confirmation](results/recipe_s3_v1/INTERPRETATION.md), randomized-depth RDT beat fixed-depth RDT by +14.3 points and escaped at 24/30 against 0/30. But each cell ran at its own tuned learning rate: randomized depth at 1e-3, fixed depth at 6e-4. **At the same learning rate (1e-3), does randomized depth still beat fixed depth?**

### Design

- **One new cell, `rdt_rope_fixed_lr1e3`:** fixed-depth (T = 16) RoPE RDT at learning rate 1e-3. It is identical to the recipe confirmation's `rdt_rope_fixed` except for the learning rate. It is identical to that study's `rdt_rope_rand` except for the training depth.
- **Seeds and data:** the recipe confirmation's 30 seeds and data (`research/data/recipe_s3_v1`), verified by hash. Runner v5, 10,000 updates, batch 32, final checkpoint, confidence readout. Evaluation is at T = 4 to 64 on held-out words of length 32, read once after the freeze.
- **Controls, not retrained:** the recipe confirmation's `rdt_rope_rand` and `rdt_rope_fixed` runs. They are paired by seed, and their evaluation files are verified against that study's recorded hashes.

### Endpoints and tests (fixed now)

Escape is defined as in the recipe confirmation: mean accuracy over positions 9–16 at T = 16 is at least 0.9.

**Declared paired tests (`rdt_rope_rand` against `rdt_rope_fixed_lr1e3`), Holm-corrected together:**
- **C1:** mean accuracy over positions 1–16 at T = 16. Two-sided paired sign-flip test, 2,000,000 seeded Monte Carlo draws with the +1 correction.
- **C2:** escape at T = 16. Exact two-sided McNemar test.

**Secondary** (reported, not corrected):
- `rdt_rope_fixed_lr1e3` against `rdt_rope_fixed` (the learning-rate effect at fixed depth), on positions 1–16 and on escape;
- randomized against fixed depth at 1e-3 on positions 17–24 at T = 32;
- correct prefix at every T.

**Decision rule.**
- If C1 or C2 is significant in favour of randomized depth, the recipe confirmation's effect is attributed to randomized depth at a matched learning rate.
- Otherwise it is reported as a joint effect of randomized depth and the learning rate it was tuned at.

### Execution

- **Checks before freezing:** the wiring checks, the RoPE checks, the runner-v5 checks and the depth-sampling checks.
- **Slots and running time:** 30 runs on four worker slots, estimated at about 9 hours. The other two slots run A₅ development probes at the same time.
- **Preflight relaxed:** because the GPUs are shared with those probes, this study's supervisor does not require idle GPUs at preflight (`busy_mib` raised). This affects only wall time, not any computation.
- **Supervision:** the restart-safe supervisor, with autostart. Failures are recorded and never retried.

### Interpretation limits

- One learning rate (1e-3), which was chosen for the randomized-depth cell. Fixed depth's own development optimum was 6e-4, with 1e-3 close behind (0.873 against 0.884).
- The controls were trained a day earlier by the same frozen runner and source.
- Sharing GPUs with development probes changes run times but not results. The runner is deterministic given its seeds, which the supervisor's replay audit checks after any interruption.
