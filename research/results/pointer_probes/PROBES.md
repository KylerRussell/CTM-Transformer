# Exploratory pointer-format probes — 2026-09-25

**Development only.** These probes were not preregistered. Each is a single Transformer run: the calibration recipe (548,660 parameters, 2 layers, width 140), final readout, and runner v3, which the parity test shows reproduces the frozen trainer exactly. Their only purpose is to find a learnable format before the next calibration is frozen. They are never paper endpoints.

In every probe:
- training maps are fresh, and each is presented once;
- the 1,024 held-out evaluation maps and the validation maps are disjoint from the training maps;
- answers are scored teacher-forced, per answer.

Code: `ctm_transformer/pointer_probes.py` and `scripts/run_pointer_probe.py`. Each run's record is in `<name>.json`, with its source hashes.

**Chance.**
- Permutation maps: 1/11 per answer. Position 1 has no earlier answers, so it cannot benefit from exclusion; **position-1 accuracy is the leak-free retrieval measure**.
- Random-value maps (`mqar_random`): 1/12 per answer.

## Results

| Probe | Format | Queries per map | Training | Answer acc. | Position 1 | All correct | Final train CE |
|---|---|---:|---|---:|---:|---:|---:|
| r1 | interleaved (`ask K:J`) | 6 | hop 1, 10k × 32 | 12.8% | 7.8% | 0% | 1.845 |
| r1 | lowercase targets (`K>j`, `ask K:j`) | 6 | hop 1, 10k × 32 | 12.4% | 9.8% | 0% | 1.841 |
| r2 | direct (`ask Kj`) | 6 | hop 1, 10k × 32 | 10.9% | 9.1% | 0% | 1.847 |
| r2 | MQAR layout (`map Kj;…;ask Kj`) | 6 | hop 1, 10k × 32 | 11.8% | 6.6% | 0% | 1.847 |
| r3a | MQAR | 6 | hop 1, **40k** × 32 | 11.5% | 7.7% | 0% | 1.845 |
| r3b | MQAR | 6 | hop 1, 10k × **128** | 12.2% | 9.5% | 0% | 1.841 |
| r4a | MQAR, map sizes 3–12 in training | 6 | hop 1, 10k × 32 | 12.1% | 8.6% | 0% | 1.381 |
| r5a | MQAR, random values | 6 | hop 1, 10k × 32 | 24.9% | 24.6% | 0% | 1.675 |
| r7a | MQAR, random values | **12** | hop 1, 5k × 32 | 38.4% | 27.3% | 0% | 1.281 |
| **r7b** | **MQAR, permutation values** | **12** | hop 1, 5k × 32 | **95.1%** | **89.8%** | **53.1%** | 0.147 |
| **r8b** | same as r7b, seed 43 | **12** | hop 1, 5k × 32 | **89.5%** | **83.6%** | 25.8% | 0.278 |
| r8a | same as r7b | 12 | **hops 1–4**, 10k × 32 | 29.0% | 9.3% | 0% | 1.471 |
| r3c | copy (answer = own query), sanity | 6 | 1k × 32 | 100% | 100% | 100% | 0.000 |
| r6a | repeat a random 20-letter string | — | 3k × 32 | 100% | — | 100% | 0.000 |
| r6b | repeat a random 8–24-letter string | — | 5k × 32 | 100% | — | 100% | 0.000 |

## What the probes show

1. **The pipeline and the model's induction ability are sound.** Copying is learned by update 50. Repeating a random string of variable length, which needs content-based induction because a fixed offset cannot solve it, is learned by update 500.
2. **With 6 queries per map, no layout escapes the shortcut basin.** Changing the layout (interleaving, disjoint target symbols, direct prediction, MQAR), updates (4×), batch (4×) or a map-size curriculum leaves every run at the permutation-exclusion level (training CE ≈ 1.84). With random values, runs sit instead at the value-frequency level (about 25%, including position 1).
3. **Querying all 12 keys per map makes retrieval learnable.** With the MQAR layout and permutation values, both seeds escape the plateau at about 2,400 updates. They reach 95.1% and 89.5% answer accuracy by 5,000 updates, and seed 43 is still rising. Position-1 accuracy of 89.8% and 83.6% rules out exclusion as the explanation.

   The random-value variant with 12 queries does *not* learn retrieval. It learns a multiset shortcut: position 12 is exactly 100% because the last answer is the one remaining value, while position 1 stays at the frequency level.
4. **Mixed-hop training (1–4) blocks learning even at hop 1** for this 2-layer Transformer within 10,000 updates: 9.3% at position 1.

## Implications for the next calibration

- One-hop retrieval is learnable, and learnable quickly, in the **all-keys permutation MQAR format**. That makes it a candidate format for a frozen Transformer calibration, with position-1 (leak-free) accuracy as the primary retrieval measure.
- Multi-hop is where the comparison should happen, and it is not yet learnable for this Transformer. That is not necessarily disqualifying: a fixed 2-layer model is expected to fail at depth. But CTM and RDT must be shown able to learn at least part of it, or the comparison sits at the floor.
- The next development probes should therefore test:
  - a hop curriculum (one-hop first, then added hops) for the Transformer;
  - CTM and RDT on one-hop and on the hop curriculum, on the calibration seeds.

  A CTM probe costs about 0.68 s per update, or about 1 hour for 5,000 updates.
