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

## Round 9: recurrent families and hop curriculum — 2026-09-25

All-keys permutation MQAR format (12 queries per map). Validation accuracy in the curves combines trained hops. Position-1 accuracy is on held-out maps.

| Probe | Family, LR | Training | Hop 1 | Position 1 | Hops 2–6 | Minutes |
|---|---|---|---:|---:|---|---:|
| r9a | CTM, 0.0003 (recipe) | hop 1, 10k × 32 | 28.5% | 7.4% | — | 118 |
| r9b | RDT, 0.001 (recipe) | hop 1, 10k × 32 | 29.2% | 12.1% | — | 60 |
| r9c | Transformer, 0.001 | curriculum: 1 → 1–2 → 1–3 → 1–4, 5k updates each | 29.0% | 8.9% | 27.8–29.9% | 12 |
| **r9d** | **Transformer, 0.0003** | same curriculum | **100%** | 24.2%* | 33.3%, 30.0%, 29.3%, 29.9%, 21.5% | 13 |

\*Position 1 in r9d is averaged over hops 1–6; hop-1 answers are 100% correct at every position.

- **The Transformer's escape depends on learning rate and schedule.** At 0.0003, hop-1 accuracy jumps from chance to 100% between 7,000 and 8,000 updates. At 0.001 with a 20,000-update schedule it never escapes, although the same rate escaped at about 2,400 updates with a 5,000-update schedule (r7b, r8b), where the rate had already decayed. Hop 2 rises only to 33–35%, just above the exclusion level (about 29%); hops 3–4 stay at it. A fixed 2-layer model is not expected to compose hops, so this is consistent with its depth limit.
- **Neither recurrent family learns one-hop retrieval within 10,000 updates at its recipe learning rate.** RDT's recipe rate (0.001) is the rate at which the Transformer also failed under a long schedule, so a 0.0003 probe is the direct test.
- **CTM has a structural reason to find this task hard.** In the reference configuration (`use_attention_residuals: false`), CTM's cross-attention keys and values are the *static* token-plus-position embeddings, computed once (`CTMTransformer`: "Embed input text → K, V for cross-attention (computed once)"). No contextual information passes between positions, so CTM cannot form an induction head that marks each value with its preceding key. Retrieving the value after key A requires locating A on one tick and then querying the next position from A's positional embedding on a later tick: learned positional arithmetic across ticks. This is possible for a recurrent model, but harder to discover. It is an architectural property of this implementation that any CTM retrieval result must report, not a tuning detail.

## Round 10: RDT learning rate and CTM variants — 2026-09-25

All-keys permutation MQAR format, one hop, 10,000 updates.

| Probe | Family / variant, LR | Answer acc. | Position 1 | All correct | Parameters | Minutes |
|---|---|---:|---:|---:|---:|---:|
| **r10b** | **RDT, 0.0003** | **100%** | **100%** | **100%** | 525,984 | 59 |
| r10a | CTM `contextual_kv`, 0.0003 | 29.7% | 7.4% | 0% | 708,935 | 116 |
| r10c | CTM `attn_res`, 0.0003 | 28.1% | 10.5% | 0% | 545,223 | 137 |

- **RDT learns one-hop retrieval at LR 0.0003.** It escapes between 6,000 and 6,500 updates and holds 100% from 8,000 on. At 0.001 (r9b) it never escapes, the same learning-rate dependence the Transformer shows.
- **Attention residuals do not enable retrieval,** as expected from their construction (mixing across layers, per position).
- **Contextual keys and values alone do not enable CTM retrieval within 10,000 updates.** A second structural limitation explains why: every position's latent starts from the same learned vector (`z = self.z0.expand(B, S, -1)`), and queries come from the latent's synchronization. Nothing gives a position's latent direct access to its own token. The output head concatenates the token embedding only at readout. A position must therefore discover its own token through attention before it can ask "what follows my token?", using a query that is initially identical at every position. RDT avoids this by re-injecting the embedded input at every recurrence step (input injection). The next candidate variant therefore adds token-conditioned latent initialization, or per-tick input injection, to contextual K/V.

## Round 11: token-conditioned CTM variants — 2026-09-25

All-keys permutation MQAR, one hop, LR 0.0003, 10,000 updates.

| Probe | CTM variant | Answer acc. | Position 1 | All correct | Parameters | Minutes |
|---|---|---:|---:|---:|---:|---:|
| r11a | `contextual_kv_token_init` | 28.4% | 10.2% | 0% | 725,319 | 119 |
| r11b | `contextual_kv_injection` | 28.6% | 7.4% | 0% | 725,319 | 120 |

Neither variant escapes the chance plateau. Giving CTM contextual keys and values *and* its own token, once or at every tick, is not sufficient within 10,000 updates, the budget at which the Transformer and RDT learn.

The remaining differences from RDT are:
- CTM latents never attend to other positions' evolving states, only to fixed text features;
- attention queries come *only* from synchronization, a decayed sum of pairwise products of neuron histories, so even a latent that contains its token must express "match my token" through pairwise products;
- the temporal objective and readout differ.

Further repair of the standalone CTM becomes open-ended. Adding CTM's mechanisms to the RDT scaffold, which learns retrieval, is the cleaner test of those mechanisms. See the [Sync-RDT design](../../SYNC_RDT_DESIGN.md). In that design, synchronization *adds* to content queries rather than replacing them, for the reason these probes suggest.
