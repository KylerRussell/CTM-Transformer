# Interpretation of the Transformer pointer calibration (Stage 1a)

## Finding under the declared rules

One-hop hop-1 accuracy on 1,024 held-out maps after 30,000 updates. Every training map was presented once, and chance is 1/11 = 9.09%.

| Peak LR | Seed 41 | Seed 43 | Mean |
|---:|---:|---:|---:|
| 0.0003 | 51.95% | 7.42% | 29.69% |
| 0.001 | 9.38% | 8.98% | 9.18% |
| 0.003 | 7.42% | 7.81% | 7.62% |

D1 selects peak LR 0.0003. **D2 is not met:** the mean is 29.69% and the lowest seed 7.42%, against a gate of at least 90% mean and at least 80% per seed. Under the frozen rules, single-query, answer-only pointer retrieval is **not learnable within 30,000 updates at this model size**. Stages 1b and 2 do not proceed on this format.

Multi-hop training (hops 1–4) stays at chance at every learning rate and seed: 6.2–12.5% at every hop from 1 to 8. This includes hop 1, which multi-hop training also contains.

## What the one successful run shows

Only one of 12 runs, LR 0.0003 with seed 41, left the plateau:
- Validation hop-1 accuracy stays at 6–15% until about 12,000 updates.
- It rises to 23% at 13,000, 44% at 19,000, and then levels off at 44–50% as the cosine schedule lowers the learning rate. The final rate is 3e-5.
- Training CE falls from the plateau value of 1.206 to 0.557.
- Its held-out hop-1 accuracy of 51.95% shows real retrieval on unseen maps, not memorization. No map repeats.
- It was trained only on one-hop queries. On hops 2–8 it scores 3.9–7.4%, below chance, consistent with answering every query with the one-hop successor.

Seed 43 at the same learning rate shows only a late stir: 14–18% at 25,000–27,000 updates, ending at 9%.

This pattern is a long plateau at chance followed by an abrupt, seed-dependent escape, then stagnation as the learning rate decays. It matches the reported phase-transition behavior of induction-style retrieval, and a behavioral observation cannot establish that mechanism. The two higher learning rates never escaped. Within this grid, the escape probability rises as the learning rate falls, but two seeds cannot estimate that relationship.

## What this does and does not answer

It answers the practical question the calibration was declared for. A cost-feasible comparison of CTM and recurrent depth cannot use this format: at 30,000 updates, runs cost about 5.7 hours (CTM) and 2.8 hours (RDT) each, even the reference Transformer succeeds in only 1 of 6 learning-rate/seed cells, and success depends on the seed. The earlier failures also have a coherent explanation: with one supervised answer per sequence, a training example carries very little signal about the retrieval rule. Every family therefore either memorized a small repeated set, or stayed on the plateau with fresh data.

It does not show that the format can never be learned. A lower learning rate, a longer budget, a different schedule or a larger model might escape more reliably; none was tested. Two seeds per cell calibrate a budget but support no rate estimate. Nothing here compares architectures.

## Consequence for the paper plan

Follow D2: move to **dense supervision**, where each sequence asks for the successor of every node rather than of one start node. This gives about 12 supervised answers per map instead of 1, the multi-query form under which associative recall is commonly learned quickly. It needs a new, versioned dataset format and trainer data path, verified to leave the existing path byte-identical. The same staged plan then applies to the dense format:
- a Transformer calibration first;
- then CTM and RDT calibration on calibration seeds, with the operating-point rules already fixed in the [calibration protocol](PLAN_BEFORE_RUNS.md);
- then the comparison on disjoint seeds.

See [results](RESULTS.md), `summary.json` and `checkpoints.json`.
