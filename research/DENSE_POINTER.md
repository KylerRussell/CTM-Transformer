# Dense-supervision pointer format and Transformer calibration

**Status: protocol declared before new scientific training — 2026-09-25.** The draft was reviewed, and all three open decisions were accepted at their defaults (below). Stage D1a is frozen in `results/dense_calibration_v1/registry.json` and launched with `research/launch_dense_calibration.sh`, a restart-safe supervisor that resumes after a container restart.

## Why a new format

The [Transformer pointer calibration](results/pointer_calibration_v1/INTERPRETATION.md) did not meet its format gate. With one query and one supervised answer per sequence, 11 of 12 Transformer runs stayed on the chance plateau for 30,000 updates. The one escape took about 12,000 updates and depended on the seed. That format cannot support an affordable, reliable comparison of CTM and recurrent depth. The declared next step (D2) is denser supervision.

## Format (`pointer_dense`, `ctm_transformer/dense_pointer.py`)

Each sequence presents one single 12-cycle map in random, noncanonical edge order. It then asks for the k-hop successor of **6 distinct nodes** in random order:

```
map H>J;L>K;D>G;...;hops 2;ask KCAFBE=<6 answer letters><EOS>
```

- **Supervision:** all 6 answers and EOS are supervised, 7 tokens per map instead of 2. The input is 77 tokens.
- **Map identity:** a map is held out by its successor table in every format, so no split shares a map with any earlier single-query or dense 12-node data.
- **Scoring:** answers are scored teacher-forced, per answer token (chance 1/11 = 9.09%). The reported metrics are answer accuracy, whole-sequence exact match, accuracy by hop, and accuracy by answer position.
- **Why 6 queries, not 12:** a k-hop successor map is a permutation, so earlier gold answers exclude candidates for later ones. With all 12 nodes queried, the last answer would be free. With 6 queried, a guesser that exploits exclusion averages at most **12.3%** per answer. Accuracy by answer position makes any use of this visible.

## Versioned runner (`shared_research_v3`, `ctm_transformer/dense_experiment.py`)

The v2 runner (`ctm_transformer/experiment.py`) is hash-frozen by every earlier study and is not modified. Runner v3 copies its training loop exactly and takes the training and evaluation datasets and the validation evaluator as arguments. Its run manifest records runner v3, the evaluator's name, and hashes of both runner versions and the dense module.

`tests/test_dense_pointer.py` checks that v3 reproduces v2 **exactly** when given v2's datasets and evaluator: losses, gradient norms, learning rates, token counts, validation rows and every final weight tensor, for CTM, recurrent depth and the Transformer. It also checks:
- dense record validation and tamper detection;
- that every answer and EOS are supervised;
- a deterministic, disjoint suite that excludes earlier maps of both formats;
- detection of a map leaked into evaluation, even after its hash is refreshed;
- the evaluator against direct scoring;
- a short dense training run that learns.

## Proposed Stage D1a: Transformer calibration on the dense format

The same staged plan and operating-point rules as the [single-query calibration](POINTER_CALIBRATION.md#operating-point-rules-for-stages-1b-and-2-fixed-now) apply.

- **Model:** Transformer recipe (548,660 parameters, final CE, final readout). Only learning rate, budget, warmup and validation interval change.
- **Grid:** peak LR 0.0003, 0.001 or 0.003 × one-hop or multi-hop (hops 1–4) × calibration seeds 41 and 43, for 12 runs.
- **Budget:** 10,000 updates, batch 32, warmup 100 (1%), cosine to 10%, and validation every 500 updates with checkpoints kept. That is a third of the single-query budget: each map now carries six supervised answers, and CTM runs at 10,000 updates cost about 1.9 hours rather than 5.7.
- **Data (generator seed 20260927):** each training file holds 10,000 × 32 = 320,000 unique maps, so every map is presented once.

  | Split | Maps | Hops |
  |---|---:|---|
  | `train_fresh_onehop` | 320,000 | 1 |
  | `train_fresh_multihop` | 320,000 | 1–4 (80,000 each) |
  | `validation` | 384 | 1–4 (96 each) |
  | `eval_id` | 1,024 | 1–4 (256 each) |
  | `eval_depth` | 1,024 | 5–8 (256 each), unseen depths |

- **Endpoints (final checkpoint at update 10,000):**
  - one-hop answer accuracy at hop 1, and the first validation update reaching 90%;
  - for multi-hop training, answer accuracy by hop 1–8;
  - accuracy by answer position against the 12.3% exclusion ceiling;
  - learning curves.
- **D1:** choose the learning rate with the highest mean final one-hop answer accuracy; ties go to the lower rate.
- **D2:** the dense format is learnable if that learning rate reaches a mean of at least 90% with every seed at least 80%. If it is not met, stop and report before any CTM or RDT calibration.
- **Cost:** about 45 minutes of wall time on both GPUs (Transformer about 0.045 s per update at this length; to be re-profiled at freeze).

If D2 is met, Stage D1b calibrates CTM and RDT on calibration seeds under the fixed operating-point rules:
- a hop band where the two-family mean lies between 20% and 80%;
- non-saturating endpoints: accuracy by hop including unseen depths, accuracy against tick budget, and sample efficiency;
- Stage 2 on seeds 47, 53 and 59.

## Decisions accepted at freeze

1. **6 queries per map.** The exclusion ceiling is 12.3%; 8 queries would raise it to 14.8%.
2. **10,000 updates per run,** with warmup 100 and validation every 500 updates.
3. **Multi-hop training on hops 1–4,** evaluated on hops 1–8.

**Execution.** GPU 0 runs seed 41 and GPU 1 runs seed 43. Before freezing and before every supervisor attempt, preflight runs:
- the dense checks, including the exact v3–v2 runner parity on both GPUs;
- the archived-trainer replay checks.

The freeze audit checks, for every run:
- all 10,000 metric rows and the learning-rate schedule;
- per-step example, token and label counts (7 supervised tokens per map);
- all 20 validation checkpoints;
- runner, evaluator, data, configuration and code hashes;
- that each map was presented once.

A genuine worker error is recorded and not retried; nothing is adaptively added or replaced.

## Outcome — 2026-09-25, 01:26 UTC

All 12 runs completed and passed the freeze audit. D1 selected peak LR 0.0003. **D2 was not met** (mean hop-1 answer accuracy 12.73%). Every run finished at the declared permutation-exclusion ceiling (12.28%), with position accuracy rising from chance to about 18% at position 6. Training CE was 1.846–1.849 in every run, close to an exclusion guesser's 1.816. No run learned retrieval; exact match was 0%. Stages D1b and D2 do not proceed on this format. See [results](results/dense_calibration_v1/RESULTS.md) and [interpretation](results/dense_calibration_v1/INTERPRETATION.md).
