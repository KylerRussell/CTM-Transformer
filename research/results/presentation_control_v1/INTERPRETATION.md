# Interpretation of the paired training-presentation control

## Finding at the declared settings

Fixed-update validation accuracy on the same 128 maps, mean ± sample SD across development seeds 23/29/31:

| Family | Trained on | Ordered evaluation | Shuffled evaluation | Final training CE (last 100 updates, mean) |
|---|---|---:|---:|---:|
| CTM | ordered | 85.16% ± 11.08 | 14.58% ± 2.74 | 0.073 |
| CTM | shuffled | 19.79% ± 6.07 | 16.15% ± 3.69 | 0.695 |
| Transformer | ordered | 100.00% ± 0.00 | 12.24% ± 1.19 | 0.000 |
| Transformer | shuffled | 19.53% ± 0.78 | 15.89% ± 4.30 | 0.002 |
| Recurrent depth | ordered | 67.71% ± 6.27 | 14.06% ± 1.35 | 0.228 |
| Recurrent depth | shuffled | 16.41% ± 3.41 | 15.36% ± 1.19 | 0.692 |

**Chance is 1/7 = 14.29%, not 1/8.** Every map is a single 8-cycle, so the successor of the start node is never the start itself. With 128 maps, one run's sampling SD at chance is about 3.1 points. Every shuffled-evaluation mean, and every ordered-evaluation mean after shuffled training, lies within 6 points of chance. The maps are shared across runs, so the three seeds are not independent samples of the maps.

Shuffled training changes shuffled-evaluation accuracy by only +1.3 to +3.7 points on average. It costs 51–80 points on ordered evaluation. No family learned presentation-independent one-hop lookup under this recipe and budget.

## Why the families fail

**The Transformer memorized the training set.** Its shuffled-training CE is 0.0005–0.0033 at the end, while validation accuracy is near chance. The training set has 2,048 fixed examples with one query each, seen about 47 times each (96,000 presentations). A 548,660-parameter model can store all 2,048 answers. Only two tokens per example (answer and EOS) are supervised. A lookup-table solution therefore fits the training loss without learning retrieval.

**CTM and recurrent depth fit neither the training nor the validation set.** Their final training CE is about 0.6–0.77 per supervised token. This is below the start of training (about 1.7–2.1), but far from memorization. Under this recipe, the same data regime that let the Transformer memorize also left these slower-optimizing models underfit. Optimizer or schedule effects under shuffled training cannot be separated from this: the recipes were selected under ordered training and were not retuned.

**The ordered-training successes were probably positional, not content-based.** In canonical order, the queried edge always sits at a fixed position determined by the start letter. A model can answer by reading that position without matching content. Every ordered-trained model scores at or below chance on shuffled presentation of the same maps (12.24–14.58%). This is consistent with position-indexed reading. It is a behavioral inference, not a measured mechanism. The earlier attention capture (`attention_temporal_v1`) is compatible with it but was not designed to test it.

## What this does and does not answer

This changes only the training edge presentation, pairing semantic maps, queries, exposure, seeds and recipes. At the existing recipes and budget, one fixed shuffled presentation per map does not produce order-robust retrieval in any family.

It does not show that these architectures cannot learn one-hop associative recall. The data regime is a strong alternative explanation for the failure: a small, fixed, repeated training set with sparse answer-only supervision. Associative recall is commonly learned from freshly generated examples, with many queries per sequence. Neither was tested here. One fixed shuffle per map is not online permutation augmentation, and recipes were not retuned for shuffled training.

The earlier ordered comparisons therefore mainly measure a presentation shortcut. The confirmation ranking (Transformer > CTM > recurrent depth on ordered maps) should not be read as a ranking of retrieval ability.

## Restart provenance

Attempt 1 was stopped by a container kill; see the interruption section of `research/PRESENTATION_CONTROL.md`. The restart-safe supervisor reran the two interrupted CTM cells from scratch. `interruption_replay.json` shows that both reruns reproduce every completed update of their interrupted predecessors exactly (1,831 and 1,722 updates, all logged fields except wall time). Every declared cell completed, and none was replaced.

## Consequence for the paper plan

Record this as a bounded negative development result. The one-hop retrieval gate is not passed by any family, and the current task cannot separate them: ordered presentation leaves a shortcut, and shuffled presentation in this data regime sends every family to chance. Multi-step composition and scaling remain blocked on reliable retrieval.

Before another architecture comparison, establish a task and data regime in which at least the ordinary Transformer reliably learns order-independent one-hop lookup. Examples are online-generated maps, multiple queries per sequence and larger node vocabularies. Then use tasks whose difficulty requires serial computation, such as varying hop counts at fixed input size with held-out deeper hops. Those tasks, rather than one-hop retrieval, are where recurrence and additional thought ticks would be expected to matter. Declare each as a separate development study. Completed test sets remain closed.

See [complete results](RESULTS.md), [frozen protocol](PLAN_BEFORE_RUNS.md), `summary.json`, `checkpoints.json` and `integrity.json`.
