# Interpretation of the dense Transformer calibration (Stage D1a)

## Finding under the declared rules

Teacher-forced hop-1 answer accuracy on 1,536 held-out answers (256 maps × 6 queries) after 10,000 updates. Every training map was presented once.

| Peak LR | Seed 41 | Seed 43 | Mean | All 6 correct |
|---:|---:|---:|---:|---:|
| 0.0003 | 13.02% | 12.43% | 12.73% | 0% |
| 0.001 | 12.37% | 11.72% | 12.04% | 0% |
| 0.003 | 12.30% | 12.70% | 12.50% | 0% |

D1 selects peak LR 0.0003. **D2 is not met.** Under the frozen rules, the dense format is not learnable by this Transformer within 10,000 updates. Stages D1b and D2 do not proceed on this format. Multi-hop training (hops 1–4) gives 11.1–13.4% at every hop from 1 to 8 and every learning rate.

## The models learned permutation exclusion, and nothing else

- Chance per answer is 9.09%. A guesser that excludes the query node and earlier gold answers averages at most 12.28%; the protocol declared this ceiling before training. Every one-hop run finishes between 11.72% and 13.02%.
- Accuracy by answer position rises from 9.0–9.4% at position 1 (chance) to 16.6–18.8% at position 6. That is the exclusion profile: a pure exclusion guesser rises from 9.09% to 16.67%.
- Final training CE is 1.846–1.849 for all 12 runs, one-hop and multi-hop alike. An exclusion guesser scores about 1.816 per supervised token; uniform guessing scores 2.055.

This is exactly the leak the format limits and reports. It is not retrieval: whole-sequence exact match is 0% everywhere. It also explains why multi-hop cells look the same as one-hop cells: a k-hop successor map is also a permutation, so exclusion helps at every hop equally.

## Why both pointer formats probably failed

Scored answers were not the limiting quantity. This study supplied about 1.92M scored answers (320,000 maps × 6), twice the 960,000 of the single-query calibration, where one run in six escaped the plateau after about 384,000.

A more likely obstacle is the lookup circuit itself. Every letter appears both as a source (`K>J`) and as a target (`X>K`). To answer "successor of K", the model must find the K that precedes `>` and copy the token two positions later. In the block format it must also first identify which listed query it is currently answering, by position. Associative-recall tasks that small Transformers learn quickly avoid both problems: keys and values use disjoint vocabularies, and each answer directly follows its query.

This explanation is a hypothesis. This study does not test it.

## What this does and does not answer

This calibration used the declared recipe: 2 layers, width 140, 548,660 parameters. It does not show that the dense format is unlearnable for larger models, longer budgets or other schedules. It does show that neither pointer format gives an affordable, reliable starting point for the CTM–RDT comparison at the current model size. Nothing here compares architectures.

## Consequence for the paper plan

Three frozen calibrations in a row each spent a full cycle discovering that the task was not learnable. The next step is a short series of **exploratory development probes**: Transformer only, minutes each, reported as development, never as endpoints. Their purpose is to find a learnable variant before the next calibration is frozen. In order of cost:

1. Interleave each answer with its query (`ask K:J;C:B;...`), removing positional query alignment.
2. Additionally mark targets with a separate symbol set (`K>j`), making one-hop lookup standard associative recall.
3. Increase Transformer depth. Families would then need rematching.

The operating-point rules and seed plan from the [calibration protocol](../pointer_calibration_v1/PLAN_BEFORE_RUNS.md) still apply to any format that passes.

See [results](RESULTS.md), `summary.json` and `checkpoints.json`.
