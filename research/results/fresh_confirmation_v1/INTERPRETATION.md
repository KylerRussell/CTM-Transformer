# Interpretation and next research controls

## What changed with CTM tuning

At development seed 17, the uniform objective with peak LR 0.0003 and confidence readout reached 100% validation accuracy, compared with 76.56% for the reused uniform LR 0.001 cell. The selected low-LR checkpoint scored 99.22% on the 512 fresh ordered maps. The best dynamic cell in the declared grid remained LR 0.001, with 73.44% validation accuracy. These comparisons select each cell's checkpoint/readout by validation CE; they are one-seed development findings.

This supports the user's concern that training choices can materially change CTM performance. It does not establish that uniform will outperform an unidentified earlier dynamic implementation or every temporal schedule. The earlier confidence-prefix monotonicity result remains a separate finding; a monotonic trajectory can still end at a weaker absolute score.

## What independent seeds revealed

The primary confirmation uses seeds 23, 29 and 31, excluding the development seed. Ordered accuracy is 87.50% ± 9.29 percentage points for CTM, 100.00% ± 0.00 for the ordinary Transformer, and 63.74% ± 8.07 for recurrent depth. These are means and sample SDs on the same 512 maps.

CTM exceeds this recurrent recipe at each of the three paired seeds, while the ordinary Transformer is best at each seed. The recurrent development reference reaches 100% on the same fresh maps, but none of its three new-seed runs reproduces that success. CTM also varies substantially across seeds (78.52–97.07%). Consequently, the previous seed-17 baseline result was insufficient to characterize the reliability of the selected recipes. This study does not identify the source of that variability: initialization and training-example order both change with the seed.

All families remain weak when the same maps are presented in shuffled edge order: means range from 13.22% to 14.52%. Their success on ordered one-hop lookup therefore does not establish robust retrieval across presentations or multistep reasoning.

## What is fair about this comparison, and what remains unmatched

All new-seed runs receive 3,000 updates, the same training examples and token exposure, and approximately matched parameter counts. Family recipes/readouts are fixed before confirmation, and checkpoints are fixed before fresh evaluation. Every declared run completed and is reported.

Training cost is unequal: approximately 31.9 minutes per CTM run, 15.8 per recurrent run, and 1.9 per ordinary Transformer run. These are measured training-loop wall times with one GPU assigned to each worker, not measured FLOPs. CTM also has auxiliary temporal supervision that the recurrent baseline lacks, and tuning histories differ. The observed ranking is therefore specific to these recipes and budgets; it does not isolate a CTM architectural advantage.

## Next bounded milestones

1. **Match temporal supervision.** Give the recurrent baseline the same declared uniform/dynamic temporal objectives and label-free readout options at T16, with verified gradient and checkpoint-selection behavior. Treat this as a new development study; the current confirmation set is now closed. Use multiple development seeds to detect recipes that succeed only at one seed.
2. **Train on varied edge presentations.** Test ordered versus shuffled training while holding semantic maps, queries, exposure and model settings fixed. Require reliable one-hop retrieval before proceeding to composition. Separate this task change from the temporal-supervision comparison so their effects can be interpreted.
3. **Then measure equal-cost tradeoffs.** Add measured FLOP accounting and compare quality at explicit training and inference budgets before scaling. The current timing results can size that study, but are not themselves FLOP matching.

These are proposed follow-ups, not additional trials silently added to the completed protocol. See [results](RESULTS.md), [CTM development](../ctm_lr_v1/RESULTS.md), and the [frozen pre-run plan](../ctm_lr_v1/PLAN_BEFORE_RUNS.md).
