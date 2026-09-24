# Interpretation of the recurrent temporal-supervision control

## Finding at the declared settings

At fixed peak LR0.001, T16, confidence readout and equal data exposure, the selected validation accuracy across development seeds23/29/31 is:

| Training objective | Mean accuracy | Sample SD | Mean training minutes |
|---|---:|---:|---:|
| Final-tick CE | 66.93% | 4.30 points | 15.8 |
| Uniform | 56.77% | 14.18 points | 24.8 |
| Dynamic aggregate | 57.03% | 20.83 points | 25.6 |

Neither auxiliary objective improves the mean validation CE or mean accuracy at these settings. Uniform lowers mean accuracy by10.16 points relative to the paired final-CE controls; dynamic lowers it by9.90 points. The extra coda decoding increases average training-loop time by roughly57% and62%, respectively. These are observed differences in this small development study, not significance claims or equal-compute comparisons.

The effect depends on the seed. At seed23, dynamic reaches73.44% versus71.88% for final CE. At seed29 it reaches33.59% versus64.84%, and at seed31 both reach64.06%. Uniform matches final-CE accuracy at seed23 and is lower at the other two seeds. Every declared cell completed and is reported.

## What this does and does not answer

The loss implementation and gradients match CTM's tested objectives, including masking and native-dtype dynamic certainty. The recurrent architecture, initialization, data-order seed, optimizer settings, training exposure and deployment readout are paired. This addresses whether adding those objectives alone improves the existing recurrent recipe at its current learning rate. It did not improve the three-seed average.

This does not show that temporal supervision is generally ineffective. LR0.001 was selected for the final-CE baseline, and different objectives may prefer different learning rates or schedules. The prior CTM study already demonstrated strong optimizer sensitivity. An objective-by-learning-rate study would be a separate development experiment; these runs should not be portrayed as optimally tuned auxiliary baselines.

The plotted trajectories are across optimizer updates, not thought ticks. This control does not retest or contradict the earlier CTM confidence-prefix monotonicity observation. Dynamic training aggregation and confidence-based deployment selection remain separate mechanisms.

These are selected validation results on128 shared maps at previously examined seeds. They are not new held-out results and should not be compared directly with the previous512-map test accuracies. No completed test set was reopened.

## Consequence for the paper plan

Retain the final-CE recurrent recipe as the current reference. Report this matched-objective control as a bounded negative development result, including all seeds, the higher variance and the added compute. It does not establish a CTM architectural advantage or eliminate optimizer/schedule confounding from the broader comparison.

The next task remains the separate **shuffled-order training control**: hold semantic maps, query difficulty, exposure, family recipes and paired seeds fixed while changing edge presentation. Use development data for that work and declare any later confirmation separately. Keep composition and scaling behind reliable one-hop retrieval. If auxiliary objectives are revisited, predeclare their optimizer/schedule study independently of the presentation control.

See [complete results](RESULTS.md), [frozen protocol](PLAN_BEFORE_RUNS.md), `summary.json`, and `integrity.json`.
