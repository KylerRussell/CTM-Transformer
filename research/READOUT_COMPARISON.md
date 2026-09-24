# Across-tick recipe test and parameter/exposure-matched comparison

## Frozen development protocol — 2026-09-23

The user authorized testing the across-tick recipe and making a fair comparison with the other model families. The exact older winning selector/configuration has not been identified. This study therefore tests explicit current implementations. Its primary comparison matches approximate parameter capacity, data exposure, and the number of new tuning trials. It is **not compute-matched**, a full reproduction of Geiping et al., or a paper-level multi-seed result.

### Architecture and exposure

| Family | Configuration | Parameters | Training depth |
|---|---|---:|---:|
| CTM | width128, two layers, independent temporal MLPs, H8, sparse synchronization128 pairs | 544,839 | T16 |
| Transformer | width140, two layers, FFN420 | 548,660 | 2 untied blocks |
| Recurrent depth | width96, FFN288, prelude1/core2/coda1, deterministic zero initial state | 525,984 | T16; 34 block applications |

The largest parameter deviation from CTM is −3.46%. Parameter counts are genuine trainable weights; no unused padding parameters. CTM uses 32 thought-layer applications per sequence. Parameter matching does not match state, FLOPs, or optimized implementation quality.

Every trial starts fresh from seed17, with data-order seed18 and depth seed19. Use the existing disjoint ordered-pointer maps, tokenizer71, batch32, 3,000 AdamW updates, 30-update warmup, cosine decay to 10% of peak, weight decay0.1, clip1, FP32 weights/moments and BF16 autocast. Each trial consumes 96,000 examples, 5,184,000 real input tokens, and 192,000 answer/EOS targets. Dropout is zero. Training budgets are completed even when an earlier checkpoint wins.

### Three declared tuning trials per family

CTM at peak LR1e-3:

1. Uniform tick CE; monotonic penalty zero.
2. Dynamic aggregation: average of per-token minimum-CE and maximum-native-certainty losses; monotonic penalty zero.
3. Same dynamic aggregation plus adjacent-tick batch-mean CE regression penalty: initial weight0.5, decaying linearly to0.05 over the first30% of updates, then fixed. This is a soft penalty, not a mathematical guarantee of monotonic predictions.

Each baseline uses final-output CE and peak LR in **3e-4, 1e-3, 3e-3**. The recurrent baseline uses fixed T16 and full backpropagation. All other within-family settings are fixed. This compares tuned complete recipes; it does not isolate architecture from auxiliary supervision. Giving recurrent depth identical temporal auxiliary losses is a subsequent mechanism control.

Three new trials each is equal trial/exposure allowance, not equal GPU-hours or equal lifetime tuning: earlier development focused more on CTM and informed this grid. No extra trial is added in response to disappointing results. Numerical failures are recorded, not silently replaced.

### Across-tick readout and checkpoint selection

Every100updates compute both final-tick and minimum-FP32-entropy readouts for CTM and recurrent depth. The standard Transformer has one readout. The recurrent core is unrolled once; apply its shared coda/head to each core state without feeding the coda back into recurrence. Tests verify these logits against independent truncations. Extra coda work is included in readout timing.

Confidence selection uses no gold answer: select the earliest minimum-entropy tick independently for each next token. It uses all16ticks, with no early stopping. The CTM training objective keeps its existing native certainty calculation; inference uses the common FP32 entropy rule. This numeric distinction is recorded rather than silently changing the training objective.

Select checkpoint **and readout** by minimum token-weighted original-validation answer/EOS CE over the declared readouts and evaluation steps. At a given step ties prefer final, then confidence; ties across steps retain the earlier checkpoint. Record unconstrained greedy answer-plus-EOS accuracy for every readout at every validation. Save every validation checkpoint and the best checkpoint per policy. The final checkpoint is also retained.

Log individual-tick CE/accuracy, confidence-selected CE/accuracy, selected-tick histograms, and gold-aware best-so-far diagnostic envelopes separately. Gold-aware envelopes must never select an inference tick or be presented as deployment scores.

### Lock selection before evaluating test maps

After **all nine** trials finish, select one trial/checkpoint/readout per family using the same minimum-validation-CE rule. Break exact trial ties by registry order. Write an immutable selection JSON containing all validation scores, selected policies, checkpoint hashes, and source/config/data identities **before any test forward pass**.

Evaluate only those three frozen winners on:

- Original validation, as a consistency check.
- Ordered held-out test maps: primary,128 examples, unrestricted answer-plus-EOS exact match and supervised CE.
- The corresponding shuffled-edge-order test maps: secondary order-robustness diagnostic, same128maps, correlated with primary.

These test splits have appeared in earlier exploratory project milestones. They are held out from the current tuning runs, but are not a never-inspected confirmatory dataset. Do not use the new test results to retune this experiment. Fresh test instances and seed replication are required before paper claims.

Report parameter counts, training and total development GPU time, peak memory, nominal block applications, and measured generation latency on the same RTX3090. Quality at equal exposure must not be labeled equal-compute efficiency. For paired test differences, any bootstrap intervals resample maps and reflect evaluation uncertainty only, not training-seed variation.

### Verification and execution

39 focused tests passed before launch: independent temporal loss/gradient calculations, penalty decay, readout tie behavior, baseline tick truncations, old final-only behavior, and policy/checkpoint persistence. Profile the actual masked batch; use both3090s. Freeze source/config/protocol hashes before training and verify them afterward.

GPU0: dynamic-plus-monotonic CTM, then uniform CTM.

GPU1: unpenalized dynamic CTM, then the three recurrent-depth and three standard Transformer trials.

Run with the environment in [README.md](README.md):

```sh
python -m scripts.run_readout_comparison --device cuda:0 --cells ctm_dynamic_mono ctm_uniform
python -m scripts.run_readout_comparison --device cuda:1 --cells ctm_dynamic recurrent_depth_lr_low recurrent_depth_lr_mid recurrent_depth_lr_high transformer_lr_low transformer_lr_mid transformer_lr_high
```

All outputs live under `research/results/readout_comparison_v1` and `research/runs/readout_comparison_v1`. See `registry.json` for exact manifests. This protocol is copied to `PLAN_BEFORE_RUNS.md` before launch.

## Completed outcome — 2026-09-23

All nine trials completed their 3,000-update budgets. The audit verified all 27,000 updates, common token/example exposure, learning-rate schedules, retained validation checkpoints, frozen source/config hashes, and selected checkpoint/readout identities. The evaluation helper's initial `test` identifier was corrected to the manifest's `test_id` before any test forward pass; see `evaluation_errata.json`. No endpoint, data, recipe or selection changed.

Dynamic confidence readout reproduced the user's qualitative monotonic pattern at its selected checkpoint: zero aggregate validation answer-token CE or accuracy regressions across the 15 adjacent transitions. Uniform had nine CE and three accuracy regressions, but its absolute generation accuracy was higher (76.6% versus dynamic 73.4%). Dynamic plus the chosen decaying penalty achieved 64.1%. These are validation results, with recipe and checkpoint selected from validation; monotonicity is observed, not guaranteed.

Frozen winners on ordered test maps: uniform CTM 75.0%, Transformer 100.0%, recurrent depth 100.0%. Shuffling the same maps' edge order reduces these to 8.6%, 8.6%, and 9.4%. Train time for the selected full-budget trials was 33.4, 1.9, and 14.9 minutes respectively. Median generation time per batch of 32 on the same RTX 3090 was 287.8, 15.1, and 185.6 ms. This is an exposure/parameter-controlled preliminary comparison, not equal compute or an architecture superiority claim.

See [full report](results/readout_comparison_v1/RESULTS.md), [tick dynamics](results/readout_comparison_v1/TICK_DYNAMICS.md), and the immutable `selection.json`. The `training_objective.validation_metric` metadata field retains the legacy default evaluator description; this study's explicit `checkpoint_selection`, `validation_readouts`, checkpoint `readout_policy`, and selection JSON define the actual policy-aware criterion.

### Next priorities

1. Test CTM optimizer sensitivity under the verified uniform/dynamic readouts on development data. The baseline learning-rate grid changed outcomes dramatically; this experiment varied CTM's objective at a fixed learning rate. Retain these current test results as a closed study rather than tuning against them.
2. For mechanism attribution, give the recurrent baseline matched auxiliary temporal objectives and study selected readouts separately from architecture. Then freeze recipes and run fresh seeds/maps with shuffled-edge training, followed by composition only after reliable lookup.
3. Add compute-matched comparisons and seed uncertainty before paper-level claims. Preserve the successful monotonic-dynamics finding without equating it with better final accuracy.
