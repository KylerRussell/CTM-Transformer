# CTM learning-rate study and fresh-map seed confirmation

## Protocol frozen before new training — 2026-09-23

This follows the completed readout comparison. That study is closed: its test outcomes do not enter selection here. The training/model/readout implementations are unchanged and their source hashes must match the earlier archive.

### Stage A: CTM optimizer development

Cross **uniform and dynamic aggregation** with peak learning rates **3e-4, 1e-3, 3e-3** at seed17. Reuse the two completed unpenalized 1e-3 trials exactly, including their metrics/checkpoints and hashes; train four new cells. Exclude the separately tested monotonic-penalty recipe from this grid and keep penalty zero. No trial is added based on intermediate results.

Everything else stays fixed: CTM544,839 parameters, width128, two layers, T16/H8, batch32, 3,000 updates, warmup30, AdamW with the existing betas/weight decay/clip, cosine decay to10% of peak, BF16 autocast and FP32 weights/moments. Each run has96,000 example presentations,5,184,000 real input tokens,192,000 answer/EOS targets. Use the existing ordered training/validation maps only.

Every100updates evaluate final and minimum-FP32-entropy readouts. Select checkpoint and policy by minimum token-weighted validation answer/EOS CE; earlier step wins ties, and final wins readout ties at the same step. Select the CTM recipe by minimum validation CE across the six cells, breaking exact recipe ties by registry order. Save all validation checkpoints. Do not evaluate any old or fresh test maps in Stage A.

GPU0 trains uniform low/high LR. GPU1 trains dynamic low/high LR. The two existing middle-LR controls are reused. This is within-CTM development with unequal historical tuning across families, not a new equal-tuning architecture claim.

### Stage B: locked recipes and independent seed replication

After all six CTM cells are audited, freeze one CTM objective/LR/readout. Carry forward the already selected standard and recurrent baseline recipes unchanged. Record complete config hashes, development checkpoint identities, the readout for each family and the new test-manifest hash in an immutable selection record before new confirmation training or test forwards.

**Primary confirmation seeds:23,29,31**, all fresh initialization/data-order seeds, paired across families. Data-order seed is initialization seed+1; thought-depth seed is seed+2. Train each of the three families at each seed from scratch for the same3,000-update budget, using the same existing train/validation splits. Thus nine new confirmation runs. For each family, freeze its selected readout across all seeds; validation chooses checkpoints under that one policy only. No recipe or policy retuning on new seeds. Numerical failures are reported, not replaced by favorable seeds.

The three seed17 development checkpoints may also receive fresh-map evaluation as a clearly separated reference. They are excluded from the primary mean and seed standard deviation because seed17 informed recipe selection.

### Fresh confirmation maps

Generate512 maps from the5040 possible directed eight-node single cycles, excluding every semantic map in every inventoried JSONL under `research/data` (including earlier OOD-depth maps and paired probes). Inventory hashes and excluded IDs are retained. Choose maps without replacement with generator seed20260924, choose a random start per map, and pair canonical ordered edges with a different random presentation. The maps, starts, answers, and input lengths are identical across the two presentations. Query depth stays one hop.

The fresh dataset contains only `test_id` and `test_shuffled`. It is never supplied to the trainer or checkpoint selector. A validator rechecks the prior-file inventory, exclusion, solutions, pairing, and manifest hashes. A focused test verifies reproducibility, semantic exclusion, pairing, rejection of reused output paths, and detection of newly added conflicting prior data.

These tests measure fresh-map generalization under ordered training and transfer to shuffled presentation. They do not test shuffled training or multistep composition. Training on shuffled edges and matched auxiliary supervision for the recurrent baseline remain separate follow-up controls.

### Final evaluation and interpretation

Wait for all nine fresh-seed runs and lock their selected checkpoints before evaluating fresh test outcomes. Evaluate all frozen new-seed checkpoints and the three development references on both512-map test presentations. Report exact answer+EOS accuracy and supervised CE per seed and family. Primary aggregate: mean and sample standard deviation over the three new training seeds. Report the development seed separately. Models and seeds share the same test maps, so outcomes are paired and are not independent replications of test-generator variability.

Report all six CTM development cells, readout choices, training/validation costs, and failures. Parameter counts and data exposure match the preceding study: CTM544,839; Transformer548,660; recurrent depth525,984. Compute, auxiliary objectives and lifetime tuning effort remain unequal. Do not claim architecture superiority or reproduce an unidentified historical recipe from this restricted task. No post-test retuning belongs to this study.

### Reproduction

Use the pinned environment and CUDA path in `README.md`.

```sh
python -m scripts.run_registry_trials --registry research/results/ctm_lr_v1/registry.json --device cuda:0 --cells ctm_uniform_lr_low ctm_uniform_lr_high
python -m scripts.run_registry_trials --registry research/results/ctm_lr_v1/registry.json --device cuda:1 --cells ctm_dynamic_lr_low ctm_dynamic_lr_high
```

Stage B registry/configs are derived only from the locked Stage A choice and original baseline choices. Source snapshots preserve the protocol, worker, manifests, existing training sources, and the prior-study references. Root outputs: `research/results/ctm_lr_v1` and `research/results/fresh_confirmation_v1`.


## Execution completed

The immutable pre-run protocol is retained in `results/ctm_lr_v1/PLAN_BEFORE_RUNS.md`. Four new LR trials and nine independent-seed confirmation runs completed, with no replaced or failed trials. The six-cell grid selected uniform CTM, peak LR 0.0003, confidence readout. Checkpoint selection used original validation data only; all 12 evaluation checkpoints (nine primary and three development references) were frozen before fresh-test forwards.

See [all LR cells](results/ctm_lr_v1/RESULTS.md), [confirmation results](results/fresh_confirmation_v1/RESULTS.md), and [interpretation](results/fresh_confirmation_v1/INTERPRETATION.md). Primary ordered accuracy is 87.50% ± 9.29 points for CTM, 100% ± 0 for Transformer, and 63.74% ± 8.07 for recurrent depth. These sample SDs span three training seeds on a shared map set. The current fresh tests are now closed for model selection.

Additional reproduction stages, in order, on a clean reconstruction of the output directories:

```sh
python -m scripts.advance_ctm_lr
python -m scripts.summarize_ctm_lr
# Execute all entries in results/fresh_confirmation_v1/registry.json with run_registry_trials.
python -m scripts.freeze_fresh_confirmation
python -m scripts.evaluate_fresh_confirmation --device cuda:0
python -m scripts.summarize_fresh_confirmation
python -m scripts.archive_fresh_confirmation
```

`continue_fresh_confirmation.py` records the actual two-GPU queue and phase transition times in `results/ctm_lr_v1/continuation.json`. Source snapshots, metric/checkpoint hashes, and the full data inventory accompany the results. Output paths are protected against accidental reuse.
