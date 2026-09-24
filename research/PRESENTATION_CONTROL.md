# Paired training-presentation control

## Protocol declared before new scientific training

This development study changes only the edge presentation in training. It follows the completed temporal-supervision control. Completed test sets remain closed; only existing train/validation semantic maps are used.

### Design and fixed recipes

Cross three families (CTM, Transformer, recurrent depth), two training presentations (ordered, shuffled), and seeds23/29/31. Reuse the nine ordered runs from `fresh_confirmation_v1`; train nine new shuffled runs. These seeds and validation maps have already been examined and are development data.

Retain the current recipes: CTM uniform temporal supervision at T16/H8, peak LR0.0003, confidence readout,544,839 parameters; Transformer final CE, LR0.0003, final readout,548,660 parameters; recurrent depth final CE at T16, LR0.001, confidence readout,525,984 parameters. No objective, architecture, optimizer or thought-depth tuning is added. Auxiliary recurrent objectives are not carried forward because they did not improve the prior fixed-LR control's mean.

Each new run uses3,000 updates, batch32, warmup30, cosine decay to10% of peak, the existing AdamW settings, FP32 parameters/moments and BF16 autocast. Each condition sees96,000 example presentations,5,184,000 input tokens,192,000 supervised answer/EOS labels. Initialization, data-order and depth seeds are seed, seed+1 and seed+2. Preserve the example sequence in the data file so the semantic maps in each minibatch remain paired with the ordered controls.

### Presentation intervention

Use all2,048 original ordered training maps and their original starts, one-hop queries and answers. For each map, sample one fixed edge permutation, rejecting canonical order. Use generator seed20260924 for training and an independent seed20260925 for validation. The permutation is common across all model families/training seeds and does not change between epochs. This is static shuffled presentation, not online permutation augmentation.

The dataset has only three splits: shuffled `train`, byte-identical ordered `validation`, and `validation_shuffled` containing the same128 validation maps, queries and answers in the same example order, with one fixed noncanonical presentation per map. Training and validation semantic IDs remain disjoint. Token lengths and masked targets match exactly between presentations. No previous test records are loaded or evaluated.

### Primary endpoint and checkpoint policy

**Primary comparison: the checkpoint after exactly3,000 updates for every run.** Evaluate its fixed family readout on both ordered and shuffled validation presentations. Freeze all18 final checkpoint identities before running these paired evaluations. This avoids selecting checkpoints differently according to evaluation presentation. Report mean and sample SD across the three paired development seeds, plus per-seed outcomes and paired training-condition differences on each evaluation presentation.

During training, retain the existing ordered-validation CE checks every100updates and save all validation checkpoints. This keeps the training/validation procedure identical except for the training prompt presentation. Minimum ordered-validation CE chooses a best checkpoint as before; report its ordered-validation score as a clearly secondary development diagnostic. Do not select or replace primary checkpoints using shuffled-validation results. Do not label these validation results as fresh held-out confirmation.

The final checkpoint file's legacy `readout_policy` metadata defaults to final; the declared family readout in this registry is authoritative for evaluation (confidence for CTM/recurrent, final for Transformer). Explicitly store both fields in the checkpoint freeze.

### Verification and cost

Five new checks verify reproducible immutable data, semantic pairing including minibatch membership/targets, detection of a changed query even after a file hash is updated, and exact short GPU replays of the archived CTM/Transformer trainer. The preceding recurrent control already verified exact archived recurrent final-CE replay. Model, optimizer, readout and training code are unchanged in this study.

Validate reused controls against their archived source bytes and saved metrics/checkpoints. New-run manifests, data and configuration hashes, all30 validation checkpoints, final checkpoint hashes and per-example predictions are retained. Report full-budget training time and memory; equal exposure and approximate parameter counts do not make this compute-matched. No new latency benchmark is part of this study.

GPU0 queue: CTM seed23, CTM seed31, Transformer seeds23/29/31. GPU1 queue: CTM seed29, recurrent seeds23/29/31. No adaptive extra trials or favorable seed replacements. Failures are recorded and reported.

### Interpretation

The paired training intervention estimates the effect of fixed shuffled presentation at these existing recipes and budgets. One shuffle realization per map does not characterize generator variability, every possible edge order, or the benefit of online augmentation. Recipes selected under ordered training may not be optimal for shuffled training. Keep optimizer retuning, temporal schedules, online shuffling, multistep composition and scaling as separately declared studies. Any later confirmatory claim needs a new locked evaluation plan.
