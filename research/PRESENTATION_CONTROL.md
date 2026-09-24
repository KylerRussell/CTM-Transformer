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

## Infrastructure interruption and restart-safe relaunch — 2026-09-24

The first launch (attempt 1, started 14:09 UTC) stopped at about 14:26 UTC when the whole container was killed; the host rebooted at 14:47. Training, logging and the agent session stopped together, and no worker error or `*.failure.json` was produced. Only the first two cells had started: `ctm_shuffled_seed23` (last complete update 1,831) and `ctm_shuffled_seed29` (1,722). The restart also removed the temporary environment under `/tmp`.

No declared cell is replaced or dropped. Training is deterministic, so each interrupted cell is rerun from scratch with its declared seeds. Its partial outputs are moved unchanged to `research/runs/presentation_control_v1_interrupted/attempt1/`, and the attempt-1 logs and continuation state are moved to `research/results/presentation_control_v1/interrupted_attempt1/`. `interruption_replay.json` compares each rerun with the updates its interrupted predecessor completed (all logged fields except wall time). A mismatch is reported as a reproducibility finding; it does not invalidate the from-scratch rerun.

Orchestration only changed; every file in `pre_run_source.json` is unchanged and verified before each attempt. `scripts/supervise_presentation_control.py` runs the frozen worker `scripts.run_registry_trials` for the unfinished cells in the declared queue order. Its summaries record the same `worker_source_sha256` that the freeze audit requires. The same frozen freeze, evaluation and summary modules follow, in the evaluation grouping of `complete_presentation_control.py`. It additionally:

- runs preflight checks: frozen hashes, dataset validation, torch 2.6.0+cu124, idle GPUs, free disk, and the five frozen checks, including exact archived-trainer replay on both GPUs;
- keeps completed cells and reruns only unfinished ones, so repeated launches are safe; a lock prevents concurrent supervisors;
- records a nonzero worker exit as `<cell>.failure.json` and never retries it; a signal-killed worker is an infrastructure interruption and is resumed on the next launch;
- records every attempt and phase in `supervisor_state.json`.

Start or resume with `research/launch_presentation_control.sh`. The environment is now under `$HOME` (`~/.venvs/ctm-research`, installed from `research/readout-comparison-environment.txt`; driver libraries in `~/.local/share/ctm-nvidia-driver`), which persists across container restarts. The script detaches the supervisor from the terminal. An XDG autostart entry (`~/.config/autostart/ctm-presentation-control.desktop`) relaunches it when the desktop session starts after a restart. The supervisor deletes that entry when the study completes.
