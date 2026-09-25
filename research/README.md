# Research development log

## Hardware and environment — 2026-09-21

Verified two RTX 3090 GPUs, each with 24 GiB VRAM, and approximately 629 GiB system RAM. `nvidia-smi topo -m` reports a PCIe host-bridge connection (PHB), not NVLink. Plan independent pilot runs/seeds on the two GPUs initially; each has its own 24 GiB memory budget. Use RAM for data preparation and caching. Profile before introducing CPU parameter offload or multi-GPU training.

**Update 2026-09-24:** `/tmp` is cleared when the container restarts. The persistent environment is now `~/.venvs/ctm-research` (installed from `research/readout-comparison-environment.txt` with the cu124 index), with driver libraries in `~/.local/share/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu`. Substitute these paths in the commands below.

The isolated Python environment for this session is `/tmp/ctm-research-venv`. It contains PyTorch 2.6.0 with CUDA 12.4, pytest, and the repository dependencies. This wheel is listed in the [official PyTorch installation instructions](https://docs.pytorch.org/get-started/previous-versions/). The environment is temporary; recreate it after `/tmp` is cleared.

Both devices were visible through NVML, but `libcuda.so` was absent from the container. The host runs driver 550.144.03. The matching `libnvidia-compute-550_550.144.03-0ubuntu1_amd64.deb` was downloaded from the [NVIDIA Ubuntu 22.04 repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/) and extracted, without system installation, under `/tmp/ctm-nvidia-driver`. Actual CUDA matrix multiplication succeeded on both GPUs after setting the library path below. No kernel driver was changed.

From the repository root, in this session:

```sh
export LD_LIBRARY_PATH=/tmp/ctm-nvidia-driver/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
. /tmp/ctm-research-venv/bin/activate
```

To create a fresh environment on a machine with working CUDA driver libraries:

```sh
python3 -m venv /tmp/ctm-research-venv
. /tmp/ctm-research-venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt pytest
```

`environment.txt` records the versions used for the initial checks. Restoring that snapshot requires the CUDA wheel index above for the `torch==2.6.0+cu124` entry. The separately extracted driver libraries are not pip packages.

## Completed correctness work

- [x] Check actual CUDA execution on both GPUs.
- [x] Test causal prefix invariance under changed future tokens, padding, and batching.
- [x] Verify order invariance without token positions and sensitivity with positions in the ordinary CTM path.
- [x] Repair existing sample-print formatting for Python 3.10 and guard version-specific optional compiler flags. Verify that the training CLI imports and accepts the new positional-encoding option under PyTorch 2.6.
- [x] Expose `--use_positional_encoding` in both training configuration paths. The legacy default remains false; new text experiments should explicitly enable it.
- [x] Correct temporal MLP sharing documentation: `nlm_groups=1` shares an MLP across all neurons; `nlm_groups=d_latent` gives independent MLPs. Test sharing and gradients.
- [x] Connect `sync_sparse_pairs` from configuration through every thought layer.
- [x] Restore the input histories of each tick during checkpoint recomputation. Before the fix, identical forward losses hid different gradients. Check every parameter gradient with and without dropout, including Hyperloop, matrix streams, and loop embeddings.
- [x] Apply learned loop embeddings in the thought update and verify that they receive gradients.
- [x] Reject nonpositive thought budgets and budgets exceeding tick-indexed parameter tables. A shared head without FiLM or loop embeddings continues to support larger T.
- [x] Verify strict state-dict round trips and inference at a larger T with a shared head.
- [x] Verify a tiny training batch can be learned.
- [x] Add a GPU timing probe with warmup, synchronized timing, BF16, peak memory, and JSON metadata.

The initial 16 research tests produced 9 failures before the fixes and all passed afterward. Additional coverage for checkpointing variants and a tiny-data learning check brings the verified set to 23 distinct tests. These are targeted checks on the actual model, not a claim that the entire pre-existing repository test suite passes.

Run the independent groups in two terminals, or launch both and wait for both exit codes:

```sh
CTM_TEST_DEVICE=cuda:0 python -m pytest tests/test_research_correctness.py \
  -k 'checkpointing or tiny_batch' -q
```

```sh
CTM_TEST_DEVICE=cuda:1 python -m pytest tests/test_research_correctness.py \
  -k 'not (checkpointing or tiny_batch)' -q
```

The tests prefer CUDA when available. An explicit unavailable CUDA device fails instead of silently falling back to CPU. CPU remains useful for lightweight checks and CI. Current gradient-equivalence checks use FP32; the timing probes exercise BF16 training, not BF16 gradient equivalence.

## Preliminary GPU timing

```sh
python -m scripts.profile_research --device cuda:0 --thought-steps 2 \
  --output research/results/preflight_gpu0_t2.json
```

```sh
python -m scripts.profile_research --device cuda:1 --thought-steps 4 \
  --output research/results/preflight_gpu1_t4.json
```

These probes use a 6.24M-parameter model, 4 layers, width 256, a 4096-token vocabulary, batch size 4, sequence length 128, shared temporal MLPs, and BF16 autocast. They include backward and AdamW steps. Parameters remain FP32, and the source records the full configuration. They measure a repeated synthetic batch with no data loading or compilation and do not estimate quality, a fully optimized throughput ceiling, or the throughput of the planned larger language models. Some probes ran alongside correctness checks on the other GPU; shared CPU resources can affect timing. Re-profile the selected actual pilot configuration before budgeting long runs.

Initial observed results (20 timed steps after 5 warmup steps):

| Device / budget | Tokens/s | Median step | Peak allocated GPU memory |
|---|---:|---:|---:|
| GPU 0, T=2 | 2,343 | 234 ms | 0.494 GiB |
| GPU 1, T=4 | 1,274 | 399 ms | 0.833 GiB |

These are separate-device timing observations; do not interpret their ratio as a controlled scaling result. The JSON files preserve all per-step timings and losses.

## Compatibility and remaining work

Applying loop embeddings changes outputs for configurations that previously enabled the unused `use_loop_pos_emb` flag. Old checkpoints with that flag contain embeddings that did not learn through this path. Reevaluate them; reproduce historical results using the original code revision. Similarly, an old nondefault `sync_sparse_pairs` configuration may describe a checkpoint that actually used 256 pairs. Recover the effective shape from the checkpoint before migration. Checkpointed training after the gradient fix should be treated as a new experiment version.

## Canonical configuration milestone

- [x] Freeze a position-aware CTM architecture, independent temporal MLPs, sparse synchronization, shared readout, and final-tick loss in [`configs/ctm_reference_v1.json`](configs/ctm_reference_v1.json).
- [x] Make attention residuals optional while retaining legacy defaults; disable them in the reference preset.
- [x] Require complete configurations, record manifest hashes, and provide a launcher with explicit data splits, seed, device, and fresh checkpoint directory.
- [x] Align research training and profiling: BF16 autocast with FP32 parameters and AdamW state.
- [x] Pass 11 canonical checks and 23 existing model checks on GPUs.
- [x] Complete a 101-step synthetic training smoke on GPU 0, strictly reload its checkpoint, verify finite FP32 parameters/optimizer moments, and run a separate evaluation forward pass.
- [x] Profile the final 29,557,841-parameter configuration on GPU 1: approximately 1,627 tokens/s, 303 ms median update, and 1.861 GiB peak allocated memory.

See [canonical model details](CANONICAL_MODEL.md) for exact choices, commands, limitations, and parameter breakdown. Machine-readable records are `results/canonical_profile_gpu1.json` and `results/canonical_training_smoke.json`. The synthetic training checkpoint lives under `/tmp`; it is not a paper result. The updated probe used the second GPU concurrently with the training smoke, so host contention can affect timing.

## Baselines and common experiment interface

- [x] Implement the standard decoder Transformer and an explicitly documented Geiping-style recurrent-depth adaptation.
- [x] Freeze baseline presets, including a separate variable-depth training variant.
- [x] Share configuration loading, model construction, training/data policy, profiling, checkpoint loading, and conditional-likelihood evaluation across all three families.
- [x] Record data/tokenizer/source hashes, training tokens, depth counts, parameter categories, and GPU timing.
- [x] Pass 19 baseline/shared-runner GPU checks plus 38 canonical/evaluation regressions. Both baseline families learn a tiny batch; all three families train and save through the shared runner.
- [x] Profile each full development preset on the 3090s. See [baseline details and results](BASELINES.md).

The shared runner uses a new windowing and learning-rate policy, documented in `BASELINES.md`. Keep those runs distinct from the earlier legacy-trainer CTM smoke. The baselines are implemented; matched budgets and meaningful task results remain to be established.

## Controlled tasks and first pilots

- [x] Generate fixed addition and pointer-chasing datasets with semantic split disjointness, difficulty annotations, and file hashes.
- [x] Add a common 71-token character vocabulary, answer/EOS-only supervision, and token-weighted accumulation and validation.
- [x] Add unconstrained greedy exact-answer evaluation with per-example predictions and difficulty breakdowns.
- [x] Pass 10 task-specific checks and 30 baseline/canonical regression checks on GPUs.
- [x] Complete six single-seed pilots: three models × two tasks, 600 updates and 19,200 examples each, using both 3090s.
- [x] Review learning curves: addition learns in distribution; pointer chasing remains near guessing. Defer seed expansion until the pointer learning gate passes.

See [task protocol and interpretation](ALGORITHMIC_TASKS.md), [results](results/algorithmic_v1/RESULTS.md), and [learning curves](results/algorithmic_v1/learning_curves.png). These are exploratory, unmatched-budget results, not paper evidence of architecture superiority.

## Pointer-learning diagnostic

- [x] Construct a balanced 32-example fitting set and a one-hop version of the original maps, with disjoint train/validation/held-out maps.
- [x] Add paired edge-order and start-node interventions, explicitly labeled as training-map probes.
- [x] Pass four construction tests; complete six GPU runs with the existing 600-update presets.
- [x] Check the gates: all three models fit the tiny set perfectly; none learns held-out one-hop lookup. Query/presentation changes largely destroy the tiny-set fit.

See [diagnostic protocol and interpretation](POINTER_DIAGNOSTICS.md) and [results](results/pointer_diagnostics_v1/RESULTS.md). The failure already occurs at single-edge retrieval in this setup, so multi-step composition and CTM-versus-recurrence claims remain premature.

## Ordered-edge lookup control

- [x] Preserve the one-hop maps, query starts, answers, splits, example order, and 600-update presets; change only edge presentation to canonical source-node order.
- [x] Add a paired held-out shuffled-order evaluation on the same 128 maps and queries.
- [x] Pass four construction checks and finish all three GPU runs using both 3090s.
- [x] Check the gate: ordered held-out accuracy rises to 24.2% / 38.3% / 21.9% (Transformer / recurrent depth / CTM), but no model reaches 90%; shuffled transfer remains weak.

See [protocol and interpretation](ORDERED_POINTER.md) and [results](results/ordered_pointer_v1/RESULTS.md). These are partial improvements at a fixed development budget, not a converged or matched-compute comparison.

## Fixed-slot value-copy control

- [x] Keep the ordered maps, splits, example order, and 600-update presets; fix the source query to A and recompute labels.
- [x] Add paired held-out shuffled-edge and unseen-query-B probes, plus the corresponding training-map probes.
- [x] Pass five construction checks and complete all three runs using both 3090s.
- [x] Pass the copying gate: all models score 100% on 128 unseen maps. Shuffled transfer is 7.8% / 7.0% / 8.6%; all retain A's answer on every query-B probe.

See [protocol and interpretation](FIXED_POINTER.md) and [results](results/fixed_pointer_v1/RESULTS.md). Basic copying is learnable under these presets; reliable query-dependent lookup remains unresolved. These single-seed controls use unmatched parameters and compute.

## Longer-budget ordered lookup

- [x] Prerecord fresh 3,000-update runs with the extended cosine horizon; verify that model presets otherwise remain unchanged.
- [x] Complete all three runs on both 3090s, with exactly five times the previous data exposure.
- [x] Verify recorded source/config/data hashes, all 9,000 learning rates, budget counters, finite losses/gradients, and validation-only checkpoint selection.
- [x] Report held-out ordered accuracy: 76.6% / 94.5% / 65.6% (Transformer / recurrent depth / CTM). Recurrent depth passes 90%; the others remain below the gate. All show weak shuffled-edge transfer.

See [protocol and interpretation](ORDERED_LONG.md) and [results](results/ordered_long_v1/RESULTS.md). Budget and schedule changed together. One seed and unmatched parameters/compute do not establish a model ranking.

## Exhaustive query-selection diagnostic

- [x] Prerecord all eight starts on each of 256 training-probe and 128 validation maps, preserving pairing and exposure labels.
- [x] Pass four construction and four independent aggregation checks; evaluate 9,216 answers across all three frozen checkpoints using both 3090s.
- [x] Reproduce all 1,152 original-query outputs exactly and verify selected checkpoint/data hashes.
- [x] Report all-start validation accuracy of 74.9% / 92.7% / 65.1% (Transformer / recurrent depth / CTM); all-eight-correct map counts are 0 / 64 / 0 out of 128.
- [x] Identify focal errors: Transformer confuses B/D/H successor values; recurrent depth struggles mainly on G; CTM has broader incomplete lookup. These are behavioral findings, not measured attention mechanisms.

See [protocol and interpretation](QUERY_DIAGNOSTIC.md) and [results](results/query_diagnostic_v1/RESULTS.md). Validation influenced checkpoint selection; queries within a map are correlated. No training was performed.

## Attention and CTM temporal-training audit

- [x] Capture all three families' answer-position attention on both GPUs; preserve full logits exactly on all 9,216 queries and pass three capture checks.
- [x] Measure CTM validation answer-token accuracy across ticks: 16.5% → 35.2% → 57.3% → 65.1%.
- [x] Verify final-loss gradients reach every tick and document the partly initialized H=8 history under T=4.
- [x] Identify that the shared runner has not tested auxiliary temporal losses/curricula.
- [x] Correct ignored-token normalization in the inactive dynamic temporal loss; pass five actual-model loss/gradient checks. Preserve pre-fix and post-fix measurements and source versions.

See [protocol, findings, and temporal ablation proposal](ATTENTION_TEMPORAL.md) and [results](results/attention_temporal_v1/RESULTS.md). No training occurred; attention associations and biological motivation are not causal or comparative evidence.

## CTM temporal-supervision experiment at T=4

- [x] Enable weighted temporal CE with independent final-tick validation scoring; pass eleven runner/loss/gradient checks.
- [x] Complete three 3,000-update CTM runs with fixed T4/H8/data/optimizer, using both GPUs.
- [x] Reproduce the earlier final-only control exactly across all 3,000 training losses and 2,688 evaluated outputs.
- [x] Select uniform by the preregistered validation-CE rule: CE 0.332454 versus 0.469648 final-only and 0.515377 later-weighted. No held-out test evaluation.

See [protocol and results](TEMPORAL_ABLATION.md). This experiment did not compare uniform with dynamic aggregation. The user's clarification directs the next study to that objective contrast at T4 versus T16.

## Objective-by-depth comparison and readout diagnosis

- [x] Freeze the uniform/dynamic × T4/T16 protocol with H8, identical data/seed/update schedule, and corrected masked dynamic loss.
- [x] Profile both T16 recipes on actual masked batches: roughly 35 minutes of training each and 2.3 GiB peak allocated memory.
- [x] Pass fifteen objective/gradient/readout/shared-runner checks and start the two T16 runs in parallel; queue the dynamic T4 control. Reuse the archived uniform T4 cell.
- [x] Complete final-tick and common confidence-readout evaluations, audit all four cells, and report the descriptive depth interaction.
- [x] Following the best-across-ticks clarification, audit historical loss versions, evaluate all fixed final checkpoints with confidence readout, and separate raw-tick curves from gold-aware envelopes. Two additional diagnostic tests passed.

See [frozen protocol](OBJECTIVE_DEPTH.md), [original results](results/objective_depth_v1/RESULTS.md), and [essential readout follow-up](results/objective_depth_v1/READOUT_FOLLOWUP.md). Dynamic reaches 54.7% at T4 and 70.3% at T16 using final checkpoints and confidence readout; last-tick CE selected much poorer checkpoints for this policy. These runs do not yet recreate the earlier successful recipe. Equal exposure, unequal compute; no held-out tests used for tuning.

## Across-tick recipe and matched baseline comparison completed

- [x] Implement explicit final/confidence validation and policy-matched checkpoint selection; retain every validation checkpoint and the best per policy.
- [x] Enable the existing monotonic penalty with validated decay settings. Verify recurrent across-tick readout against separate truncations. Pass39 focused tests.
- [x] Freeze three trials per family, equal exposure, seed17, and approximately matched parameters: CTM544,839, Transformer548,660, recurrent depth525,984.
- [x] Execute both GPU queues under the [frozen protocol](READOUT_COMPARISON.md); keep test evaluation downstream of immutable winner selection.
- [x] Finish nine trials and audit all 27,000 updates; freeze three winners before test evaluation and measure generation costs.
- [x] Observe monotonic aggregate confidence-readout CE/accuracy for dynamic CTM, but higher absolute validation accuracy for uniform. Ordered test: CTM 75%, both baselines 100%; shuffled-order test: all 8.6–9.4%. See [results](results/readout_comparison_v1/RESULTS.md) and [tick dynamics](results/readout_comparison_v1/TICK_DYNAMICS.md).

This study tests an explicit current recipe; the exact historical winning version remains unspecified. Equal trial count and data exposure do not match compute or lifetime tuning. See `results/readout_comparison_v1/registry.json` for all declared configurations.

## CTM LR study and fresh-map confirmation completed

- [x] Freeze the uniform/dynamic × three-LR development grid; reuse the two middle-LR controls and train four new cells on both GPUs.
- [x] Generate and validate 512 fresh semantic maps, excluding all 2,560 maps in the prior inventory; pair ordered and shuffled presentations.
- [x] Declare confirmation seeds 23, 29 and 31, with a fixed recipe/readout per family and separate seed-17 development references.
- [x] Complete all six development cells and freeze uniform CTM, peak LR 0.0003, confidence readout alongside the previously selected baselines.
- [x] Complete nine new-seed training runs, freeze all checkpoints, and evaluate paired fresh maps.
- [x] Report seed means/SDs, per-example outcomes, costs and archived provenance.

See [protocol](CTM_LR_CONFIRMATION.md), [CTM LR results](results/ctm_lr_v1/RESULTS.md), [fresh confirmation](results/fresh_confirmation_v1/RESULTS.md), and [interpretation](results/fresh_confirmation_v1/INTERPRETATION.md). Across three new seeds, ordered accuracy is 87.50% ± 9.29 points for CTM, 100% ± 0 for Transformer, and 63.74% ± 8.07 for recurrent depth. Shuffled means are 13.22–14.52%. The development reference reached 99.22% for CTM and 100% for both baselines, illustrating why independent seed replication matters. This confirmation keeps ordered training; shuffled-order training and matched recurrent temporal supervision remain separate follow-ups.

## Recurrent temporal-supervision control completed

- [x] Implement uniform and dynamic temporal losses with exact CTM masking/certainty semantics and zero monotonic penalty.
- [x] Pass eleven GPU checks, including gradient parity and exact archived final-CE replay; profile the full-size objectives.
- [x] Freeze six new runs at seeds 23/29/31, paired with three archived final-CE controls, at fixed LR0.001/T16/confidence readout.
- [x] Start both GPU queues with all validation checkpoints retained.
- [x] Complete and audit all nine cells (six new, three reused); report paired validation differences, seed variability and added compute.

See [protocol](RECURRENT_TEMPORAL_CONTROL.md), [results](results/recurrent_temporal_v1/RESULTS.md), and [interpretation](results/recurrent_temporal_v1/INTERPRETATION.md). At fixed LR0.001, mean validation accuracy is 66.93% ± 4.30 points for final CE, 56.77% ± 14.18 for uniform, and 57.03% ± 20.83 for dynamic. Neither auxiliary objective improves the mean; training time rises from 15.8 to 24.8–25.6 minutes per run. This is development using previously examined seeds and original validation maps. Completed tests remain closed; objective-specific optimizer tuning and shuffled training are separate controls.

## Paired shuffled-training control completed

- [x] Generate one fixed noncanonical presentation per existing training/validation map, preserving semantic example order, queries, answers, target masks and token counts.
- [x] Pass five new data/replay checks, including exact archived CTM/Transformer GPU replay; retain the earlier recurrent replay evidence.
- [x] Freeze nine new shuffled-training runs and nine ordered controls at seeds23/29/31 with unchanged family recipes.
- [x] Start both GPU queues; declare the primary endpoint at exactly3,000 updates for every model.
- [x] Finish training, freeze all18 final checkpoints, and evaluate paired ordered/shuffled validation maps.
- [x] Report paired training-condition differences, seed variability, secondary ordered-selection diagnostics, and archived provenance.

- [x] Record the attempt-1 infrastructure interruption (container killed at about 14:26 UTC, 2026-09-24; two CTM cells partially trained); move its partial outputs aside unchanged.
- [x] Rebuild the pinned environment under `$HOME` so it survives container restarts, and relaunch through a restart-safe supervisor that uses only frozen study code and resumes automatically after a restart.

- [x] Verify that both interrupted CTM cells rerun exactly (1,831 and 1,722 matching updates) and audit all54,000 updates.

See [protocol](PRESENTATION_CONTROL.md), [results](results/presentation_control_v1/RESULTS.md), and [interpretation](results/presentation_control_v1/INTERPRETATION.md). Chance is 1/7 (14.29%) because every map is a single 8-cycle. After shuffled training, ordered-evaluation accuracy is 19.79% (CTM), 19.53% (Transformer) and 16.41% (recurrent depth); shuffled-evaluation accuracy is 15.36–16.15%. Every family is therefore near chance. Shuffled training costs 51–80 points on ordered maps and gains only 1–4 points on shuffled maps. The shuffled-trained Transformer reaches about 0.002 training CE, which is memorization of the 2,048 fixed examples (about 47 passes each). CTM and recurrent depth remain underfit (about 0.69 CE). Ordered-trained models score at or below chance on shuffled presentation, consistent with a positional shortcut. This is development on existing maps with fixed per-map shuffles, not online augmentation. Completed tests remain closed. See the interruption section of the protocol for the relaunch procedure.

## Fresh-map studies and Transformer calibration

- [x] Declare the fresh-map 12-node study (repeated vs fresh vs multi-hop). Only 5,040 8-node single-cycle maps exist, so fresh 8-node data is impossible.
- [x] Stop it after 4 of 27 cells: fresh-map Transformers sat exactly on the chance plateau (training CE 1.205 = ln 11 / 2), while the repeated-map Transformer memorized. See [stop record](ONLINE_POINTER.md#stopped-at-user-direction--2026-09-24-1900-utc).
- [x] Declare and run the [Transformer calibration](POINTER_CALIBRATION.md): peak LR 0.0003/0.001/0.003 × 30,000 updates × one-hop/multi-hop × seeds 41/43, every map presented once. Also fix the operating-point rules for the later CTM–RDT comparison.
- [x] Apply the declared rules: D1 selects LR 0.0003; **D2 not met** (mean hop-1 accuracy 29.69%). Only one of 12 runs escaped the plateau, at about 12,000 updates, reaching 51.95%. Multi-hop training stayed at chance. See [results](results/pointer_calibration_v1/RESULTS.md) and [interpretation](results/pointer_calibration_v1/INTERPRETATION.md).

## Dense-format Transformer calibration

- [x] Implement the dense `pointer_dense` format (6 queries per map) and runner v3; verify exact v3–v2 parity for all three families.
- [x] Freeze and run the [dense calibration](DENSE_POINTER.md) (12 Transformer runs, 10,000 updates, every map presented once).
- [x] Apply the declared rules: **D2 not met** (mean hop-1 answer accuracy 12.73%). Every run learned only permutation exclusion: it finished at the 12.28% exclusion ceiling, with training CE 1.846–1.849 against 1.816 for an exclusion guesser and 0% exact match. See [results](results/dense_calibration_v1/RESULTS.md) and [interpretation](results/dense_calibration_v1/INTERPRETATION.md).

## Next tasks, in order

The evaluation-scoring milestone is implemented and validated; see [evaluation validation](EVALUATION.md). Historical benchmark scores still need reevaluation from their original checkpoints.

1. Run short exploratory Transformer probes, reported as development only, to find a learnable retrieval variant: interleaved query/answer pairs, distinct target symbols, and more depth. Freeze the next calibration only for a variant that clearly learns, then calibrate CTM and RDT under the fixed operating-point rules.
2. Then test serial computation directly: vary hop count at fixed map size, train on bounded hops, and evaluate held-out deeper hops and extra thought ticks. One-hop retrieval alone is not expected to separate recurrent from fixed-depth models.
3. Use the measured costs to select comparable model/budget variants, add FLOP accounting, and begin the main experiment registry.

The controlled-task development comparison, compact fresh-map three-seed confirmation, recurrent temporal-supervision control, and paired presentation control are complete. No paper-scale training run has been launched. See `../RESEARCH_PLAN.md` for the overall study design.
