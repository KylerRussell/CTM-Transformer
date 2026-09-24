# Recurrent-depth temporal-supervision control

## Protocol declared before scientific training

This is a new **development** study following the closed fresh-map confirmation. It isolates the temporal objective within the existing recurrent-depth model. It does not reopen completed test sets or claim a fully matched architecture comparison.

### Declared cells and pairing

Cross objectives **final_ce, uniform, dynamic_aggregate** with seeds **23, 29, 31**. Reuse the three completed final-CE recurrent runs from `fresh_confirmation_v1` unchanged, including their confidence-readout checkpoint choices. Train six new auxiliary-objective cells. These previously examined seeds are now development seeds, not new confirmatory replications. No favorable seed replacement, adaptive objective addition, or learning-rate sweep is allowed in this study.

Keep the recurrent configuration fixed: width96, FFN288, four heads, prelude1/core2/coda1, T16, deterministic zero recurrent initial state, positional embeddings, dropout0, 525,984 parameters. Peak LR0.001, AdamW betas0.9/0.95, weight decay0.1, clip1, 30-update warmup and cosine decay to10% of peak. Train3,000 updates at batch32, FP32 parameters/moments and BF16 autocast. Initialization seed is the declared seed, data-order seed is seed+1, depth seed is seed+2. Each cell receives96,000 example presentations,5,184,000 input tokens,192,000 supervised answer/EOS labels. Train/validation maps are the existing ordered_pointer_v1 splits.

GPU0 queue: uniform seed23, dynamic seed29, uniform seed31. GPU1 queue: dynamic seed23, uniform seed29, dynamic seed31. Every seed's two new objectives run on different identical GPU models. All validation checkpoints are retained.

### Exact objective control

- **Final CE:** train only the final tick, unchanged from the existing baseline.
- **Uniform:** mean supervised-token CE across all16 ticks, matching CTM's uniform ramp with zero monotonic penalty.
- **Dynamic aggregate:** for each supervised token, average the CE at the minimum-CE tick and the maximum-certainty tick, then average over supervised tokens. Match CTM's normalized native-logits-dtype entropy and earliest-tie behavior exactly. Ground truth chooses one branch of the training objective; it never enters the deployment readout.

The shared coda/head decodes each recurrent state. The decoded state never feeds back into recurrence. Gradients through recurrence remain connected. No new parameters, monotonic penalty, schedule, distillation, or curriculum are introduced.

Dynamic training certainty follows CTM's existing BF16 arithmetic under autocast. Inference confidence remains the existing minimum-FP32-entropy policy. These different precision conventions are explicit and covered by parity tests; changing them would be a separate ablation.

### Fixed readout and selection

Use **confidence readout only** for checkpoint selection across all nine cells. This policy was already fixed for the three reused controls. It also matches the selected CTM's label-free readout. Holding it fixed isolates the objective and avoids giving new cells additional policy-selection opportunities.

Every100updates evaluate original validation answer/EOS CE and unrestricted exact answer+EOS generation. Select minimum token-weighted validation CE, retaining the earlier checkpoint on exact ties. No held-out/test forward belongs to this study. Report every objective and every seed; mean and sample SD span the three development seeds. Paired objective differences use the same seed and same validation maps. These selected validation scores are development estimates, not generalization confirmation.

### Verification, cost, and interpretation

Eleven GPU checks verify loss/gradient parity against the actual CTM at T16 in FP32/BF16, earliest-tick selection and prompt masking, recurrent unroll/gradient equivalence, unchanged initialization/parameter state/inference, checkpoint objective metadata, and exact final-CE replay against the prior archived runner.

Full-size GPU profiling precedes training. Auxiliary training executes49 decoder-block applications per sequence (1 prelude +16×[2 core +1 coda]), compared with34 for final CE. Blocks differ from CTM blocks; counts are not FLOPs. Record actual training-loop wall time and peak memory. This is equal exposure and identical recurrent architecture, not equal compute. LR0.001 remains fixed even if another objective would prefer a different LR.

The previous model, readout, optimizer and data-loader code stay unchanged; the shared runner gains an explicit recurrent-objective argument and accurate auxiliary-coda block accounting. Existing checkpoint/config schemas still load for inference. New checkpoints and run manifests record the objective separately from the unchanged baseline architecture config. The previous study's source archives preserve its earlier runner; reused-control provenance is checked against those archived bytes.

Failures must be recorded and never replaced by favorable runs. Before any later architectural claim, use these development findings to declare a separate matched-supervision, optimizer/compute study and new confirmation plan. Shuffled-training and multistep-task controls remain separate milestones.
