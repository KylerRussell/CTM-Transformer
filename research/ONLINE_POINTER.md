# Fresh-map pointer study: training-data regime and multi-hop depth

## Protocol declared before new scientific training — 2026-09-24

This development study follows the [paired presentation control](results/presentation_control_v1/INTERPRETATION.md). In that control, every family reached near-chance accuracy after shuffled-presentation training. The Transformer memorized its 2,048 fixed examples (training CE about 0.002), and CTM and recurrent depth stayed underfit. This study asks two questions:

1. **Data regime.** Does one-hop retrieval under random presentation become learnable when each training map is seen once instead of about 47 times? Budget, recipes and seeds stay unchanged.
2. **Serial depth.** When models are trained on 1–3 hops of fresh maps, how does accuracy change at unseen depths of 4–8 hops? For CTM and recurrent depth, how does it change with the inference tick budget?

These seeds and recipes have been examined before. All splits are development data; this study defines no locked test set. Any later confirmatory claim needs its own locked plan.

### Why 12 nodes

Eight-node single-cycle maps number only 7! = 5,040, and earlier studies already used 2,560 of them. The 2,048-map training set was about 40% of every possible map, so fresh 8-node data is impossible. The study therefore uses single 12-cycles: there are 11! ≈ 39.9M of them, and chance is **1/11 = 9.09%**. Because the node count changes, the data-regime question is answered by a 12-node repeated-map control inside this study, not by comparison with the 8-node runs. The 512 earlier 12-node maps (from `algorithmic_v1`) are excluded from every split.

### Data (`research/data/online_pointer_v1`, generator seed 20260925)

Every example is a single 12-cycle with a random, noncanonical edge presentation and one query, in the existing `algorithmic_char_v1` format. Every map appears in exactly one split, except for the nested repeated control. Hop counts are balanced within each split. Prompts are 68 characters: 70 input tokens and 2 supervised tokens (answer and EOS).

| Split | Maps | Hops | Use |
|---|---:|---|---|
| `train_repeated_onehop` | 2,048 | 1 | First 2,048 records of `train_fresh_onehop`; seen about 47 times each |
| `train_fresh_onehop` | 96,000 | 1 | Each map seen once |
| `train_fresh_multihop` | 96,000 | 1, 2, 3 (32,000 each) | Each map seen once |
| `validation` | 384 | 1, 2, 3 (128 each) | Shared by all runs for the unchanged training-time validation every 100 updates |
| `eval_id` | 768 | 1, 2, 3 (256 each) | Primary endpoints at trained depths |
| `eval_depth` | 1,280 | 4–8 (256 each) | Unseen depths |

The two 96,000-map files are deterministic from the manifest seed. They are hashed in the registry and pre-run record but not archived.

### Design and fixed recipes

Cross three training conditions (repeated, fresh, multi-hop) with three families (CTM, Transformer, recurrent depth) and seeds 23/29/31, for 27 new runs. Family recipes are the ones used in the presentation control, with only data paths and output directories changed:

- **CTM:** uniform temporal supervision, T16/H8, peak LR 0.0003, confidence readout, 544,839 parameters.
- **Transformer:** final CE, LR 0.0003, final readout, 548,660 parameters.
- **Recurrent depth:** final CE, T16, LR 0.001, confidence readout, 525,984 parameters.

The recipes were selected under ordered 8-node training and are not retuned here.

Each run uses the frozen shared runner (`scripts/run_registry_trials.run_trial`) without code changes. Settings: 3,000 updates, batch 32, warmup 30, cosine decay to 10% of peak, existing AdamW settings, FP32 parameters/moments and BF16 autocast. Every run sees 96,000 example presentations, 6,720,000 input tokens and 192,000 supervised labels. Initialization, data-order and depth seeds are seed, seed+1 and seed+2. Validation (all 384 maps, declared readout) runs every 100 updates, and every validation checkpoint is kept. The measured per-update cost at 12 nodes is about 0.68 s (CTM), 0.33 s (recurrent depth) and about 0.03 s (Transformer).

### Endpoints

All primary endpoints use the **checkpoint after exactly 3,000 updates**, the declared family readout and the trained thought depth. All 27 checkpoint identities are frozen before any evaluation. Summaries report mean and sample SD across the three seeds, per-seed values and paired differences. Three seeds support no significance claim.

1. **Data regime (primary).** Hop-1 exact accuracy on `eval_id` for fresh versus repeated training, paired by family and seed.
2. **Retrieval gate (primary).** A family passes if fresh one-hop hop-1 accuracy has a mean of at least 90% with every seed at least 80%. Multi-hop conclusions are drawn only for families that pass. For other families, multi-hop results are reported descriptively.
3. **Serial depth (primary).** For multi-hop training, exact accuracy by hop: 1–3 on `eval_id` (trained depths) and 4–8 on `eval_depth` (unseen depths).

**Secondary and exploratory analyses:**
- Accuracy for CTM and recurrent-depth checkpoints at inference budgets T = 4, 8, 16 and 32. T16 is trained; T32 is beyond the training budget. Only the tick budget changes. Earlier checks confirmed that both architectures run at T up to 64 without tick-indexed parameters.
- Multi-hop-trained hop-1 accuracy.
- Training curves and minimum-validation-CE checkpoints.
- Final training CE as a memorization indicator.
- Measured training time and memory.

Validation influences nothing except the secondary best-checkpoint report.

### Verification

Study checks, which must pass before freezing:
- the scaled-down generator is deterministic, nested, disjoint and excludes prior maps;
- leaked maps are detected even after a manifest hash is updated;
- the supervisor recovers partial cells and passes its replay audit;
- the inference tick override changes only the depth, on both architectures on the GPU.

The exact archived-trainer replay checks from the presentation control also run on both GPUs, both before freezing and before every supervisor attempt.

The freeze audit checks, for every run:
- all 3,000 metric rows, the learning-rate schedule, and finite losses and gradients;
- per-step example, token and label counts;
- the saved validation checkpoints;
- data, configuration and code hashes;
- the final checkpoint.

### Execution

`research/launch_online_pointer.sh` starts the restart-safe supervisor (`scripts/study_supervisor.py`) detached from the terminal. An autostart entry resumes it after a container restart. Completed cells are kept, and interrupted cells are rerun from scratch; each rerun is checked against the interrupted updates. A genuine worker error is recorded as `<cell>.failure.json` and is not retried or replaced.

The GPU queues are fixed in `registry.json`. One-hop cells run first so the data-regime contrast finishes earliest. Evaluation starts only after all 27 checkpoints are frozen. No adaptive extra trials or seed replacements are allowed.

### Interpretation limits

One query per map and answer-only supervision are kept, because the frozen dataset format allows one record per map. Training on several queries per sequence, larger node sets and optimizer retuning remain separate studies. The fresh–repeated contrast changes only map diversity at a fixed update budget. It cannot separate the benefit of diversity from the regularizing effect of never repeating an example. Equal exposure and approximate parameter matching do not equalize compute: a CTM run costs about 20 times a Transformer run. Tick-sweep results beyond T16 are exploratory extrapolation, not a trained condition.
