# CTM-Transformer comparative research plan

Draft: 2026-09-21. Based on static inspection of repository revision `73f90c4`, the saved evaluation JSON files, and the primary literature linked below. Existing metrics have not been reproduced. Hardware is now verified: two RTX 3090 GPUs (24 GiB each), approximately 629 GiB system RAM, and PCIe connectivity. Total compute budget and submission date remain unconfirmed; sizes and schedules below are planning proposals. See [research development log](research/README.md) for completed correctness fixes, GPU setup, and preliminary timings; the audit below records the original revision.

**Working title:** *Does Temporal Synchronization Improve Recurrent Computation in Language Models?*

**Central question:** At comparable training and inference budgets, do explicit activation histories and synchronization improve prediction, iterative problem solving, or generalization beyond what recurrent depth alone provides?

The contribution should be a precise architecture and a controlled empirical answer. A broad claim that the model “thinks” or that recurrence itself is novel would be difficult to defend. A useful negative result could identify when temporal history fails to help, provided the baselines, controls, and analysis are strong.

## 1. Establish what the current model actually does

The default CTM path maintains a latent vector and pre/post-activation histories for every token position and thought layer. Synchronization of post-activation histories generates queries that causally attend to token embeddings. A gated synapse combines retrieved information with the current latent state; a temporal MLP processes pre-activation history. Layers repeat across thought ticks. The output head reads the final latent vector concatenated with the current token embedding.

Distinguish three axes in the paper: token position, layer index, and thought tick. Temporal history here is across internal computation; buffers reset on each forward call. It is not persistent memory across generated tokens. The current implementation also mixes previous layer outputs through attention residuals.

The matrix-stream option replaces both the temporal MLP and synchronization. It is a separate architecture variant and cannot substantiate a claim about those mechanisms.

Before committing to paper-scale training, resolve these issues:

| Observation from the current code | Required action and scientific implication |
|---|---|
| `NeuronLevelModels` stores one MLP per `nlm_groups`; with the default value 1, all neurons share that MLP. This contradicts comments describing unique per-neuron weights. | Specify shared, grouped, and independent temporal MLPs accurately. In this implementation, `nlm_groups=d_latent` gives independent temporal MLPs. Benchmark the parameter and speed costs before choosing the canonical variant. |
| Token positional embeddings are disabled by default, while attention reads static token embeddings. | Test order sensitivity. For the ordinary path without other order-aware features, the final position should be invariant to permutations of earlier tokens that preserve their multiset and the final token: its attention reads an unordered set of embeddings. This is a static-code inference to verify. Thought time does not supply token order. Establish a position-aware main configuration. |
| Synchronization generates attention queries; the final head reads latent state plus token embedding, not synchronization directly. | Describe this explicitly as an adaptation of CTM. Do not claim an exact reproduction of the original CTM readout. |
| FiLM parameters and per-tick adapters are indexed by thought tick and allocated only up to configured T. | Use a shared readout, or define and validate an extrapolatable conditioning rule before evaluating beyond trained T. Merely allocating extra untrained rows is not valid depth extrapolation. |
| `loop_pos_emb` is allocated but has no forward use found in this revision. | Audit whether intended configuration switches affect execution; remove claims about inactive features or implement and verify them in a new version. |
| `sync_sparse_pairs` is configurable but is not passed through `ThoughtLayer` to `SynchronizationComputer`. | Wire and verify this before interpreting a sweep over the number of synchronization pairs. |
| Attention residuals run unconditionally; optional FEEC, Hyperloop, Engram, sparse attention, and quantization create additional differences. | Make the main architecture reproducible and attribute each retained addition. Use switches or controls to isolate the CTM mechanisms. Preserve the current recipe as a named reference configuration. |
| The harness tokenizes context and continuation separately, omits the first continuation token when context is empty, and permits missing checkpoint keys. | Check scoring against the reference harness adapter for the exact tokenizer, including whitespace boundaries, empty contexts, truncation, padding, and shifts. Require strict checkpoint loading or an explicit validated migration. These are audit findings, not a demonstrated explanation for the saved scores. |
| Generation reruns the prefix, and the model projects vocabulary logits every tick. | Profile the actual implementation. Make caching and final-only readout explicit engineering tasks if claiming practical inference efficiency; verify numerical equivalence after optimization. |

Useful correctness gates include future-token perturbation invariance, identical-prefix logits under padding/batching, tiny-dataset overfitting, checkpoint round trips, and gradient agreement with checkpointing enabled versus disabled. Stateful FIFO buffers make the last check particularly relevant. Add generation and rolling likelihood support only when the selected evaluation requires them.

## 2. Interpret existing evidence conservatively

The stored `eval_t_sweep.json` reports the following raw accuracy percentages:

| Task | T=1 | T=4 | T=8 |
|---|---:|---:|---:|
| ARC-Easy | 27.02 | 29.80 | 29.04 |
| HellaSwag | 26.09 | 25.96 | 26.14 |
| PIQA | 52.88 | 53.10 | 52.72 |
| LAMBADA | 0.0194 | 0.0194 | 0.0194 |

LAMBADA perplexity is approximately 0.883M, 3.175M, and 2.184M respectively. This is a task-specific perplexity and should not be presented as general corpus perplexity. These results give mixed evidence about additional ticks, with weak performance on several tasks. They do not establish superiority over another architecture or a consistent improvement with compute.

The JSON files do not include the checkpoint identity, complete training budget, tokenizer revision, or per-example predictions. Recover those if possible, but treat these runs as exploratory. Do not infer statistical significance from differences between aggregate scores or treat the reported example-level standard errors as training-seed uncertainty.

## 3. Position the work against the literature

The original [Continuous Thought Machines paper](https://arxiv.org/abs/2505.05522) supplies the temporal neuron processing and synchronization ideas. The paper must identify which mechanisms are retained, changed, or omitted.

Use [Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*](https://arxiv.org/abs/2502.05171) as the principal recurrent-depth reference. Its prelude, recurrent core, input injection, coda, and variable-depth training matter; repeating an arbitrary block is an incomplete reproduction. Consult the [authors' implementation](https://github.com/seal-rg/recurrent-pretraining) and document every small-scale adaptation. Their released 3.5B model is contextual evidence, not a controlled baseline for a smaller model trained on different data.

[Universal Transformers](https://arxiv.org/abs/1807.03819) establish earlier depth recurrence. More recent work also narrows the novelty claim: [LoopFormer](https://arxiv.org/abs/2602.11451) addresses variable compute through trajectory conditioning and shortcut consistency; [Parcae](https://arxiv.org/abs/2604.12946) studies stability and scaling of looped language models; [SpiralFormer](https://arxiv.org/abs/2602.11698) studies recursion across sequence resolutions. Review these before freezing the design, and repeat the literature search before submission. This initial search is not an exhaustive novelty review.

Choose one recent comparator that most directly overlaps the eventual claim: LoopFormer for budget flexibility, or Parcae for stability. Additional architectures should earn their place by testing a specific alternative explanation.

## 4. Hypotheses and baselines

Freeze primary hypotheses after development pilots and before confirmatory runs:

1. **Prediction at a fixed budget:** CTM improves held-out next-token cross-entropy or task accuracy over recurrent depth at comparable training and inference compute.
2. **Useful additional computation:** a single CTM checkpoint improves with additional ticks, including on harder examples, and improves more per unit of compute than the recurrent baseline.
3. **Mechanism attribution:** temporal processing and synchronization explain the benefit after controlling for state size, readout conditioning, auxiliary losses, and other architecture changes.

Train the following core families from scratch on identical data:

| Family/control | Purpose |
|---|---|
| Standard decoder-only Transformer with untied layers | Establish ordinary language-model quality and compute efficiency. |
| Recurrent-depth Transformer adapted from Geiping et al. | Test whether repeating a well-designed Transformer core explains the result. |
| Canonical CTM-Transformer | Test the complete proposed mechanism. |
| CTM scaffold with temporal MLP and synchronization both replaced | Preserve its cross-attention, input injection, residual structure, and head while removing the claimed mechanisms. This is not automatically equivalent to the published recurrent-depth model. |
| Recurrent/scaffold control with a simple history buffer or larger state | Test whether extra memory capacity explains a gain attributed to synchronization. |

Use parameter-matched and compute-matched Transformer configurations where needed. A T=1 CTM is a useful ablation, but it is not the standard Transformer baseline. Preserve strong conventional design choices in the ordinary baseline; use the scaffold controls for strict mechanism isolation.

## 5. Fair comparison protocol

Use one tokenizer, document-level data splits, fixed corpus revisions, the same sequence lengths and packing policy, and identical evaluation examples. Count unique trainable parameters, including embeddings and heads, once; also report non-embedding parameters and recurrent state memory. Record effective depth as actual block applications, including nested loops.

Run two complementary training comparisons:

- **Equal training FLOPs:** stop each model at common cumulative compute checkpoints. Faster or cheaper architectures may see more tokens. This is the main compute-efficiency comparison.
- **Equal training tokens:** compare common data-prefix milestones and report the compute spent. This measures data efficiency and is not an equal-cost experiment.

A trajectory can supply checkpoints for both analyses where its budget reaches them. Report inference quality versus measured FLOPs and wall time, not simply versus tick count. CTM and recurrent-depth ticks may have very different costs. Include vocabulary projections, temporal MLPs, synchronization, attention residuals, recomputation, and teacher costs when applicable. The shortcut `6 × unique parameters × tokens` is not a sufficient compute estimate for these recurrent models.

Use the same hardware, precision, batch sizes, context lengths, and warmup/timing procedures for speed comparisons. Measure prefill separately from decoding, peak memory, median latency and a tail percentile, and throughput. Record caching policy and include each model's required recurrent/KV state. Until comparable optimized decoding exists, qualify latency as implementation performance.

Give each family an equal documented tuning budget, allowing learning rates and stability settings to differ when validation supports it. Log all attempted configurations. Start with ordinary cross-entropy and no distillation for architecture attribution. If teacher supervision is needed, provide identical teacher signals to every family and report cache generation cost and provenance separately as well as in total resource accounting.

## 6. Experiments in stages

**Stage A: correctness and feasibility.** Complete the audits, implement the two main baselines, and profile small configurations. Select a compact position-aware CTM with explicit temporal MLP sharing semantics and a shared output head. Use ordinary dense attention and a single level of thought recurrence initially; assess additional components only when needed. Confirm all three main families can learn a tiny dataset and an elementary task.

**Stage B: controlled algorithmic pilot.** Use two complementary generated tasks, for example multi-digit arithmetic and pointer chasing or graph reachability. Train on bounded lengths/depths and reserve longer inputs and greater reasoning depths for generalization tests. Hold graph size fixed while varying path depth in a separate test so length and difficulty are not always confounded. Use disjoint instances, fixed generators, exact answer metrics, and identical answer-only supervision. Treat chain-of-thought supervision as a separate experiment if introduced.

Run five seeds at a small scale, initially about 5–20M non-embedding parameters subject to profiling. Evaluate trained thought budgets such as 1, 2, 4, and 8. Test 12 and 16 only after eliminating tick-index limits and clearly distinguish depths seen during training from unseen depths. Diagnose both trained depth-specific models and truncation of a single checkpoint; these answer different questions.

**Stage C: language modeling.** Choose a fixed, documented slice of the available training corpus with document-disjoint validation and test sets, plus one held-out domain for transfer. The primary metric is final-tick token-weighted cross-entropy; report perplexity derived from it. Synthetic answer-only losses and language-model all-token losses must be reported separately.

A provisional main scale is 30–100M non-embedding parameters and 100–500M tokens for the initial language pilot. These are feasibility proposals, not a claim of sufficient training. Use learning curves and measured throughput to set the final budget. Train the three principal families with at least three seeds, using common FLOP and token checkpoints. Keep the existing ARC-Easy, HellaSwag, PIQA, and LAMBADA suite as secondary evaluation after repairing/validating the adapter. Add difficult generation benchmarks only if pilot competence makes them informative.

**Stage D: confirmation and scale transfer.** Run the decisive ablations below, then repeat the three main families at a second feasible size, provisionally 100–300M non-embedding parameters. A second scale strengthens generality; if unavailable, scope the paper to the measured regime. Include the selected recent recurrent comparator when the intended claim requires it. Do not spend most of the budget on one large CTM checkpoint before the controlled pilot succeeds.

## 7. Ablations that establish causality

First run a 2×2 factorial study inside the same CTM scaffold:

| Temporal MLP over history | Synchronization queries | Interpretation |
|---|---|---|
| Off | Off | Scaffold control |
| On | Off | Temporal processing contribution |
| Off | On | Synchronization contribution |
| On | On | Full CTM and interaction between mechanisms |

“Temporal MLP off” replaces that update with a current-state transformation. Keep the post-activation buffer when synchronization requires it. “Synchronization off” derives queries from the current state using a comparable projection. Train each variant from scratch; give removed capacity an appropriately matched control rather than adding unused parameters.

Then prioritize history lengths 1/4/8/16; shared/grouped/independent temporal MLPs; synchronization summary versus sparse pairs; and fixed versus variable-depth training. Test loss/readout changes separately: final-only CE versus multi-tick supervision, with and without the monotonic penalty, and shared readout versus tick-specific FiLM. Apply the same auxiliary recipe to recurrent baselines when feasible. An objective rewarding improvement across ticks cannot itself establish a unique architectural advantage.

The temporal objective must also be crossed with trained thought depth. The user reports that uniform and dynamic aggregation behave similarly at T4 but diverge at T16; a T4-only comparison cannot assess that interaction. The [frozen development protocol](research/OBJECTIVE_DEPTH.md) tests uniform/dynamic × T4/T16 with H8 fixed, matched exposure, and both final-tick and label-free minimum-entropy readouts. Keep training objective, inference policy, and checkpoint-selection criterion explicit. Treat this one-seed development study as a basis for replication, not a confirmatory result or evidence of biological fidelity.

For mechanism analysis, measure state changes, prediction changes, and held-out loss at every tick. Freeze a latent state while varying FiLM/readout tick to test how much apparent refinement comes from the head. Combine trained ablations with inference interventions such as erasing old history, perturbing synchronization, or shuffling temporal order. Shuffling history tests the temporal MLP but may leave an unweighted Gram synchronization statistic unchanged; choose interventions based on the actual statistic. Report intervention distribution shift and avoid treating attractive trajectory plots as proof of reasoning.

Adaptive stopping is optional follow-up work. Choose thresholds only on validation data and include their overhead, calibration, average ticks, and tail latency. The main claim should first hold at fixed compute budgets.

## 8. Statistical protocol and decision gates

Save per-example outputs and compare models on the same examples. Report means and spread across training seeds. For evaluation uncertainty, use paired bootstrap intervals over independent examples or documents; avoid treating correlated tokens as independent observations. For synthetic data, distinguish training seed from test-generator seed. Preselect a small primary endpoint set and label other analyses exploratory.

Use development pilots to estimate variance and select a practically meaningful effect threshold before confirmation. For example, decide whether an accuracy improvement at fixed compute or a compute reduction at fixed quality is large enough to matter for the chosen application. There is no universal percentage that guarantees publishability.

Progression gates:

1. **Correctness:** resolve order sensitivity, scoring, configuration, state, and checkpoint issues; all families learn elementary tasks.
2. **Signal:** pilots show a reproducible advantage or an informative, reproducible failure mode. Flat ticks alone are a diagnosis target.
3. **Attribution:** history/synchronization controls distinguish the mechanism from extra compute, memory, readout conditioning, and training losses.
4. **Generalization:** confirm on fresh seeds and locked test data, preferably across two task families and a second scale.
5. **Claim selection:** choose an architecture paper if the mechanism adds value, an efficiency paper if quality-versus-cost improves, or a focused empirical study if the main finding is a well-supported limitation. Admission to any venue remains uncertain.

## 9. Budget, deliverables, and paper structure

Allocate an initial planning budget of roughly 15% to correctness/profiling/pilots, 45% to main comparisons, 25% to decisive ablations, and 15% to confirmation and reruns. Estimate actual GPU hours from measured aggregate training throughput: `GPU hours = accelerator count × tokens / aggregate tokens per second / 3600`, summed per stage when T changes. Add tuning, evaluation, failed runs, and teacher preparation explicitly. Existing run-command throughput comments are not a measured budget for new experiments.

An illustrative sequence is one week for specification/audits, one to two weeks for baselines and pilots, two to four weeks for main runs, and two weeks for confirmation and writing. This is an organizational sequence, not a hardware-backed completion estimate. Prefer fewer well-controlled experiments with seeds over many unsupported feature combinations.

Required figures/tables:

1. Architecture diagram with token, layer, and thought axes; show all recurrent state and readout paths.
2. Held-out quality versus cumulative training FLOPs, with seed variation.
3. Quality versus inference FLOPs and latency, including a frontier of ordinary Transformer configurations.
4. Task difficulty/length versus accuracy at several thought budgets, marking unseen depths.
5. Factorial ablation table with parameters, memory, compute, and uncertainty.
6. Tick-wise dynamics plus an intervention that tests the proposed explanation.

Draft the paper as: problem and hypotheses; related work; exact update equations and complexity; experimental protocol; main comparisons; ablations and mechanism analysis; limitations and conclusion. Write methods during implementation, but write the abstract and final claims after confirmation. Derive cost for each synchronization method rather than assuming every variant materializes a D×D matrix.

Release the canonical configuration, baseline configurations, pinned environment, data manifests and generator seeds, evaluation adapter checks, strict checkpoint loader, compute accounting, raw metrics, per-example predictions where distributable, and plotting scripts. Record code commit, tokenizer, dataset revision, seed, tokens, training FLOPs, thought-depth distribution, checkpoint, and hardware for every result. Store a machine-readable experiment registry and distinguish exploratory runs from confirmatory runs.

**First concrete milestone:** a reproducible small-scale comparison of standard Transformer, recurrent-depth Transformer, and canonical CTM on two controlled tasks, accompanied by a verified evaluation harness and measured cost. Use that evidence to set the language-model scale and finalize the paper's claim.

## Development update — 2026-09-23

The [across-tick recipe and matched baseline pilot](research/READOUT_COMPARISON.md) completed nine GPU trials with approximately matched parameters, equal token exposure, and three new trials per family. Dynamic CTM confidence readout exhibits the reported monotonic aggregate validation trend across T16, but uniform CTM has the better absolute validation score in this grid. Frozen ordered-test results are 75% for CTM and 100% for both baselines; shuffled-order transfer is 8.6–9.4% for all. Measured costs strongly differ. These results motivate CTM optimizer tuning, matched auxiliary-supervision controls, shuffled-order training and fresh seed/map confirmation before scaling or paper claims. They do not invalidate the earlier version-specific CTM findings or establish broad architecture rankings.


## Fresh-map seed confirmation completed

The [CTM learning-rate grid](research/results/ctm_lr_v1/RESULTS.md) selected uniform temporal supervision with peak LR 0.0003 and confidence readout at T16. Its development checkpoint reaches 100% validation and 99.22% fresh ordered-map accuracy. The best dynamic cell remains LR 0.001 in this restricted grid; this does not identify or refute the earlier historical recipe.

The [locked-recipe confirmation](research/results/fresh_confirmation_v1/RESULTS.md) completed nine new runs at seeds 23/29/31, with all checkpoints frozen before evaluating 512 new maps. Ordered accuracy mean ± sample SD: CTM **87.50% ± 9.29 points**, ordinary Transformer **100% ± 0**, recurrent depth **63.74% ± 8.07 points**. Shuffled-order means are **13.22–14.52%**. Seed 17 is a separate development reference, excluded from these aggregates. The strong seed-17 recurrent result did not replicate at the three new seeds; CTM is also seed-sensitive. Equal exposure and approximate parameter counts are verified, while compute, temporal objectives and tuning history remain unequal.

The next bounded study should match recurrent temporal supervision, followed by a separate shuffled-training control before composition or scaling. See [interpretation and remaining controls](research/results/fresh_confirmation_v1/INTERPRETATION.md). The current test set is closed for further selection.


## Recurrent temporal-supervision control completed — 2026-09-24

The [paired objective control](research/results/recurrent_temporal_v1/RESULTS.md) completed six new GPU runs and reused three verified final-CE controls at seeds23/29/31. The same recurrent architecture, LR0.001, T16, confidence readout and data exposure were used across objectives. Eleven GPU checks verified CTM loss/gradient parity, recurrent behavior, checkpoint metadata and unchanged final-CE training.

Selected validation accuracy mean ± sample SD is **66.93% ± 4.30 points** for final CE, **56.77% ± 14.18** for uniform, and **57.03% ± 20.83** for dynamic aggregation. Auxiliary supervision raises training time from15.8 to24.8–25.6 minutes per run without improving the mean at this learning rate. All27,000 updates, including9,000 reused control updates, were audited. These are development results on128 validation maps; no test forward belongs to this study.

This is a bounded negative result for adding the objectives at fixed optimizer settings. It does not establish their attainable performance after objective-specific LR/schedule tuning, nor does it retest CTM tick-prefix monotonicity. Retain final CE as the current recurrent reference and proceed to the separate shuffled-training task control before composition or scaling. See [interpretation](research/results/recurrent_temporal_v1/INTERPRETATION.md).


## Paired presentation control completed — 2026-09-24

The [paired shuffled-training control](research/results/presentation_control_v1/RESULTS.md) trained nine new runs with one fixed noncanonical edge order per training map and paired them with the nine ordered-training confirmation runs (seeds 23/29/31, unchanged recipes, equal exposure). A container kill interrupted the first attempt. A restart-safe supervisor reran the two affected cells from scratch, reproducing every completed update exactly; no cell was replaced.

Chance is 1/7 because every map is a single 8-cycle. After shuffled training, all families are near chance on both presentations: ordered-map accuracy 16.41–19.79%, shuffled-map accuracy 15.36–16.15%. Relative to ordered training, this costs 51–80 points on ordered maps and gains only 1–4 points on shuffled maps. The shuffled-trained Transformer memorizes the 2,048 fixed examples (training CE about 0.002); CTM and recurrent depth remain underfit. Ordered-trained models are at or below chance on shuffled presentation, so the earlier ordered successes most likely use a positional shortcut rather than content-based retrieval.

The one-hop retrieval gate is not passed by any family, and the current task cannot separate the architectures. The next development study should first establish a data regime in which an ordinary Transformer reliably learns order-independent lookup: online-generated maps, several queries per sequence and larger node sets. Architectures should then be compared on tasks whose difficulty requires serial computation, such as hop count at fixed input size with held-out deeper hops and inference-time tick extrapolation. See [interpretation](research/results/presentation_control_v1/INTERPRETATION.md).


## Fresh-map calibration completed — 2026-09-24

Fresh 8-node data is impossible: only 5,040 single-cycle maps exist, and the 2,048-map training set was about 40% of them. The studies therefore moved to 12-node maps (chance 1/11). A fresh-map study was [stopped](research/ONLINE_POINTER.md#stopped-at-user-direction--2026-09-24-1900-utc) after its Transformers sat exactly on the chance plateau for 3,000 updates.

The follow-up [Transformer calibration](research/results/pointer_calibration_v1/RESULTS.md) trained 12 runs for 30,000 updates each on 960,000 unique maps. Only one run (LR 0.0003, seed 41) escaped the plateau, at about 12,000 updates, reaching 51.95% held-out hop-1 accuracy. Multi-hop training stayed at chance. Under the declared gate, single-query, answer-only retrieval is not learnable at a cost that permits a CTM–RDT comparison.

The [calibration protocol](research/POINTER_CALIBRATION.md) fixed the rules for that comparison before any CTM or RDT calibration:
- a difficulty band in which the mean of the two families lies between 20% and 80%;
- non-saturating primary endpoints: accuracy by hop including unseen depths, accuracy against tick budget, and sample efficiency;
- disjoint confirmation seeds.

The next step is a dense-supervision pointer format that asks for every node's successor in each sequence. See [interpretation](research/results/pointer_calibration_v1/INTERPRETATION.md).


## Dense-format calibration completed — 2026-09-25

The [dense-supervision calibration](research/results/dense_calibration_v1/RESULTS.md) asked for 6 successors per map and trained 12 Transformer runs for 10,000 updates each on runner v3. Runner v3 reproduces the frozen trainer exactly. Every run learned only permutation exclusion: accuracy stayed at the 12.28% exclusion ceiling, and exact match was 0%. The format gate was not met.

Supervision volume was not the limit. The likelier obstacle is the lookup circuit: keys and values share one alphabet, and in the block format queries must be aligned with answers by position. Exploratory Transformer probes will look for a learnable variant before the next calibration is frozen. See [interpretation](research/results/dense_calibration_v1/INTERPRETATION.md).


## CTM variants and Sync-RDT retrieval study — 2026-09-26

Exploratory probes found a learnable retrieval format (all-keys MQAR) and two structural limits of the reference CTM for in-context retrieval: static cross-attention keys and values, and a shared start state that gives no position its own token. Four repair variants (attention residuals; contextual K/V; contextual K/V with token initialization; contextual K/V with per-tick injection) all stay at chance within 10,000 updates. Meanwhile, the Transformer and RDT learn at LR 0.0003.

To test CTM's mechanisms where retrieval is possible, [Sync-RDT](research/SYNC_RDT_DESIGN.md) adds them to the RDT scaffold. With both mechanisms off it reproduces RDT exactly. A frozen [six-cell, five-seed study](research/results/syncrdt_retrieval_v1/RESULTS.md) found **no declared effect** of either mechanism on retrieval sample efficiency. A single-seed development advantage for synchronization did not replicate. The width-matched RDT control was the only cell to learn at every seed.

The mechanisms still need testing on multi-step computation, which retrieval does not require. Next is a serial-depth task that needs iteration without first needing in-context retrieval. See [interpretation](research/results/syncrdt_retrieval_v1/INTERPRETATION.md).


## First positive mechanism result: synchronization extends serial state — 2026-09-26

On the S₃ word problem (running products, dense labels, fresh words of lengths 1–16), a frozen [seven-cell, five-seed study](research/results/group_s3_v1/RESULTS.md) found:
- **CTM's synchronization-derived query terms inside the recurrent-depth scaffold roughly double the correct running-product prefix at the same step budget** (median 10 against 5). The effect holds at every seed against both RDT and a width-matched RDT, and sync's gain keeps growing with inference steps (median 3, 7, 10, 11 at T = 4, 8, 16, 32).
- CTM's temporal MLPs do not help alone and dilute the effect when combined.
- The standalone reference CTM does not use its ticks and matches the fixed-depth Transformer.
- Plain recurrence uses its steps but does not reliably beat fixed depth. No model extrapolates past its trained lengths.

This supports a specific claim: synchronization of recurrent state histories, used to steer attention, makes recurrent depth more effective. It is not a claim for the CTM architecture as a whole. Before it becomes a paper claim, it needs locked confirmation and mechanism ablations; see the [plan](research/SYNC_CONFIRMATION_PLAN.md) and [interpretation](research/results/group_s3_v1/INTERPRETATION.md).


## What in synchronization helps: pairwise state products in the queries — 2026-09-27

[Mechanism ablations](research/results/sync_ablation_v1/INTERPRETATION.md) of the `sync` cell on the S₃ data found:
- removing learned decay, or computing synchronization from the **current state only** (no history), **preserves** the gain (median correct prefix 11 against 10);
- replacing the pairwise products with linear features of the same history **removes** it (median 3, below plain RDT);
- adding the term to the recurrent state instead of the queries diverges at every seed.

The claim therefore narrows: **pairwise multiplicative features of the recurrent state, used to form attention queries, roughly double how far recurrent depth tracks state.** CTM's temporal machinery (history window, decay, ordering) does not carry the benefit here. Next come A7 (self-pairs only), then a locked confirmation of `sync` and `current` with seven new seeds.


## Pre-LM review and course correction — 2026-09-28

An external [deep-research review](research/DEEP_RESEARCH_REPORT.md) concluded that the language-model phase should wait. The synchronization result must first be separated from learning-rate, query-scale and fixed-depth-training confounds, and tested on the non-solvable A₅. The earlier CTM departs from the published CTM (readout, loss, ticks, budget), so its failures are weak evidence about CTM itself.

The project keeps a CTM-inspired but **maximally faithful, scalable** design, [CTM-LM](research/CTM_LM_DESIGN.md), and adopts the review's pre-LM plan:
1. CTM-LM capability checks;
2. the Sync-RDT confound sweep (per-cell learning rates and scale controls, plus a gated-attention competitor);
3. randomized-depth training;
4. A₅ with a curriculum, plus equal-compute comparisons;
5. a mini language-model stability gate.

Future protocols add continuous primary metrics and exact paired p-values with Holm correction. The bio-inspired factor is dropped.


## Locked confirmation: the synchronization effect did not replicate — 2026-09-28

On seven never-used seeds with a locked test evaluated once, **neither `sync` nor `current` exceeded the RDT controls**: median correct prefix 5–6 for every cell ([interpretation](research/results/sync_confirmation_v1/INTERPRETATION.md)). Outcomes are bimodal by seed. Every recurrent cell sometimes escapes to a serial solution of positions 9–16 (at 1–2 of 7 seeds) and otherwise stays near the fixed-depth profile. The development study's 5-of-5 pattern was most plausibly chance, which is the statistical weakness the external review flagged. **The claim that second-order query features extend state tracking is withdrawn**, and the ablation conclusions built on it are superseded.

What remains: recurrent models use extra steps; nothing generalizes past the trained length; the CTM-inspired reference does not use its ticks; and the retrieval findings stand. The central open problem is now **making serial learning reliable**. Architecture comparisons need escape-probability endpoints with many more seeds, and training changes (randomized depth, curriculum, learning rate) come before any mechanism claim or language-model phase.


## Reliability baseline and randomized depth — 2026-09-28

The [reliability study](research/results/reliability_s3_v1/INTERPRETATION.md) (3 cells × 20 new seeds) established the baseline:
- **RDT beats the fixed-depth Transformer by 10.1 points** on positions 1–16 (Holm p = 0.022). This is the project's first result under exact paired tests with multiplicity control.
- **CTM-LM is 13.9 points below RDT** (Holm p = 0.0013).
- Escape to the serial solution is rare: 2/20 for RDT and 0/20 for the others.

[Randomized-depth training](research/results/randdepth_s3_v1/INTERPRETATION.md) (log-normal Poisson, mean about 16) was then compared on the same seeds.
- **No declared test was significant after Holm.**
- For RDT, the direction is favourable: escape 7/20 against 2/20, +9.9 points, 13 of 20 seeds improved. It also made progress per step faster and extra steps harmless.
- For CTM-LM, randomized depth under CTM's loss removed tick use entirely.

**Positional-encoding flaw.** The study also showed that **no S₃ study so far could have measured length extrapolation**. Every model uses learned absolute position embeddings, and training lengths stop at 16, so position rows 17 and later are untrained. The "nothing extrapolates" finding is withdrawn as an architectural statement. The next S₃ protocol uses NoPE or RoPE, keeps randomized depth for RDT, and adds a length curriculum and per-cell learning rates.


## A reliable recipe for serial learning — 2026-09-30

[Development probes](research/results/recipe_probes/RECIPE_PROBES.md) chose RoPE with no absolute position table. They also found that **the learning rate was the dominant factor**: randomized-depth RDT escaped at 1 of 6 seeds at 3e-4 and 3 of 3 at 1e-3.

The frozen [recipe confirmation](research/results/recipe_s3_v1/INTERPRETATION.md) used 4 RoPE cells × 30 new seeds, each cell at its tuned learning rate. **All four declared tests passed after Holm correction.**
- Randomized-depth RDT beats fixed-depth RDT by **+14.3 points** on positions 1–16 and **escapes at 24/30 against 0/30**.
- It stays **+26.0 points above the Transformer on positions 17–24**. This is short extrapolation: accuracy is 0.75 and 0.55 at positions 17 and 18, then above chance. It is not length generalization.
- CTM-LM is **22.7 points below fixed-depth RDT**. Under the declared rule, it leaves the serial-task track.
- As a secondary result, fixed-depth RDT again beats the Transformer, by +15.7 points.

**Adopted recurrent-depth recipe:** RoPE, log-normal-Poisson randomized depth, and a per-cell tuned learning rate.

**Caveat resolved (2026-10-01):** a [learning-rate-matched control](research/results/lrcontrol_s3_v1/INTERPRETATION.md) put fixed depth at 1e-3 on the same 30 seeds. Randomized depth still wins by +11.7 points and escapes at 24/30 against 1/30 (Holm p < 10⁻⁶). The effect belongs to randomized depth, not to the learning rate.

**Next:** that control, optionally; then A₅ with the adopted recipe; CTM-LM retrieval; and the mini language-model gate.


## Scope and path to pretraining — 2026-10-01

**Paper scope.** The paper is about the **feasibility and benefits of the CTM compared with standard Transformers and recurrent-depth Transformers (RDTs)**. The synthetic studies were preliminary testing, done to choose recipes before small-scale pretraining of each model type at sizes this system can afford (2× RTX 3090, 24 GiB each).

**What the preliminary phase settled:**
- **Recurrent-depth recipe:** RoPE, log-normal-Poisson randomized depth (confirmed against a learning-rate-matched control), and a learning rate tuned per architecture.
- **CTM-LM:** a CTM as faithful as language-model scale allows. It uses its ticks, but at tiny scale (0.6M parameters) it trails RDT on serial state tracking. It has not yet shown one-hop retrieval.
- **Width matters at tiny scale.** On A₅, RDT at width 96 never learned, while width 192 with randomized depth and 30,000 updates solved it on 2 of 3 development seeds. A matched Transformer handled 2–4 positions. So tiny-model failures can reflect capacity, not design, and that applies to reading CTM-LM's results too. A₅ is kept as a [development observation](research/results/recipe_probes/RECIPE_PROBES.md). A frozen A₅ study is not planned, because it tests recurrent depth rather than the CTM.

**Change to the gate:** CTM-LM joins pretraining **regardless of the synthetic gates**, because it is the paper's subject. The preliminary findings become stated expectations (tick use; weaker serial tracking than RDT at tiny scale; retrieval unproven), to be tested at language-model scale.

**New track: CTM-inspired additions to RDT.** If CTM-LM trains worse than RDT, the paper's constructive question is whether CTM mechanisms improve an RDT. Candidates, with the evidence so far:

| Candidate | Source in the CTM | Prior evidence here |
|---|---|---|
| Synchronization-derived query terms in the RDT core | action synchronization driving attention | positive in development; **not replicated** in the locked confirmation, under the old recipe (learned positions, LR 3e-4, fixed depth). Worth one re-screen under the new recipe |
| Output-synchronization readout | the CTM readout from pairwise state products | untested in RDT |
| Tick-selection loss and certainty-based early exit | min-loss plus max-certainty loss; adaptive compute | in CTM-LM with randomized depth it collapsed tick use. In RDT it may enable per-token adaptive depth at inference |
| Learned initial state | CTM's learned start state and history | untested in RDT (RDT starts from zeros) |
| Neuron-level temporal models over step history | per-neuron temporal MLPs | the `history` cell did not help (development) |

Synthetic S₃ is near ceiling for randomized-depth RDT (95.8%). So candidates are screened where there is headroom:
- validation loss in the mini language-model gate (the outcome that matters);
- S₃ at a reduced update budget, or on positions 17–24, as a secondary measure.

At most one or two candidates go forward as a "CTM-augmented RDT" arm, each needing a locked confirmation before it counts as a claim.

**Next steps:**
1. **Cost profiling.** Throughput, memory and feasible batch size for Transformer, RDT and CTM-LM at about 10M, 25M and 50M parameters, at language-model sequence lengths, on one RTX 3090. This fixes the feasible model sizes and token budgets.
2. **CTM-LM retrieval at larger width (optional, development).**
3. **Mini language-model gate.** Each architecture, plus the screened CTM-augmented RDT candidates, at about 10–20M parameters on a few hundred million FineWeb-Edu tokens, with two learning rates per arm. It checks stability and picks the learning rates.
4. **Pretraining.** Transformer, RDT, CTM-LM and at most one CTM-augmented RDT, at the sizes and budgets that step 1 shows are feasible.

**Update — 2026-10-01: 500M arms, token budgets and the learning-rate sweep.**
- **Model size.** The user set about 500M parameters as the minimum. Recurrent families get two designs each:
  - "heavy", where most parameters recur;
  - "compute-aware", where most parameters are applied once.

  All five arms run on both RTX 3090s ([profile](research/results/lm_cost_profile/PROFILE.md)).
- **Token budgets (user decision):**
  - heavy arms (RDT, CTM-LM): 1B tokens;
  - compute-aware arms: 2B;
  - Transformer: both 1B and 2B, as the reference for each group.

  That is about 20 days of both GPUs.
- **Initialization.** Transformer and RDT arms use the width-scaled std √(2/5d). With the recipe's fixed 0.02, RDT-heavy's gradients grew about 1.5× per recurrence at initialization.
- **Learning rates.** Step 3 is replaced by a learning-rate scaling sweep ([protocol](research/LR_SCALING.md)). Each arm's optimum is measured at three narrower widths and at two training lengths, then extrapolated to its 500M size and token budget. Its cost is about 6 days of both GPUs.

  The CTM-augmented RDT screen follows the sweep, at the sweep's widths and with the fitted rates.

