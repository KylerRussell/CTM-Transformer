# CTM-augmented RDT screen — protocol (2026-10-01)

## Question

If CTM-LM trains worse than the RDT, can CTM mechanisms improve the RDT? This screen runs on language modelling, which is the outcome that matters, and picks at most two mechanisms for a confirmation. It is a development screen, not a paper claim.

## Mechanisms (`ctm_transformer/ctm_rdt.py`)

The core output x_t of each recurrence step plays the role of the CTM's post-activation state.

| Mechanism | What is added | Source in the CTM |
|---|---|---|
| `sync_query` | The decayed pairwise products of x_0..x_{t−1} over P = d random pairs add W·S to every core attention query. | Action synchronization driving attention (the LM-scale Sync-RDT cell, with CTM's full decayed history in place of an 8-step window) |
| `sync_readout` | The decayed pairwise products of x_1..x_T add W·S to the coda input. | Output synchronization readout |
| `learned_init` | A learned start state x_0 replaces the zero vector. | The CTM's learned start state |

**Exactness.** Every added weight and the start state are zero-initialized. At initialization each variant computes exactly the RDT: same loss, same logits, and the same gradients for the shared parameters (`tests/test_ctm_rdt.py`). With the same seed, a variant and the RDT share the initialization of all RDT parameters and see the same data and the same depth sequence. The comparison is therefore paired.

**Not screened:**
- **Neuron-level history models:** they did not help in development.
- **The tick-selection loss with certainty-based early exit:** this is an inference-time analysis of the trained RDT (per-token adaptive depth), not a training change.

## Design

| Setting | Value |
|---|---|
| Model | The compute-aware RDT (11/2/11 layout) at width d 512, the middle width of the [learning-rate sweep](LR_SCALING.md) |
| Training | 100M FineWeb-Edu tokens at that width's best learning rate from the sweep; otherwise the sweep's settings |
| Arms | The RDT and each of the three single-mechanism variants |
| Seeds | 1234 and 1235. The sweep's seed-1234 RDT run is reused. |

This adds seven runs of about 1.5 hours each.

**Metric:**
- held-out cross-entropy at depth 16, on the sweep's validation windows (offset 40,000, 1,024 windows);
- the paired difference from the RDT with the same seed.

Training throughput relative to the RDT is reported too, because a mechanism that costs compute must earn it.

**Decision rule (fixed before the runs):**
- A mechanism advances if it lowers held-out loss on **both** seeds **and** by at least **0.01 nats** on average.
- At most the two best advance.

An advancing mechanism (or the pair of them) then gets:
- a check at neighbouring learning rates;
- a confirmation at the 768 width;
- only then, a 500M "CTM-augmented RDT" arm, under a locked protocol.

## Running

The screen is the last stage of the learning-rate sweep's queue (`scripts/lr_sweep.py`). Its runs start once the d 512 RDT width is resolved. Results are written to [results/ctm_rdt_screen/SCREEN.md](results/ctm_rdt_screen/SCREEN.md) by `scripts/summarize_ctm_rdt_screen.py`.
