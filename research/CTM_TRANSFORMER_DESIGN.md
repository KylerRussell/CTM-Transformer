# The CTM Transformer: design (2026-10-14)

## Goal

The user's words: "take the CTM arch and apply it so that we can merge it to work properly with language modeling in an LLM, which is frankly stupid to move away from the transformer. In this way we are basically just slightly adapting a transformer using the theory proposed in the CTM and CMM papers."

The paper compares the CTM Transformer's best with the RDT's best, with the plain Transformer as the baseline.

## Why the earlier CTM-LM is not this

The earlier CTM-LM, and the "matched CTM" of 2026-10-13, were a Transformer backbone feeding a separate per-position CTM head. The Transformer ran once, and the head read it through one cross-attention per tick. Diagnostics on 2026-10-14 (`research/RDT_RECIPE.md`) showed the head's ticks contributed nothing:
- Every gain came from pairwise products of a token-conditioned start state.
- The tick states were about 30 times smaller than that start state, so the multiplicative readout gave them almost no gradient.
- The synchronization decays were stuck at 0.

## Principle

Start from the exact Transformer and adapt its own components with CTM and CMM mechanisms. Every mechanism is zero-initialized, so that:
- the untrained CTM Transformer at 1 tick is exactly the 8-layer Transformer at d 704 (47.2M non-embedding parameters);
- at any tick count it is exactly Dans, the RDT's best (`ctm_transformer/rdt_recipe.py`).

Any difference from those two references therefore comes from the CTM/CMM mechanisms.

## The model (`ctm_transformer/ctm_transformer.py`)

Layers 1–2 and 7–8 run once. Layers 3–6 form the thinking group and run for T internal ticks. The residual stream at the group's output is the state, and it is normalized and added to the input every tick, as in Dans.

| CTM / CMM idea | Where it lives in the Transformer |
|---|---|
| Internal ticks, decoupled from tokens (CTM) | The thinking group runs T ticks. Training draws T per step (log-normal Poisson, mean 15, at most 48), as for Dans. |
| Neuron-level models: each neuron processes the history of its own pre-activations with private weights (CTM) | In each thinking layer's feed-forward network. Each of the 1,856 hidden neurons keeps the history of its gate pre-activation over the last 4 ticks. A private per-neuron MLP (hidden width 4) maps that history to an activation, which is added to the SwiGLU activation with a per-neuron scale initialized to zero. |
| Synchronization: neurons' co-activation over internal time is the representation (CTM), normalized as a proper average whose scale does not depend on the tick count (CMM's α/β) | The state's neurons, normalized, are paired at random: 704 pairs for action and 704 for output. Each pair keeps a decayed running average of the products of its two neurons over ticks. The learned decay is softplus-parametrized so it can move away from 0, unlike the CTM's clamp. |
| Synchronization drives attention (CTM: the query comes from action synchronization) | In each thinking layer, the attention query is the Transformer's query plus a zero-initialized projection of the action synchronization from earlier ticks. |
| Synchronization is the output (CTM) | The output synchronization is added to the state before the last two layers through a zero-initialized projection. It reaches the logits through the residual stream, never only through the multiplicative path that trapped the CTM-LM. |

The extra parameters are the neuron-level models, 4 query projections (2.0M) and the output projection (0.5M). The total is about 50.4M non-embedding parameters, against 47.2M.

**Implementation decisions for speed and memory (2026-10-14):**
- **Neuron-level models at hidden width 4, not 8.** Per-token, per-neuron MLPs over 1,856 neurons are memory-bound. At hidden width 8 they cost 24 ms per layer per tick, against 4 ms for the whole feed-forward network. They run as one fused `torch.compile` kernel, which matches the reference computation to within 4e-7.
- **Pair products as two one-hot matrix multiplies**, twice as fast as gathering by index.
- **Synchronization recomputed in the backward pass**, like the layers: only its running sums are stored, and the attention query reads them inside the checkpointed layer.
- **Micro-batch 8 with 32 accumulation steps** (Dans uses 16 × 16). Tokens per step and the averaged gradient are the same.


**Options for later tests:**
- the CMM attention sink;
- the CTM loss over several ticks with a most-certain-tick readout.

The first build keeps Dans's training (final-tick loss, randomized depth), so that only the architecture differs.

## Testing

1. **Correctness (unit tests):**
   - At initialization the model equals Dans at any depth, and the 8-layer Transformer at depth 1.
   - Every mechanism receives a gradient, and changes the output once its zero-initialized parameter is nonzero.
   - Checkpointed and plain execution agree.
2. **Liveness (diagnostics on training checkpoints).** Remove each mechanism at evaluation time (neuron-level models, synchronization in the queries, synchronization at the output) and measure the loss increase. A mechanism the model does not use is reported as dead. This is the check that caught the CTM-LM's dead ticks.
3. **Language modelling:**
   - the full CTM Transformer at 400M tokens, lr 2e-3, seeds 1234 and 2, the same data, schedule and evaluation as Dans;
   - references: Dans (3.569, 3.579), the 8-layer Transformer (3.618, 3.638), and the 20-layer Transformer at equal compute to 4 ticks (3.562).
   - **Rule, fixed before the runs:** the CTM mechanisms add value if the two-seed mean is at least 0.02 below Dans's (3.574), about one seed spread. A smaller difference is a tie.
4. **Then:**
   - ablations of each mechanism;
   - the CMM sink and the CTM loss options;
   - a learning-rate check;
   - the scaling ladder with the Transformer, the RDT's best and the CTM Transformer's best.

## Progress (2026-10-14)

Two seeds of CTfull (all three mechanisms) are training at 400M tokens. The step-381 snapshots were not written: snapshots are saved only on checkpoint steps (every 50), so the snapshot interval is now 400. The liveness check therefore used the rolling checkpoint at step 500, on CPU, with 4 held-out windows (`research/results/ctm_transformer/diagnose_*`).

| | Seed 1234 | Seed 2 |
|---|---:|---:|
| Held-out loss at step 381 (Dans / 8-layer Transformer) | 4.470 (4.443 / 4.422) | 4.463 (4.465 / 4.455) |
| Loss at 1 / 4 / 8 / 16 / 32 ticks, step 500 | 4.293 / 4.196 / 4.193 / 4.193 / 4.193 | 4.297 / 4.173 / 4.169 / 4.169 / 4.169 |
| Loss increase when sync_query is removed | **0.050** | **0.058** |
| Loss increase when sync_output is removed | 0.009 | 0.004 |
| Loss increase when neuron_models are removed | −0.000 (unused) | 0.001 (unused) |

- **Synchronization drives the attention.** The model relies on the synchronization-derived queries, the CTM's central mechanism.
- **The output synchronization is used weakly.**
- **The neuron-level models are unused** although their scales grew. The thinking group converges by 4–8 ticks and stays flat to 32, so each neuron's pre-activation history is nearly constant at the fixed point. A per-neuron model of a constant history adds nothing that the activation function does not already provide.
- **Using a mechanism is not the same as gaining from it.** Against Dans on the same tokens, training loss over the last 80 steps is +0.023 (seed 1234) and +0.004 (seed 2).

## Result at 400M tokens (2026-10-15)

| | Seed 1234 | Seed 2 | Mean |
|---|---:|---:|---:|
| **CTM Transformer (CTfull)** | **3.586** | **3.590** | **3.588** |
| Dans (RDT's best) | 3.569 | 3.579 | 3.574 |
| 8-layer Transformer (baseline) | 3.618 | 3.638 | 3.628 |

- **Against the baseline:** the CTM Transformer beats the 8-layer Transformer on both seeds (−0.032, −0.048; mean −0.040).
- **Against Dans:** +0.014. Under the pre-registered rule (at least 0.02 below Dans) this is a **tie, on the worse side**. The CTM mechanisms do not add value beyond the RDT's best at this scale.
- **Compute:** the CTM Transformer costs about 2.9 times Dans per tick.

**Liveness at the final checkpoint** (32 held-out windows, loss increase when removed at 16 ticks): all three mechanisms are used.

| Mechanism | Seed 1234 | Seed 2 |
|---|---:|---:|
| sync_query | 0.063 | 0.070 |
| sync_output | 0.011 | 0.011 |
| neuron_models | 0.010 | 0.005 |

The neuron-level models, unused at step 500, came into use late. The model relies on the mechanisms but does not end up better than Dans, which suggests they replace capacity Dans gets elsewhere rather than adding to it. The per-mechanism ablations (CTsq, CTso, CTnlm) test this directly.

On these 32 windows, loss converges by 8 ticks and is flat to 32 (seed 1234: 3.835 at 1 tick, 3.580 at 4, 3.569 at 8, 3.568 at 16 and 32), with no drift.

## Ablations (2026-10-15, running)

- **Runs:** CTsq (synchronization in the queries only), CTso (synchronization at the output only) and CTnlm (neuron-level models only), each on seeds 1234 and 2, at 400M tokens. They run in that order on each GPU.
- **Rules, fixed before the runs** (in `scripts/ctm_transformer_study.py`):
  - a variant beats Dans if its two-seed mean is at least 0.02 below 3.574;
  - against CTfull (3.588), a difference under 0.02 is a tie.
