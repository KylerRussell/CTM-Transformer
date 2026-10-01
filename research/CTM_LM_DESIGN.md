# CTM-LM: a CTM as close to the published design as language-model scale allows

**Status: design adopted 2026-09-28. The tiny implementation is in `ctm_transformer/ctm_lm.py`.**

## Intent

The project's CTM should be **CTM-inspired, but as faithful to the published Continuous Thought Machine (Darlow et al., [arXiv 2505.05522](https://arxiv.org/abs/2505.05522)) as scaling to language models allows**. Some features cannot scale, just as full attention does not scale for ordinary Transformers at LLM size. Those are adapted *in kind* and stated explicitly.

The [deep-research review](DEEP_RESEARCH_REPORT.md) showed that the project's earlier "reference CTM" departed from the published design in ways that plausibly explain its failures. From now on it is called the **CTM-inspired reference**, and it is dropped from the language-model phase.

## Component mapping

The published details are as the review verified them from the CTM paper.

| Published CTM | CTM-inspired reference (earlier) | **CTM-LM (this design)** | Scaling note |
|---|---|---|---|
| Cross-attention to features from a backbone (e.g. a ResNet over the input) | Static token + position embeddings as keys and values | **A causal Transformer backbone** whose outputs are the keys and values: the language-model analogue of CTM's feature backbone. This supplies contextual features, and so induction-style lookup, faithfully. | Standard |
| Internal ticks, decoupled from the input | 16 fixed | Ticks per token: tiny scale 16 in training; larger T and randomized depth under test; certainty-based halting at inference | The paper's 50–100 ticks are not affordable per token |
| Neuron-level models: a private MLP per neuron over its pre-activation history | Per-neuron MLP with a repo-specific gate blending the latest input | **Private per-neuron MLPs** over the pre-activation history, with GLU nonlinearities; post-activation = NLM output, with no blending gate | Linear in width |
| Synapse: a U-Net-style MLP over the post-activations and the attention output | Plain MLP (or a repo-specific convolutional U-Net) | **A U-Net-style MLP** (down to a bottleneck and back up, with skip connections and LayerNorm) over [attention output ; post-activations] | Standard |
| Synchronization: decayed pairwise products of post-activation histories over randomly sampled pairs, normalized by √(sum of decays), learnable decay per pair | Recomputed over an 8-step buffer each tick; one pair set | **Recursive** decayed products (α ← e^{−r}α + z_i z_j, β ← e^{−r}β + 1, S = α/√β), with r ≥ 0 learned per pair; no D×D matrix and no buffer | O(pairs) per tick |
| Two pair sets: S_action gives the attention queries, S_out gives the outputs | One set; queries only; output from latent + token embedding | **Both sets.** Queries q = W_in·S_action; logits = W_out·S_out, with no token-embedding shortcut in the readout | A normal output head |
| Loss: mean of the minimum-loss tick and the maximum-certainty tick (certainty = 1 − normalized entropy) | Uniform across ticks | **The published loss, per supervised token** | Needs per-tick logits; a fused cross-entropy is planned for large vocabularies |
| Learnable initial post-activation state and initial pre-activation history | Learnable start state only | **Both learnable** | — |
| Adaptive compute by certainty | Not used | Certainty-based tick selection and halting at inference | — |
| Each input processed by one CTM; no latent–latent communication | Same (per position) | **Same:** each position thinks independently over the causal backbone's features. Training parallelizes over positions, and generation needs only a KV cache of backbone features. | Very scalable |

**Language-model-specific additions, each stated as a departure:**
- **Rotary position embeddings on the CTM cross-attention queries and keys.** The first query comes from the learned initial state, so it is identical at every position. With absolute positions it cannot target "my own token"; with RoPE it can prefer nearby or self positions. This is the second limitation the retrieval probes found, and its effect is a hypothesis for the capability checks to test.
- **Causal masking:** position p attends to backbone features at positions ≤ p.

**Explicitly not matched:** 50–100 ticks per token, and the paper's large pair counts and widths. Only the kind is matched, not the amount.

## Tiny configuration (capability checks)

The tiny configuration is built from the CTM recipe config (vocabulary, width 128, 4 heads, sequence window, recipe learning rate), with the loss recorded as `dynamic_aggregate`: the min-loss plus max-certainty rule, which is CTM's loss.
- **Backbone:** 2 causal Transformer blocks (width 128, learned absolute positions, as the baselines).
- **CTM:** 128 neurons, memory length 8, per-neuron hidden width 16, 128 action pairs, 128 output pairs, 4 attention heads with RoPE, and a U-Net synapse of widths 256 → 128 → 64 → 128 → 128.
- **Ticks:** 16 in training. Parameter count and throughput are measured by the tests and the first probes.

## Adopted pre-LM plan (revised after the review)

The language-model phase starts only if the synchronization effect survives steps 2–4. Each step gets a frozen protocol with a **continuous primary metric** (mean accuracy over positions or an area measure) alongside the correct prefix, **exact paired p-values** with Holm correction across declared comparisons, and threshold-sensitivity reporting.

0. **Build the tiny CTM-LM** with tests: exact recursive synchronization, the published loss, RoPE, causality, gradients, round trips and runner compatibility. *(This change.)*
1. **CTM-LM capability checks (development).** One-hop all-keys MQAR retrieval, and S₃ running products, with tick use at T = 4 to 32. These test whether the earlier CTM failures came from the reference's departures. Later, if affordable, one longer-tick S₃ run nearer the paper's regime.
2. **Confound sweep for Sync-RDT.**
   - A per-cell learning-rate sweep {1e-4, 2e-4, 3e-4, 6e-4, 1e-3} × 3 seeds for RDT, width-matched RDT and `current`.
   - At each cell's best learning rate, the scale-control ladder: RMS-normalized sync, QK-norm, norm-matched linear features, random quadratic features, a learned temperature only, a low-rank bilinear query, and gated attention.
   - Log the maximum attention logit, query norm and attention entropy.
3. **Randomized depth.** Log-normal-Poisson depth with truncated backpropagation, against fixed-T twins; evaluation at T up to 64 and at T = sequence length; frontier speed (prefix / T). *Done on S₃ ([interpretation](results/randdepth_s3_v1/INTERPRETATION.md)), with full rather than truncated backpropagation. RDT: favourable but not significant. CTM-LM: loses tick use. Extrapolation was unmeasurable because of learned absolute positions. With RoPE and tuned learning rates, the [recipe confirmation](results/recipe_s3_v1/INTERPRETATION.md) confirmed randomized depth for RDT: 24/30 escapes against 0/30. CTM-LM was 22.7 points below fixed-depth RDT, so it leaves the serial-task track.*
4. *(Revised 2026-10-01: done as development only. RDT solves A₅ at width 192 with randomized depth and 30,000 updates; see the [probes](results/recipe_probes/RECIPE_PROBES.md). A frozen A₅ study is not planned; see the [scope update](../RESEARCH_PLAN.md).)* **A₅ as the primary hard task:** a length curriculum, T ≥ n and longer training, with RDT, Sync-RDT and CTM-LM. Also an equal-compute comparison (RDT at more steps and wider, against the synchronization variants).
5. **Mini language-model stability and signal gate:** 10–20M parameters, 200–300M tokens of FineWeb-Edu with synthetic probes mixed in, RoPE with QK-norm, and 2 learning rates. It proceeds only if nothing diverges and the synthetic-probe gaps have confidence intervals that exclude 0.
6. **The language-model phase**, designed as in the review: about 40–60M parameters, 1–2B tokens, 32k BPE, 5–10% synthetic data, Parcae-style stable injection, randomized depth, per-arm learning-rate tuning, a gated-attention arm, and measured compute accounting. The CTM-LM joins only if step 1 shows it learns retrieval and uses its ticks. *(Revised 2026-10-01: CTM-LM joins pretraining regardless, as the paper's subject. At most one CTM-augmented RDT arm may join after screening; see the [scope update](../RESEARCH_PLAN.md).)*

**Dropped:** the bio-inspired homeostasis and structural-plasticity factor. The review found no Transformer evidence for it, and its source used SGD on one-hidden-layer MLPs. It is deferred to a separate side study, if pursued at all.
