# CTM architecture variants and the change of main task

**Status: design notes, 2026-09-25.** The variants below are implemented in `ctm_transformer/ctm_variants.py`, except `bio`, which is a design. Probe results are development only; see [`results/pointer_probes/PROBES.md`](results/pointer_probes/PROBES.md).

## Why variants are needed

The retrieval probes showed that the reference CTM, as implemented, cannot form an induction head. Its cross-attention keys and values are the *static* token-plus-position embeddings, computed once per forward pass, and no other path moves contextual information between positions. Retrieving "the value after key A" therefore requires locating A on one tick and then querying the next position from A's positional embedding on a later tick. Transformers and the recurrent-depth prelude do this in one step, by composing a previous-token head with a matching head.

For a paper about language models, this is central. In-context copying and retrieval are core language-model capabilities, and the limitation is a plausible contributor to the historical near-zero LAMBADA accuracy. Any CTM result must therefore be reported alongside a variant that removes this limitation.

## Variants

Variants are selected by name, not by `CTMConfig` fields, so every older fully specified config still loads unchanged. Runner v3 accepts `model_factory`, and each run manifest records the factory name (for example `variant_factory[contextual_kv]`). With no factory, the runner still reproduces v2 exactly; `tests/test_dense_pointer.py` checks this.

| Variant | Change | Cross-position information | Extra parameters (d=128) |
|---|---|---|---|
| `reference` | none | static embeddings only | 0 |
| `attn_res` | existing `use_attention_residuals=True` (Kimi-style AttnRes) | **none added**: attention over each position's earlier *layer outputs* | a few per layer |
| `contextual_kv` | one causal self-attention and SwiGLU block (hidden 2d) over the embeddings before CTM reads them | contextual keys, values and output-head text features | 164,096 (544,839 → 708,935) |

`attn_res` is included as a control. By construction it is expected *not* to enable induction, and the probe checks that empirically.

`contextual_kv` adds parameters. Any comparison must report parameter counts and, where needed, a width-matched reference.

`tests/test_ctm_variants.py` checks, for all three variants on the GPU:
- future-token invariance;
- that gradients reach the new parameters;
- strict state-dict round trips;
- explicit variant/config consistency;
- that the parameter overhead equals exactly the prelude.

## Planned `bio` variant (after the retrieval probes)

The best combination will be chosen from the probe results: contextual K/V if it enables retrieval, with or without attention residuals. It will then add the two mechanisms that [Leon (2026)](../reference_papers/) found most consistently useful for moving MLPs from memorization to generalization. Both act on the MLPs inside CTM:

1. **Homeostasis (primary).** Each hidden unit of the temporal (neuron-level) MLPs, and of the synapse MLP, gets a non-gradient threshold state θ subtracted before its nonlinearity. Once per update, θᵢ ← clip(θᵢ + η(ρ̂ᵢ − ρ\*), −θmax, θmax), where ρ̂ᵢ is the unit's activity rate on the minibatch. The paper uses ρ\* = 0.12, η = 0.01, θmax = 2. θ is a buffer that is saved in checkpoints and is not trained. A variant applying the same rule to CTM's latent neurons, whose post-activations define synchronization, is a natural second option.
2. **Structural plasticity (secondary, ablated).** A magnitude-based top-k connection mask (paper density 0.35), refreshed periodically, on the synapse MLP, and optionally on the temporal MLP's first layer. The paper finds that it helps most when a compact circuit exists, but can slow convergence when combined with homeostasis, so it is a separate switch.

The remaining mechanisms (gain modulation, input gating, fast threshold modulation, lateral inhibition, decorrelation) were secondary or inconsistent in the paper. They are not planned.

**Caveats to state with any result.** The paper studies one-hidden-layer MLPs on sparse parity and noisy XOR, trained with SGD. It reports that Adam gave substantially different and generally worse grokking behavior. Our runners use AdamW. Mechanism benefits may therefore not transfer, and optimizer choice becomes a factor to test, not an assumption.

## Change of main comparison task

Pointer retrieval stays as a cheap **capability check** for every family and variant: all-keys MQAR format, one hop. It is no longer the main comparison task. For the question of whether temporal synchronization improves recurrent computation, the next controlled task should need serial computation without first needing in-context retrieval, for example cumulative parity or prefix sums (also used by the CTM paper and the recurrence-extrapolation literature). The language-model pilot (Stage C) follows, including the reference and contextual-K/V CTM variants.
