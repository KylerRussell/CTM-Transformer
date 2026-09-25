# Design draft: CTM mechanisms inside a recurrent-depth scaffold (Sync-RDT)

**Status: DRAFT for review, 2026-09-25. Nothing here is implemented or frozen.**

## Motivation

The paper asks whether CTM's temporal machinery (per-neuron activation histories, neuron-level temporal MLPs and synchronization) improves recurrent computation *beyond recurrent depth alone* (research plan, §4 and §7). Comparing the CTM with RDT as whole architectures confounds that question with unrelated differences. The retrieval probes found two such differences, neither involving synchronization:

1. CTM's cross-attention reads static embeddings, so it cannot form induction heads.
2. Every CTM position starts from the same latent, so its queries cannot express its own token.

RDT has neither limitation and learns one-hop retrieval (r10b: 100% held-out). A third difference remains even in the repaired CTM variants: RDT's recurrent states attend to *each other's* evolving states at every step, while CTM latents only read fixed text features. Passing intermediate results between positions matters for multi-step computation.

Sync-RDT keeps RDT's scaffold unchanged and adds CTM's mechanisms inside its recurrent core as independent switches. The "both off" cell *is* RDT, which makes the research plan's 2×2 mechanism factorial (§7) exact.

## Scaffold (unchanged from the RDT recipe)

- Width d = 96, 4 heads, SwiGLU hidden 288, pre-norm with sandwich norms, untied embeddings, no dropout.
- Prelude P: 1 block. Shared recurrent core R: 2 blocks. Coda C: 1 block.
- Fixed T = 16 recurrence steps in training.
- Final CE, peak LR 0.0003 (0.001 fails; probes r9b and r10b), confidence readout.

For each position, with e = P(embed(x)) and s₀ = 0:

```
u_t = W_inj [s_{t-1} ; e]             input injection (RDT)
p_t = R(u_t)                           shared core; self-attention over all positions' u_t (causal)
s_t = p_t                              RDT state update
logits_t = head(norm(C(norm(s_t))))   per-step readout (confidence readout picks a step per token)
```

## CTM mechanisms as switches

Both mechanisms act per position and per recurrence step, on the core output p_t ∈ R^d, treating its d channels as CTM's neurons. A FIFO buffer keeps each position's last H = 8 core outputs `P_t = [p_{t-7} … p_t]`, zero-filled before step 1 as in CTM.

**A. Neuron-level temporal MLPs (history processing).** This is CTM's `NeuronLevelModels`, reused unchanged: independent per-neuron MLPs (H → 16 → 1, GELU) over each channel's history, with CTM's learned gate.

```
s_t = σ(g) ⊙ tanh(NLM(P_t)) + (1 − σ(g)) ⊙ p_t
```

- **Off:** s_t = p_t (RDT).
- **Parameters:** d·(8·16 + 16 + 16 + 1 + 1) = 96 · 162 = 15,552.
- **Initialization:** CTM's own, unchanged. The NLM output starts near zero (small second-layer init) and the gate starts at σ(0) = 0.5, so the state starts near 0.5·p_t: a rescaled RDT update, not RDT exactly.

**B. Synchronization-derived queries.** This is CTM's `SynchronizationComputer` (sparse_decay), reused unchanged: 128 random channel pairs (i, j), each with a learned decay rate r:

```
sync_t = Σ_τ exp(−r·τ) · s_{t−τ,i} · s_{t−τ,j} / sqrt(Σ_τ exp(−r·τ))
```

It is computed over the history of post-update states s. In every core self-attention layer, the query gains an additive sync term:

```
q = W_q · norm(u) + W_sync · sync_{t−1}
```

- **Initialization:** W_sync starts at zero, so the cell starts as RDT.
- **Off:** no W_sync (RDT).
- **Parameters:** 2 core blocks × 128 × 96 + 128 decay rates = 24,704.

This keeps CTM's defining role for synchronization (it decides *where to attend*) while leaving RDT's content-based attention intact. A "replace" variant, with queries from sync alone as in CTM, can be a later ablation. It is not the primary cell, because it would reintroduce the query problem that blocked CTM retrieval.

| Cell | Temporal MLP | Sync queries | Parameters |
|---|---|---|---:|
| RDT (control) | off | off | 525,984 |
| + history | on | off | 541,536 |
| + sync | off | on | 550,688 |
| Sync-RDT (full) | on | on | 566,240 |

The differences (3–8%) are reported with every result. If a cell wins, a width-matched RDT control (d = 100) checks whether parameters alone explain the gain. Compute is measured, not assumed: both mechanisms are elementwise or small matrix products next to attention.

## Bio factor (Leon, 2026), crossed after the 2×2

The paper's mechanisms target MLP hidden units, and every cell has two MLP sites: the SwiGLU FFN in the core (all cells) and the temporal MLP (cells with A).

- **Homeostasis:** a non-gradient per-unit threshold θ subtracted before the nonlinearity, updated once per step by θ ← clip(θ + η(ρ̂ − ρ\*), ±θmax), with the paper's ρ\* = 0.12, η = 0.01, θmax = 2. θ is stored as a buffer and saved in checkpoints.
- **Structural plasticity:** a magnitude top-k mask (density 0.35) on the core FFN, refreshed periodically. This is a separate switch, because the paper finds it can conflict with homeostasis.

To avoid confounding the two questions, the bio factor is applied to **both** RDT and the best CTM-mechanism cell. The paper's SGD-versus-Adam caveat is carried forward: an optimizer contrast is part of the bio ablation.

## Verification before any study

- **Exact parity:** Sync-RDT with both switches off reproduces `BaselineTransformer` (recurrent_depth) exactly, in losses, gradients and final weights, like the v3–v2 test. This makes the control cell provably RDT.
- Future-token invariance, gradients reaching every new parameter, and strict round trips for each cell.
- History semantics: zero-filled before step 1, FIFO order, identical under per-tick logits.
- Per-step logits under confidence readout match separate truncations at depths 1–16, as already verified for RDT.

## Study sequence

1. Implement `ctm_transformer/sync_rdt.py`, reusing `NeuronLevelModels`, `SynchronizationComputer` and the baseline blocks without modifying them. Build through a named factory, as with the CTM variants; runner v3 records the factory.
2. **Capability check (development):** all four cells on one-hop all-keys MQAR at LR 0.0003. Each must learn retrieval before the main comparison.
3. **Main task:** a serial-depth task where the fixed-depth Transformer fails, for example cumulative parity / prefix sums, or multi-hop with a hop curriculum. It is calibrated under the fixed operating-point rules: difficulty band with the two-family mean between 20% and 80%, non-saturating endpoints (accuracy by depth including unseen depths, accuracy against inference steps, sample efficiency), and disjoint confirmation seeds.
4. **Frozen factorial:** 4 cells × 3 seeds. The Transformer and the best repaired CTM variant are external references, reported with their structural limitations.
5. The bio factor on RDT and the best cell.
6. The language-model pilot with the chosen cells.

## Open questions for review

1. **Sync as an additive query term (primary) versus a replacement (ablation).** The additive form keeps RDT's retrieval ability. The replacement form is closer to CTM.
2. **Where the history is taken:** from the core output p_t (proposed), or from each core block separately.
3. **Readout:** RDT's coda on the state (proposed), or additionally CTM's readout from synchronization.
4. **Learning rate:** 0.0003 for all cells (proposed; it is the rate at which RDT learns), or a small per-cell learning-rate check.
