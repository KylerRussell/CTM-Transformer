"""
AdaMuon Optimizer

Reference:
    Si, Zhang, Shen. "AdaMuon: Adaptive Muon Optimizer." arXiv:2507.11005 (2025).
    https://arxiv.org/abs/2507.11005

AdaMuon = Muon (sign-stabilized polar-decomposition update) + element-wise
second momentum on the orthogonal output O_t + RMS alignment so the update
norm matches Adam's empirical ~0.2 (lets you reuse Adam's LR schedule).

Per Algorithm 1 of the paper:
    M_t = β · M_{t-1} + G_t                              # heavy-ball first moment
    O_t = NewtonSchulz( sign(M_t), T )                   # sign-stabilized polar
    V_t = β · V_{t-1} + (1 - β) · O_t ⊙ O_t              # second moment EMA
    Ô_t = O_t ⊘ (√V_t + ε)                                # variance normalize
    γ_t = 0.2 · √(mn) / ||Ô_t||_F                         # RMS align to Adam
    W_{t+1} = W_t − η · (γ_t · Ô_t + λ · W_t)             # decoupled WD update

Restrictions (matches the paper):
- Operates ONLY on 2D parameter tensors. 1D params (biases, norms),
  embeddings, the LM head, and >2D weight stacks (e.g. grouped-NLM
  [groups, h, w] tensors) should be routed to AdamW via build_param_groups.
- No bias correction on V_t — the RMS-alignment step (γ ∝ √V_t) exactly
  cancels any constant multiplicative bias in V_t, making explicit
  correction redundant. See paper Appendix B.
- No bias correction on M_t either — sign(M_t) is invariant to positive
  rescaling, and the polar factor is globally scale-invariant. See
  paper Appendix B last paragraph.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


# ──────────────────────────────────────────────────────────────────────────
# Newton-Schulz polar factor
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def newton_schulz5(X: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Quintic Newton-Schulz iteration for the polar factor of X.

    Computes U @ V^T from the SVD X = U S V^T (without doing SVD), which is
    Muon's "orthogonalized" update direction. The fixed iteration polynomial

        f(x) = a·x + b·x^3 + c·x^5,    (a, b, c) = (3.4445, -4.7750, 2.0315)

    is tuned (Jordan et al. 2024) so the fixed point sits near 1, which gives
    the small singular values an aggressive boost while keeping the large ones
    bounded away from divergence. T=5 iterations is the standard choice.

    Runs internally in bf16 — the Muon literature has shown this is safe and
    saves ~2× memory + compute on the matmul-heavy NS body. Output is cast
    back to the input dtype.

    Args:
        X: 2D tensor [n, m]. Frobenius-normalized internally.
        steps: NS iterations (default 5).
        eps: numerical floor on the Frobenius normalization.

    Returns:
        Polar factor of X, same shape, same dtype as input.
    """
    assert X.ndim == 2, f"newton_schulz5 expects 2D input; got shape {tuple(X.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = X.dtype

    X = X.to(torch.bfloat16)

    # If X is taller than wide, work on X^T (smaller XX^T). Saves one matmul
    # axis dim per step. Equivalent because polar(X^T) = polar(X)^T.
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    # Frobenius normalize so all singular values land in [0, 1] — required
    # for the NS polynomial's convergence basin.
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T              # [r, r] where r = min(n, m)
        # X_{k+1} = a·X + (b·A + c·A²)·X
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)


# ──────────────────────────────────────────────────────────────────────────
# AdaMuon optimizer
# ──────────────────────────────────────────────────────────────────────────

class AdaMuon(Optimizer):
    """
    AdaMuon optimizer for 2D parameter matrices.

    Args:
        params: iterable of 2D parameters (or param groups). Any non-2D
            parameter encountered at step time will raise.
        lr: learning rate η. Thanks to RMS alignment (γ → ||γ·Ô||_F = 0.2·√mn),
            this is on the same scale as AdamW's LR.
        weight_decay: decoupled weight-decay coefficient λ. Paper uses 0.1.
        beta: shared first/second momentum coefficient β. Paper uses 0.95.
        eps: small constant in the V_t denominator, prevents 0-division
            when a coordinate has been zero throughout training.
        ns_steps: Newton-Schulz iterations (default 5).
        rms_target: target RMS magnitude after rescaling (default 0.2,
            matching Adam's empirical update RMS per Liu et al. 2025).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        beta: float = 0.95,
        eps: float = 1e-8,
        ns_steps: int = 5,
        rms_target: float = 0.2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            eps=eps,
            ns_steps=ns_steps,
            rms_target=rms_target,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group["lr"]
            wd        = group["weight_decay"]
            beta      = group["beta"]
            eps       = group["eps"]
            ns_steps  = group["ns_steps"]
            rms_target = group["rms_target"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim != 2:
                    raise ValueError(
                        f"AdaMuon only supports 2D parameters; got shape "
                        f"{tuple(p.shape)}. Route 1D params (biases, norms), "
                        f"embeddings, LM head, and >2D weight stacks to AdamW. "
                        f"See adamuon.build_param_groups()."
                    )

                g = p.grad
                state = self.state[p]

                # Lazy init of momentum buffers — match dtype/device of param.
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                state["step"] += 1

                # ── 1) First momentum (heavy-ball, no (1-β) factor) ─────
                #     M_t = β·M_{t-1} + G_t
                m.mul_(beta).add_(g)

                # ── 2) Sign-stabilized orthogonal direction ─────────────
                #     O_t = NS( sign(M_t) )
                # Theorem 1 in the paper: among element-wise transforms
                # that are scale-invariant + sign-consistent + odd +
                # bounded, sign(·) is the unique canonical choice.
                # Stabilizes element-wise volatility before the polar
                # decomposition, which is required for V_t accumulation
                # to actually mean anything (ablation Fig 3).
                signed = torch.sign(m)
                O = newton_schulz5(signed, steps=ns_steps)

                # ── 3) Second momentum on O (not on G or M) ─────────────
                #     V_t = β·V_{t-1} + (1-β)·O_t ⊙ O_t
                # Accumulating on O rather than G/M is critical:
                #   - G is ill-conditioned / direction-noisy
                #   - M is volatile element-wise during early training
                #   - O is geometrically normalized → stable variance basis
                v.mul_(beta).addcmul_(O, O, value=1.0 - beta)

                # ── 4) Variance-normalize: Ô = O ⊘ (√V + ε) ─────────────
                # We want a dtype-matched eps and sqrt — avoid promoting
                # to fp32 and back.
                O_hat = O / (v.sqrt() + eps)

                # ── 5) RMS alignment: γ = rms_target · √(mn) / ||Ô||_F ──
                # Result: ||γ·Ô||_F = rms_target·√(mn), i.e. RMS = rms_target.
                # This both rescales to Adam's update norm AND cancels any
                # constant bias in V_t (paper Appendix B), so we don't need
                # explicit bias correction.
                fro = O_hat.norm()
                if fro < 1e-12:
                    # Degenerate (e.g. all-zero gradient & zero variance).
                    # Skip the parameter update but still apply weight decay.
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    continue

                # gamma is a 0-dim tensor (fro is a tensor). Fold it into
                # O_hat in-place so the final p.add_ uses a Python-scalar
                # alpha — avoids the 0-dim-tensor-as-alpha path and keeps
                # the inner loop sync-free.
                gamma = (rms_target * (O_hat.numel() ** 0.5)) / fro
                O_hat.mul_(gamma)

                # ── 6) Update with decoupled weight decay ───────────────
                #     W ← W - lr·(γ·Ô + λ·W)
                # WD as an in-place scale: W ← (1 - lr·λ)·W, then add update.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(O_hat, alpha=-lr)

        return loss


# ──────────────────────────────────────────────────────────────────────────
# Parameter routing
# ──────────────────────────────────────────────────────────────────────────

def build_param_groups(
    model: torch.nn.Module,
    verbose: bool = False,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Split a model's parameters into:
      * muon_params  — 2D weight matrices for AdaMuon
      * adamw_params — everything else (1D, embeddings, LM head, NLM stacks)

    Routing rules (checked top-down, first match wins):
      1. param.ndim != 2  → AdamW
            covers biases, LayerNorm/RMSNorm scales, learned scalar gates,
            and the NLM weight stacks (shape [nlm_groups, history_len, hidden]).
      2. param lives inside a NeuronLevelModels module  → AdamW
            NLM params are *per-group MLP tensors* packed as 2D/3D arrays
            for vectorized compute. Even when they happen to be 2D (e.g.
            b1 of shape [nlm_groups, nlm_hidden_dim]), they're stacked
            biases — not weight matrices in the spectral-norm sense, and
            polar decomposition does not respect that geometry.
      3. name contains "embedding" or "embed.weight"  → AdamW
            token + positional embeddings (very wide, lookup-style).
      4. param is the final Linear weight inside `model.output_proj`  → AdamW
            this is the LM head — wide-output, conventionally on Adam.
      5. otherwise  → AdaMuon

    Args:
        model: the model whose parameters to split.
        verbose: if True, print the split for inspection.

    Returns:
        (muon_params, adamw_params) tuple of two parameter lists.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []

    # ── Collect IDs of parameters that must go to AdamW regardless of shape ─
    # Any param living inside a NeuronLevelModels module: its tensors are
    # per-group MLP packs, not single weight matrices.
    nlm_param_ids: set[int] = set()
    try:
        from ctm_transformer.nlm import NeuronLevelModels
        for m in model.modules():
            if isinstance(m, NeuronLevelModels):
                for p in m.parameters(recurse=True):
                    nlm_param_ids.add(id(p))
    except Exception:
        # If the import path differs (e.g. running tests from a different
        # working directory), fall back to a name-based heuristic below.
        pass

    # All nn.Embedding weights → AdamW. Catches both the token embedding
    # and the Engram lookup tables (which are also nn.Embedding internally,
    # under EngramTable.tables). isinstance is more reliable than the
    # name-substring fallback below — Engram's parameter is named
    # `engram_table.tables.weight`, which doesn't contain "embedding".
    embedding_weight_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            embedding_weight_ids.add(id(m.weight))

    # Identify the LM head: the *last* nn.Linear nested anywhere inside
    # `model.output_proj`. Identity-check via id() so we don't accidentally
    # exclude lookalikes from elsewhere in the model.
    lm_head_weight_id: int | None = None
    if hasattr(model, "output_proj"):
        for m in reversed(list(model.output_proj.modules())):
            if isinstance(m, torch.nn.Linear):
                lm_head_weight_id = id(m.weight)
                break

    muon_log: list[str] = []
    adamw_log: list[str] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        nlc = name.lower()

        # Rule 1: non-2D
        if p.ndim != 2:
            adamw_params.append(p)
            adamw_log.append(f"  [adamw, ndim={p.ndim}] {name}  {tuple(p.shape)}")
            continue

        # Rule 2: NLM-owned param (handles stacked-bias 2D edge cases like b1, b2)
        if id(p) in nlm_param_ids or ".nlm." in f".{nlc}.":
            adamw_params.append(p)
            adamw_log.append(f"  [adamw, nlm]        {name}  {tuple(p.shape)}")
            continue

        # Rule 3: any nn.Embedding weight (token emb, Engram tables, etc.)
        if id(p) in embedding_weight_ids:
            adamw_params.append(p)
            adamw_log.append(f"  [adamw, embed]      {name}  {tuple(p.shape)}")
            continue

        # Rule 4: name-based fallback for embeddings (handles renamed/wrapped cases)
        if "embedding" in nlc or "embed.weight" in nlc:
            adamw_params.append(p)
            adamw_log.append(f"  [adamw, embed-name] {name}  {tuple(p.shape)}")
            continue

        # Rule 5: LM head
        if lm_head_weight_id is not None and id(p) == lm_head_weight_id:
            adamw_params.append(p)
            adamw_log.append(f"  [adamw, lm_head]    {name}  {tuple(p.shape)}")
            continue

        # Default: 2D hidden weight → AdaMuon
        muon_params.append(p)
        muon_log.append(f"  [adamuon]           {name}  {tuple(p.shape)}")

    if verbose:
        n_muon = sum(p.numel() for p in muon_params)
        n_adam = sum(p.numel() for p in adamw_params)
        print(f"AdaMuon params: {len(muon_params)} tensors, {n_muon:,} elements")
        for line in muon_log:
            print(line)
        print(f"AdamW params:   {len(adamw_params)} tensors, {n_adam:,} elements")
        for line in adamw_log:
            print(line)

    return muon_params, adamw_params