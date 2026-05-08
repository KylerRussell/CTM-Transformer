"""
CTM-Transformer Optional / Reusable Components

Standalone modules that aren't part of the model architecture proper but
are wired into training when their corresponding flag is set:

  1. adamuon.py          — AdaMuon optimizer + parameter routing helper
                           (used when --optimizer adamuon).
  2. cpu_offload_engine.py — CTMCPUOffloadEngine: stores ThoughtLayer
                             params on CPU pinned memory and streams
                             them to a single GPU template per layer call.
                             For models too large to fit on a single GPU.

These could just as well live next to model.py, but keeping them in their
own file makes the dependency story clearer: the model file is pure
architecture, the train file is pure orchestration, and this file is
where 'opt-in helpers' live.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer
import copy
import logging
import torch.nn as nn



# After consolidation, NeuronLevelModels and TernaryLinear live in model.py.
# build_param_groups (below) needs both for its routing-rule isinstance checks.
from ctm_transformer.model import NeuronLevelModels, TernaryLinear

# ═════════════════════════════════════════════════════════════════════════
# adamuon.py
# ═════════════════════════════════════════════════════════════════════════

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

    # TernaryLinear latent weights → AdamW.
    # AdaMuon would be a poor fit here: its RMS-aligned update has a fixed
    # magnitude (0.2 by default), independent of the gradient. For a
    # *ternarized* layer, the latent weight's role is purely "which side of
    # ±Δ am I on" — and a fixed-magnitude update of 0.2 per step would flip
    # entries across the threshold far too aggressively, causing the
    # ternarization pattern to thrash from step to step instead of slowly
    # crystallizing. TWN's original paper uses SGD+momentum, which produces
    # naturally small updates that respect the Δ threshold. AdamW is a
    # reasonable compromise — its variance scaling adapts to the small
    # gradient regime that ternary training tends toward.
    ternary_param_ids: set[int] = set()
    try:
        for m in model.modules():
            if isinstance(m, TernaryLinear):
                for p in m.parameters(recurse=True):
                    ternary_param_ids.add(id(p))
    except Exception:
        pass

    # Identify the LM head: this is the final vocabulary projection. Two
    # possible layouts depending on `config.per_tick_heads`:
    #   - Legacy: model.output_proj is an nn.Sequential ending in the LM
    #     head Linear; the per-tick adapters don't exist.
    #   - per_tick_heads=True: model.output_proj is None, model.lm_head is
    #     the shared vocabulary Linear, and model.tick_adapters is a
    #     ModuleList of small per-tick projections.
    # In both cases, we want the LM head's weight routed to AdamW (large
    # vocabulary projections typically benefit from AdamW's variance
    # adaptation more than from AdaMuon's polar-decomposition update).
    # Identity-check via id() so we don't accidentally match lookalikes
    # from elsewhere in the model.
    lm_head_weight_id: int | None = None
    if getattr(model, "output_proj", None) is not None:
        # Legacy path: walk the Sequential, find the last Linear.
        for m in reversed(list(model.output_proj.modules())):
            if isinstance(m, torch.nn.Linear):
                lm_head_weight_id = id(m.weight)
                break
    elif getattr(model, "lm_head", None) is not None:
        # Per-tick heads path: lm_head IS the LM head.
        if isinstance(model.lm_head, torch.nn.Linear):
            lm_head_weight_id = id(model.lm_head.weight)

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

        # Rule 3: TernaryLinear latent weight (or bias) → AdamW
        # Excluded from AdaMuon because its fixed-RMS update magnitude
        # would thrash the ternarization across the ±Δ threshold every step.
        if id(p) in ternary_param_ids:
            adamw_params.append(p)
            adamw_log.append(f"  [adamw, ternary]    {name}  {tuple(p.shape)}")
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

# ═════════════════════════════════════════════════════════════════════════
# cpu_offload_engine.py
# ═════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


class _StreamedLayerProxy(nn.Module):
    """Proxy that loads weights from CPU and runs on GPU template.

    On each forward call:
    1. H2D copy: pinned CPU flat → GPU flat buffer → template params
    2. Forward through template with requires_grad=True
    3. After backward completes globally, engine.collect_grads() copies
       template param grads to CPU params (NOT via hooks to avoid
       double-counting with gradient checkpointing).
    """

    def __init__(self, engine, layer_idx):
        super().__init__()
        self.engine = engine
        self.layer_idx = layer_idx

    @property
    def memory(self):
        return self.engine.gpu_template.memory

    @property
    def stream(self):
        return self.engine.gpu_template.stream

    def reset_memory(self, *args, **kwargs):
        self.engine.gpu_template.reset_memory(*args, **kwargs)

    def forward(self, *args, **kwargs):
        engine = self.engine
        idx = self.layer_idx

        # Track which layer was last loaded (for grad collection)
        engine._last_layer_idx = idx

        # 1. H2D: copy pinned flat → GPU flat → unflatten to template
        flat = engine.layer_pinned_flats[idx]
        n = engine.layer_numels[idx]
        engine.gpu_flat_buffer[:n].copy_(flat, non_blocking=False)

        template = engine.gpu_template
        offset = 0
        for p, shape, numel in zip(
            template.parameters(),
            engine.layer_param_shapes[idx],
            engine.layer_param_numels[idx],
        ):
            p.data.copy_(engine.gpu_flat_buffer[offset:offset + numel].view(shape))
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None
            offset += numel

        # 2. Forward — autograd graph connects through template params
        return template(*args, **kwargs)


class CTMCPUOffloadEngine:
    """CPU-backed training engine for CTM-Transformer.

    Stores ThoughtLayer parameters on CPU pinned memory and streams them to
    a single GPU template for each layer call. Static modules (embedding,
    output head, FEEC, etc.) stay GPU-resident.

    The engine patches model._get_layers_sequence() to return proxy layers,
    so model.forward() works unmodified with gradient checkpointing.

    IMPORTANT: After loss.backward(), you must call engine.collect_grads()
    to copy the template's accumulated gradients back to CPU params.

    Usage:
        engine = CTMCPUOffloadEngine(model, device='cuda:0')
        result = model(input_ids, targets=targets)
        result['loss'].backward()
        # Template grads flow through autograd; but they live on GPU template.
        # The optimizer holds CPU params. We need to bridge them:
        engine.collect_grads()
        optimizer.step()
        engine.sync_params_from_cpu()
    """

    def __init__(self, model, device="cuda:0", dtype=torch.bfloat16):
        self.model = model
        self.config = model.config
        self.device = torch.device(device)
        self.dtype = dtype
        self._last_layer_idx = -1

        self.layers = model._get_layers_sequence()
        self.n_layers = len(self.layers)

        self._move_static_to_gpu()
        self._setup_cpu_layers()
        self._setup_gpu_template()
        self._patch_model()

        # Accumulated grads per-layer (indexed by layer_idx)
        # Shape: list of flat tensors on CPU, same size as pinned flats
        self._grad_accum = [
            torch.zeros_like(flat) for flat in self.layer_pinned_flats
        ]

        print(
            f"  Engine: {self.n_layers} layers, "
            f"max layer {self.max_layer_numel * 2 / 1e6:.1f} MB (bf16), "
            f"GPU resident: {self._gpu_resident_mb:.1f} MB"
        )

    def _move_static_to_gpu(self):
        """Move small, always-needed modules to GPU."""
        m = self.model
        layer_param_ids = set()
        for layer in self.layers:
            for p in layer.parameters():
                layer_param_ids.add(id(p))

        gpu_mb = 0
        for name, param in m.named_parameters():
            if id(param) not in layer_param_ids:
                param.data = param.data.to(self.device, self.dtype)
                gpu_mb += param.numel() * param.element_size() / 1e6

        for name, buf in m.named_buffers():
            buf.data = buf.data.to(self.device)

        self._gpu_resident_mb = gpu_mb

    def _setup_cpu_layers(self):
        """Move layer params to CPU pinned memory."""
        self.layer_pinned_flats = []
        self.layer_param_shapes = []
        self.layer_param_numels = []
        self.layer_numels = []
        self.layer_cpu_params = []

        for layer in self.layers:
            for p in layer.parameters():
                p.data = p.data.cpu().to(self.dtype)

            shapes = [p.shape for p in layer.parameters()]
            numels = [p.numel() for p in layer.parameters()]
            total = sum(numels)
            cpu_params = list(layer.parameters())

            flat = torch.empty(total, dtype=self.dtype).pin_memory()
            offset = 0
            for p in layer.parameters():
                n = p.numel()
                flat[offset:offset + n].copy_(p.data.flatten())
                offset += n

            self.layer_pinned_flats.append(flat)
            self.layer_param_shapes.append(shapes)
            self.layer_param_numels.append(numels)
            self.layer_numels.append(total)
            self.layer_cpu_params.append(cpu_params)

        self.max_layer_numel = max(self.layer_numels) if self.layer_numels else 0

    def _setup_gpu_template(self):
        """Create a single GPU-resident layer shell."""
        if not self.layers:
            self.gpu_template = None
            self.gpu_flat_buffer = None
            return

        self.gpu_flat_buffer = torch.empty(
            self.max_layer_numel, dtype=self.dtype, device=self.device
        )
        self.gpu_template = copy.deepcopy(self.layers[0])
        self.gpu_template = self.gpu_template.to(self.device, self.dtype)

    def _patch_model(self):
        """Replace model layers with streaming proxies."""
        proxies = nn.ModuleList([
            _StreamedLayerProxy(self, i) for i in range(self.n_layers)
        ])
        self._original_get_layers = self.model._get_layers_sequence
        self.model._get_layers_sequence = lambda: proxies
        self._proxies = proxies

    def zero_grad(self):
        """Zero accumulated gradients for all CPU layer params."""
        for acc in self._grad_accum:
            acc.zero_()
        for params in self.layer_cpu_params:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    def collect_grads(self):
        """After backward, copy template grads to CPU params.

        NOTE: With gradient checkpointing, the template is shared across all
        layers and ticks. During backward, PyTorch recomputes the forward
        (which re-loads weights into the template for each layer) and
        accumulates grads on the template params. However, since all 26
        layers share one template, the grads on the template at the END of
        backward only reflect the LAST layer that was recomputed.

        This is a fundamental limitation of the single-template approach
        with gradient checkpointing. The grad hooks approach (previous
        version) tried to fix this but double-counted due to recomputation.

        For correct grad collection with a single template, we would need
        to either:
        a) Use separate templates per layer (26 × 25 MB = 650 MB GPU)
        b) Not use gradient checkpointing (OOM)
        c) Use manual backward with explicit recompute (original engine v1)

        For now, this is a best-effort implementation.
        """
        template = self.gpu_template
        if template is None:
            return

        # The template grads belong to whatever layer was last computed
        # This is only correct without gradient checkpointing
        idx = self._last_layer_idx
        if idx < 0 or idx >= self.n_layers:
            return

        offset = 0
        for p, cpu_p in zip(template.parameters(), self.layer_cpu_params[idx]):
            if p.grad is not None:
                n = p.grad.numel()
                if cpu_p.grad is None:
                    cpu_p.grad = p.grad.cpu().to(cpu_p.dtype)
                else:
                    cpu_p.grad.add_(p.grad.cpu().to(cpu_p.dtype))

    def sync_params_from_cpu(self):
        """Refresh pinned flats from CPU params (after optimizer step)."""
        for i, layer in enumerate(self.layers):
            flat = self.layer_pinned_flats[i]
            offset = 0
            for p in layer.parameters():
                n = p.numel()
                flat[offset:offset + n].copy_(p.data.flatten())
                offset += n

    def get_all_parameters(self):
        """Return all trainable params (CPU layers + GPU static)."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        layer_param_ids = set()
        for layer in self.layers:
            for p in layer.parameters():
                layer_param_ids.add(id(p))
        for p in self.model.parameters():
            if id(p) not in layer_param_ids:
                params.append(p)
        return params

    def shutdown(self):
        """Restore original model state."""
        if hasattr(self, '_original_get_layers'):
            self.model._get_layers_sequence = self._original_get_layers

