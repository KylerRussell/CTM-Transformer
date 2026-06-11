"""
Validation metrics for CTM-Transformer.

Two complementary signals computed on a held-out slice of the cached
teacher dataset (so no extra data prep needed — uses what
CachedTeacherDataset(is_eval=True) gives you):

  1. Held-out next-token cross-entropy / perplexity, computed on the
     model's final-tick logits. This is "what does the deployed model
     do" PPL, not the training-time mix-of-losses (dynamic_aggregate,
     etc). Use it as a learning curve.

  2. Distillation top-k alignment: fraction of tokens where the student's
     argmax matches the teacher's top-1 (or is in the teacher's top-k).
     Goes up if distillation is doing what it's supposed to.

Plus a `compute_validation_metrics_t_sweep` helper for the T-ablation
(deliverable #4): runs the same eval at multiple thought-step budgets so
you can see whether the recurrent compute is earning its keep.

Designed to be called from the multi-GPU worker on rank 0 only — weights
stay synced across ranks because of the flat-allreduce on grads, so
evaluating on one rank is correct.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _unwrap_model(model):
    """Strip torch.compile / DDP wrappers."""
    inner = getattr(model, "_orig_mod", model)
    inner = getattr(inner, "module", inner)
    return inner


@torch.no_grad()
def compute_validation_metrics(
    model,
    eval_loader: DataLoader,
    device,
    config,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 50,
    max_thought_steps: Optional[int] = None,
    topk_for_alignment: int = 5,
) -> dict:
    """
    Run validation over `max_batches` from `eval_loader`.

    Returns:
        dict with keys:
            ce_loss      — held-out cross-entropy (nats/token), final tick
            ppl          — exp(ce_loss); the honest "deployment PPL"
            top1_agree   — fraction of tokens where student argmax equals
                           teacher argmax (from cached teacher top-k)
            topk_agree   — fraction where student argmax is in teacher's
                           top-`topk_for_alignment`
            n_tokens     — tokens scored
            n_batches    — batches consumed
            T            — thought steps used at eval time

    Notes:
        - Caller is responsible for whatever rank gating they want.
        - eval_loader yields (ids, tgt, top_idx, top_val, _res[, teacher_z])
          tuples per cached_teacher_collate. top_idx and top_val may be empty
          tensors if the cache wasn't built with distillation; we detect
          that and skip alignment metrics. The optional 6th `teacher_z`
          element is ignored here — validation measures pure LM quality,
          not the training-time predictive-coding signal.
        - PPL is capped to e^30 to avoid overflow on early-step nonsense
          checkpoints — if you see ppl == 1.07e13, treat it as +inf.
    """
    was_training = model.training
    model.eval()

    # ── Resolve T ──────────────────────────────────────────────────
    prospective_active = getattr(config, "use_prospective_config", False)
    if max_thought_steps is None:
        if prospective_active:
            T = None
            T_label = "dyn"
        else:
            unwrapped = _unwrap_model(model)
            if hasattr(unwrapped, "_train_step") and hasattr(config, "resolve_thought_steps"):
                T = config.resolve_thought_steps(int(unwrapped._train_step.item()))
            else:
                T = config.max_thought_steps
            T_label = str(T)
    else:
        T = max_thought_steps
        T_label = str(T)

    device_type = (
        device.type if isinstance(device, torch.device)
        else (device.split(":")[0] if ":" in device else device)
    )

    total_ce_sum = 0.0       # sum of per-token CE across all tokens
    total_top1_correct = 0
    total_topk_correct = 0
    total_tokens_for_ce = 0
    total_tokens_for_align = 0
    n_batches = 0

    for batch in eval_loader:
        if n_batches >= max_batches:
            break
        # Robust to teacher-free 2-tuples (CurriculumDataset → (x, y)) as well
        # as the 5/6-tuple cached-teacher collate. We only need x/y for CE; the
        # teacher top-k (if present) drives the optional alignment metric below.
        ids, tgt = batch[0], batch[1]
        top_idx = batch[2] if len(batch) > 2 else None
        top_val = batch[3] if len(batch) > 3 else None
        ids = ids.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # NOTE: deliberately NOT wrapping in torch.amp.autocast.
        # The multi-GPU worker explicitly casts the model to bf16 via
        # model.to(device, dtype), so autocasting on top of that creates
        # dtype mismatches in attention kernels (autocast promotes some
        # ops to fp32 for stability, which clashes with the bf16 model
        # weights). The training forward doesn't autocast either.
        # NOTE 2: deliberately NOT passing cached_top_indices/values.
        # Validation measures pure LM quality, not the training-time
        # distillation-mixed loss. Alignment is computed separately
        # below using the same cached top-k.
        result = model(ids, max_thought_steps=T)

        logits = result["logits"]  # [B, S, V] — final tick
        if logits is None:
            # Edge case: T=0 short-circuit (shouldn't happen in practice)
            continue

        # ── Held-out CE / PPL on final-tick logits ────────────────
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, S]
        # Sum over all tokens; we'll divide by total_tokens at the end so
        # the CE is correctly weighted by sequence length even if the
        # last batch is short.
        n_tok = target_lp.numel()
        total_ce_sum += float(-target_lp.sum().item())
        total_tokens_for_ce += n_tok

        # ── Distillation alignment ────────────────────────────────
        if top_idx is not None and top_idx.numel() > 0:
            top_idx = top_idx.to(device, non_blocking=True)  # [B, S, K_data]
            B, S, K_data = top_idx.shape
            K = min(K_data, topk_for_alignment)

            student_top1 = logits.argmax(dim=-1)  # [B, S]
            teacher_top1 = top_idx[:, :, 0]
            top1_match = (student_top1 == teacher_top1)

            teacher_topk = top_idx[:, :, :K]
            topk_match = (student_top1.unsqueeze(-1) == teacher_topk).any(dim=-1)

            total_top1_correct += int(top1_match.sum().item())
            total_topk_correct += int(topk_match.sum().item())
            total_tokens_for_align += B * S

        n_batches += 1

    if was_training:
        model.train()

    # ── Aggregate ─────────────────────────────────────────────────
    ce = total_ce_sum / max(total_tokens_for_ce, 1)
    ppl = math.exp(min(ce, 30.0))

    out = {
        "ce_loss": ce,
        "ppl": ppl,
        "n_tokens": total_tokens_for_ce,
        "n_batches": n_batches,
        "T": T_label,
    }
    if total_tokens_for_align > 0:
        out["top1_agree"] = total_top1_correct / total_tokens_for_align
        out["topk_agree"] = total_topk_correct / total_tokens_for_align
    else:
        out["top1_agree"] = float("nan")
        out["topk_agree"] = float("nan")
    return out


@torch.no_grad()
def compute_validation_metrics_t_sweep(
    model,
    eval_loader_factory: Callable[[], DataLoader],
    device,
    config,
    amp_dtype: torch.dtype = torch.bfloat16,
    T_values=(1, 4, 8),
    max_batches: int = 30,
    topk_for_alignment: int = 5,
) -> dict:
    """T-ablation: same eval at multiple thought-step budgets.

    Args:
        eval_loader_factory: callable returning a fresh DataLoader. Called
            once per T so each T sees the same sequence of batches (assuming
            the underlying dataset is deterministic — for IterableDatasets
            with shuffle, you'll want shuffle_shards=False or a fixed seed).

    Returns:
        dict[T] -> per-T metrics dict (same shape as
        compute_validation_metrics).
    """
    out = {}
    for T in T_values:
        loader = eval_loader_factory()
        out[T] = compute_validation_metrics(
            model, loader, device, config,
            amp_dtype=amp_dtype, max_batches=max_batches,
            max_thought_steps=T, topk_for_alignment=topk_for_alignment,
        )
    return out


def _svca_spectrum(
    Z: torch.Tensor,
    n_components: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Cross-validated (SVCA2) variance spectrum of an activation matrix.

    Splits neurons into two equal halves (a, b) and reads the shared variance
    of each covarying mode. The *neuron* split is the cross-validation: only
    covariation between the two disjoint populations survives, so private
    per-neuron noise collapses to ~0 in the tail instead of inflating it.

    This is SVCA2 — NOT time-split SVCA. The cross-covariance and the variance
    readout both use ALL timepoints; there is no train/test split over time.
    Stringer/Pachitariu found the extra time split inflates the power-law
    exponent (their Fig S4: a symmetric-connectivity ground truth of α≈0.69
    reads ≈0.86 with the time split vs ≈0.68 without), so it is omitted. This
    matters for comparison against the biological target α≈0.67; for pure
    self-comparison across runs the estimator choice only needs to be consistent.

    Args:
        Z: [N, d] activations (N timepoints, d neurons), fp32/64.
        n_components: cap on returned modes (default: d // 2).
        generator: optional RNG for the neuron split → reproducible α.

    Returns:
        1D tensor of cross-validated shared variances, descending order.
    """
    Z = Z.double()
    Z = Z - Z.mean(dim=0, keepdim=True)
    N, d = Z.shape

    nperm = torch.randperm(d, generator=generator)
    da = d // 2
    A = Z[:, nperm[:da]]
    B = Z[:, nperm[da:2 * da]]

    # SVCA2: cross-covariance over ALL timepoints (no train/test time split).
    C = (A.t() @ B) / max(N - 1, 1)
    U, _, Vh = torch.linalg.svd(C)
    V = Vh.t()
    k = n_components or da
    k = min(k, U.shape[1], V.shape[1])
    U, V = U[:, :k], V[:, :k]

    # Shared variance per mode = covariance of the paired projections on the
    # same data (≈ the singular values of C; the paired form is kept so
    # per-component temporal traces stay inspectable if ever needed).
    pa = A @ U
    pb = B @ V
    pa = pa - pa.mean(dim=0, keepdim=True)
    pb = pb - pb.mean(dim=0, keepdim=True)
    shared = (pa * pb).mean(dim=0)  # [k]
    return shared.sort(descending=True).values


def _fit_power_law(
    spec: torch.Tensor, fit_lo: int = 10, fit_hi: Optional[int] = 500
) -> Optional[dict]:
    """Weighted log-log fit of var(rank) ∝ rank^(-alpha) over [fit_lo, fit_hi).

    Uses the paper's regression weights 1/log(rank), which upweight the
    more-reliable head of the spectrum. Returns a metrics dict, or None when
    too few reliable (positive) modes exist to fit.
    """
    n_pos = int((spec > 0).sum().item())
    if n_pos < fit_lo + 5:
        return None
    hi = min(fit_hi or n_pos, n_pos)
    lo = fit_lo
    if hi - lo < 5:
        return None

    rank = torch.arange(lo, hi, dtype=torch.float64) + 1.0   # 1-based PC index
    x = torch.log(rank)
    y = torch.log(spec[lo:hi].double())
    w = 1.0 / torch.log(rank)                                # paper weighting
    w = w / w.sum()

    xm = (w * x).sum()
    ym = (w * y).sum()
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return None
    slope = (w * (x - xm) * (y - ym)).sum() / sxx
    yhat = slope * x + (ym - slope * xm)
    ss_res = (w * (y - yhat) ** 2).sum()
    ss_tot = (w * (y - ym) ** 2).sum().clamp_min(1e-12)
    return {
        "power_law_alpha": float(-slope.item()),
        "fit_r2": float((1.0 - ss_res / ss_tot).item()),
        "n_components": n_pos,
        "fit_lo": lo,
        "fit_hi": int(hi),
    }


@torch.no_grad()
def compute_spectrum_metrics(
    model,
    input_ids: torch.Tensor,
    device,
    fit_lo: int = 10,
    fit_hi: Optional[int] = 500,
    max_thought_steps: Optional[int] = None,
    seed: Optional[int] = 0,
) -> dict:
    """Power-law exponent of the latent-activation variance spectrum (SVCA2).

    Captures the per-tick post-norm latent z from every thought layer (the
    recurrent compute state) and fits a power law var(rank) ∝ rank^(-alpha)
    over PC indices [fit_lo, fit_hi] using the cross-validated SVCA2 spectrum.

    A well-conditioned critical-symmetric scaffold should sit near alpha ≈ 2/3
    (≈0.67); drift toward alpha ≫ 1 (collapsing/low-dimensional) or alpha ≈ 0
    (white/uncorrelated) is the pathology to watch.

    The headline `power_law_alpha` pools activations across all layers and
    ticks. Pooling conflates across-layer dimensionality with across-tick
    dynamics, so a per-layer breakdown (`per_layer`) is also returned — that's
    what reveals whether one specific layer is collapsing while others are
    healthy. Each layer/tick contributes the same B*S rows, so ticks are
    weighted equally (no late-tick bias).

    Args:
        seed: RNG seed for the neuron split, so repeated calls on the same data
            give the same alpha (stable learning-curve plots). None → global RNG.

    Returns:
        dict with headline keys: power_law_alpha, fit_r2, n_components,
        n_samples, fit_lo, fit_hi; plus `per_layer` (dict layer_idx -> fit dict)
        and `per_layer_alpha` (dict layer_idx -> alpha) for the breakdown.
    """
    nan = float("nan")
    out = {
        "power_law_alpha": nan, "fit_r2": nan,
        "n_components": 0, "n_samples": 0, "fit_lo": fit_lo, "fit_hi": fit_hi,
        "per_layer": {}, "per_layer_alpha": {},
    }

    unwrapped = _unwrap_model(model)
    if not hasattr(unwrapped, "_get_layers_sequence"):
        return out

    # One hook per unique post_norm module (de-dup so hyperloop's reused middle
    # block isn't double-counted), keyed by its first layer index.
    captured: dict[int, list[torch.Tensor]] = {}
    handles = []
    seen: dict[int, int] = {}

    def _make_hook(idx):
        def _hook(_m, _i, o):
            captured.setdefault(idx, []).append(
                o.detach().reshape(-1, o.shape[-1]).float().cpu()
            )
        return _hook

    for idx, layer in enumerate(unwrapped._get_layers_sequence()):
        pn = getattr(layer, "post_norm", None)
        if pn is None or id(pn) in seen:
            continue
        seen[id(pn)] = idx
        handles.append(pn.register_forward_hook(_make_hook(idx)))

    was_training = model.training
    model.eval()
    try:
        model(input_ids.to(device), max_thought_steps=max_thought_steps)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    if not captured:
        return out

    def _gen(s):
        if s is None:
            return None
        g = torch.Generator()
        g.manual_seed(int(s))
        return g

    # ── Per-layer spectra ────────────────────────────────────────────────
    per_layer: dict = {}
    per_layer_alpha: dict = {}
    for idx in sorted(captured):
        Zl = torch.cat(captured[idx], dim=0)
        if Zl.shape[0] < 8 or Zl.shape[1] < 4:
            continue
        spec_l = _svca_spectrum(Zl, generator=_gen(None if seed is None else seed + idx + 1))
        fit_l = _fit_power_law(spec_l, fit_lo, fit_hi)
        if fit_l is not None:
            per_layer[idx] = fit_l
            per_layer_alpha[idx] = fit_l["power_law_alpha"]
    out["per_layer"] = per_layer
    out["per_layer_alpha"] = per_layer_alpha

    # ── Pooled headline spectrum ─────────────────────────────────────────
    Z = torch.cat([c for caps in captured.values() for c in caps], dim=0)
    out["n_samples"] = int(Z.shape[0])
    if Z.shape[0] < 8 or Z.shape[1] < 4:
        return out

    spec = _svca_spectrum(Z, generator=_gen(seed))
    fit = _fit_power_law(spec, fit_lo, fit_hi)
    if fit is not None:
        out.update(fit)
    else:
        out["n_components"] = int((spec > 0).sum().item())
    return out


def format_spectrum_report(metrics: dict, prefix: str = "  ") -> str:
    """One-line formatted report for the activation-spectrum probe (SVCA2)."""
    a = metrics.get("power_law_alpha", float("nan"))
    if math.isnan(a):
        return prefix + f"spectrum: n/a (n_components={metrics.get('n_components', 0)})"
    line = (
        prefix
        + f"spectrum: α={a:.3f}  R²={metrics.get('fit_r2', float('nan')):.3f}  "
        f"(PCs {metrics.get('fit_lo')}–{metrics.get('fit_hi')}, "
        f"{metrics.get('n_components')} modes, {metrics.get('n_samples')} samples)"
    )
    per_layer = metrics.get("per_layer_alpha", {})
    if per_layer:
        layers = "  ".join(f"L{i}={per_layer[i]:.3f}" for i in sorted(per_layer))
        line += f"\n{prefix}  per-layer α: {layers}"
    return line


def format_validation_report(metrics: dict, prefix: str = "  ") -> str:
    """One-line-ish formatted report for log printing."""
    parts = [
        f"T={metrics['T']}",
        f"CE={metrics['ce_loss']:.4f}",
        f"PPL={metrics['ppl']:.2f}",
    ]
    if "top1_agree" in metrics and not math.isnan(metrics["top1_agree"]):
        parts.extend([
            f"top1={metrics['top1_agree']*100:.1f}%",
            f"top5={metrics['topk_agree']*100:.1f}%",
        ])
    parts.append(f"({metrics['n_tokens']} tok)")
    return prefix + " | ".join(parts)


def format_t_sweep_report(t_sweep: dict, prefix: str = "  ") -> str:
    """Multi-line table for T-ablation results."""
    lines = []
    lines.append(prefix + f"{'T':>4s}  {'CE':>8s}  {'PPL':>8s}  {'top1':>7s}  {'top5':>7s}")
    lines.append(prefix + "-" * 42)
    for T in sorted(t_sweep.keys()):
        m = t_sweep[T]
        top1 = f"{m['top1_agree']*100:.1f}%" if not math.isnan(m["top1_agree"]) else "  n/a"
        top5 = f"{m['topk_agree']*100:.1f}%" if not math.isnan(m["topk_agree"]) else "  n/a"
        lines.append(
            prefix + f"{str(T):>4s}  {m['ce_loss']:>8.4f}  {m['ppl']:>8.2f}  "
            f"{top1:>7s}  {top5:>7s}"
        )
    return "\n".join(lines)