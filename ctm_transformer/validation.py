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
    if max_thought_steps is None:
        unwrapped = _unwrap_model(model)
        if hasattr(unwrapped, "_train_step") and hasattr(config, "resolve_thought_steps"):
            T = config.resolve_thought_steps(int(unwrapped._train_step.item()))
        else:
            T = config.max_thought_steps
    else:
        T = max_thought_steps

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
        # Robust to both 5-tuple (legacy) and 6-tuple (with teacher_z)
        # collate outputs — we only need the first 4 elements here.
        ids, tgt, top_idx, top_val = batch[0], batch[1], batch[2], batch[3]
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
        "T": T,
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
            prefix + f"{T:>4d}  {m['ce_loss']:>8.4f}  {m['ppl']:>8.2f}  "
            f"{top1:>7s}  {top5:>7s}"
        )
    return "\n".join(lines)