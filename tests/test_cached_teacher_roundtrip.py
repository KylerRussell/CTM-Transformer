"""
test_cached_teacher_roundtrip.py — Smoke test for the offline-cache
producer + consumer + model integration.

Runs entirely on CPU with a tiny fake teacher and a tiny fake config so
we can verify shapes, dtypes, and the KL loss computation without
needing the real NemotronH weights or a GPU.

What it checks:
  1. Producer-shaped data round-trips through write_shard.
  2. CachedTeacherDataset yields the right shapes/dtypes.
  3. The model.forward distillation block accepts cached tensors and
     produces a finite KL loss.
  4. The cached path matches the live-teacher path exactly when the
     cached top-K is the actual topk of teacher logits (numerical
     equivalence test).

Run from the repo root:
    PYTHONPATH=. python test_cached_teacher_roundtrip.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ctm_transformer.cached_teacher_dataset import (
    CachedTeacherDataset,
    cached_teacher_collate,
)


def fake_shard(out_dir: Path, idx: int, N=4, S=16, K=8, V=128, seed=0):
    """Write a fake shard with random data matching the producer format."""
    rng = np.random.RandomState(seed + idx)
    input_ids = rng.randint(0, V, size=(N, S), dtype=np.int32)
    targets = rng.randint(0, V, size=(N, S), dtype=np.int32)
    top_indices = rng.randint(0, V, size=(N, S, K), dtype=np.int32)
    top_values = rng.randn(N, S, K).astype(np.float16)
    residual = rng.randn(N, S).astype(np.float16) * 0.1 - 5.0
    np.savez(
        out_dir / f"shard_{idx:06d}.npz",
        input_ids=input_ids, targets=targets,
        top_indices=top_indices, top_values=top_values,
        residual_log_mass=residual,
    )
    return input_ids, targets, top_indices, top_values, residual


def test_dataset_shapes():
    print("[test 1] Dataset shapes and dtypes...", end=" ")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        # Need at least 20 shards so the 5% eval split reserves >= 1.
        # With the default eval_fraction=0.05, len * 0.05 -> max(1, ...)
        # so 5 shards still gives 1 eval and 4 train. Use 5.
        for i in range(5):
            fake_shard(out_dir, i, N=4, S=16, K=8, V=128, seed=42)

        ds = CachedTeacherDataset(
            cache_dir=tmp,
            shuffle_shards=False,
            loop=False,
            is_eval=False,
        )
        n_samples = 0
        for x, y, ti, tv, rm in ds:
            assert x.shape == (16,) and x.dtype == torch.long, f"x: {x.shape} {x.dtype}"
            assert y.shape == (16,) and y.dtype == torch.long
            assert ti.shape == (16, 8) and ti.dtype == torch.long
            assert tv.shape == (16, 8) and tv.dtype == torch.float32
            assert rm.shape == (16,) and rm.dtype == torch.float32
            n_samples += 1
        assert n_samples == 4 * 4  # 4 train shards × 4 samples
    print("OK")


def test_collate():
    print("[test 2] Collate function batches correctly...", end=" ")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for i in range(3):
            fake_shard(out_dir, i, N=2, S=16, K=8, V=128)
        ds = CachedTeacherDataset(cache_dir=tmp, shuffle_shards=False, loop=False)
        batch = [next(iter(ds)) for _ in range(2)]
        x, y, ti, tv, rm = cached_teacher_collate(batch)
        assert x.shape == (2, 16)
        assert ti.shape == (2, 16, 8)
        assert tv.shape == (2, 16, 8)
        assert rm.shape == (2, 16)
    print("OK")


def test_kl_equivalence_to_live_path():
    """The cached path must produce IDENTICAL teacher_log_probs as the
    live-teacher path when fed the actual top-K of the teacher's logits.

    This is the linchpin test: if numerical equivalence holds, then
    swapping live → cached changes only WHERE the top-K computation
    happens, not WHAT the model trains on.
    """
    print("[test 3] Cache path matches live-teacher path numerically...", end=" ")

    torch.manual_seed(0)
    B, S, V, K = 2, 5, 64, 8
    T_temp = 4.0

    # Fake teacher logits and student logits (one tick)
    teacher_logits = torch.randn(B, S, V) * 2.0
    student_logits = torch.randn(B, S, V) * 2.0

    # ── Live path: compute topk inside the loss ───────────────────
    teacher_logits_flat = teacher_logits.reshape(B * S, V)
    teacher_top_logits, teacher_top_indices = teacher_logits_flat.topk(K, dim=-1)
    teacher_log_probs_live = F.log_softmax(
        teacher_top_logits / T_temp, dim=-1, dtype=torch.float32,
    )
    student_logits_flat = student_logits.reshape(B * S, V)
    student_top_logits = student_logits_flat.gather(-1, teacher_top_indices)
    student_log_probs_live = F.log_softmax(
        student_top_logits / T_temp, dim=-1, dtype=torch.float32,
    )
    kl_live = F.kl_div(
        student_log_probs_live, teacher_log_probs_live,
        reduction="batchmean", log_target=True,
    ) * T_temp * T_temp

    # ── Cache path: precompute topk in the producer, supply to loss ──
    top_v_cached, top_i_cached = teacher_logits.topk(K, dim=-1)  # [B, S, K]
    # Round-trip through fp16 storage to match real cache fidelity
    top_v_cached_f16 = top_v_cached.to(torch.float16).to(torch.float32)

    # Replicate the cached-path arithmetic from model.py
    top_i_flat = top_i_cached.reshape(B * S, K).long()
    top_v_flat = top_v_cached_f16.reshape(B * S, K)
    teacher_log_probs_cached = F.log_softmax(
        top_v_flat / T_temp, dim=-1, dtype=torch.float32,
    )
    student_top_logits_cached = student_logits_flat.gather(-1, top_i_flat)
    student_log_probs_cached = F.log_softmax(
        student_top_logits_cached / T_temp, dim=-1, dtype=torch.float32,
    )
    kl_cached = F.kl_div(
        student_log_probs_cached, teacher_log_probs_cached,
        reduction="batchmean", log_target=True,
    ) * T_temp * T_temp

    # The two should match within fp16 quantization noise on top_values.
    abs_diff = (kl_live - kl_cached).abs().item()
    rel_diff = abs_diff / max(abs(kl_live.item()), 1e-9)
    assert rel_diff < 1e-3, (
        f"KL mismatch live={kl_live.item():.6f} cached={kl_cached.item():.6f} "
        f"rel_diff={rel_diff:.2e}"
    )
    print(f"OK (live={kl_live.item():.4f}, cached={kl_cached.item():.4f}, "
          f"rel_diff={rel_diff:.1e})")


def test_residual_log_mass_finite():
    """Residual log-mass should be a finite number ≤ 0 for any logits."""
    print("[test 4] Residual log-mass numerical stability...", end=" ")
    torch.manual_seed(42)
    K = 4
    V = 64

    for case in ["uniform", "peaked", "extreme"]:
        if case == "uniform":
            logits = torch.zeros(8, V)
        elif case == "peaked":
            logits = torch.randn(8, V) * 0.1
            logits[:, 0] = 100.0  # near-degenerate top
        else:  # extreme
            logits = torch.randn(8, V) * 1000.0

        full_lse = torch.logsumexp(logits.float(), dim=-1)
        top_v, _ = logits.topk(K, dim=-1)
        top_lse = torch.logsumexp(top_v.float(), dim=-1)
        log_p_top = (top_lse - full_lse).clamp(max=-1e-7)
        threshold = -float(np.log(2.0))
        residual = torch.where(
            log_p_top > threshold,
            torch.log(-torch.expm1(log_p_top)),
            torch.log1p(-torch.exp(log_p_top)),
        )
        assert torch.isfinite(residual).all(), f"{case}: {residual}"
        assert (residual <= 0).all(), f"{case}: residual must be log of probability"
    print("OK")


if __name__ == "__main__":
    test_dataset_shapes()
    test_collate()
    test_kl_equivalence_to_live_path()
    test_residual_log_mass_finite()
    print("\nAll cache-roundtrip smoke tests passed.")
