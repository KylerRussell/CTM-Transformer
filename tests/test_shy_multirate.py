"""
test_shy_multirate.py — Spectral-radius homeostasis (SHY) + multi-rate streams.

Covers two mechanisms:

  1. `--use_spectral_renorm`: `renormalize_spectral_radius` projects each
     synapse z-recurrence matrix back to spectral radius ≤ target. Verifies it
     shrinks supercritical matrices (square dendritic/unet via eigenvalues,
     non-square mlp via singular value) and never amplifies sub-critical ones.

  2. `--use_multi_rate_streams`: matrix streams update on a power-of-2 tick
     cadence [1,1,2,4,...]. Verifies the period schedule and the freeze
     semantics of `apply_update_cadence` (slow streams hold their value), and
     that the off-path is an identity passthrough with no extra buffer.

Plus a forward/backward smoke with both features + critical init enabled.
Runs on CPU (no CUDA required).
"""

import math
import sys

import torch

sys.path.insert(0, ".")

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer


def make_config(**overrides):
    defaults = dict(
        vocab_size=512, d_model=64, d_latent=64, n_heads=4, n_layers=3,
        nlm_hidden_dim=16, history_len=4, max_thought_steps=5, max_seq_len=32,
        seq_len=16, batch_size=2, sync_method="diag_summary", dropout=0.0,
    )
    defaults.update(overrides)
    return CTMConfig(**defaults)


def _rho(slice_, square):
    A = slice_.detach().float()
    if square:
        return torch.linalg.eigvals(A).abs().max().real.item()
    return torch.linalg.svdvals(A).max().item()


def test_spectral_renorm_shrinks(device):
    for st, square in [("dendritic", True), ("mlp", False), ("unet", True)]:
        m = CTMTransformer(make_config(synapse_type=st)).to(device)
        layer = m._get_layers_sequence()[0]
        with torch.no_grad():
            layer._synapse_recurrent_slice().mul_(50.0)   # force supercritical
        rho0 = _rho(layer._synapse_recurrent_slice(), square)
        radii = m.renormalize_spectral_radius(target=1.0)
        rho1 = _rho(layer._synapse_recurrent_slice(), square)
        assert rho0 > 1.0 and rho1 <= 1.0 + 1e-3, (st, rho0, rho1)
        assert len(radii) == 3
    print("  [renorm] supercritical z-slice shrunk to ρ≤1 (dendritic/mlp/unet) ✓")


def test_spectral_renorm_only_shrinks(device):
    m = CTMTransformer(make_config(
        synapse_type="dendritic", use_critical_init=True,
        critical_init_spectral_radius=0.5,
    )).to(device)
    layer = m._get_layers_sequence()[0]
    before = layer._synapse_recurrent_slice().detach().clone()
    m.renormalize_spectral_radius(target=1.0)
    after = layer._synapse_recurrent_slice().detach()
    assert torch.allclose(before, after)
    print("  [renorm] sub-critical matrix left untouched (only shrinks) ✓")


def test_multi_rate_schedule_and_cadence(device):
    m = CTMTransformer(make_config(
        use_matrix_streams=True, n_streams=4, use_multi_rate_streams=True,
    )).to(device)
    stream = m._get_layers_sequence()[0].stream
    assert stream.stream_update_period.tolist() == [1, 1, 2, 4]

    old = torch.zeros(2, 16, 4, 64, device=device)
    new = torch.ones(2, 16, 4, 64, device=device)
    cases = {0: [1, 1, 1, 1], 1: [1, 1, 0, 0], 2: [1, 1, 1, 0],
             3: [1, 1, 0, 0], 4: [1, 1, 1, 1]}
    for t, exp in cases.items():
        out = stream.apply_update_cadence(t, old, new)
        got = [int(out[0, 0, i, 0].item()) for i in range(4)]
        assert got == exp, (t, got, exp)
    print("  [multi-rate] periods [1,1,2,4] + freeze cadence correct ✓")


def test_multi_rate_off_path(device):
    m = CTMTransformer(make_config(use_matrix_streams=True, n_streams=4)).to(device)
    s = m._get_layers_sequence()[0].stream
    assert not hasattr(s, "stream_update_period")
    old = torch.zeros(2, 16, 4, 64, device=device)
    new = torch.ones(2, 16, 4, 64, device=device)
    assert torch.equal(s.apply_update_cadence(3, old, new), new)
    print("  [multi-rate off] no buffer, cadence is identity ✓")


def test_forward_backward(device):
    m = CTMTransformer(make_config(
        use_matrix_streams=True, n_streams=4, use_multi_rate_streams=True,
        use_critical_init=True, synapse_type="dendritic",
    )).to(device)
    ids = torch.randint(0, 512, (2, 16), device=device)
    tgt = torch.randint(0, 512, (2, 16), device=device)
    loss = m(ids, targets=tgt)["loss"]
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    assert math.isfinite(loss.item()) and gn > 0
    print(f"  [fwd/bwd] multi-rate + critical init: loss={loss.item():.4f} "
          f"grad_norm={gn:.1f} ✓")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)
    test_spectral_renorm_shrinks(device)
    test_spectral_renorm_only_shrinks(device)
    test_multi_rate_schedule_and_cadence(device)
    test_multi_rate_off_path(device)
    test_forward_backward(device)
    print("\nALL SHY + MULTI-RATE TESTS PASSED ✓")


if __name__ == "__main__":
    main()
