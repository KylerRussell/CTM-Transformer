"""
test_critical_init.py — Critical-symmetric dynamics-prior init.

Covers the `--use_critical_init` feature:

  1. `critical_symmetric_init_` produces the requested spectral radius and
     (for square inputs) a symmetric, zero-diagonal matrix; non-square
     inputs fall back to spectral-norm scaling (asymmetric).
  2. The init is applied to the synapse z-recurrence slice for every
     synapse_type AND survives the global `_init_weights` pass (it runs in
     the `_reset_special_inits` hook afterward). mlp's z-slice is non-square
     → scaled-asymmetric; dendritic/unet are square → critical-symmetric.
  3. With the flag OFF the path is unchanged: z-slice keeps normal_(0.02),
     no Hebbian `M_init` buffer, `init_state` returns zeros.
  4. HebbianSynapse pre-warms M_0 with a critical buffer (flat + compartments).
  5. A model with critical init does a clean forward/backward.
  6. `compute_spectrum_metrics` returns a finite power-law fit.

Runs on CPU (no CUDA required).
"""

import math
import sys

import torch

sys.path.insert(0, ".")

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.biological import critical_symmetric_init_
from ctm_transformer.validation import (
    compute_spectrum_metrics,
    format_spectrum_report,
)


def make_config(**overrides):
    defaults = dict(
        vocab_size=512, d_model=64, d_latent=64, n_heads=4, n_layers=3,
        nlm_hidden_dim=16, history_len=4, max_thought_steps=4, max_seq_len=32,
        seq_len=16, batch_size=2, sync_method="diag_summary", dropout=0.0,
    )
    defaults.update(overrides)
    return CTMConfig(**defaults)


def test_utility():
    W = torch.empty(128, 128)
    critical_symmetric_init_(W, sym_frac=1.0, spectral_radius=0.9)
    rho = torch.linalg.eigvals(W).abs().max().real.item()
    assert abs(rho - 0.9) < 1e-3, rho
    assert (W - W.t()).abs().max().item() < 1e-5      # symmetric
    assert W.diag().abs().max().item() < 1e-6         # zero diagonal

    Wn = torch.empty(128, 64)
    critical_symmetric_init_(Wn, sym_frac=1.0, spectral_radius=0.8)
    assert abs(torch.linalg.svdvals(Wn).max().item() - 0.8) < 1e-3
    print("  [utility] spectral radius / symmetry / non-square scaling ✓")


def test_synapse_zslice(device):
    for st, square in [("mlp", False), ("dendritic", True), ("unet", True)]:
        m = CTMTransformer(make_config(
            synapse_type=st, use_critical_init=True,
            critical_init_sym_frac=1.0, dendritic_n_branches=4,
        )).to(device)
        layer = m._get_layers_sequence()[0]
        dm = layer.d_model
        if st == "dendritic":
            W = layer.synapse.soma.weight
        elif st == "unet":
            W = layer.synapse.proj[1].weight
        else:
            W = layer.synapse[0].weight
        z = W[:, dm:].detach().float().cpu()
        if square:
            rho = torch.linalg.eigvals(z).abs().max().real.item()
            assert abs(rho - 0.999) < 1e-2, f"{st}: rho={rho}"
        else:
            sv = torch.linalg.svdvals(z).max().item()
            assert abs(sv - 0.999) < 1e-2, f"{st}: sigma={sv}"
    print("  [synapse] z-slice critically initialized for mlp/dendritic/unet ✓")


def test_off_path_unchanged(device):
    m = CTMTransformer(make_config(
        synapse_type="dendritic", use_critical_init=False,
        use_hebbian_synapse=True, hebbian_bottleneck_dim=32,
    )).to(device)
    layer = m._get_layers_sequence()[0]
    z = layer.synapse.soma.weight[:, layer.d_model:].detach().float()
    assert z.std().item() < 0.05, z.std().item()      # still normal_(0.02)
    heb = layer.hebbian
    assert not hasattr(heb, "M_init")
    st = heb.init_state(2, 16, torch.device(device), torch.float32)
    assert st.abs().sum().item() == 0.0
    print("  [off] normal init + zero Hebbian state preserved ✓")


def test_hebbian_prior(device):
    m = CTMTransformer(make_config(
        use_hebbian_synapse=True, hebbian_bottleneck_dim=32,
        use_critical_init=True, critical_init_sym_frac=1.0,
    )).to(device)
    heb = m._get_layers_sequence()[0].hebbian
    assert hasattr(heb, "M_init") and heb.M_init.shape == (32, 32)
    rho = torch.linalg.eigvals(heb.M_init).abs().max().real.item()
    assert abs(rho - 0.999) < 1e-2, rho
    st = heb.init_state(2, 16, torch.device(device), torch.float32)
    assert st.shape == (2, 16, 32, 32) and st.abs().sum().item() > 0

    mc = CTMTransformer(make_config(
        use_hebbian_synapse=True, hebbian_bottleneck_dim=32,
        hebbian_n_compartments=4, use_critical_init=True,
    )).to(device)
    assert mc._get_layers_sequence()[0].hebbian.M_init.shape == (4, 8, 8)
    print("  [hebbian] M_0 pre-warmed (flat 32x32 + compartments 4x8x8) ✓")


def test_forward_backward_and_spectrum(device):
    m = CTMTransformer(make_config(
        synapse_type="dendritic", use_critical_init=True,
        use_hebbian_synapse=True, hebbian_bottleneck_dim=32,
        use_matrix_streams=True, n_streams=4,
    )).to(device)
    ids = torch.randint(0, 512, (2, 16), device=device)
    tgt = torch.randint(0, 512, (2, 16), device=device)
    loss = m(ids, targets=tgt)["loss"]
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    assert math.isfinite(loss.item()) and gnorm > 0
    print(f"  [fwd/bwd] loss={loss.item():.4f} grad_norm={gnorm:.1f} ✓")

    sm = compute_spectrum_metrics(m, ids, device, fit_lo=4, fit_hi=None)
    assert math.isfinite(sm["power_law_alpha"]) and sm["n_samples"] > 0
    print("  " + format_spectrum_report(sm).strip() + " ✓")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)
    test_utility()
    test_synapse_zslice(device)
    test_off_path_unchanged(device)
    test_hebbian_prior(device)
    test_forward_backward_and_spectrum(device)
    print("\nALL CRITICAL-INIT TESTS PASSED ✓")


if __name__ == "__main__":
    main()
