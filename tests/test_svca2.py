"""
test_svca2.py — SVCA2 spectrum probe (validation.compute_spectrum_metrics).

The probe was changed from time-split SVCA to SVCA2 (no train/test time split),
which the Stringer/Pachitariu paper found avoids inflating the power-law
exponent. This test pins the SVCA2 behavior:

  1. Math identity: the paired-projection "shared" variances equal the singular
     values of the FULL-data neuron cross-covariance (up to the 1/N vs 1/(N-1)
     normalization) — i.e. no time split is used.
  2. Monotonic response: a larger planted shared-latent exponent yields a larger
     estimated α (we assert ordering, not exact recovery — the cross-covariance
     singular-value exponent isn't identical to the latent-variance exponent for
     non-orthonormal loadings, so exact recovery would require replicating the
     paper's connectivity simulation).
  3. The estimator genuinely differs from the old time-split version.
  4. Seeded calls are reproducible (stable learning curves); different seeds differ.
  5. Per-layer α breakdown is returned and the report renders it.

CAVEAT (documented behavior, not tested as a target): SVCA2 measures shared
variance on the same data, so on near-white / low-SNR inputs the SVD overfits
and reports a spuriously decaying spectrum. That noise-robustness was the
dropped time split's job; SVCA2 is appropriate for high-SNR structured
activations and for self-comparison across runs with a fixed seed.

Runs on CPU.
"""

import math
import sys

import torch

sys.path.insert(0, ".")

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.validation import (
    _svca_spectrum,
    _fit_power_law,
    compute_spectrum_metrics,
    format_spectrum_report,
)


def _shared_latent(alpha, r=200, d=800, N=6000, noise=0.1):
    lam = (torch.arange(1, r + 1, dtype=torch.float64)) ** (-alpha)
    F = torch.randn(N, r, dtype=torch.float64) * lam.sqrt()
    L = torch.randn(d, r, dtype=torch.float64) / math.sqrt(r)
    return F @ L.t() + noise * torch.randn(N, d, dtype=torch.float64)


def _timesplit_spectrum(Z, generator):
    """The OLD (now-replaced) time-split estimator, for comparison only."""
    Z = Z.double() - Z.double().mean(0, keepdim=True)
    N, d = Z.shape
    nperm = torch.randperm(d, generator=generator)
    da = d // 2
    A, B = Z[:, nperm[:da]], Z[:, nperm[da:2 * da]]
    tperm = torch.randperm(N, generator=generator)
    nt = N // 2
    tr, te = tperm[:nt], tperm[nt:]
    C = (A[tr].t() @ B[tr]) / max(nt - 1, 1)
    U, _, Vh = torch.linalg.svd(C)
    V = Vh.t()
    pa, pb = A[te] @ U, B[te] @ V
    pa -= pa.mean(0, keepdim=True)
    pb -= pb.mean(0, keepdim=True)
    return (pa * pb).mean(0).sort(descending=True).values


def test_no_time_split_identity():
    N, d = 6000, 800
    Z = _shared_latent(0.7, N=N, d=d)
    Zc = Z.double() - Z.double().mean(0, keepdim=True)
    g = torch.Generator(); g.manual_seed(3)
    spec = _svca_spectrum(Z, generator=g)
    # Reproduce the exact split and compare to full-data cross-cov singular values.
    g2 = torch.Generator(); g2.manual_seed(3)
    nperm = torch.randperm(d, generator=g2); da = d // 2
    A, B = Zc[:, nperm[:da]], Zc[:, nperm[da:2 * da]]
    svals = torch.linalg.svdvals((A.t() @ B) / (N - 1)).sort(descending=True).values
    rel = ((spec[:50] - svals[:50]).abs().max() / svals[:50].abs().max()).item()
    assert rel < 2.0 / N, rel
    print(f"  [identity] shared == full cross-cov singular values (rel-err {rel:.1e}) ✓")


def test_monotonic_response():
    est = {}
    for alpha in (0.4, 0.7, 1.1):
        g = torch.Generator(); g.manual_seed(0)
        fit = _fit_power_law(_svca_spectrum(_shared_latent(alpha), generator=g), 10, 180)
        est[alpha] = fit["power_law_alpha"]
        assert fit["fit_r2"] > 0.8
    assert est[0.4] < est[0.7] < est[1.1], est
    print(f"  [monotonic] α(0.4)<α(0.7)<α(1.1): "
          f"{est[0.4]:.2f}<{est[0.7]:.2f}<{est[1.1]:.2f} ✓")


def test_differs_from_timesplit():
    Z = _shared_latent(0.7)
    g1 = torch.Generator(); g1.manual_seed(0)
    g2 = torch.Generator(); g2.manual_seed(0)
    a2 = _fit_power_law(_svca_spectrum(Z, generator=g1), 10, 180)["power_law_alpha"]
    asp = _fit_power_law(_timesplit_spectrum(Z, g2), 10, 180)["power_law_alpha"]
    assert abs(a2 - asp) > 1e-3
    print(f"  [differs] SVCA2 α={a2:.3f} ≠ time-split α={asp:.3f} ✓")


def test_reproducible_and_per_layer(device):
    cfg = CTMConfig(
        vocab_size=512, d_model=64, d_latent=64, n_heads=4, n_layers=3,
        nlm_hidden_dim=16, history_len=4, max_thought_steps=5, max_seq_len=32,
        seq_len=16, batch_size=2, sync_method="diag_summary", dropout=0.0,
    )
    m = CTMTransformer(cfg).to(device)
    ids = torch.randint(0, 512, (2, 16), device=device)
    r1 = compute_spectrum_metrics(m, ids, device, fit_lo=4, fit_hi=None, seed=0)
    r2 = compute_spectrum_metrics(m, ids, device, fit_lo=4, fit_hi=None, seed=0)
    r3 = compute_spectrum_metrics(m, ids, device, fit_lo=4, fit_hi=None, seed=99)
    assert r1["power_law_alpha"] == r2["power_law_alpha"]   # deterministic per seed
    assert r1["power_law_alpha"] != r3["power_law_alpha"]   # seed actually matters
    assert len(r1["per_layer_alpha"]) == 3
    assert "\n" in format_spectrum_report(r1)               # per-layer line present
    print(f"  [repro+per-layer] seed deterministic; per-layer α="
          + ",".join(f"{v:.2f}" for v in r1['per_layer_alpha'].values()) + " ✓")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)
    test_no_time_split_identity()
    test_monotonic_response()
    test_differs_from_timesplit()
    test_reproducible_and_per_layer(device)
    print("\nALL SVCA2 TESTS PASSED ✓")


if __name__ == "__main__":
    main()
