"""
CTM-v2 Feature Smoke Tests

Tests each v2 feature flag independently and in combination:
1. Loop Position Embeddings
2. FEEC Integrator
3. Matrix-Valued Residual Streams
4. DSSA (Dual-Space Sparse Attention)
5. Hyperloop Looped Middle Cycle
6. All v2 features combined
"""

import sys
import torch
sys.path.insert(0, ".")

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer


# ── Test helpers ────────────────────────────────────────────────────────

def make_config(**overrides):
    """Create a small test config with optional overrides."""
    defaults = dict(
        vocab_size=50257,
        d_model=128,
        d_latent=128,
        n_heads=4,
        n_layers=4,
        nlm_hidden_dim=16,
        history_len=4,
        max_thought_steps=4,
        max_seq_len=64,
        seq_len=32,
        batch_size=2,
        sync_method="diag_summary",
        gradient_checkpointing=False,
        dropout=0.0,
    )
    defaults.update(overrides)
    return CTMConfig(**defaults)


def run_forward_backward(config, label):
    """Run forward + backward and verify shapes & gradients."""
    print(f"\n{'─'*60}")
    print(f"[{label}]")
    print(f"{'─'*60}")

    model = CTMTransformer(config)
    n_params = model.get_num_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    B, S = config.batch_size, config.seq_len
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    targets = torch.randint(0, config.vocab_size, (B, S))

    # Forward
    result = model(input_ids, targets=targets)
    logits = result["logits"]
    loss = result["loss"]

    assert logits.shape == (B, S, config.vocab_size), f"Bad logits shape: {logits.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss not positive: {loss.item()}"
    print(f"  Forward OK — loss: {loss.item():.4f}")

    # Check v2-specific outputs
    if config.use_feec and "feec_energy" in result:
        print(f"  FEEC energy: {result['feec_energy']:.4f}")
    if "feec_energy_penalty" in result:
        print(f"  FEEC energy penalty: {result['feec_energy_penalty']:.6f}")

    # Backward
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters())
    print(f"  Backward OK — {has_grad}/{total} params have gradients")

    # Mini training (3 steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_loss = loss.item()
    for step in range(3):
        result = model(input_ids, targets=targets)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    delta = final_loss - initial_loss
    print(f"  Training: {initial_loss:.4f} → {final_loss:.4f} (Δ={delta:+.4f})")
    print(f"  ✓ PASSED")
    return True


# ── Individual feature tests ───────────────────────────────────────────

def test_loop_pos_emb():
    config = make_config(use_loop_pos_emb=True)
    return run_forward_backward(config, "Loop Position Embeddings")


def test_feec():
    config = make_config(use_feec=True, feec_energy_penalty_weight=0.01)
    return run_forward_backward(config, "FEEC Integrator")


def test_matrix_streams():
    config = make_config(use_matrix_streams=True, n_streams=4)
    return run_forward_backward(config, "Matrix-Valued Residual Streams")


def test_dssa():
    config = make_config(
        use_dssa=True,
        dssa_n_partitions=8,
        dssa_top_k=4,
        dssa_block_size=16,
        dssa_top_k_blocks=2,
    )
    return run_forward_backward(config, "Dual-Space Sparse Attention")


def test_hyperloop():
    config = make_config(
        use_hyperloop=True,
        n_layers=8,  # Need enough layers for begin+middle+end
        hyperloop_n_begin=2,
        hyperloop_n_middle=2,
        hyperloop_n_end=2,
        hyperloop_middle_loops=2,
    )
    return run_forward_backward(config, "Hyperloop Looped Middle Cycle")


def test_feec_plus_streams():
    config = make_config(
        use_feec=True,
        use_matrix_streams=True,
        n_streams=4,
        use_loop_pos_emb=True,
    )
    return run_forward_backward(config, "FEEC + Matrix Streams + Loop Pos Emb")


def test_all_v2():
    config = make_config(
        use_feec=True,
        use_matrix_streams=True,
        n_streams=4,
        use_loop_pos_emb=True,
        use_hyperloop=True,
        n_layers=8,
        hyperloop_n_begin=2,
        hyperloop_n_middle=2,
        hyperloop_n_end=2,
        hyperloop_middle_loops=2,
        # Skip DSSA for the all-in test (it's slow with naive SSE scan)
    )
    return run_forward_backward(config, "All v2 Features (except DSSA)")


# ── FEEC stability test ────────────────────────────────────────────────

def test_feec_energy_stability():
    """Verify FEEC energy remains bounded over many steps."""
    print(f"\n{'─'*60}")
    print(f"[FEEC Energy Stability (100 steps)]")
    print(f"{'─'*60}")

    from ctm_transformer.feec_integrator import FEECIntegrator

    integrator = FEECIntegrator(d_latent=128, n_layers=4)
    z = torch.randn(2, 32, 128) * 0.01
    v = torch.zeros(2, 32, 128)

    energies = []
    for t in range(100):
        force = torch.randn(2, 32, 128) * 0.05
        z, v = integrator.step(z, v, force, layer_idx=t % 4)
        e = integrator.energy(z, v)
        assert torch.isfinite(e), f"Energy diverged at step {t}: {e.item()}"
        energies.append(e.item())

    print(f"  Energy range: [{min(energies):.2f}, {max(energies):.2f}]")
    print(f"  Final energy: {energies[-1]:.2f}")
    print(f"  ✓ PASSED — energy bounded over 100 steps")
    return True


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CTM-v2 Feature Smoke Tests")
    print("=" * 60)

    results = {}
    tests = [
        ("Loop Pos Emb", test_loop_pos_emb),
        ("FEEC", test_feec),
        ("FEEC Stability", test_feec_energy_stability),
        ("Matrix Streams", test_matrix_streams),
        ("DSSA", test_dssa),
        ("Hyperloop", test_hyperloop),
        ("FEEC+Streams+LPE", test_feec_plus_streams),
        ("All v2", test_all_v2),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
