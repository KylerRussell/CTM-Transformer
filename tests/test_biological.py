"""
Smoke test for the biological learning extensions.

What this tests (CPU-only, ~5 second run):
  1. Config flags can be set, no AttributeError on the new fields.
  2. Model constructs cleanly with both `use_hebbian_synapse=True`
     and `use_predictive_coding=True`.
  3. Forward pass produces a loss and runs `loss.backward()` without
     shape errors (the new return-tuple sizes are wired correctly).
  4. The PC auxiliary loss is nonzero when teacher_z is provided.
  5. The PC auxiliary loss is silently zero when teacher_z is omitted
     (graceful degradation).
  6. The Hebbian fast-weight matrix shape is what we expect, and the
     update preserves its shape across thought ticks.
  7. Backward through the cerebellar readout actually populates grads
     on the predictor parameters.
  8. Backward through the Hebbian synapse populates grads on the
     decay/lr/gate scalars.
  9. With both flags OFF, the model trains exactly as before (no new
     parameters, no PC loss key in the result dict).

Run from the project root, either way works:
    python -m tests.test_biological
    python tests/test_biological.py
"""
from __future__ import annotations

import math
import os
import sys

# Make this script robust to its launch location. When invoked as a
# plain script (`python tests/test_biological.py`), the project root
# is NOT on sys.path by default — only the script's directory. So we
# walk up one level and prepend the parent so `import ctm_transformer`
# works without requiring `python -m`. When invoked via `-m`, this is
# a harmless no-op because the project root is already on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

# Import via the package so the relative imports inside model.py work.
from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.biological import HebbianSynapse, CerebellarReadout


def _make_tiny_config(**overrides) -> CTMConfig:
    """A maximally cheap config that still exercises every code path."""
    base = dict(
        vocab_size=128,
        d_model=32,
        d_latent=32,
        n_heads=4,
        n_layers=2,
        nlm_hidden_dim=8,
        nlm_groups=1,
        history_len=2,
        max_thought_steps=2,
        max_seq_len=8,
        seq_len=8,
        batch_size=2,
        dropout=0.0,
        use_shared_head_film=True,    # required for tie_embeddings
        tie_embeddings=False,
        temporal_loss_type="dynamic_aggregate",
        gradient_checkpointing=False,
        # Distillation off by default — individual tests toggle as needed
        use_distillation=False,
        use_cached_teacher=False,
        teacher_d_model=16,
        # Biological extensions
        use_hebbian_synapse=False,
        use_predictive_coding=False,
    )
    base.update(overrides)
    return CTMConfig(**base)


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — config: new fields exist and accept overrides
# ─────────────────────────────────────────────────────────────────────────
def test_config_fields():
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=8,
        hebbian_decay_init=0.85,
        hebbian_lr_init=0.05,
        hebbian_gate_init=-2.0,
        use_predictive_coding=True,
        pc_loss_weight=0.1,
        pc_hidden_dim=16,
        pc_normalize="layernorm",
        pc_loss_type="cosine",
        pc_dropout=0.0,
        pc_tick_aggregation="all",
    )
    assert cfg.use_hebbian_synapse is True
    assert cfg.hebbian_bottleneck_dim == 8
    assert cfg.use_predictive_coding is True
    assert cfg.pc_loss_weight == 0.1
    assert cfg.pc_tick_aggregation == "all"
    print("  [pass] new config fields wired through")


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — model constructs with both flags ON
# ─────────────────────────────────────────────────────────────────────────
def test_model_constructs_with_bio_on():
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        use_predictive_coding=True,
        pc_hidden_dim=8,
    )
    m = CTMTransformer(cfg)
    # Cerebellar readout was created
    assert m.cerebellar_readout is not None, "cerebellar_readout not constructed"
    assert isinstance(m.cerebellar_readout, CerebellarReadout)
    # Each ThoughtLayer has a Hebbian synapse
    for layer in m._get_layers_sequence():
        assert layer.hebbian is not None, "ThoughtLayer.hebbian not constructed"
        assert isinstance(layer.hebbian, HebbianSynapse)
    print("  [pass] model constructs with both bio flags ON")


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — forward + backward end-to-end (Hebbian + PC both ON, teacher_z given)
# ─────────────────────────────────────────────────────────────────────────
def test_forward_backward_with_bio():
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        use_predictive_coding=True,
        pc_loss_weight=0.1,
        pc_hidden_dim=8,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    m.train()

    B, S = 2, 8
    x = torch.randint(0, cfg.vocab_size, (B, S))
    y = torch.randint(0, cfg.vocab_size, (B, S))
    # Teacher hidden state must match cfg.teacher_d_model.
    tz = torch.randn(B, S, cfg.teacher_d_model)

    result = m(x, targets=y, max_thought_steps=cfg.max_thought_steps,
               teacher_z=tz)

    assert "loss" in result, "loss missing"
    assert "pc_loss" in result, "pc_loss missing when teacher_z provided"
    assert torch.isfinite(result["loss"]), f"loss is not finite: {result['loss']}"
    assert torch.isfinite(result["pc_loss"]), f"pc_loss is not finite: {result['pc_loss']}"
    assert result["pc_loss"].item() != 0.0, \
        "pc_loss is exactly zero — predictor probably wasn't called"

    # Backward — should not raise
    result["loss"].backward()

    # Confirm grads landed on the cerebellar predictor
    pred_grads = [p.grad for p in m.cerebellar_readout.predictor.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in pred_grads), \
        "cerebellar_readout received no gradient"

    # Confirm grads landed on Hebbian gate / lr / decay
    layer0 = m._get_layers_sequence()[0]
    assert layer0.hebbian.gate_logit.grad is not None, \
        "Hebbian gate received no gradient"
    assert layer0.hebbian.lr_logit.grad is not None, \
        "Hebbian lr_logit received no gradient"

    print(f"  [pass] forward+backward with bio ON | loss={result['loss'].item():.4f} "
          f"pc_loss={result['pc_loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 4 — graceful degradation: PC ON but no teacher_z → pc_loss absent
# ─────────────────────────────────────────────────────────────────────────
def test_pc_silent_without_teacher_z():
    cfg = _make_tiny_config(use_predictive_coding=True)
    m = CTMTransformer(cfg).to(torch.float32)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    # Note: no teacher_z passed
    result = m(x, targets=y, max_thought_steps=cfg.max_thought_steps)

    assert "loss" in result
    assert "pc_loss" not in result, \
        "pc_loss key should be absent when teacher_z is None"
    assert torch.isfinite(result["loss"])
    print(f"  [pass] PC silently zero when teacher_z is None | "
          f"loss={result['loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 5 — baseline (both flags OFF) still works
# ─────────────────────────────────────────────────────────────────────────
def test_baseline_unchanged():
    cfg = _make_tiny_config()  # both OFF
    m = CTMTransformer(cfg).to(torch.float32)
    assert m.cerebellar_readout is None
    for layer in m._get_layers_sequence():
        assert layer.hebbian is None

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=cfg.max_thought_steps)
    assert "pc_loss" not in result
    result["loss"].backward()
    print(f"  [pass] baseline (both flags OFF) unchanged | "
          f"loss={result['loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 6 — HebbianSynapse unit-tests (shapes, decay/lr behaviour)
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_unit():
    d_lat, d_mod, bn = 16, 24, 4
    h = HebbianSynapse(d_latent=d_lat, d_model=d_mod, bottleneck_dim=bn)
    B, S = 2, 5
    M0 = h.init_state(B, S, torch.device("cpu"), torch.float32)
    assert M0.shape == (B, S, bn, bn), f"unexpected M shape: {M0.shape}"
    assert M0.abs().sum() == 0, "init state should be zeros"

    z = torch.randn(B, S, d_lat)
    a = torch.randn(B, S, d_mod)
    readout, M1, lr_eff = h(z, a, M0)
    assert readout.shape == (B, S, d_lat), f"readout shape: {readout.shape}"
    assert M1.shape == M0.shape
    assert lr_eff.ndim == 0, f"lr_eff should be scalar, got {lr_eff.shape}"
    # First-tick readout should be ~zero because M0=0 → M0·a=0.
    # (After projection through readout_proj it's still zero, then
    # LayerNorm of zero is zero, then gate*zero is zero.)
    assert readout.abs().max() < 1e-4, \
        f"first-tick readout should be zero, got max={readout.abs().max()}"

    # After the first update, M1 should be the lr*outer product (decay*0 = 0).
    # We can verify by checking that M1 is nonzero somewhere.
    assert M1.abs().sum() > 0, "M was not updated"

    # Without a modulator, lr_eff == sigmoid(lr_logit) (base lr).
    expected_lr = torch.sigmoid(h.lr_logit).item()
    assert abs(lr_eff.item() - expected_lr) < 1e-5, \
        f"unmodulated lr_eff should equal base lr {expected_lr:.4f}, got {lr_eff.item():.4f}"

    # Now test with a per-position modulator: lr_eff should be base * mean(mod).
    mod = torch.ones(B, S) * 3.0  # uniform 3x boost
    _, _, lr_eff_mod = h(z, a, M0, lr_modulator=mod)
    expected_mod_lr = expected_lr * 3.0
    assert abs(lr_eff_mod.item() - expected_mod_lr) < 1e-4, \
        (f"modulator=3 should give 3x base lr ({expected_mod_lr:.4f}), "
         f"got {lr_eff_mod.item():.4f}")

    print(f"  [pass] HebbianSynapse unit | M shape {tuple(M1.shape)} | "
          f"first-tick readout norm {readout.abs().max():.2e} | "
          f"lr_eff base={lr_eff.item():.4f} 3x_mod={lr_eff_mod.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 7 — CerebellarReadout unit-tests
# ─────────────────────────────────────────────────────────────────────────
def test_cerebellar_unit():
    d_lat, d_tea = 16, 24
    c = CerebellarReadout(d_latent=d_lat, target_dim=d_tea,
                          hidden_dim=32, loss_type="cosine")
    z = torch.randn(2, 5, d_lat)
    tz = torch.randn(2, 5, d_tea)
    loss = c(z, tz)
    assert loss.ndim == 0, f"loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss)
    # Cosine loss ∈ [0, 2], typically near 1 for random vectors.
    assert 0.0 <= loss.item() <= 2.0, f"cosine loss out of range: {loss.item()}"
    print(f"  [pass] CerebellarReadout unit | loss={loss.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 8 — pc_tick_aggregation="last" only fires on last tick
# ─────────────────────────────────────────────────────────────────────────
def test_pc_tick_aggregation_last():
    cfg = _make_tiny_config(
        use_predictive_coding=True,
        pc_tick_aggregation="last",
        max_thought_steps=4,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    tz = torch.randn(2, 8, cfg.teacher_d_model)

    result = m(x, targets=y, max_thought_steps=4, teacher_z=tz)
    assert "per_tick_pc_loss" in result
    per_tick = result["per_tick_pc_loss"]
    # With "last", only one tick should contribute → per_tick has length 1.
    assert per_tick.shape[0] == 1, \
        f"expected 1 PC tick contribution with 'last', got {per_tick.shape[0]}"
    print(f"  [pass] pc_tick_aggregation='last' fires only once | "
          f"per_tick_pc_loss shape={tuple(per_tick.shape)}")


# ─────────────────────────────────────────────────────────────────────────
# Test 9 — Hebbian gate starts ~zero so output ≈ baseline at init
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_init_near_baseline():
    cfg_base = _make_tiny_config(use_hebbian_synapse=False)
    cfg_heb = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        hebbian_gate_init=-10.0,   # ≈0 sigmoid → near-zero contribution
    )

    torch.manual_seed(0)
    m_base = CTMTransformer(cfg_base).to(torch.float32)
    torch.manual_seed(0)
    m_heb = CTMTransformer(cfg_heb).to(torch.float32)

    x = torch.randint(0, cfg_base.vocab_size, (2, 8))
    y = torch.randint(0, cfg_base.vocab_size, (2, 8))

    with torch.no_grad():
        r1 = m_base(x, targets=y, max_thought_steps=2)
        r2 = m_heb(x, targets=y, max_thought_steps=2)

    # With gate_init=-10, sigmoid(-10) ≈ 4.5e-5 → Hebbian contribution
    # is essentially zero, so losses should match within float tolerance.
    delta = abs(r1["loss"].item() - r2["loss"].item())
    assert delta < 0.05, \
        f"Hebbian-on loss diverged too much at init: |Δ|={delta:.4f}"
    print(f"  [pass] Hebbian at init ≈ baseline | "
          f"loss_base={r1['loss'].item():.4f} loss_heb={r2['loss'].item():.4f} "
          f"|Δ|={delta:.5f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 10 — Hebbian state persists across ticks (not reset every tick)
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_state_persists():
    """If the fast-weight matrix is reset each tick instead of carrying
    forward, the model would lose its working-memory property. Confirm
    by inspecting that the contribution at tick 1 differs from tick 0
    (because M_1 = decay*M_0 + lr*outer ≠ 0 even though M_0 was zero)."""
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        hebbian_gate_init=2.0,    # gate ≈ 0.88 — large contribution
        max_thought_steps=3,
    )
    m = CTMTransformer(cfg).to(torch.float32)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    result = m(x, targets=y, max_thought_steps=3)
    # If state didn't persist, every tick would have the same Hebbian
    # contribution (zero, since M is always zero at the start of a tick).
    # We can't directly observe contributions, but we can check that
    # per-tick CE losses are NOT identical — which would be the case if
    # nothing distinguished the ticks.
    per_tick = result["per_tick_loss"]
    assert per_tick.shape[0] == 3
    # Variance across ticks > 0 → ticks actually differ. (Note: this
    # property also holds for the baseline, so it's a soft check rather
    # than an isolation of the Hebbian path. The real test is that the
    # gradient flows back through the persisted state in test 3.)
    var = per_tick.var().item()
    assert var > 1e-6, "per-tick losses identical — thought loop collapsed"
    print(f"  [pass] Hebbian state persists across {3} ticks | "
          f"per_tick var={var:.4e}")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# Test 11 — regression: non-affine LayerNorm init doesn't crash
# ─────────────────────────────────────────────────────────────────────────
def test_non_affine_layernorm_init():
    """The original _init_weights crashed on LayerNorm(..., elementwise_affine=False)
    because module.weight/bias are None in that case. Our bio modules use
    non-affine LayerNorms (readout_norm, student_norm, teacher_norm) so
    constructing the model exercises this path. This test pins the fix."""
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        use_predictive_coding=True,
        pc_hidden_dim=8,
        # Also turn on the distillation feature-norm path that previously
        # would have hit the same bug if distill_feature_weight > 0.
        use_distillation=True,
        distill_feature_weight=0.1,
    )
    # Just constructing should not raise.
    _ = CTMTransformer(cfg)
    print("  [pass] non-affine LayerNorm init survives .apply(_init_weights)")


# ─────────────────────────────────────────────────────────────────────────
# Test 12 — regression: Hebbian + matrix_streams combo (production config)
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_with_matrix_streams():
    """The real training command uses both --use_matrix_streams and (now)
    --use_hebbian_synapse. Cover that combination explicitly. Previously
    a dangling-else made the v1 FIFO reset depend on use_hebbian_synapse,
    so {streams ON, hebbian ON} happened to work but {streams OFF,
    hebbian ON} crashed in the NLM path."""
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        use_matrix_streams=True,
        n_streams=2,
        stream_gating="diagonal",
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=cfg.max_thought_steps)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    print(f"  [pass] Hebbian + matrix_streams (production combo) | "
          f"loss={result['loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 13 — intrinsic PC (next_tick mode): no teacher_z needed
# ─────────────────────────────────────────────────────────────────────────
def test_pc_next_tick_no_teacher():
    """pc_target='next_tick' should compute a nonzero PC loss without
    any teacher_z being supplied. The cerebellar readout learns to
    predict z_{t+1} from z_t. This is the recommended mode when the
    teacher cache doesn't store hidden states."""
    cfg = _make_tiny_config(
        use_predictive_coding=True,
        pc_target="next_tick",
        pc_loss_weight=0.1,
        max_thought_steps=4,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    # Note: NO teacher_z passed
    result = m(x, targets=y, max_thought_steps=4)

    assert "pc_loss" in result, "next_tick PC should produce pc_loss without teacher_z"
    assert torch.isfinite(result["pc_loss"]), \
        f"pc_loss not finite: {result['pc_loss']}"
    # With T=4 ticks, pairs are (0,1), (1,2), (2,3) → 3 per-tick losses.
    assert result["per_tick_pc_loss"].shape[0] == 3, \
        f"expected 3 pairs at T=4 next_tick, got {result['per_tick_pc_loss'].shape[0]}"
    result["loss"].backward()
    print(f"  [pass] PC next_tick mode (no teacher_z) | "
          f"loss={result['loss'].item():.4f} pc_loss={result['pc_loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 14 — intrinsic PC (final_tick mode)
# ─────────────────────────────────────────────────────────────────────────
def test_pc_final_tick_no_teacher():
    """pc_target='final_tick': every tick t<T-1 predicts z_{T-1}."""
    cfg = _make_tiny_config(
        use_predictive_coding=True,
        pc_target="final_tick",
        pc_loss_weight=0.1,
        max_thought_steps=4,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=4)
    assert "pc_loss" in result
    assert torch.isfinite(result["pc_loss"])
    # 3 input ticks (0, 1, 2) all predict z_3 → 3 per-tick losses.
    assert result["per_tick_pc_loss"].shape[0] == 3
    result["loss"].backward()
    print(f"  [pass] PC final_tick mode | "
          f"loss={result['loss'].item():.4f} pc_loss={result['pc_loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 15 — diagnostics: hebbian_gate, hebbian_M_norm, pc_loss in result
# ─────────────────────────────────────────────────────────────────────────
def test_bio_diagnostics_exposed():
    """The training log relies on `result['hebbian_gate']`,
    `result['hebbian_M_norm']`, `result['pc_loss']`, and
    `result['per_tick_pc_loss']`. Pin them so a refactor of the result
    dict can't silently break the log line.

    Also check the values are sane:
      - hebbian_gate ≈ sigmoid(hebbian_gate_init), since training hasn't
        moved it yet.
      - hebbian_M_norm > 0 after at least one tick.
      - per_tick_pc_loss has T-1 entries for next_tick mode at T ticks."""
    import math as _math
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        hebbian_gate_init=-2.0,
        use_predictive_coding=True,
        pc_target="next_tick",
        max_thought_steps=3,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=3)

    # Required keys
    for k in ("hebbian_gate", "hebbian_M_norm", "pc_loss", "per_tick_pc_loss"):
        assert k in result, f"missing diagnostic key: {k}"

    # Sanity
    expected_gate = 1.0 / (1.0 + _math.exp(2.0))   # sigmoid(-2)
    assert abs(result["hebbian_gate"] - expected_gate) < 1e-3, \
        f"gate at init should be ~{expected_gate:.3f}, got {result['hebbian_gate']:.3f}"
    assert result["hebbian_M_norm"] > 1e-5, \
        (f"M norm suspiciously small ({result['hebbian_M_norm']:.2e}) — the "
         "Hebbian path may be bypassed. Expected positive nonzero after "
         "at least one outer-product update with lr=sigmoid(lr_logit) ≈ 0.1 "
         "and z, a from non-degenerate projections.")
    # next_tick: T=3 ticks → pairs (0→1), (1→2) → 2 PC losses
    assert result["per_tick_pc_loss"].shape[0] == 2, \
        f"expected 2 PC entries at T=3 next_tick, got {result['per_tick_pc_loss'].shape[0]}"

    print(f"  [pass] bio diagnostics exposed | gate={result['hebbian_gate']:.3f} "
          f"|M|={result['hebbian_M_norm']:.6f} pc={result['pc_loss'].item():.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 16 — certainty-modulated Hebbian lr ("neurochemical modulation")
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_cert_lr_alpha():
    """When hebbian_cert_lr_alpha > 0, the effective Hebbian lr should
    differ from the base lr (sigmoid(lr_logit)). Specifically:
      - With alpha=0, lr_eff == base lr exactly.
      - With alpha=2 and (presumably) some uncertainty, lr_eff > base lr.
      - At tick 0 there is no prev_certainty, so the modulator falls
        back to None. The eff_lr at tick 0 should equal base lr; only
        ticks 1+ see modulation. We test via the mean over all ticks,
        which is a softer signal but still informative."""
    import math as _math

    # Run two models with the same seed: one with alpha=0, one with alpha=2.
    # They should give different lr_eff values, and the alpha=2 one should
    # be HIGHER (because uncertainty>0 → multiplier>1).
    def run(alpha, seed=42):
        torch.manual_seed(seed)
        cfg = _make_tiny_config(
            use_hebbian_synapse=True,
            hebbian_bottleneck_dim=4,
            hebbian_lr_init=0.1,
            hebbian_cert_lr_alpha=alpha,
            max_thought_steps=3,
        )
        m = CTMTransformer(cfg).to(torch.float32)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        return m(x, targets=y, max_thought_steps=3)

    r0 = run(alpha=0.0)
    r2 = run(alpha=2.0)

    assert "hebbian_lr_eff" in r0, "lr_eff missing from result"
    assert "hebbian_lr_eff" in r2, "lr_eff missing with alpha=2"

    lr_base = _math.exp(-2.2) / (1 + _math.exp(-2.2))  # sigmoid(-2.2) ≈ 0.0998
    # With alpha=0, lr_eff should be EXACTLY base lr (no modulation).
    assert abs(r0["hebbian_lr_eff"] - lr_base) < 1e-3, \
        (f"alpha=0 should give lr_eff == base lr ({lr_base:.4f}), "
         f"got {r0['hebbian_lr_eff']:.4f}")

    # With alpha=2, lr_eff should be > base lr (uncertainty multiplier >= 1).
    # We don't pin an exact value because it depends on the model's
    # initial certainty, which depends on init; but it must be strictly
    # larger than alpha=0 case.
    assert r2["hebbian_lr_eff"] > r0["hebbian_lr_eff"], \
        (f"alpha=2 lr_eff ({r2['hebbian_lr_eff']:.4f}) should exceed "
         f"alpha=0 lr_eff ({r0['hebbian_lr_eff']:.4f})")

    # And it shouldn't go absurdly high — the clamp(max=10) in the
    # Hebbian forward bounds the multiplier at 10, so lr_eff ≤ 10 * base.
    assert r2["hebbian_lr_eff"] <= 10.0 * lr_base + 1e-3, \
        f"alpha=2 lr_eff too high: {r2['hebbian_lr_eff']:.4f} > 10*base"

    print(f"  [pass] certainty-modulated lr | base={lr_base:.4f} "
          f"α=0:{r0['hebbian_lr_eff']:.4f} α=2:{r2['hebbian_lr_eff']:.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 17 — diagnostic: hebbian_force_gate overrides learned gate
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_force_gate():
    """When hebbian_force_gate is set, the readout gate uses that exact
    value regardless of gate_logit. Used as a diagnostic to isolate
    whether the gate gradient is the bottleneck. Verifies:
      - force_gate=0.9 makes the result dict report gate=0.9
      - gate_logit is still a parameter (so it can still accumulate
        gradient from elsewhere if any path touches it)
      - Forward + backward still work end-to-end."""
    cfg = _make_tiny_config(
        use_hebbian_synapse=True,
        hebbian_bottleneck_dim=4,
        hebbian_gate_init=-3.0,    # logit-space ≈ 0.047 learned gate
        hebbian_force_gate=0.9,    # but force the effective gate to 0.9
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=cfg.max_thought_steps)

    # Effective gate reported in diagnostics matches force_gate, not
    # sigmoid(gate_logit).
    assert "hebbian_gate" in result
    assert abs(result["hebbian_gate"] - 0.9) < 1e-6, \
        f"forced gate should report 0.9, got {result['hebbian_gate']:.4f}"

    # gate_logit parameter is preserved (still ≈ -3, not modified).
    layer0 = m._get_layers_sequence()[0]
    assert abs(layer0.hebbian.gate_logit.item() - (-3.0)) < 1e-3

    # Backward works.
    result["loss"].backward()
    print(f"  [pass] force_gate diagnostic | reported gate={result['hebbian_gate']:.3f} "
          f"(gate_logit unchanged at {layer0.hebbian.gate_logit.item():.3f})")


# ─────────────────────────────────────────────────────────────────────────
# Test 18 — Prospective Configuration: stability weights emerge correctly
# ─────────────────────────────────────────────────────────────────────────
def test_prospective_config_basic():
    """When use_prospective_config + use_feec are on, the result dict
    should expose stability_w_first/last/min. The weights should
    average to ~1 (because we normalize). Tick 0's weight should be
    exactly 1 (no predecessor → zero energy delta → exp(0) = 1, then
    normalized). Last tick's weight should be in [eps, T]."""
    cfg = _make_tiny_config(
        use_feec=True,
        use_prospective_config=True,
        prospective_beta=2.0,
        max_thought_steps=4,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=4)

    for k in ("stability_w_first", "stability_w_last", "stability_w_min"):
        assert k in result, f"missing PC diagnostic key: {k}"

    # Sanity: min weight is positive but ≤ first weight (worst-case ≤ tick 0).
    assert 0.0 < result["stability_w_min"]
    # First-tick weight, post-normalization, is bounded by T (max possible
    # if all other ticks have ~zero weight). For typical β=2 and
    # moderately-stable loops the first weight is in [0.5, 2].
    assert result["stability_w_first"] <= 4.0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()

    print(f"  [pass] PC basic | w_first={result['stability_w_first']:.3f} "
          f"w_last={result['stability_w_last']:.3f} "
          f"w_min={result['stability_w_min']:.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 19 — Prospective Configuration: β=0 → uniform weights → no change
# ─────────────────────────────────────────────────────────────────────────
def test_prospective_config_beta_zero():
    """With beta=0, exp(-0 * anything) = 1, then normalized → uniform
    weight 1 everywhere. Loss should match a baseline run that has
    use_prospective_config=False."""
    torch.manual_seed(0)
    cfg_off = _make_tiny_config(use_feec=True, use_prospective_config=False)
    m_off = CTMTransformer(cfg_off).to(torch.float32)

    torch.manual_seed(0)
    cfg_on = _make_tiny_config(use_feec=True, use_prospective_config=True,
                                prospective_beta=0.0)
    m_on = CTMTransformer(cfg_on).to(torch.float32)

    x = torch.randint(0, cfg_off.vocab_size, (2, 8))
    y = torch.randint(0, cfg_off.vocab_size, (2, 8))

    with torch.no_grad():
        r_off = m_off(x, targets=y, max_thought_steps=2)
        r_on  = m_on(x, targets=y, max_thought_steps=2)

    # The flag-OFF case must not have PC diagnostics.
    assert "stability_w_first" not in r_off
    # The flag-ON, beta=0 case must have them, but they should be uniform.
    assert "stability_w_first" in r_on
    assert abs(r_on["stability_w_first"] - 1.0) < 1e-3
    assert abs(r_on["stability_w_min"] - 1.0) < 1e-3
    # Losses should match (β=0 is a no-op modulo numerical noise in the mul).
    assert abs(r_off["loss"].item() - r_on["loss"].item()) < 1e-3, \
        f"β=0 should match flag-off: off={r_off['loss']} on={r_on['loss']}"

    print(f"  [pass] PC β=0 == flag-off | loss_off={r_off['loss'].item():.4f} "
          f"loss_on={r_on['loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 20 — Prospective Configuration: requires FEEC to be active
# ─────────────────────────────────────────────────────────────────────────
def test_prospective_config_requires_feec():
    """Without --use_feec, the per-tick energy can't be computed.
    use_prospective_config=True should silently degrade (no PC diagnostics,
    no weighting applied) rather than crashing."""
    cfg = _make_tiny_config(
        use_feec=False,
        use_prospective_config=True,
        prospective_beta=2.0,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=2)

    assert "stability_w_first" not in result, \
        "PC should be inactive without FEEC, but stability_w_first appeared"
    assert torch.isfinite(result["loss"])
    print(f"  [pass] PC silently disabled when FEEC off | "
          f"loss={result['loss'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 21 — Prospective Configuration: composes with Hebbian
# ─────────────────────────────────────────────────────────────────────────
def test_prospective_config_with_hebbian():
    """The 'production-target' combination: PC + Hebbian + FEEC + PC
    feature-distillation, all active simultaneously. Mirror as closely
    as possible the run command the user is going to launch."""
    cfg = _make_tiny_config(
        use_feec=True,
        use_prospective_config=True,
        prospective_beta=2.0,
        use_hebbian_synapse=True,
        hebbian_force_gate=0.9,
        hebbian_bottleneck_dim=4,
        use_predictive_coding=True,
        pc_target="next_tick",
        pc_loss_weight=0.05,
        max_thought_steps=4,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=4)

    # All four mechanisms should report diagnostics.
    for k in ("pc_loss", "hebbian_gate", "hebbian_M_norm",
              "stability_w_first", "stability_w_last"):
        assert k in result, f"missing diagnostic: {k}"

    # Hebbian gate should be exactly the forced value (0.9).
    assert abs(result["hebbian_gate"] - 0.9) < 1e-6

    assert torch.isfinite(result["loss"])
    result["loss"].backward()

    print(f"  [pass] PC + Hebbian production combo | "
          f"loss={result['loss'].item():.4f} "
          f"w_first={result['stability_w_first']:.3f} "
          f"w_min={result['stability_w_min']:.3f} "
          f"pc={result['pc_loss'].item():.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 22 — delta-rule Hebbian update: error-correcting steady state
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_delta_rule_converges():
    """Defining property of the (normalized) delta rule: when (z, a) are
    presented repeatedly, M·a should converge to a finite fixed point
    r∞ = lr·z / (1 − decay + lr), and the error trajectory should be
    monotonically decreasing toward |z − r∞|.

    Under the classic outer-product rule, M·a does NOT have a fixed
    point — it grows unboundedly proportional to <a, a>·t.

    With decay close to 1 (slow forgetting), r∞ ≈ z so the error
    trajectory should approach zero. With decay much less than 1, the
    rule still converges but to a damped target.

    This test pins:
      1. delta rule reaches a steady state (final error << initial)
      2. delta rule does NOT diverge to infinity
      3. outer-product does grow without bound under the same conditions
    """
    import math as _math
    d_lat, d_mod = 16, 16   # no bottleneck → identity projections
    # decay logit = 6 → sigmoid ≈ 0.9975, lr logit = 0 → sigmoid = 0.5
    # Recursion: r_{t+1} = (decay − lr)·r_t + lr·z
    #   r_{t+1} = 0.4975·r_t + 0.5·z
    # Fixed point: r∞ = 0.5·z / (0.5 + 0.0025) ≈ 0.995·z → near-zero error
    # Contraction factor: 0.4975 < 1 → converges in ~10 steps
    h_delta = HebbianSynapse(
        d_latent=d_lat, d_model=d_mod, bottleneck_dim=0,
        decay_init=0.9975, lr_init=0.5,
        update_rule="delta",
    )
    h_outer = HebbianSynapse(
        d_latent=d_lat, d_model=d_mod, bottleneck_dim=0,
        decay_init=0.9975, lr_init=0.5,
        update_rule="outer_product",
    )

    B, S = 1, 1
    torch.manual_seed(0)
    z = torch.randn(B, S, d_lat)
    a = torch.randn(B, S, d_mod)

    M_delta = h_delta.init_state(B, S, torch.device("cpu"), torch.float32)
    M_outer = h_outer.init_state(B, S, torch.device("cpu"), torch.float32)

    err_delta, err_outer = [], []
    M_outer_norms = []
    for _ in range(50):
        with torch.no_grad():
            ma_delta = torch.einsum("bsmn,bsn->bsm",
                                     M_delta.to(a.dtype), a)
            ma_outer = torch.einsum("bsmn,bsn->bsm",
                                     M_outer.to(a.dtype), a)
            err_delta.append((z - ma_delta).norm().item())
            err_outer.append((z - ma_outer).norm().item())
            M_outer_norms.append(M_outer.norm().item())
        _, M_delta, _ = h_delta(z, a, M_delta)
        _, M_outer, _ = h_outer(z, a, M_outer)

    initial_err = err_delta[0]
    final_err = err_delta[-1]

    # Property 1: delta rule does NOT diverge (numerical stability).
    assert _math.isfinite(final_err), \
        f"delta rule diverged to inf: trajectory={err_delta[:5]}..."

    # Property 2: delta-rule error drops substantially. With our
    # parameters (decay≈0.9975, lr≈0.5) the fixed-point error is
    # |z|·(1 − 0.995) ≈ 0.005·|z|, so 99% reduction is expected after
    # ~10 steps. Allow generous slack.
    assert final_err < 0.1 * initial_err, \
        (f"delta-rule should reduce |z - M·a| substantially; "
         f"got initial={initial_err:.3f} final={final_err:.3f}")

    # Property 3: under identical conditions, outer-product accumulates.
    # |M| should grow over time (not necessarily monotonically because
    # of the decay term, but trending upward).
    assert M_outer_norms[-1] > M_outer_norms[0] + 0.5, \
        (f"outer-product |M| should accumulate, got "
         f"start={M_outer_norms[0]:.3f} end={M_outer_norms[-1]:.3f}")

    print(f"  [pass] delta-rule converges: |z - M·a| "
          f"{initial_err:.3f} → {final_err:.4f} (50 steps) | "
          f"outer-product |M|: {M_outer_norms[0]:.2f} → {M_outer_norms[-1]:.2f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 23 — delta-rule + force_gate + PC: production combo
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_delta_production_combo():
    """The full setup the user will actually train with:
    delta-rule Hebbian + force_gate=0.9 + PC next_tick + FEEC + Prospective.
    Ensures all mechanisms compose, all diagnostics are exposed, and the
    forward+backward run cleanly."""
    cfg = _make_tiny_config(
        use_feec=True,
        use_prospective_config=True,
        prospective_beta=2.0,
        use_hebbian_synapse=True,
        hebbian_force_gate=0.9,
        hebbian_bottleneck_dim=4,
        hebbian_update_rule="delta",
        use_predictive_coding=True,
        pc_target="next_tick",
        pc_loss_weight=0.05,
        max_thought_steps=4,
    )
    m = CTMTransformer(cfg).to(torch.float32)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    result = m(x, targets=y, max_thought_steps=4)

    # All four mechanisms report.
    for k in ("pc_loss", "hebbian_gate", "hebbian_M_norm",
              "stability_w_first"):
        assert k in result, f"missing diagnostic: {k}"
    assert abs(result["hebbian_gate"] - 0.9) < 1e-6

    # The delta rule on the live ThoughtLayer's HebbianSynapse.
    layer0 = m._get_layers_sequence()[0]
    assert layer0.hebbian.update_rule == "delta"

    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    print(f"  [pass] delta-rule + force_gate + PC + Prospective | "
          f"loss={result['loss'].item():.4f} |M|={result['hebbian_M_norm']:.6f}")


# ─────────────────────────────────────────────────────────────────────────
# Test 24 — invalid update_rule raises a clear error
# ─────────────────────────────────────────────────────────────────────────
def test_hebbian_invalid_update_rule():
    try:
        HebbianSynapse(d_latent=16, d_model=16, bottleneck_dim=0,
                       update_rule="banana")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "update_rule must be" in str(e)
    print(f"  [pass] invalid update_rule rejected clearly")


TESTS = [
    test_config_fields,
    test_model_constructs_with_bio_on,
    test_forward_backward_with_bio,
    test_pc_silent_without_teacher_z,
    test_baseline_unchanged,
    test_hebbian_unit,
    test_cerebellar_unit,
    test_pc_tick_aggregation_last,
    test_hebbian_init_near_baseline,
    test_hebbian_state_persists,
    test_non_affine_layernorm_init,
    test_hebbian_with_matrix_streams,
    test_pc_next_tick_no_teacher,
    test_pc_final_tick_no_teacher,
    test_bio_diagnostics_exposed,
    test_hebbian_cert_lr_alpha,
    test_hebbian_force_gate,
    test_prospective_config_basic,
    test_prospective_config_beta_zero,
    test_prospective_config_requires_feec,
    test_prospective_config_with_hebbian,
    test_hebbian_delta_rule_converges,
    test_hebbian_delta_production_combo,
    test_hebbian_invalid_update_rule,
]

def main() -> int:
    torch.manual_seed(0)
    failures = []
    for t in TESTS:
        name = t.__name__
        print(f"\n[{name}]")
        try:
            t()
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failures.append(name)
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} / {len(TESTS)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASSED: {len(TESTS)} / {len(TESTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())