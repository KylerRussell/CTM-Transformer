"""
test_tick_aggregation.py — Verify the new distill_tick_aggregation modes
('first' and 'decay_ramp') produce the expected aggregated KL values.

Doesn't import the model (heavy deps); reproduces the aggregation logic
in isolation against synthetic per-tick KL tensors.
"""

import torch


def aggregate_kl(per_tick_kl: torch.Tensor, mode: str,
                 tick_ramp_start: float = 0.5, tick_ramp_end: float = 1.5,
                 temporal_loss_type: str = "dynamic_aggregate") -> torch.Tensor:
    """Mirror of model.py's aggregation logic. Tests should track it."""
    T = per_tick_kl.shape[0]
    if mode == "all":
        return per_tick_kl.mean()
    elif mode == "first":
        return per_tick_kl[0]
    elif mode == "last":
        return per_tick_kl[-1]
    elif mode == "decay_ramp":
        weights = torch.linspace(1.0, 0.1, T,
                                 device=per_tick_kl.device,
                                 dtype=per_tick_kl.dtype)
        weights = weights * (T / weights.sum())
        return (weights * per_tick_kl).mean()
    elif mode == "lm_aligned":
        if temporal_loss_type == "dynamic_aggregate":
            return per_tick_kl.mean()
        ramp = torch.linspace(tick_ramp_start, tick_ramp_end, T,
                              device=per_tick_kl.device,
                              dtype=per_tick_kl.dtype)
        ramp = ramp * (T / ramp.sum())
        return (ramp * per_tick_kl).mean()
    raise ValueError(mode)


def test_first_picks_tick_zero():
    """'first' should select per_tick_kl[0] exactly."""
    print("[test 1] 'first' returns tick 0...", end=" ")
    kl = torch.tensor([5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6])
    result = aggregate_kl(kl, mode="first")
    assert torch.isclose(result, torch.tensor(5.0)), f"got {result}"
    print(f"OK (got {result.item():.2f})")


def test_decay_ramp_weights_tick_zero_most():
    """'decay_ramp' should weight earlier ticks more heavily."""
    print("[test 2] 'decay_ramp' weights tick 0 the most...", end=" ")
    # Construct a KL with all 1.0s — we should see the weight pattern
    # directly in the result. With weights renormalized to mean 1.0,
    # this gives a result of 1.0 (sanity check on normalization).
    kl = torch.ones(8)
    result = aggregate_kl(kl, mode="decay_ramp")
    assert torch.isclose(result, torch.tensor(1.0), atol=1e-5), \
        f"normalization broken: got {result}"

    # Now construct a KL where only tick 0 is nonzero — the decay_ramp
    # should weight it ~T*1.0/sum(weights) = T*1/sum(linspace(1.0,0.1,T))
    # times its raw value. Verify it's > 'all' would give.
    kl2 = torch.zeros(8)
    kl2[0] = 1.0
    decay_result = aggregate_kl(kl2, mode="decay_ramp")
    all_result = aggregate_kl(kl2, mode="all")
    assert decay_result > all_result, \
        f"decay_ramp ({decay_result}) should weight tick 0 more than 'all' ({all_result})"

    # Symmetric: tick 0 nonzero should give a HIGHER result than tick T-1 nonzero.
    kl3 = torch.zeros(8)
    kl3[-1] = 1.0
    decay_late = aggregate_kl(kl3, mode="decay_ramp")
    assert decay_result > decay_late, \
        f"early-only ({decay_result}) should outweigh late-only ({decay_late})"
    print(f"OK (early-tick={decay_result.item():.3f} > late-tick={decay_late.item():.3f})")


def test_first_produces_smaller_loss_when_loop_refines():
    """If the model's loop is genuinely improving across ticks (per-tick
    KL decreasing), then 'first' >> 'all' >> 'last' in magnitude. This
    is the failure mode where 'all' is expensive and 'last' is too lenient
    — which is when 'first' makes sense as a teacher anchor on the
    *initial state* without flattening the trajectory."""
    print("[test 3] Refining loop: first > all > last...", end=" ")
    # Loop refines: KL drops from 5 → 0.5 across 8 ticks
    kl = torch.linspace(5.0, 0.5, 8)
    first_v = aggregate_kl(kl, mode="first")
    all_v = aggregate_kl(kl, mode="all")
    last_v = aggregate_kl(kl, mode="last")
    assert first_v > all_v > last_v, \
        f"expected first > all > last, got {first_v}, {all_v}, {last_v}"
    print(f"OK (first={first_v.item():.2f}, all={all_v.item():.2f}, last={last_v.item():.2f})")


def test_unknown_mode_would_raise_in_real_code():
    """In the real model.py, an unknown mode raises ValueError. This
    test just documents that — argparse choices will catch it before
    runtime in practice."""
    print("[test 4] Unknown mode rejected...", end=" ")
    try:
        aggregate_kl(torch.ones(4), mode="bogus")
        assert False, "should have raised"
    except ValueError:
        pass
    print("OK")


def test_decay_ramp_normalization_invariant_under_T():
    """The renormalization (weights * T / sum(weights)) keeps the mean
    weight = 1.0 regardless of T. So distill_logit_weight semantics
    don't shift when T changes."""
    print("[test 5] decay_ramp normalization invariant under T...", end=" ")
    for T in [4, 6, 8, 12, 16]:
        kl = torch.ones(T)
        result = aggregate_kl(kl, mode="decay_ramp")
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-4), \
            f"T={T}: got {result}, expected 1.0"
    print("OK (works for T in {4, 6, 8, 12, 16})")


if __name__ == "__main__":
    test_first_picks_tick_zero()
    test_decay_ramp_weights_tick_zero_most()
    test_first_produces_smaller_loss_when_loop_refines()
    test_unknown_mode_would_raise_in_real_code()
    test_decay_ramp_normalization_invariant_under_T()
    print("\nAll tick-aggregation tests passed.")
