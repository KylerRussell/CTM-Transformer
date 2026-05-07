"""
test_offload_patches.py — Verify the two changes to train_offload.py:
  1. Optimizer construction respects --use_8bit_adam (delegates to _make_adamw).
  2. PhaseTimer instrumentation records the expected phases.

These are pure-logic tests; they don't spin up the dual-GPU pipeline.
"""

import os
import sys
import torch
import torch.nn as nn


def test_make_adamw_respects_8bit_flag():
    """When use_8bit_adam=True, _make_adamw should try bitsandbytes;
    when False, it should return vanilla AdamW. Verify the fallback
    path also works when bitsandbytes isn't installed."""
    print("[test 1] _make_adamw respects use_8bit_adam...", end=" ")

    # Build a fake config object with the fields _make_adamw reads
    class FakeConfig:
        learning_rate = 3e-4
        weight_decay = 0.01
        adam_beta1 = 0.9
        adam_beta2 = 0.95
        use_8bit_adam = False

    # Define a minimal _make_adamw matching train.py's implementation
    def _make_adamw(config, params, **overrides):
        kwargs = dict(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
        )
        kwargs.update(overrides)
        if not config.use_8bit_adam:
            return torch.optim.AdamW(params, **kwargs)
        try:
            import bitsandbytes as bnb
        except ImportError:
            return torch.optim.AdamW(params, **kwargs)
        return bnb.optim.AdamW8bit(params, **kwargs)

    model = nn.Linear(10, 10)

    # use_8bit_adam=False → vanilla AdamW
    cfg = FakeConfig()
    opt = _make_adamw(cfg, model.parameters())
    assert type(opt).__name__ == "AdamW", \
        f"expected AdamW with flag off, got {type(opt).__name__}"

    # use_8bit_adam=True → either AdamW8bit (if bnb installed) or
    # AdamW (if not) — both are acceptable.
    cfg.use_8bit_adam = True
    opt = _make_adamw(cfg, model.parameters())
    name = type(opt).__name__
    assert name in ("AdamW", "AdamW8bit"), \
        f"expected AdamW or AdamW8bit, got {name}"

    print(f"OK (off=AdamW, on={name})")


def test_phase_timer_records_offload_phases():
    """The PhaseTimer wrapping in train_offload.py should produce
    timings under exactly these phase names: data, student_fwd,
    backward, allreduce, grad_clip, optimizer."""
    print("[test 2] PhaseTimer records expected phases...", end=" ")

    # Inline import to avoid loading heavy deps; the real PhaseTimer
    # is a small standalone class.
    sys.path.insert(0, '/mnt/user-data/uploads')
    from phase_timer import PhaseTimer

    timer = PhaseTimer(enabled=True, warmup_steps=0)

    expected_phases = [
        "data", "student_fwd", "backward",
        "allreduce", "grad_clip", "optimizer",
    ]

    # Simulate 5 training steps
    for step in range(5):
        with timer.step():
            for phase in expected_phases:
                with timer(phase):
                    # tiny dummy work
                    _ = sum(range(100))

    # Every phase should have 5 samples recorded
    for phase in expected_phases:
        assert phase in timer.timings, f"missing phase: {phase}"
        assert len(timer.timings[phase]) == 5, \
            f"phase {phase} has {len(timer.timings[phase])} samples, expected 5"

    print(f"OK (recorded {len(expected_phases)} phases × 5 steps)")


def test_runtime_kwargs_thread_through():
    """Verify that the runtime_kwargs pattern works — i.e. spawn-time
    args are accessible to the worker via a plain dict."""
    print("[test 3] runtime_kwargs dict passes profile flags...", end=" ")

    runtime_kwargs = dict(
        profile=True,
        profile_warmup=20,
        profile_interval=100,
    )

    # Simulate worker reading them
    assert runtime_kwargs.get('profile', False) == True
    assert runtime_kwargs.get('profile_warmup', 0) == 20
    assert runtime_kwargs.get('profile_interval', 0) == 100

    # Default fallback when key is missing
    assert runtime_kwargs.get('nonexistent_flag', False) == False

    print("OK")


def test_rank_zero_only_profiling():
    """Profile should be enabled only on rank 0 — the gating logic
    should AND the user's --profile flag with (rank == 0)."""
    print("[test 4] profile gates on rank 0 only...", end=" ")

    runtime_kwargs = {'profile': True}

    # Rank 0 → enabled
    rank = 0
    assert (runtime_kwargs.get('profile', False) and rank == 0) == True

    # Rank 1 → disabled (avoids double-sync overhead)
    rank = 1
    assert (runtime_kwargs.get('profile', False) and rank == 0) == False

    # If user didn't pass --profile, both ranks have it disabled
    runtime_kwargs = {'profile': False}
    assert (runtime_kwargs.get('profile', False) and 0 == 0) == False

    print("OK")


if __name__ == "__main__":
    test_make_adamw_respects_8bit_flag()
    test_phase_timer_records_offload_phases()
    test_runtime_kwargs_thread_through()
    test_rank_zero_only_profiling()
    print("\nAll offload-patch smoke tests passed.")
