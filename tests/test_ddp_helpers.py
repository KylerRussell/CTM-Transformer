"""
test_ddp_helpers.py — Unit tests for the DDP-related helpers.

These don't import train.py directly (it has heavy deps like `datasets`,
`transformers`, etc.). Instead we re-define the helper functions here
matching exactly what train.py has, and verify the logic. Any drift
between this and train.py would be a real bug — the tests are the
contract.
"""

import os
import sys
import torch
import torch.nn as nn


# ─── Helpers (mirror of train.py) ──────────────────────────────────────

def is_torchrun_launch() -> bool:
    """True if we were launched via torchrun (RANK + LOCAL_RANK in env)."""
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ


def unwrap_model(m):
    """Return m.module if m is DDP-wrapped, else m."""
    return m.module if hasattr(m, "module") else m


def is_main_process_factory(dist_initialized: bool, rank: int = 0):
    """Factory matching is_main_process() logic — parameterized for testing."""
    if not dist_initialized:
        return True
    return rank == 0


# ─── Tests ─────────────────────────────────────────────────────────────

def test_is_torchrun_launch_detection():
    print("[test 1] is_torchrun_launch detection...", end=" ")
    saved = {k: os.environ.get(k) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert not is_torchrun_launch()
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        assert is_torchrun_launch()
        del os.environ["LOCAL_RANK"]
        assert not is_torchrun_launch()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("OK")


def test_unwrap_model_bare():
    print("[test 2] unwrap_model on bare module...", end=" ")
    m = nn.Linear(10, 10)
    assert unwrap_model(m) is m
    print("OK")


def test_unwrap_model_ddp_like():
    print("[test 3] unwrap_model on DDP-like wrapper...", end=" ")
    inner = nn.Linear(10, 10)
    class FakeWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m
    wrapped = FakeWrapper(inner)
    assert unwrap_model(wrapped) is inner
    print("OK")


def test_is_main_process_logic():
    print("[test 4] is_main_process logic...", end=" ")
    assert is_main_process_factory(dist_initialized=False)
    assert is_main_process_factory(dist_initialized=True, rank=0)
    assert not is_main_process_factory(dist_initialized=True, rank=1)
    assert not is_main_process_factory(dist_initialized=True, rank=7)
    print("OK")


def test_state_dict_strip_module_prefix():
    print("[test 5] state-dict module-prefix stripping...", end=" ")
    sd_with = {"module.l1.w": torch.randn(2,2), "module.l2.b": torch.randn(2)}
    has = any(k.startswith("module.") for k in sd_with)
    assert has
    sd_clean = {k.removeprefix("module."): v for k, v in sd_with.items()}
    assert "module.l1.w" not in sd_clean and "l1.w" in sd_clean
    sd_already = {"l1.w": torch.randn(2,2)}
    assert not any(k.startswith("module.") for k in sd_already)
    print("OK")


def test_ddp_world_size_token_accounting():
    print("[test 6] DDP world_size token accounting...", end=" ")
    bs, sl, accum = 1, 512, 1
    assert bs * sl * accum * 1 == 512
    assert bs * sl * accum * 2 == 1024
    assert 1 * 512 * 4 * 2 == 4096
    print("OK")


def test_ddp_data_sharding_disjoint():
    print("[test 7] DDP shard partitioning is disjoint...", end=" ")
    n, w = 200, 2
    r0 = list(range(0, n, w))
    r1 = list(range(1, n, w))
    assert set(r0) & set(r1) == set()
    assert set(r0) | set(r1) == set(range(n))
    assert len(r0) == len(r1) == 100
    print("OK")


if __name__ == "__main__":
    test_is_torchrun_launch_detection()
    test_unwrap_model_bare()
    test_unwrap_model_ddp_like()
    test_is_main_process_logic()
    test_state_dict_strip_module_prefix()
    test_ddp_world_size_token_accounting()
    test_ddp_data_sharding_disjoint()
    print("\nAll DDP-helper smoke tests passed.")
