"""Checks for the pretraining components (ctm_transformer/pretrain.py)."""
import numpy as np
import pytest
import torch

from ctm_transformer.depth_sampling import lognormal_poisson
from ctm_transformer.pretrain import OffloadedAdamW, TokenWindows, lr_at, micro_indices, step_depth

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def shards(tmp_path, lengths=(1000, 777, 2048)):
    paths, start = [], 0
    for i, n in enumerate(lengths):
        p = tmp_path / f'train_{i:03d}.bin'
        np.arange(start, start + n, dtype=np.uint16).tofile(p)
        paths.append(p)
        start += n
    return paths


def test_windows_are_shifted_pairs_from_one_seeded_permutation(tmp_path):
    w = TokenWindows(shards(tmp_path), 64, seed=3)
    assert len(w) == sum((n - 1) // 64 for n in (1000, 777, 2048))
    assert sorted(w.order.tolist()) == list(range(len(w)))
    assert np.array_equal(w.order, TokenWindows(shards(tmp_path), 64, seed=3).order)
    for i in range(len(w)):
        x, y = w.window(i)
        assert x.shape == y.shape == (64,) and torch.equal(x[1:], y[:-1]) and int(y[0]) == int(x[0]) + 1  # contiguous, within one shard
    x, y = w.batch([0, 5])
    assert x.shape == (2, 64)


def test_global_batches_partition_the_stream_without_overlap():
    world, micro_batch, accumulation = 2, 3, 4
    seen = []
    for step in range(3):
        for micro in range(accumulation):
            for rank in range(world):
                seen += micro_indices(step, micro, rank, world, micro_batch, accumulation)
    assert seen == sorted(seen) and seen == list(range(3 * world * micro_batch * accumulation))


def test_step_depth_is_a_function_of_seed_and_step():
    sampler = lognormal_poisson(15, 0.5, 48)
    a = [step_depth(sampler, 7, s) for s in range(200)]
    assert a == [step_depth(sampler, 7, s) for s in range(200)] and len(set(a)) > 10
    assert a != [step_depth(sampler, 8, s) for s in range(200)] and step_depth(None, 7, 3) is None


def test_lr_schedule_warms_up_and_decays_to_the_floor():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1) and lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(10, 1.0, 10, 100) == pytest.approx(1.0) and lr_at(100, 1.0, 10, 100) == pytest.approx(0.1)


@needs_cuda
def test_offloaded_adamw_reproduces_torch_adamw():
    torch.manual_seed(0)
    a = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).cuda()
    b = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).cuda()
    b.load_state_dict(a.state_dict())
    reference = torch.optim.AdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    offloaded = OffloadedAdamW(b.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for step in range(5):
        x = torch.randn(4, 16, device='cuda')
        for model, opt in ((a, reference), (b, offloaded)):
            opt.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)
    state = offloaded.state_dict()
    c = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).cuda()
    restored = OffloadedAdamW(c.parameters(), lr=1e-2)
    restored.load_state_dict(state)
    assert all(torch.equal(p, q) for p, q in zip(b.parameters(), c.parameters())) and restored.step_count == 5
