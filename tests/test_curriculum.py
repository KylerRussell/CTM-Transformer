"""Checks for length curricula and runner v5."""
import json
import os
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ctm_transformer.curriculum import length_curriculum, ramp_limit
from ctm_transformer.depth_sampling import lognormal_poisson
from tests.test_depth_sampling import configs, data

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def pool(steps, batch, maximum=16):
    return SimpleNamespace(records=[{'steps': 1 + i % maximum} for i in range(steps * batch)])


def test_ramp_rises_from_two_to_the_maximum_over_the_first_half():
    limits = [ramp_limit(u, 1000, 2, 0.5, 16) for u in range(1000)]
    assert limits[0] == 2 and limits[499] == 16 and set(limits[500:]) == {16}
    assert all(a <= b for a, b in zip(limits, limits[1:])) and set(limits) == set(range(2, 17))


def test_curriculum_is_a_reproducible_permutation_that_respects_the_ramp():
    steps, batch = 400, 32
    data_ = pool(steps, batch)
    sampler = length_curriculum('linear_half', steps, batch)
    a = sampler(torch.Generator().manual_seed(4), data_)
    b = sampler(torch.Generator().manual_seed(4), data_)
    assert torch.equal(a, b) and sorted(a.tolist()) == list(range(steps * batch))
    lengths = [data_.records[i]['steps'] for i in a.tolist()]
    for u in range(steps // 2):
        assert max(lengths[u * batch:(u + 1) * batch]) <= ramp_limit(u, steps, 2, 0.5, 16)
    early, late = Counter(lengths[:batch * 20]), Counter(lengths[-batch * 20:])
    assert max(early) <= 3 and max(late) == 16
    with pytest.raises(ValueError):
        sampler(torch.Generator(), pool(steps, batch - 1))
    with pytest.raises(ValueError):
        length_curriculum('unknown', steps, batch)


@needs_cuda
@pytest.mark.parametrize('kind', ['rdt', 'ctm_lm'])
def test_runner_v5_without_order_sampler_reproduces_v4_exactly(kind, tmp_path):
    from ctm_transformer.dense_experiment_v4 import train_dense_experiment as v4
    from ctm_transformer.dense_experiment_v5 import train_dense_experiment as v5
    from ctm_transformer.dense_pointer import evaluate_dense
    torch.set_num_threads(4)
    train, valid = data(tmp_path)
    c, identity, factory, policy = configs(kind)
    c = replace(c, batch_size=8, max_steps=10, warmup_steps=1, eval_interval=5, log_interval=5, max_thought_steps=4, device=DEVICE)
    runs = []
    for name, runner in (('v4', v4), ('v5', v5)):
        out = tmp_path / name
        runner(replace(c, checkpoint_dir=str(out)), identity, 11, train, valid, evaluate_dense, [policy], model_factory=factory,
               depth_sampler=lognormal_poisson(4, 0.5, 8))
        rows = [{k: r[k] for k in ('loss', 'gradient_norm', 'lr', 'thought_steps', 'validation_readouts') if k in r}
                for r in map(json.loads, (out / 'metrics.jsonl').read_text().splitlines())]
        runs.append((rows, torch.load(out / 'final.pt', map_location='cpu', weights_only=False)['model_state_dict']))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(runs[0][1][k], runs[1][1][k]) for k in runs[0][1])


@needs_cuda
def test_runner_v5_presents_the_curriculum_order(tmp_path):
    from ctm_transformer.dense_experiment_v5 import train_dense_experiment as v5
    from ctm_transformer.dense_pointer import evaluate_dense
    train, valid = data(tmp_path)  # 64 words of lengths 1..8
    c, identity, factory, policy = configs('rdt')
    steps, batch = 8, 8
    c = replace(c, batch_size=batch, max_steps=steps, warmup_steps=1, eval_interval=4, log_interval=4, max_thought_steps=2,
                device=DEVICE, checkpoint_dir=str(tmp_path / 'run'))
    sampler = length_curriculum('linear_half', steps, batch)
    v5(c, identity, 5, train, valid, evaluate_dense, [policy], model_factory=factory, order_sampler=sampler)
    order = sampler(torch.Generator().manual_seed(5 + 1), train).tolist()
    rows = [json.loads(x) for x in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    expected, seen = [], 0
    for u in range(steps):
        index = torch.tensor(order[u * batch:(u + 1) * batch])
        seen += int(train.lengths[index].sum())
        expected.append(seen)
    assert [r['tokens_seen'] for r in rows] == expected
    run = json.loads((tmp_path / 'run/research_run.json').read_text())
    assert run['runner'] == 'shared_research_v5' and run['presentation_order'] == sampler.description
