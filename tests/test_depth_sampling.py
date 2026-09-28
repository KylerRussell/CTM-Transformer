"""Checks for training-depth samplers and runner v4."""
import json
import os
from dataclasses import replace

import pytest
import torch

from ctm_transformer.depth_sampling import fixed, lognormal_poisson

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_lognormal_poisson_is_reproducible_bounded_and_centered():
    sampler = lognormal_poisson(16, 0.5, 64)
    draw = lambda seed: [sampler(g) for g in [torch.Generator().manual_seed(seed)] for _ in range(4000)]
    a, b = draw(7), draw(7)
    assert a == b and min(a) >= 1 and max(a) <= 64
    assert 15.0 < sum(a) / len(a) < 18.5  # about mean + 1 before capping
    assert len(set(a)) > 20  # genuinely variable
    assert fixed(16)(torch.Generator()) == 16 and 'lognormal_poisson(mean=16' in sampler.description


def data(tmp_path):
    from ctm_transformer.algorithmic import AlgorithmicTokenizer
    from ctm_transformer.group_word import Group, GroupWordDataset, write_group_split
    g = Group('S3')
    ev = write_group_split(tmp_path / 'v.jsonl', g, [4, 8], 8, 1)
    write_group_split(tmp_path / 't.jsonl', g, list(range(1, 9)), 64, 2, exclude=ev)
    tok = AlgorithmicTokenizer()
    return GroupWordDataset(tmp_path / 't.jsonl', tok, 128), GroupWordDataset(tmp_path / 'v.jsonl', tok, 128)


def configs(kind):
    from ctm_transformer.ctm_lm import ctm_lm_factory, lm_config
    from ctm_transformer.research import load_research_config
    from ctm_transformer.sync_rdt import cell_factory
    if kind == 'rdt':
        c, identity = load_research_config('research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json')
        return c, identity, cell_factory('rdt'), 'confidence'
    c, identity = load_research_config('research/configs/presentation_control_v1/ctm_shuffled_seed23.json')
    return lm_config(c), identity, ctm_lm_factory(), 'confidence'


@needs_cuda
@pytest.mark.parametrize('kind', ['rdt', 'ctm_lm'])
def test_runner_v4_without_sampler_reproduces_v3_exactly(kind, tmp_path):
    from ctm_transformer.dense_experiment import train_dense_experiment as v3
    from ctm_transformer.dense_experiment_v4 import train_dense_experiment as v4
    from ctm_transformer.dense_pointer import evaluate_dense
    torch.set_num_threads(4)
    train, valid = data(tmp_path)
    c, identity, factory, policy = configs(kind)
    c = replace(c, batch_size=8, max_steps=6, warmup_steps=1, eval_interval=3, log_interval=3, max_thought_steps=4, device=DEVICE)
    runs = []
    for name, runner in (('v3', v3), ('v4', v4)):
        out = tmp_path / name
        runner(replace(c, checkpoint_dir=str(out)), identity, 11, train, valid, evaluate_dense, [policy], model_factory=factory)
        rows = [{k: r[k] for k in ('loss', 'gradient_norm', 'lr', 'thought_steps', 'validation_readouts') if k in r}
                for r in map(json.loads, (out / 'metrics.jsonl').read_text().splitlines())]
        runs.append((rows, torch.load(out / 'final.pt', map_location='cpu', weights_only=False)['model_state_dict']))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(runs[0][1][k], runs[1][1][k]) for k in runs[0][1])


@needs_cuda
@pytest.mark.parametrize('kind', ['rdt', 'ctm_lm'])
def test_runner_v4_samples_and_records_depth(kind, tmp_path):
    from ctm_transformer.dense_experiment_v4 import train_dense_experiment as v4
    from ctm_transformer.dense_pointer import evaluate_dense
    train, valid = data(tmp_path)
    c, identity, factory, policy = configs(kind)
    c = replace(c, batch_size=8, max_steps=12, warmup_steps=1, eval_interval=6, log_interval=6, max_thought_steps=4,
                device=DEVICE, checkpoint_dir=str(tmp_path / 'run'))
    sampler = lognormal_poisson(4, 0.5, 12)
    summary = v4(c, identity, 5, train, valid, evaluate_dense, [policy], model_factory=factory, depth_sampler=sampler)
    rows = [json.loads(x) for x in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    depths = [r['thought_steps'] for r in rows]
    expected = [sampler(g) for g in [torch.Generator().manual_seed(5 + 2)] for _ in range(12)]
    assert depths == expected and len(set(depths)) > 1 and summary['complete']
    run = json.loads((tmp_path / 'run/research_run.json').read_text())
    assert run['runner'] == 'shared_research_v4' and run['depth_sampling'] == sampler.description
    assert sum(summary['depth_counts'].values()) == 12
