"""Checks for CTM-LM against the published CTM definitions (research/CTM_LM_DESIGN.md)."""
import json
import math
import os
import random
from dataclasses import replace

import pytest
import torch

from ctm_transformer.ctm_lm import CTMLM, Synchronization, ctm_loss, ctm_lm_factory, lm_config, rope
from ctm_transformer.research import load_research_config

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RECIPE = 'research/configs/presentation_control_v1/ctm_shuffled_seed23.json'


def config():
    return lm_config(load_research_config(RECIPE)[0])


def test_recursive_synchronization_equals_the_direct_decayed_sum():
    sync = Synchronization(12, 20, torch.Generator().manual_seed(0))
    with torch.no_grad():
        sync.decay.uniform_(0, 2)
    history = [torch.randn(3, 12) for _ in range(6)]
    alpha, beta = sync.start(history[0])
    for z in history[1:]:
        alpha, beta = sync.step(alpha, beta, z)
    t = len(history)
    weights = torch.exp(-sync.decay[None] * torch.arange(t - 1, -1, -1, dtype=torch.float32)[:, None])  # [t, pairs], newest weight 1
    products = torch.stack([sync.products(z) for z in history], dim=1)  # [3, t, pairs]
    direct = (products * weights[None]).sum(1) / torch.sqrt(weights.sum(0))[None]
    assert torch.allclose(Synchronization.read(alpha, beta), direct, atol=1e-5)


def test_loss_is_the_mean_of_min_loss_and_max_certainty_ticks():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 4, 7)
    targets = torch.randint(0, 7, (2, 5))
    targets[0, :2] = -100
    out = ctm_loss(logits, targets)
    expected = []
    for b in range(2):
        for s in range(5):
            if targets[b, s] == -100:
                continue
            logp = logits[b, s].log_softmax(-1)
            ce = -logp[:, targets[b, s]]
            certainty = 1 + (logp.exp() * logp).sum(-1) / math.log(7)
            expected.append((ce[ce.argmin()] + ce[certainty.argmax()]) / 2)
    assert torch.allclose(out['loss'], torch.stack(expected).mean(), atol=1e-6)


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(2)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    score = lambda i, j: (rope(q, torch.tensor([i])) * rope(k, torch.tensor([j]))).sum().item()
    assert abs(score(5, 2) - score(13, 10)) < 1e-4 and abs(score(3, 3) - score(40, 40)) < 1e-4
    assert abs(score(5, 2) - score(5, 4)) > 1e-3


def test_parameters_and_no_tick_indexed_tables():
    model = CTMLM(config())
    names = {n for n, _ in model.named_parameters()}
    assert {'z_init', 'history_init', 'sync_action.decay', 'sync_out.decay', 'lm_head.weight', 'query.weight'} <= names
    assert not any('tick' in n or 'film' in n for n in names)
    assert model.lm_head.in_features == config().sync_sparse_pairs  # the readout is the output synchronization
    assert not torch.equal(model.sync_action.left, model.sync_out.left)  # separately sampled pair sets


@needs_cuda
def test_causal_trains_varies_ticks_and_round_trips():
    torch.manual_seed(0)
    c = config()
    model = CTMLM(c).to(DEVICE)
    x = torch.randint(3, c.vocab_size, (2, 30), device=DEVICE)
    y = x.clone()
    y[:, 20:] = torch.randint(3, c.vocab_size, (2, 10), device=DEVICE)
    model.eval()
    with torch.no_grad():
        a = model(x, max_thought_steps=5)
        assert torch.allclose(a['logits'][:, :20], model(y, max_thought_steps=5)['logits'][:, :20], atol=1e-5)
        assert len(model(x, max_thought_steps=32)['all_logits']) == 32 and len(a['all_logits']) == 5
    model.train()
    targets = torch.full_like(x, -100)
    targets[:, -4:] = x[:, -4:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(x, targets=targets, max_thought_steps=6)
    out['loss'].backward()
    assert out['per_tick_loss'].shape == (6,)
    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, missing
    clone = CTMLM(c).to(DEVICE)
    clone.load_state_dict(model.state_dict(), strict=True)


@needs_cuda
def test_runner_v3_trains_ctm_lm_on_probe_data(tmp_path):
    from ctm_transformer.algorithmic import AlgorithmicTokenizer
    from ctm_transformer.dense_experiment import train_dense_experiment
    from ctm_transformer.dense_pointer import evaluate_dense
    from ctm_transformer.pointer_probes import ProbeDataset, write_probe_split
    val = write_probe_split(tmp_path / 'v.jsonl', 'mqar', [1], 8, 1, queries=12)
    write_probe_split(tmp_path / 't.jsonl', 'mqar', [1], 160, 2, exclude=val, queries=12)
    tok = AlgorithmicTokenizer()
    train, valid = ProbeDataset(tmp_path / 't.jsonl', tok, 128), ProbeDataset(tmp_path / 'v.jsonl', tok, 128)
    c, identity = load_research_config(RECIPE)
    c = replace(lm_config(c), batch_size=8, max_steps=20, warmup_steps=2, eval_interval=10, log_interval=10,
                max_thought_steps=4, device=DEVICE, checkpoint_dir=str(tmp_path / 'run'))
    summary = train_dense_experiment(c, identity, 3, train, valid, evaluate_dense, ['confidence'], model_factory=ctm_lm_factory())
    rows = [json.loads(x) for x in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    run = json.loads((tmp_path / 'run/research_run.json').read_text())
    assert summary['complete'] and run['model_factory'] == 'ctm_transformer.ctm_lm.ctm_lm_factory[tiny]'
    assert run['training_objective']['type'] == 'dynamic_aggregate' and 'validation' in rows[-1]
    assert all(math.isfinite(r['loss']) for r in rows)
