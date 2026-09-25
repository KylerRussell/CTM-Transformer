"""Checks for Sync-RDT: exact RDT parity when off, and correct mechanisms when on."""
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ctm_transformer.baselines import BaselineTransformer
from ctm_transformer.research import load_research_config
from ctm_transformer.sync_rdt import CELLS, SyncRDT, cell_factory

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RECIPE = 'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'
EXPECTED_PARAMETERS = {'rdt': 525984, 'history': 541536, 'sync': 550688, 'sync_rdt': 566240}


def config():
    return load_research_config(RECIPE)[0]


def test_parameter_counts_match_design():
    for cell, count in EXPECTED_PARAMETERS.items():
        assert sum(p.numel() for p in cell_factory(cell)(config()).parameters()) == count, cell


def test_off_cell_has_baseline_initialization_and_outputs():
    c = config()
    torch.manual_seed(7)
    baseline = BaselineTransformer(c)
    torch.manual_seed(7)
    hybrid = SyncRDT(c)
    a, b = baseline.state_dict(), hybrid.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    x = torch.randint(3, c.vocab_size, (2, 30))
    out_a = baseline(x, max_thought_steps=5, return_all_logits=True)
    out_b = hybrid(x, max_thought_steps=5, return_all_logits=True)
    assert torch.equal(out_a['logits'], out_b['logits'])
    assert all(torch.equal(p, q) for p, q in zip(out_a['all_logits'], out_b['all_logits']))


@needs_cuda
def test_off_cell_training_reproduces_baseline_exactly(tmp_path):
    from ctm_transformer.algorithmic import AlgorithmicTokenizer, AnswerDataset
    from ctm_transformer.dense_experiment import train_dense_experiment
    from ctm_transformer.readout_selection import evaluate_readouts
    torch.set_num_threads(4)
    paths = []
    for split in ('train', 'validation'):
        path = tmp_path / f'{split}.jsonl'
        path.write_text('\n'.join(Path(f'research/data/ordered_pointer_v1/pointer/{split}.jsonl').read_text().splitlines()[:4]) + '\n')
        paths.append(path)
    c, identity = load_research_config(RECIPE)
    c = replace(c, batch_size=2, max_steps=3, warmup_steps=0, eval_interval=3, log_interval=3, device=DEVICE)
    tokenizer = AlgorithmicTokenizer()
    runs = []
    for name, factory in (('baseline', None), ('hybrid', cell_factory('rdt'))):
        out = tmp_path / name
        train_dense_experiment(replace(c, checkpoint_dir=str(out)), identity, 23, AnswerDataset(paths[0], tokenizer, c.seq_len),
                               AnswerDataset(paths[1], tokenizer, c.seq_len), evaluate_readouts, ['confidence'], model_factory=factory)
        rows = [{k: r[k] for k in ('loss', 'gradient_norm', 'lr', 'validation_readouts') if k in r}
                for r in map(json.loads, (out / 'metrics.jsonl').read_text().splitlines())]
        runs.append((rows, torch.load(out / 'final.pt', map_location='cpu', weights_only=False)['model_state_dict']))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(runs[0][1][k], runs[1][1][k]) for k in runs[0][1])


@needs_cuda
@pytest.mark.parametrize('cell', ['history', 'sync', 'sync_rdt'])
def test_active_cells_are_causal_train_and_decode_every_step(cell):
    torch.manual_seed(0)
    c = config()
    model = cell_factory(cell)(c).to(DEVICE).eval()
    x = torch.randint(3, c.vocab_size, (2, 40), device=DEVICE)
    y = x.clone()
    y[:, 25:] = torch.randint(3, c.vocab_size, (2, 15), device=DEVICE)
    with torch.no_grad():
        full = model(x, max_thought_steps=6, return_all_logits=True)
        assert torch.allclose(full['logits'][:, :25], model(y, max_thought_steps=6)['logits'][:, :25], atol=1e-5)
        # Per-step logits equal separately truncated depths.
        for depth in (1, 3, 6):
            assert torch.allclose(full['all_logits'][depth - 1], model(x, max_thought_steps=depth)['logits'], atol=1e-5)
    model.train()
    targets = torch.full_like(x, -100)
    targets[:, -3:] = x[:, -3:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, targets=targets, max_thought_steps=6)['loss']
    loss.backward()
    history, sync = CELLS[cell]
    if history:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.nlm.parameters())
    if sync:
        # W_sync starts at zero, so its gradient is nonzero once the state history is nonzero.
        assert all(q.weight.grad is not None and q.weight.grad.abs().sum() > 0 for q in model.sync_query)
    clone = cell_factory(cell)(c).to(DEVICE)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert all(torch.equal(clone.state_dict()[k], model.state_dict()[k]) for k in model.state_dict())


def test_active_cells_start_from_the_baseline_scaffold():
    c = config()
    torch.manual_seed(3)
    base = BaselineTransformer(c).state_dict()
    torch.manual_seed(3)
    full = SyncRDT(c, history=True, sync=True).state_dict()
    assert all(torch.equal(base[k], full[k]) for k in base)
    assert all(torch.count_nonzero(full[k]) == 0 for k in full if k.startswith('sync_query'))


def test_lowgate_history_control_differs_only_in_gate_initialization():
    c = config()
    torch.manual_seed(5)
    standard = cell_factory('history')(c).state_dict()
    torch.manual_seed(5)
    lowgate = cell_factory('history_lowgate')(c).state_dict()
    assert standard.keys() == lowgate.keys()
    assert all(torch.equal(standard[k], lowgate[k]) for k in standard if k != 'nlm.gate')
    assert torch.all(standard['nlm.gate'] == 0) and torch.all(lowgate['nlm.gate'] == -4.0)
    assert abs(torch.sigmoid(lowgate['nlm.gate'][0]).item() - 0.01799) < 1e-4
