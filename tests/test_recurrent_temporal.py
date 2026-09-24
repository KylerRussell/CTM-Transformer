"""Objective parity, gradient routing, unchanged inference, and runner regression."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import types
import zipfile
import pytest
import torch
from torch.nn import functional as F
from ctm_transformer.baselines import BaselineTransformer
from ctm_transformer.experiment import train_experiment
from ctm_transformer.recurrent_temporal import RecurrentTemporalTransformer, temporal_loss
from ctm_transformer.research import load_research_config, build_model, parameter_counts
from scripts.eval_harness import _load_checkpoint

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')


def small(**changes):
    c, identity = load_research_config('research/configs/fresh_confirmation_v1/recurrent_depth_seed23.json')
    c = replace(c, d_model=32, n_heads=4, ffn_hidden_dim=64, max_thought_steps=4,
                train_depth_min=4, train_depth_max=4, max_seq_len=64, seq_len=64,
                batch_size=2, device=DEVICE, **changes)
    return c, identity


@pytest.mark.parametrize('objective', ['uniform', 'dynamic_aggregate'])
@pytest.mark.parametrize('bf16', [False, True])
def test_loss_and_parameter_gradient_match_actual_ctm(objective, bf16):
    torch.set_num_threads(4); torch.manual_seed(87)
    name = 'dynamic' if objective == 'dynamic_aggregate' else 'uniform'
    c, _ = load_research_config(f'research/configs/ctm_{name}_t16_v1.json')
    c = replace(c, d_model=32, d_latent=32, n_heads=4, n_layers=1, nlm_groups=32,
                nlm_hidden_dim=8, sync_sparse_pairs=16)
    model = build_model(c).to(DEVICE).eval()
    x = torch.randint(3, 71, (2, 7), device=DEVICE)
    y = torch.full_like(x, -100); y[:, -2:] = x[:, -2:]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        native = model(x, targets=y)
        matched = temporal_loss(native['all_logits'], y, objective)
    torch.testing.assert_close(matched['loss'], native['loss'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(matched['per_tick_loss'], native['per_tick_loss'])
    if objective == 'dynamic_aggregate':
        torch.testing.assert_close(matched['certainties'], native['certainties'], atol=0, rtol=0)
    parameter = model.layers[0].nlm.w1
    actual = torch.autograd.grad(native['loss'], parameter, retain_graph=True)[0]
    expected = torch.autograd.grad(matched['loss'], parameter)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize('objective', ['uniform', 'dynamic_aggregate'])
def test_recurrent_ticks_and_gradients_match_independent_truncations(objective):
    torch.set_num_threads(4); torch.manual_seed(88)
    config, _ = small(dtype='float32')
    model = RecurrentTemporalTransformer(config, objective).to(DEVICE).eval()
    base = BaselineTransformer(config).to(DEVICE).eval(); base.load_state_dict(model.state_dict())
    x = torch.randint(3, 71, (2, 6), device=DEVICE)
    y = torch.full_like(x, -100); y[:, -2:] = x[:, -2:]
    result = model(x, y)
    independent = [base(x, max_thought_steps=t)['logits'] for t in range(1, 5)]
    for a, b in zip(result['all_logits'], independent):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    loss = temporal_loss(independent, y, objective)['loss']
    result['loss'].backward(); loss.backward()
    for (name, a), (other, b) in zip(model.named_parameters(), base.named_parameters()):
        assert name == other and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)
    assert result['block_applications'] == 13


def test_dynamic_routes_gradient_to_best_and_most_certain_earliest_ticks():
    # Tick 1 is correct; tick 2 is confidently wrong; tick 3 repeats tick 1.
    ticks = [torch.tensor([[[2., 0., 0.], [100., -100., 0.]]], device=DEVICE, requires_grad=True),
             torch.tensor([[[0., 8., 0.], [-100., 100., 0.]]], device=DEVICE, requires_grad=True),
             torch.tensor([[[2., 0., 0.], [0., 0., 0.]]], device=DEVICE, requires_grad=True)]
    y = torch.tensor([[0, -100]], device=DEVICE)
    result = temporal_loss(ticks, y, 'dynamic_aggregate')
    expected = (F.cross_entropy(ticks[0][:, 0], y[:, 0]) + F.cross_entropy(ticks[1][:, 0], y[:, 0])) / 2
    torch.testing.assert_close(result['loss'], expected)
    result['loss'].backward()
    assert ticks[0].grad[:, 0].abs().sum() > 0 and ticks[1].grad[:, 0].abs().sum() > 0
    assert ticks[2].grad.count_nonzero() == 0
    assert all(z.grad[:, 1].count_nonzero() == 0 for z in ticks)
    with pytest.raises(ValueError, match='at least one supervised'):
        temporal_loss(ticks, torch.full_like(y, -100), 'uniform')


def test_same_initialization_state_and_inference():
    from ctm_transformer.confidence_readout import ConfidenceReadout
    config, _ = load_research_config('research/configs/fresh_confirmation_v1/recurrent_depth_seed23.json')
    torch.set_num_threads(4); torch.manual_seed(23)
    base = BaselineTransformer(config).to(DEVICE).eval(); rng = torch.get_rng_state()
    torch.manual_seed(23)
    model = RecurrentTemporalTransformer(config, 'dynamic_aggregate').to(DEVICE).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert parameter_counts(model) == parameter_counts(base) and parameter_counts(model)['total'] == 525984
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    x = torch.tensor([[1, 2, 3, 4]], device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(base(x)['logits'], model(x)['logits'], atol=0, rtol=0)
        torch.testing.assert_close(ConfidenceReadout(base)(x)['logits'], ConfidenceReadout(model)(x)['logits'], atol=0, rtol=0)
    with pytest.raises(ValueError, match='dropout=0'):
        RecurrentTemporalTransformer(replace(config, dropout=.1), 'uniform')
    with pytest.raises(ValueError, match='Unknown'):
        RecurrentTemporalTransformer(config, 'unsupported')


def fixture_data(tmp_path):
    paths = []
    for split in ('train', 'validation'):
        p = tmp_path / (split + '.jsonl')
        source = Path(f'research/data/ordered_pointer_v1/pointer/{split}.jsonl')
        p.write_text('\n'.join(source.read_text().splitlines()[:4]) + '\n'); paths.append(p)
    return paths


@pytest.mark.parametrize('objective', ['uniform', 'dynamic_aggregate'])
def test_runner_records_objective_and_loads_inference_checkpoint(objective, tmp_path):
    train, val = fixture_data(tmp_path)
    config, identity = small(max_steps=2, warmup_steps=0, eval_interval=1, log_interval=2,
                             data_path=str(train), eval_data_path=str(val), checkpoint_dir=str(tmp_path/'run'))
    summary = train_experiment(config, identity, 23, 'algorithmic', ['confidence'], True, recurrent_objective=objective)
    assert summary['complete'] and summary['parameters']['total'] > 0
    root = tmp_path/'run'; record = json.loads((root/'research_run.json').read_text())
    assert record['training_objective']['type'] == objective
    assert 'ctm_transformer/recurrent_temporal.py' in record['code_sha256']
    rows = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    assert all(row['block_applications_per_sequence'] == 13 and len(row['per_tick_supervised_ce']) == 4 for row in rows)
    assert summary['token_block_applications'] == rows[-1]['padded_token_positions'] * 13
    checkpoint = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['training_objective'] == record['training_objective']
    loaded, _ = _load_checkpoint(root/'best.pt'); loaded.to(DEVICE).eval()
    restored = RecurrentTemporalTransformer(config, objective).to(DEVICE).eval()
    restored.load_state_dict(checkpoint['model_state_dict'])
    x = torch.tensor([[1, 2, 3, 4]], device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x)['logits'], restored(x)['logits'], atol=0, rtol=0)


def test_final_ce_runner_matches_archived_training_exactly(tmp_path):
    # Execute the closed study's archived runner on the same small fixture.
    root = Path('research/results/fresh_confirmation_v1')
    manifest = json.loads((root/'pre_run_source.json').read_text())
    with zipfile.ZipFile(root/'pre_run_source.zip') as archive:
        source = archive.read('ctm_transformer/experiment.py')
    assert hashlib.sha256(source).hexdigest() == manifest['files']['ctm_transformer/experiment.py']
    old = types.ModuleType('archived_experiment'); exec(compile(source, '<archived experiment>', 'exec'), old.__dict__)
    train, val = fixture_data(tmp_path)
    config, identity = small(max_steps=3, warmup_steps=0, eval_interval=3, log_interval=3,
                             data_path=str(train), eval_data_path=str(val))
    states = []; trajectories = []
    for name, runner, kwargs in [('archived', old.train_experiment, {}), ('default', train_experiment, {}),
                                 ('explicit_final', train_experiment, {'recurrent_objective': 'final_ce'})]:
        directory = tmp_path/name
        runner(replace(config, checkpoint_dir=str(directory)), identity, 23, 'algorithmic', ['confidence'], True, **kwargs)
        payload = torch.load(directory/'final.pt', map_location='cpu', weights_only=False)
        states.append(payload['model_state_dict'])
        trajectories.append([{k: r[k] for k in ('step','loss','lr','gradient_norm','validation_readouts') if k in r}
                             for r in map(json.loads, (directory/'metrics.jsonl').read_text().splitlines())])
    assert trajectories[0] == trajectories[1] == trajectories[2]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]) and torch.equal(states[0][key], states[2][key]), key
