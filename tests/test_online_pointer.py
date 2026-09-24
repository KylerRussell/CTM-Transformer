"""Checks for the fresh-map pointer study: data design, recovery and tick override."""
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ctm_transformer.online_pointer as op
import scripts.study_supervisor as sup
from ctm_transformer.algorithmic import canonical_json, pointer_record, load_algorithmic_split


def small_design(monkeypatch, tmp_path):
    monkeypatch.setattr(op, 'SPLITS', {'train_fresh_onehop': (60, [1]), 'train_fresh_multihop': (10, [1, 2, 3]),
                                       'validation': (4, [1, 2, 3]), 'eval_id': (5, [1, 2, 3]), 'eval_depth': (3, [4, 5, 6, 7, 8])})
    monkeypatch.setattr(op, 'REPEATED', ('train_repeated_onehop', 16))
    prior = tmp_path / 'prior'
    (prior / 'x').mkdir(parents=True)
    import random
    rng = random.Random(0)
    rows = [pointer_record(rng, 12, 1) for _ in range(5)] + [pointer_record(rng, 8, 1) for _ in range(5)]
    (prior / 'x' / 'old.jsonl').write_text(''.join(canonical_json(r) + '\n' for r in rows))
    return prior, {r['id'] for r in rows[:5]}


def test_online_suite_is_deterministic_nested_and_disjoint(tmp_path, monkeypatch):
    prior, prior_ids = small_design(monkeypatch, tmp_path)
    a = op.create_online_pointer_suite(tmp_path / 'a', data_root=prior)
    op.create_online_pointer_suite(tmp_path / 'b', data_root=prior)
    assert a['excluded_prior_maps'] == 5
    for split in a['tasks']['pointer']:
        assert (tmp_path / 'a/pointer' / f'{split}.jsonl').read_bytes() == (tmp_path / 'b/pointer' / f'{split}.jsonl').read_bytes()
    data = op.validate_online_pointer_suite(tmp_path / 'a')
    assert all(not (d.ids & prior_ids) for d in data.values())
    assert data['train_repeated_onehop'].ids <= data['train_fresh_onehop'].ids
    assert {r['steps'] for r in data['eval_depth'].records} == {4, 5, 6, 7, 8}
    with pytest.raises(ValueError, match='empty'):
        op.create_online_pointer_suite(tmp_path / 'a', data_root=prior)


def test_online_validation_detects_map_overlap(tmp_path, monkeypatch):
    prior, _ = small_design(monkeypatch, tmp_path)
    op.create_online_pointer_suite(tmp_path / 'a', data_root=prior)
    root = tmp_path / 'a'
    train = (root / 'pointer/train_fresh_multihop.jsonl').read_text().splitlines()
    evaluation = (root / 'pointer/eval_id.jsonl').read_text().splitlines()
    leaked = json.loads(train[0])
    leaked['split'] = 'eval_id'
    evaluation[0] = canonical_json(leaked)
    (root / 'pointer/eval_id.jsonl').write_text('\n'.join(evaluation) + '\n')
    manifest = json.loads((root / 'manifest.json').read_text())
    import hashlib
    manifest['tasks']['pointer']['eval_id']['sha256'] = hashlib.sha256((root / 'pointer/eval_id.jsonl').read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        op.validate_online_pointer_suite(root)


def test_online_supervisor_recovers_partial_cells(tmp_path):
    root, runs = tmp_path / 'results', tmp_path / 'runs'
    root.mkdir()
    entries = []
    for cell, status in [('done', 'complete'), ('partial', 'partial'), ('pending', 'pending')]:
        run = runs / cell
        entries.append({'cell': cell, 'run_directory': str(run), 'summary_path': str(root / f'{cell}.summary.json')})
        if status == 'complete':
            run.mkdir(parents=True)
            for f in ('summary.json', 'final.pt'):
                (run / f).write_text('{}')
            Path(entries[-1]['summary_path']).write_text('{}')
        elif status == 'partial':
            run.mkdir(parents=True)
            (run / 'metrics.jsonl').write_text(''.join(json.dumps({'step': s, 'loss': s, 'update_seconds': 1}) + '\n' for s in (1, 2)) + '{"st')
    (root / 'registry.json').write_text(json.dumps({'entries': entries, 'gpu_queues': {}}))
    study = sup.Study(root=root, runs=runs, worker_module='', freeze_module='', evaluate_module='',
                      summarize_module='', autostart=tmp_path / 'autostart')
    s = sup.Supervisor(study)
    assert s.attempt['attempt'] == 1
    s.archive_interrupted()
    assert (runs / 'done/final.pt').exists() and not (runs / 'partial').exists()
    record = s.state['interrupted_runs'][0]
    assert record['last_complete_step'] == 2 and Path(record['moved_to']).name == 'partial'
    (runs / 'partial').mkdir()
    (runs / 'partial/metrics.jsonl').write_text(''.join(json.dumps({'step': s, 'loss': s, 'update_seconds': 9}) + '\n' for s in (1, 2, 3)))
    s.replay_audit()
    assert json.loads((root / 'interruption_replay.json').read_text())['runs'][0]['exact']
    assert sup.Supervisor(study).attempt['attempt'] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_online_tick_override_changes_only_inference_depth(tmp_path, monkeypatch):
    from ctm_transformer.research import build_model, load_research_config
    from ctm_transformer.readout_selection import evaluate_readouts
    prior, _ = small_design(monkeypatch, tmp_path)
    op.create_online_pointer_suite(tmp_path / 'a', data_root=prior)
    data = load_algorithmic_split(tmp_path / 'a', 'pointer', 'eval_depth', 128)
    device = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
    for family in ('ctm', 'recurrent_depth'):
        config, _ = load_research_config(f'research/configs/presentation_control_v1/{family}_shuffled_seed23.json')
        torch.manual_seed(0)
        model = build_model(config).to(device)
        for depth in (4, 32):
            result = evaluate_readouts(model, data, replace(config, max_thought_steps=depth), device, ('confidence',))
            assert result['confidence']['thought_steps'] == depth
            assert len(result['diagnostics']['raw_tick_ce']) == depth
            assert len(result['confidence']['predictions']) == len(data)
