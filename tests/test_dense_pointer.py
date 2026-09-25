"""Checks for the dense pointer format and the versioned v3 runner."""
import json
import os
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ctm_transformer.algorithmic import AlgorithmicTokenizer, AnswerDataset, canonical_json, pointer_record
from ctm_transformer import dense_pointer as dp

DESIGN = {'train_fresh_onehop': (40, [1]), 'train_fresh_multihop': (10, [1, 2, 3, 4]),
          'validation': (3, [1, 2, 3, 4]), 'eval_id': (4, [1, 2, 3, 4]), 'eval_depth': (4, [5, 6, 7, 8])}
DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def write(path, rows):
    path.write_text(''.join(canonical_json(r) + '\n' for r in rows))
    return path


def test_dense_records_supervise_every_answer_and_reject_tampering(tmp_path):
    rng = random.Random(3)
    rows = [dp.dense_record(rng, k) for k in (1, 2, 3, 7)]
    for r in rows:
        dp.validate_dense_record(r)
        assert len(r['answer']) == dp.QUERIES and all(a != q for a, q in zip(r['answer'], r['queries']))
        assert r['answer'] == ''.join(dp.hop(r['successors'], q, r['steps']) for q in r['queries'])
    data = dp.DensePointerDataset(write(tmp_path / 'x.jsonl', rows), AlgorithmicTokenizer(), 128)
    assert data.metadata['supervised_tokens'] == len(rows) * (dp.QUERIES + 1)
    x, y, _ = data.batch(slice(0, 4))
    decoded = AlgorithmicTokenizer().decode([int(t) for t in y[0] if t >= 3])
    assert decoded == rows[0]['answer'] and int((y[0] == AlgorithmicTokenizer.eot_token).sum()) == 1
    bad = dict(rows[0], answer=rows[0]['answer'][::-1])
    with pytest.raises(ValueError, match='inconsistent'):
        dp.validate_dense_record(bad)
    assert abs(dp.exclusion_guess_ceiling() - sum(1 / (11 - i) for i in range(6)) / 6) < 1e-12


def test_dense_suite_is_deterministic_disjoint_and_excludes_both_formats(tmp_path):
    rng = random.Random(0)
    prior_single = [pointer_record(rng, 12, 1) for _ in range(4)]
    prior_dense = [dp.dense_record(rng, 2) for _ in range(3)]
    (tmp_path / 'prior').mkdir()
    write(tmp_path / 'prior/a.jsonl', prior_single)
    write(tmp_path / 'prior/b.jsonl', prior_dense)
    a = dp.create_dense_suite(tmp_path / 'a', DESIGN, 'x', 5, data_root=tmp_path / 'prior')
    dp.create_dense_suite(tmp_path / 'b', DESIGN, 'x', 5, data_root=tmp_path / 'prior')
    assert a['excluded_prior_maps'] == 7
    for split in DESIGN:
        assert (tmp_path / 'a/pointer' / f'{split}.jsonl').read_bytes() == (tmp_path / 'b/pointer' / f'{split}.jsonl').read_bytes()
    data = dp.validate_dense_suite(tmp_path / 'a')
    prior = {dp.map_key(r) for r in prior_single + prior_dense}
    assert all(not ({dp.map_key(r) for r in d.records} & prior) for d in data.values())
    # A map leaked from training into evaluation is detected even with a refreshed hash.
    rows = (tmp_path / 'a/pointer/eval_id.jsonl').read_text().splitlines()
    leaked = json.loads((tmp_path / 'a/pointer/train_fresh_multihop.jsonl').read_text().splitlines()[0])
    target = json.loads(rows[0])
    moved = dict(leaked, split='eval_id', steps=target['steps'], difficulty={'nodes': 12, 'steps': target['steps']})
    moved['prompt'] = dp.dense_prompt(moved)
    moved['answer'] = ''.join(dp.hop(moved['successors'], q, moved['steps']) for q in moved['queries'])
    rows[0] = canonical_json(moved)
    (tmp_path / 'a/pointer/eval_id.jsonl').write_text('\n'.join(rows) + '\n')
    import hashlib
    manifest = json.loads((tmp_path / 'a/manifest.json').read_text())
    manifest['tasks'][dp.TASK]['eval_id']['sha256'] = hashlib.sha256((tmp_path / 'a/pointer/eval_id.jsonl').read_bytes()).hexdigest()
    (tmp_path / 'a/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='overlap'):
        dp.validate_dense_suite(tmp_path / 'a')


@needs_cuda
@pytest.mark.parametrize('family', ['ctm', 'recurrent_depth', 'transformer'])
def test_dense_runner_reproduces_frozen_v2_runner(family, tmp_path):
    """Given v2 data and evaluator, v3 matches v2 metrics and weights exactly."""
    from ctm_transformer.experiment import train_experiment
    from ctm_transformer.dense_experiment import train_dense_experiment
    from ctm_transformer.readout_selection import evaluate_readouts
    from ctm_transformer.research import load_research_config
    torch.set_num_threads(4)
    paths = []
    for split in ('train', 'validation'):
        path = tmp_path / f'{split}.jsonl'
        path.write_text('\n'.join(Path(f'research/data/ordered_pointer_v1/pointer/{split}.jsonl').read_text().splitlines()[:4]) + '\n')
        paths.append(path)
    config, identity = load_research_config(f'research/configs/presentation_control_v1/{family}_shuffled_seed23.json')
    config = replace(config, batch_size=2, max_steps=3, warmup_steps=0, eval_interval=3, log_interval=3,
                     data_path=str(paths[0]), eval_data_path=str(paths[1]), device=DEVICE)
    policy = 'final' if family == 'transformer' else 'confidence'
    tokenizer = AlgorithmicTokenizer()
    runs = {'v2': lambda c: train_experiment(c, identity, 23, 'algorithmic', [policy], True),
            'v3': lambda c: train_dense_experiment(c, identity, 23, AnswerDataset(paths[0], tokenizer, c.seq_len),
                                                   AnswerDataset(paths[1], tokenizer, c.seq_len), evaluate_readouts, [policy])}
    trajectories, states = [], []
    for name, run in runs.items():
        out = tmp_path / name
        run(replace(config, checkpoint_dir=str(out)))
        states.append(torch.load(out / 'final.pt', map_location='cpu', weights_only=False)['model_state_dict'])
        trajectories.append([{k: r[k] for k in ('loss', 'gradient_norm', 'lr', 'tokens_seen', 'validation_readouts') if k in r}
                             for r in map(json.loads, (out / 'metrics.jsonl').read_text().splitlines())])
    assert trajectories[0] == trajectories[1]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key


@needs_cuda
def test_dense_evaluator_matches_direct_scoring_and_trains(tmp_path):
    from ctm_transformer.dense_experiment import train_dense_experiment
    from ctm_transformer.research import build_model, load_research_config
    rng = random.Random(11)
    train = write(tmp_path / 'train.jsonl', [dict(dp.dense_record(rng, 1), split='train') for _ in range(64)])
    valid = write(tmp_path / 'valid.jsonl', [dict(dp.dense_record(rng, k), split='validation') for k in (1, 2, 3, 1, 2, 3, 1, 2)])
    tokenizer = AlgorithmicTokenizer()
    train_data, valid_data = (dp.DensePointerDataset(p, tokenizer, 128) for p in (train, valid))
    config, identity = load_research_config('research/configs/presentation_control_v1/transformer_shuffled_seed23.json')
    config = replace(config, batch_size=4, device=DEVICE)
    torch.manual_seed(0)
    model = build_model(config).to(DEVICE)
    result = dp.evaluate_dense(model, valid_data, config, DEVICE, ('final',))['final']
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        x, y, _ = valid_data.batch(slice(0, len(valid_data)))
        logits = model(x.to(DEVICE), max_thought_steps=1, return_all_logits=True)['all_logits'][-1].float()
    y = y.to(DEVICE)
    mask = y.ne(-100) & y.ne(tokenizer.eot_token)
    direct = logits.argmax(-1)[mask].eq(y[mask]).float().mean().item()
    assert abs(result['answer_accuracy'] - direct) < 1e-9
    assert len(result['answer_accuracy_by_position']) == dp.QUERIES and set(result['by_hop']) == {'1', '2', '3'}
    summary = train_dense_experiment(replace(config, max_steps=40, warmup_steps=4, eval_interval=20, log_interval=20,
                                             checkpoint_dir=str(tmp_path / 'run')), identity, 5, train_data, valid_data,
                                     dp.evaluate_dense, ['final'], data_policy='test')
    rows = [json.loads(x) for x in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['supervised_tokens_seen'] == 40 * 4 * (dp.QUERIES + 1)
    assert rows[-1]['loss'] < rows[0]['loss'] and 'by_hop' in rows[-1]['validation']
    run = json.loads((tmp_path / 'run/research_run.json').read_text())
    assert run['runner'] == 'shared_research_v3' and run['evaluator'] == 'ctm_transformer.dense_pointer.evaluate_dense'
