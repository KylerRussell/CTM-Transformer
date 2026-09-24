"""Independent checks of task solutions, splits, supervision, and generation."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ctm_transformer.algorithmic import (AlgorithmicTokenizer, AnswerDataset, carry_count,
    addition_record, pointer_record, generate_suite, load_algorithmic_split, validate_record, instance_identity)
from ctm_transformer.algorithmic_eval import greedy_answers, evaluate_checkpoint
from ctm_transformer.experiment import train_experiment, evaluate_loss
from ctm_transformer.research import load_research_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def device():
    value = os.environ.get('CTM_TEST_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.empty(1, device=value)
    torch.set_num_threads(4)
    return value


@pytest.fixture(scope='module')
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp('algorithmic') / 'suite'
    generate_suite(root, 1234)
    return root


def write_rows(path, records):
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))


def test_known_carries_and_semantic_identity():
    assert [carry_count(a,b) for a,b in [(0,0),(9,1),(19,81),(999,1),(12,23)]] == [0,1,2,3,0]
    rng = random.Random(1)
    row = addition_record(rng, 2, [0,1,2])
    swapped = row | {'a': row['b'], 'b': row['a']}
    assert instance_identity(row) == instance_identity(swapped)
    graph = pointer_record(rng, 8, 2)
    assert instance_identity(graph) == instance_identity(graph | {'start': 'B', 'steps': 3})


def test_generated_splits_are_disjoint_and_correct(suite):
    manifest = json.loads((suite / 'manifest.json').read_text())
    for task, splits in manifest['tasks'].items():
        seen = set()
        for split in splits:
            data = load_algorithmic_split(suite, task, split, 128)
            assert not seen & data.ids
            seen.update(data.ids)
            for row in data.records:
                if task == 'addition':
                    assert int(row['answer']) == sum((row['a'], row['b']))
                    if split in ('train','validation','test_id'):
                        assert row['difficulty']['digits'] <= 2 and row['difficulty']['carries'] <= 1
                    elif split == 'ood_carry':
                        assert row['difficulty'] == {'digits':2, 'carries':2}
                else:
                    # Parse the rendered graph rather than using its stored map.
                    parts = row['prompt'].split(';')
                    edges = parts[:-2]; edges[0] = edges[0][4:]
                    graph = dict(edge.split('>') for edge in edges)
                    state = parts[-2].split(' ')[1]
                    count = int(parts[-1].split(' ')[1][:-1])
                    for _ in range(count):
                        state = graph[state]
                    assert state == row['answer']
                    if split == 'ood_depth':
                        assert len(graph) == 8 and 4 <= count <= 6


def test_generator_is_byte_reproducible(suite, tmp_path):
    second = tmp_path / 'repeat'
    generate_suite(second, 1234)
    assert (suite / 'manifest.json').read_bytes() == (second / 'manifest.json').read_bytes()
    for path in suite.rglob('*.jsonl'):
        assert path.read_bytes() == (second / path.relative_to(suite)).read_bytes()
    with pytest.raises(ValueError, match='empty'):
        generate_suite(second, 1234)


def test_answer_mask_shift_and_padding(tmp_path):
    rng = random.Random(2); tok = AlgorithmicTokenizer()
    rows = [addition_record(rng, 1, [0,1]), addition_record(rng, 4, [0,1,2,3,4])]
    path = tmp_path / 'examples.jsonl'; write_rows(path, rows)
    data = AnswerDataset(path, tok, 128)
    inputs, targets, real = data.batch(slice(None))
    assert real == int(data.lengths.sum())
    for i, row in enumerate(rows):
        prefix = [tok.bos_token] + tok.encode(row['prompt'])
        answer = tok.encode(row['answer']) + [tok.eot_token]
        assert inputs[i, :len(prefix)].tolist() == prefix
        assert targets[i, :len(prefix)-1].tolist() == [-100] * (len(prefix)-1)
        assert targets[i, targets[i] != -100].tolist() == answer
        assert inputs[i, len(prefix):int(data.lengths[i])].tolist() == answer[:-1]
        assert (targets[i, int(data.lengths[i]):] == -100).all()
    with pytest.raises(ValueError, match='truncation'):
        AnswerDataset(path, tok, 3)


def test_corrupt_and_duplicate_records_rejected(tmp_path):
    row = addition_record(random.Random(4), 2, [0,1])
    with pytest.raises(ValueError, match='inconsistent'):
        validate_record(row | {'answer': '999'})
    path = tmp_path / 'duplicates.jsonl'; write_rows(path, [row, row])
    with pytest.raises(ValueError, match='Duplicate'):
        AnswerDataset(path, AlgorithmicTokenizer(), 128)


class AdditionOracle(nn.Module):
    def __init__(self, fail=False):
        super().__init__(); self.fail = fail
    def forward(self, ids, **kwargs):
        tok = AlgorithmicTokenizer()
        logits = torch.full((*ids.shape, tok.n_vocab), -10., device=ids.device)
        for row in range(len(ids)):
            tokens = [t for t in ids[row].tolist() if t >= 3]
            text = tok.decode(tokens)
            prompt, suffix = text.split('=')
            a, b = map(int, prompt[4:].split('+'))
            answer = str(a+b)
            next_id = tok.encode(answer[len(suffix)])[0] if len(suffix) < len(answer) else tok.eot_token
            if self.fail: next_id = tok.encode('Z')[0]
            logits[row, :, next_id] = 10.
        return {'logits': logits}


def test_generation_uses_no_gold_tokens_and_requires_eos(device):
    rng = random.Random(5)
    records = [addition_record(rng, d, list(range(d+1))) for d in (1,2,4)]
    config, _ = load_research_config(ROOT / 'research/configs/transformer_algorithmic_v1.json')
    result = greedy_answers(AdditionOracle(), records, config, device, 1)
    assert all(r['correct'] and r['terminated'] for r in result)
    # Altering the recorded answer cannot change generated predictions.
    wrong_labels = [r | {'answer': '99999'} for r in records]
    other = greedy_answers(AdditionOracle(), wrong_labels, config, device, 1)
    assert [r['prediction'] for r in other] == [r['prediction'] for r in result]
    failed = greedy_answers(AdditionOracle(fail=True), records, config, device, 1)
    assert all(not r['correct'] and not r['terminated'] for r in failed)


@pytest.mark.parametrize('family', ['transformer','recurrent_depth','ctm'])
def test_masked_training_and_checkpoint_evaluation(device, family, tmp_path, suite):
    from scripts.eval_harness import _load_checkpoint
    config, identity = load_research_config(ROOT / f'research/configs/{family}_algorithmic_v1.json')
    changes = dict(d_model=16, n_heads=2, batch_size=2, max_steps=2, warmup_steps=0,
        eval_interval=1, log_interval=2, gradient_accumulation_steps=2,
        device=device, checkpoint_dir=str(tmp_path / 'run'))
    if family == 'ctm': changes.update(d_latent=16, n_layers=1, nlm_groups=16, nlm_hidden_dim=4, sync_sparse_pairs=16, max_thought_steps=2)
    elif family == 'recurrent_depth': changes.update(core_layers=1, ffn_hidden_dim=32, max_thought_steps=2)
    else: changes.update(n_layers=1, ffn_hidden_dim=32)
    config = replace(config, **changes)
    rng = random.Random(99)
    train_rows = [addition_record(rng, d, list(range(d+1))) for d in (1,2,3,4)]
    eval_rows = [addition_record(rng, d, list(range(d+1))) for d in (1,2,4)]
    train_path, eval_path = tmp_path/'train.jsonl', tmp_path/'validation.jsonl'
    write_rows(train_path, train_rows); write_rows(eval_path, eval_rows)
    config.data_path, config.eval_data_path = str(train_path), str(eval_path)
    summary = train_experiment(config, identity, 17, data_format='algorithmic')
    expected = 2 * sum(len(r['answer']) + 1 for r in train_rows)
    assert summary['supervised_tokens_seen'] == expected
    assert summary['examples_seen'] == 8
    model, saved = _load_checkpoint(tmp_path/'run/final.pt'); model.to(device).eval()
    data = AnswerDataset(eval_path, AlgorithmicTokenizer(), 128)
    measured = evaluate_loss(model, data, config, device)
    x,y,_ = data.batch(slice(None)); x,y=x.to(device),y.to(device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)['logits']
    expected_loss = F.cross_entropy(logits.float().reshape(-1,71), y.reshape(-1)).item()
    assert measured['loss'] == pytest.approx(expected_loss, abs=0.005)
    # Actual checkpoint evaluator with one complete held-out split.
    result = evaluate_checkpoint(tmp_path/'run/final.pt', suite, 'addition', device,
                                tmp_path/'evaluation.json', splits=['validation'])
    assert result['complete']
    values = next(iter(result['results']['validation']['depths'].values()))
    assert values['examples'] == 136 and len(values['predictions']) == 136
    assert 0 <= values['exact_match'] <= 1
    # Cross-file semantic leakage is rejected even under a different filename.
    leak = tmp_path/'leak.jsonl'; write_rows(leak, train_rows[:2])
    config.eval_data_path = str(leak); config.checkpoint_dir = str(tmp_path/'other')
    with pytest.raises(ValueError, match='overlapping'):
        train_experiment(config, identity, 17, data_format='algorithmic')


def test_pointer_generation_oracle_across_depth_and_size(device):
    class PointerOracle(nn.Module):
        def forward(self, ids, **kwargs):
            tok = AlgorithmicTokenizer()
            logits = torch.full((*ids.shape, tok.n_vocab), -10., device=ids.device)
            for row in range(len(ids)):
                text = tok.decode([t for t in ids[row].tolist() if t >= 3])
                prompt, generated = text.split('=')
                parts = prompt.split(';')
                edges = parts[:-2]; edges[0] = edges[0][4:]
                mapping = dict(edge.split('>') for edge in edges)
                node = parts[-2].split(' ')[1]
                for _ in range(int(parts[-1].split(' ')[1])):
                    node = mapping[node]
                next_id = tok.encode(node)[0] if not generated else tok.eot_token
                logits[row, :, next_id] = 10.
            return {'logits': logits}
    rng = random.Random(6)
    records = [pointer_record(rng, nodes, steps) for nodes, steps in [(8,1),(8,3),(8,6),(12,6)]]
    config, _ = load_research_config(ROOT / 'research/configs/transformer_algorithmic_v1.json')
    result = greedy_answers(PointerOracle(), records, config, device, 1)
    assert all(row['correct'] and row['terminated'] for row in result)
