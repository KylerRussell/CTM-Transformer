"""Checks for the group word problem (running products) task."""
import itertools
import os
import random
from dataclasses import replace

import pytest
import torch

from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.group_word import (MIN_HELD_OUT, Group, GroupWordDataset, compose, group_elements,
                                        group_record, validate_group_record, write_group_split)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('name,size', [('Z2', 2), ('S3', 6), ('A5', 60)])
def test_groups_are_closed_associative_with_identity_and_inverses(name, size):
    elements = group_elements(name)
    assert len(elements) == size
    members = set(elements)
    identity = tuple(range(len(elements[0])))
    assert identity in members
    sample = elements if size <= 6 else random.Random(0).sample(elements, 12)
    for a, b, c in itertools.product(sample, repeat=3):
        assert compose(a, b) in members
        assert compose(compose(a, b), c) == compose(a, compose(b, c))
    assert all(any(compose(a, b) == identity for b in elements) for a in sample)
    if name != 'Z2':
        assert any(compose(a, b) != compose(b, a) for a, b in itertools.product(sample, repeat=2))


def test_running_products_follow_the_left_to_right_convention():
    group = Group('A5')
    rng = random.Random(1)
    for _ in range(20):
        word = ''.join(rng.choice(list(group.element)) for _ in range(rng.randint(1, 30)))
        product, expected = None, []
        for ch in word:
            g = group.element[ch]
            product = g if product is None else tuple(g[product[x]] for x in range(5))  # apply product, then g
            expected.append(group.symbol[product])
        assert group.running_products(word) == ''.join(expected)


def test_labels_align_and_never_appear_in_inputs(tmp_path):
    group = Group('S3')
    rng = random.Random(2)
    rows = [group_record(rng, group, n) for n in (1, 5, 16, 40)]
    path = tmp_path / 'x.jsonl'
    path.write_text(''.join(__import__('json').dumps(r) + '\n' for r in rows))
    data = GroupWordDataset(path, AlgorithmicTokenizer(), 128)
    tok = AlgorithmicTokenizer()
    for i, r in enumerate(rows):
        x, y = data.inputs[i, :data.lengths[i]], data.targets[i, :data.lengths[i]]
        assert tok.decode(x[1:].tolist()) == r['word'] + '='
        assert y[0] == -100 and tok.decode(y[1:-1].tolist()) == r['products'] and y[-1] == tok.eot_token
        # Each label is predicted at the position that has just read its element.
        assert all(tok.decode([int(x[j])]) == r['word'][j - 1] for j in range(1, len(r['word']) + 1))
    with pytest.raises(ValueError, match='Wrong running products'):
        validate_group_record(dict(rows[1], products=rows[1]['products'][::-1] if rows[1]['products'] != rows[1]['products'][::-1] else 'x' * 5), group)


def test_split_writer_keeps_long_sequences_disjoint_and_is_deterministic(tmp_path):
    group = Group('S3')
    lengths = list(range(1, 13))
    ev = write_group_split(tmp_path / 'e.jsonl', group, lengths, 600, 5)
    tr = write_group_split(tmp_path / 't.jsonl', group, lengths, 3000, 6, exclude=ev)
    write_group_split(tmp_path / 't2.jsonl', group, lengths, 3000, 6, exclude=ev)
    assert (tmp_path / 't.jsonl').read_bytes() == (tmp_path / 't2.jsonl').read_bytes()
    assert not (tr & ev) and all(len(i) for i in (tr, ev))
    data = GroupWordDataset(tmp_path / 't.jsonl', AlgorithmicTokenizer(), 128)
    assert data.metadata['lengths'] == lengths and all(r['steps'] >= MIN_HELD_OUT for r in data.records if r['id'] in data.ids)


@needs_cuda
def test_dense_evaluator_groups_by_length_and_position(tmp_path):
    from ctm_transformer.dense_pointer import evaluate_dense
    from ctm_transformer.research import build_model, load_research_config
    group = Group('A5')
    write_group_split(tmp_path / 'v.jsonl', group, [3, 7, 20], 12, 9)
    data = GroupWordDataset(tmp_path / 'v.jsonl', AlgorithmicTokenizer(), 128)
    config, _ = load_research_config('research/configs/presentation_control_v1/transformer_shuffled_seed23.json')
    config = replace(config, batch_size=4, device=os.environ.get('CTM_TEST_DEVICE', 'cuda:0'))
    torch.manual_seed(0)
    result = evaluate_dense(build_model(config).to(config.device), data, config, config.device, ('final',))['final']
    assert set(result['by_hop']) == {'3', '7', '20'} and len(result['answer_accuracy_by_position']) == 20
    assert result['target_tokens'] == sum(r['steps'] + 1 for r in data.records)
