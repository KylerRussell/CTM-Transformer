"""Checks for the pointer calibration: suite design and decision helpers."""
import hashlib
import json
import random

import pytest

from ctm_transformer.algorithmic import canonical_json, pointer_record
from ctm_transformer.pointer_calibration import create_pointer_suite, validate_pointer_suite
from scripts.summarize_pointer_calibration import first_reaching

DESIGN = {'train_fresh_onehop': (40, [1]), 'train_fresh_multihop': (10, [1, 2, 3, 4]),
          'validation': (3, [1, 2, 3, 4]), 'eval_id': (4, [1, 2, 3, 4]), 'eval_depth': (4, [5, 6, 7, 8])}


def prior(tmp_path):
    rng = random.Random(0)
    rows = [pointer_record(rng, 12, 2) for _ in range(6)] + [pointer_record(rng, 8, 1) for _ in range(4)]
    (tmp_path / 'prior').mkdir()
    (tmp_path / 'prior/old.jsonl').write_text(''.join(canonical_json(r) + '\n' for r in rows))
    return tmp_path / 'prior', {r['id'] for r in rows[:6]}


def test_calibration_suite_is_deterministic_balanced_and_disjoint(tmp_path):
    root, prior_ids = prior(tmp_path)
    a = create_pointer_suite(tmp_path / 'a', DESIGN, 'x', 7, data_root=root)
    create_pointer_suite(tmp_path / 'b', DESIGN, 'x', 7, data_root=root)
    assert a['excluded_prior_maps'] == 6
    for split in DESIGN:
        assert (tmp_path / 'a/pointer' / f'{split}.jsonl').read_bytes() == (tmp_path / 'b/pointer' / f'{split}.jsonl').read_bytes()
    data = validate_pointer_suite(tmp_path / 'a')
    assert all(not (d.ids & prior_ids) for d in data.values())
    assert a['tasks']['pointer']['train_fresh_multihop']['steps'] == {1: 10, 2: 10, 3: 10, 4: 10}
    with pytest.raises(ValueError, match='empty'):
        create_pointer_suite(tmp_path / 'a', DESIGN, 'x', 7, data_root=root)


def test_calibration_validation_detects_leaked_training_map(tmp_path):
    root, _ = prior(tmp_path)
    create_pointer_suite(tmp_path / 'a', DESIGN, 'x', 7, data_root=root)
    suite = tmp_path / 'a'
    leaked = json.loads((suite / 'pointer/train_fresh_onehop.jsonl').read_text().splitlines()[0])
    rows = (suite / 'pointer/eval_id.jsonl').read_text().splitlines()
    replaced = json.loads(rows[0])
    leaked.update(split='eval_id')
    # Keep the hop balance intact so only the overlap check can fail.
    leaked = {**leaked, 'steps': replaced['steps']}
    from ctm_transformer.algorithmic import solve
    leaked['prompt'] = leaked['prompt'].rsplit('steps', 1)[0] + f"steps {leaked['steps']}="
    leaked['answer'] = solve(leaked)
    leaked['difficulty'] = {'nodes': 12, 'steps': leaked['steps']}
    rows[0] = canonical_json(leaked)
    (suite / 'pointer/eval_id.jsonl').write_text('\n'.join(rows) + '\n')
    manifest = json.loads((suite / 'manifest.json').read_text())
    manifest['tasks']['pointer']['eval_id']['sha256'] = hashlib.sha256((suite / 'pointer/eval_id.jsonl').read_bytes()).hexdigest()
    (suite / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='overlap'):
        validate_pointer_suite(suite)


def test_calibration_first_reaching_threshold():
    curve = [{'step': 1000, 'by_hop': {'nodes=12,steps=1': 0.5}}, {'step': 2000, 'by_hop': {'nodes=12,steps=1': 0.95}},
             {'step': 3000, 'by_hop': {'nodes=12,steps=1': 0.85}}]
    assert first_reaching(curve, 1) == 2000
    assert first_reaching(curve, 2) is None
