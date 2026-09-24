"""Check paired diagnostic construction without changing model or task semantics."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from ctm_transformer.algorithmic import canonical_json, load_algorithmic_split
from ctm_transformer.pointer_diagnostics import create_diagnostics, revise_query, validate_diagnostic_suite

SOURCE = Path(__file__).resolve().parents[1]/'research/data/algorithmic_v1'


@pytest.fixture
def diagnostic(tmp_path):
    root = tmp_path/'diagnostic'
    create_diagnostics(SOURCE, root)
    return root


def test_onehop_preserves_maps_and_presentation(diagnostic):
    data = validate_diagnostic_suite(diagnostic/'onehop')
    for split in ('train','validation','test_id'):
        original = {r['id']:r for r in load_algorithmic_split(SOURCE,'pointer',split,128).records}
        assert data[split].ids == set(original)
        for row in data[split].records:
            old = original[row['id']]
            assert row['steps'] == 1
            assert row['successors'] == old['successors']
            assert row['start'] == old['start'] and row['presentation'] == old['presentation']
            assert row['answer'] == row['successors'][row['start']]


def test_tiny_set_is_balanced_and_probes_are_paired(diagnostic):
    data = validate_diagnostic_suite(diagnostic/'tiny')
    original = {r['id']:r for r in load_algorithmic_split(SOURCE,'pointer','train',128).records}
    assert len(data['train']) == 32
    assert Counter(row['answer'] for row in data['train'].records) == dict.fromkeys('ABCDEFGH',4)
    for row in data['train'].records:
        assert row['prompt'] == original[row['id']]['prompt']
        assert row['answer'] == original[row['id']]['answer']
    assert data['train'].ids == data['train_probe'].ids == data['train_reordered'].ids == data['train_new_query'].ids
    for split in ('train_reordered','train_new_query'):
        for row in data[split].records:
            # Independent transition iteration validates labels after each edit.
            node = row['start']
            for _ in range(row['steps']): node = row['successors'][node]
            assert node == row['answer']


def test_byte_reproducibility_and_immutable_output(diagnostic, tmp_path):
    second=tmp_path/'repeat'
    create_diagnostics(SOURCE,second)
    for path in diagnostic.rglob('*.json*'):
        assert path.read_bytes() == (second/path.relative_to(diagnostic)).read_bytes()
    with pytest.raises(ValueError,match='empty'):
        create_diagnostics(SOURCE,second)


def test_rejects_intervention_that_changes_two_factors(diagnostic):
    root=diagnostic/'onehop'
    path=root/'pointer/train_reordered.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    row=rows[0]
    row=revise_query(row,start=next(node for node in row['successors'] if node != row['start']))
    rows[0]=row
    raw=''.join(canonical_json(r)+'\n' for r in rows).encode(); path.write_bytes(raw)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['tasks']['pointer']['train_reordered']['sha256']=hashlib.sha256(raw).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='edge-order intervention'):
        validate_diagnostic_suite(root)
