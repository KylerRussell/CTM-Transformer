"""Verify that the ordered lookup control changes presentation only."""
import hashlib
import json
from pathlib import Path

import pytest

from ctm_transformer.algorithmic import canonical_json
from ctm_transformer.ordered_pointer import create_ordered_suite, validate_ordered_suite
from ctm_transformer.pointer_diagnostics import revise_query, validate_diagnostic_suite

SOURCE=Path(__file__).resolve().parents[1]/'research/data/pointer_diagnostics_v1/onehop'


@pytest.fixture
def ordered(tmp_path):
    root=tmp_path/'ordered'
    create_ordered_suite(SOURCE,root)
    return root


def test_ordering_preserves_map_query_labels_and_example_order(ordered):
    old=validate_diagnostic_suite(SOURCE)
    new=validate_ordered_suite(ordered)
    for split in ('train','validation','test_id','train_probe','train_new_query'):
        assert [r['id'] for r in new[split].records]==[r['id'] for r in old[split].records]
        for actual,previous in zip(new[split].records,old[split].records):
            assert all(actual[k]==previous[k] for k in ('id','successors','start','steps','answer','difficulty'))
            assert actual['presentation']==list('ABCDEFGH')
            assert len(actual['prompt'])==len(previous['prompt'])
            assert actual['answer']==actual['successors'][actual['start']]


def test_heldout_transfer_is_paired_and_disjoint_from_training(ordered):
    data=validate_ordered_suite(ordered)
    assert data['test_id'].ids==data['test_shuffled'].ids
    assert not data['test_shuffled'].ids & (data['train'].ids | data['validation'].ids)
    by_id={r['id']:r for r in data['test_id'].records}
    for row in data['test_shuffled'].records:
        reference=by_id[row['id']]
        assert row['presentation']!=reference['presentation']
        assert row['answer']==reference['answer'] and row['start']==reference['start']


def test_reproducible_and_immutable(ordered,tmp_path):
    second=tmp_path/'second'
    create_ordered_suite(SOURCE,second)
    for path in ordered.rglob('*.json*'):
        assert path.read_bytes()==(second/path.relative_to(ordered)).read_bytes()
    with pytest.raises(ValueError,match='empty'):
        create_ordered_suite(SOURCE,ordered)


def test_rejects_corrupted_heldout_intervention(ordered):
    path=ordered/'pointer/test_shuffled.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    row=rows[0]
    rows[0]=revise_query(row,start=next(k for k in row['successors'] if k!=row['start']))
    raw=''.join(canonical_json(r)+'\n' for r in rows).encode();path.write_bytes(raw)
    manifest=json.loads((ordered/'manifest.json').read_text())
    manifest['tasks']['pointer']['test_shuffled']['sha256']=hashlib.sha256(raw).hexdigest()
    (ordered/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='presentation only'):
        validate_ordered_suite(ordered)
