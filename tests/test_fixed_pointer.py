"""Verify fixed-slot copying and the deliberately unseen-query probes."""
import hashlib
import json
from pathlib import Path
import pytest
from ctm_transformer.algorithmic import canonical_json
from ctm_transformer.fixed_pointer import create_fixed_suite, validate_fixed_suite
from ctm_transformer.ordered_pointer import validate_ordered_suite
from ctm_transformer.pointer_diagnostics import revise_query

SOURCE=Path(__file__).resolve().parents[1]/'research/data/ordered_pointer_v1'

@pytest.fixture
def fixed(tmp_path):
    root=tmp_path/'fixed'
    create_fixed_suite(SOURCE,root)
    return root


def test_fixed_copy_preserves_maps_splits_order_and_lengths(fixed):
    old=validate_ordered_suite(SOURCE)
    new=validate_fixed_suite(fixed)
    for split in ('train','validation','test_id','train_probe','train_reordered','test_shuffled'):
        assert [r['id'] for r in new[split].records]==[r['id'] for r in old[split].records]
        for actual,previous in zip(new[split].records,old[split].records):
            assert all(actual[k]==previous[k] for k in ('id','successors','steps','presentation','difficulty'))
            assert actual['start']=='A'
            assert actual['answer']==actual['successors']['A']
            assert len(actual['prompt'])==len(previous['prompt'])
    for split in ('train','validation','test_id'):
        for row in new[split].records:
            assert row['prompt'].startswith('map A>'+row['answer']+';')


def test_paired_probes_and_disjoint_maps(fixed):
    data=validate_fixed_suite(fixed)
    assert not data['train'].ids & (data['validation'].ids | data['test_id'].ids)
    for source,probe in [('train_probe','train_new_query'),('test_id','test_new_query')]:
        by_id={r['id']:r for r in data[source].records}
        assert data[probe].ids==set(by_id)
        for row in data[probe].records:
            old=by_id[row['id']]
            assert row['start']=='B' and row['answer']==row['successors']['B']
            assert row['answer']!=old['answer']
            assert row['presentation']==old['presentation'] and row['successors']==old['successors']
    assert data['test_shuffled'].ids==data['test_id'].ids


def test_reproducible_and_immutable(fixed,tmp_path):
    second=tmp_path/'second'
    create_fixed_suite(SOURCE,second)
    for path in fixed.rglob('*.json*'):
        assert path.read_bytes()==(second/path.relative_to(fixed)).read_bytes()
    with pytest.raises(ValueError,match='empty'):
        create_fixed_suite(SOURCE,fixed)


@pytest.mark.parametrize('split,start,message',[
    ('validation','C','query A'),('test_new_query','C','only start A to B')])
def test_rejects_semantically_wrong_but_valid_records(fixed,split,start,message):
    path=fixed/'pointer'/f'{split}.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]=revise_query(rows[0],start=start)
    raw=''.join(canonical_json(r)+'\n' for r in rows).encode();path.write_bytes(raw)
    manifest=json.loads((fixed/'manifest.json').read_text())
    manifest['tasks']['pointer'][split]['sha256']=hashlib.sha256(raw).hexdigest()
    (fixed/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match=message):
        validate_fixed_suite(fixed)
