"""Check map pairing, correct labels, exposure annotations, and immutable generation."""
import hashlib
import json
from pathlib import Path
import pytest
from ctm_transformer.query_diagnostic import create_query_suite,validate_query_suite,GROUPS,NODES
from ctm_transformer.algorithmic import canonical_json
from ctm_transformer.ordered_pointer import validate_ordered_suite

SOURCE=Path(__file__).resolve().parents[1]/'research/data/ordered_pointer_v1'
@pytest.fixture
def grid(tmp_path):
    root=tmp_path/'grid';create_query_suite(SOURCE,root);return root


def test_exhaustive_paired_labels_and_exposure(grid):
    data=validate_query_suite(grid,SOURCE);original=validate_ordered_suite(SOURCE)
    assert sum(len(d) for d in data.values())==3072
    for group in GROUPS:
        for index,old in enumerate(original[group].records):
            rows=[data[f'{group}_start_{n}'].records[index] for n in NODES]
            assert {r['answer'] for r in rows}==set(NODES)
            assert sum(r['start']==r['original_start'] for r in rows)==1
            for row in rows:
                assert row['answer']==old['successors'][row['start']]
                assert row['id']==old['id'] and row['presentation']==old['presentation']
                assert len(row['prompt'])==len(old['prompt'])
                if group=='validation':assert 'trained' not in row['query_exposure']
    assert not data['train_probe_start_A'].ids & data['validation_start_A'].ids


def test_reproducible_immutable(grid,tmp_path):
    second=tmp_path/'second';create_query_suite(SOURCE,second)
    for path in grid.rglob('*.json*'):assert path.read_bytes()==(second/path.relative_to(grid)).read_bytes()
    with pytest.raises(ValueError,match='empty'):create_query_suite(SOURCE,grid)


def test_rejects_missing_start(grid):
    path=grid/'manifest.json';m=json.loads(path.read_text());del m['tasks']['pointer']['validation_start_H'];path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='all eight'):validate_query_suite(grid,SOURCE)


def test_rejects_false_training_exposure_even_with_updated_hash(grid):
    path=grid/'pointer/validation_start_A.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    rows[0]['query_exposure']='trained_query'
    raw=''.join(canonical_json(r)+'\n' for r in rows).encode();path.write_bytes(raw)
    mp=grid/'manifest.json';m=json.loads(mp.read_text());m['tasks']['pointer']['validation_start_A']['sha256']=hashlib.sha256(raw).hexdigest();mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='annotation'):validate_query_suite(grid,SOURCE)
