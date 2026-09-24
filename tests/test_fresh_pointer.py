import json,random
from pathlib import Path
import pytest
from ctm_transformer.algorithmic import pointer_record,canonical_json,load_algorithmic_split
from ctm_transformer.fresh_pointer import create_fresh_suite,validate_fresh_suite


def test_fresh_maps_exclusion_pairing_and_reproducibility(tmp_path):
    prior=tmp_path/'prior';prior.mkdir();rng=random.Random(1)
    rows=[pointer_record(rng,8,1) for _ in range(12)]
    (prior/'old.jsonl').write_text(''.join(canonical_json(r)+'\n' for r in rows))
    a,b=tmp_path/'a',tmp_path/'b'
    ma=create_fresh_suite(a,prior,23,32);mb=create_fresh_suite(b,prior,23,32)
    assert ma==mb and ma['universe_maps']==5040
    ds=load_algorithmic_split(a,'pointer','test_id',128)
    assert not ds.ids&{r['id'] for r in rows} and len(ds)==32
    for split in ('test_id','test_shuffled'):
        assert (a/'pointer'/f'{split}.jsonl').read_bytes()==(b/'pointer'/f'{split}.jsonl').read_bytes()
    assert all(r['steps']==1 for r in ds.records)
    with pytest.raises(ValueError,match='Fresh output'):create_fresh_suite(a,prior,23,32)
    (prior/'new.jsonl').write_text(canonical_json(ds.records[0])+'\n')
    with pytest.raises(ValueError,match='inventory changed'):validate_fresh_suite(a)
