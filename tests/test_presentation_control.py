"""Validate the presentation-only intervention and unchanged archived training."""
from dataclasses import replace
import hashlib,json,os,types,zipfile
from pathlib import Path
import pytest
import torch
from ctm_transformer.algorithmic import canonical_json
from ctm_transformer.presentation_control import create_presentation_control,validate_presentation_control,digest
from ctm_transformer.pointer_diagnostics import revise_query
from ctm_transformer.experiment import train_experiment
from ctm_transformer.research import load_research_config


def test_pairing_reproducibility_and_immutability(tmp_path):
    source='research/data/ordered_pointer_v1'
    create_presentation_control(source,tmp_path/'a');create_presentation_control(source,tmp_path/'b')
    for split in ('train','validation','validation_shuffled'):
        assert (tmp_path/'a/pointer'/f'{split}.jsonl').read_bytes()==(tmp_path/'b/pointer'/f'{split}.jsonl').read_bytes()
    data=validate_presentation_control(tmp_path/'a')
    assert len(data['train'])==2048 and len(data['validation'])==128
    assert data['validation'].ids==data['validation_shuffled'].ids
    with pytest.raises(ValueError,match='empty output'):create_presentation_control(source,tmp_path/'a')


def test_semantic_mutation_caught_even_with_updated_file_hash(tmp_path):
    create_presentation_control('research/data/ordered_pointer_v1',tmp_path/'data')
    root=tmp_path/'data';p=root/'pointer/train.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
    old=rows[0];start=next(k for k in old['successors'] if k!=old['start'])
    rows[0]=revise_query(old,start=start)
    p.write_text(''.join(canonical_json(r)+'\n' for r in rows))
    manifest=json.loads((root/'manifest.json').read_text());manifest['tasks']['pointer']['train']['sha256']=digest(p)
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Semantic examples'):validate_presentation_control(root)


def test_presentation_does_not_change_paired_batch_semantics(tmp_path):
    from ctm_transformer.algorithmic import load_algorithmic_split
    create_presentation_control('research/data/ordered_pointer_v1',tmp_path/'data')
    shuffled=validate_presentation_control(tmp_path/'data')['train']
    ordered=load_algorithmic_split('research/data/ordered_pointer_v1','pointer','train',128)
    a=torch.randperm(len(ordered),generator=torch.Generator().manual_seed(24))
    b=torch.randperm(len(shuffled),generator=torch.Generator().manual_seed(24))
    assert torch.equal(a,b)
    for offset in range(0,len(a),32):
        indexes=a[offset:offset+32]
        assert [ordered.records[i]['id'] for i in indexes]==[shuffled.records[i]['id'] for i in indexes]
        x,y,n=ordered.batch(indexes);xx,yy,nn=shuffled.batch(indexes)
        assert x.shape==xx.shape and torch.equal(y,yy) and n==nn


@pytest.mark.parametrize('family',['ctm','transformer'])
def test_current_default_runner_replays_archived_runner(family,tmp_path):
    torch.set_num_threads(4)
    old_root=Path('research/results/fresh_confirmation_v1');pre=json.loads((old_root/'pre_run_source.json').read_text())
    with zipfile.ZipFile(old_root/'pre_run_source.zip') as z:source=z.read('ctm_transformer/experiment.py')
    assert hashlib.sha256(source).hexdigest()==pre['files']['ctm_transformer/experiment.py']
    module=types.ModuleType('archived_experiment');exec(compile(source,'<archived runner>','exec'),module.__dict__)
    paths=[]
    for split in ('train','validation'):
        path=tmp_path/f'{split}.jsonl';path.write_text('\n'.join(Path(f'research/data/ordered_pointer_v1/pointer/{split}.jsonl').read_text().splitlines()[:4])+'\n');paths.append(path)
    config,identity=load_research_config(f'research/configs/fresh_confirmation_v1/{family}_seed23.json')
    config=replace(config,batch_size=2,max_steps=3,warmup_steps=0,eval_interval=3,log_interval=3,
        data_path=str(paths[0]),eval_data_path=str(paths[1]),device=os.environ.get('CTM_TEST_DEVICE','cuda:0'))
    states=[];trajectories=[]
    policy='confidence' if family=='ctm' else 'final'
    for name,runner in [('old',module.train_experiment),('current',train_experiment)]:
        out=tmp_path/name
        runner(replace(config,checkpoint_dir=str(out)),identity,23,'algorithmic',[policy],True)
        payload=torch.load(out/'final.pt',map_location='cpu',weights_only=False);states.append(payload['model_state_dict'])
        trajectories.append([{k:r[k] for k in ('loss','gradient_norm','lr','validation_readouts') if k in r}
            for r in map(json.loads,(out/'metrics.jsonl').read_text().splitlines())])
    assert trajectories[0]==trajectories[1]
    for key in states[0]:assert torch.equal(states[0][key],states[1][key]),key
