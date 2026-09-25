"""Seeded all-keys MQAR suites (single 12-cycle maps, every key queried, one hop).

Format (from `pointer_probes`, ``mqar``): ``map Al;Gf;...;hops 1;ask Kd;Ia;...;`` with
lowercase values; all 12 keys are queried in random order and only answers plus a
final EOS are supervised. Position 1 is leak-free (chance 1/11); later positions can
use exclusion of earlier gold answers.

Each seed gets its own training, validation and evaluation maps, disjoint by map
within the seed. Every training map is presented once. No model carries over
from earlier studies, so maps are not excluded against earlier data.
"""
import hashlib,json
from pathlib import Path
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.dense_pointer import NODES,hop
from ctm_transformer.pointer_probes import ProbeDataset,write_probe_split

FORMAT,QUERIES='mqar',NODES


def create_mqar_suite(root,seeds,train_maps,validation_maps,evaluation_maps,name):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty dataset directory; generated splits are immutable')
    root.mkdir(parents=True,exist_ok=True)
    manifest={'schema_version':1,'name':name,'format':FORMAT,'queries_per_map':QUERIES,'nodes':NODES,'hops':[1],
        'generator_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('ctm_transformer/mqar_suite.py','ctm_transformer/pointer_probes.py')},
        'tokenizer':AlgorithmicTokenizer().snapshot(),'role':'development study data; no locked test split','seeds':{}}
    for seed in seeds:
        directory=root/f'seed{seed}';directory.mkdir()
        base=int(hashlib.sha256(f'{name}:{seed}'.encode()).hexdigest()[:8],16)
        val=write_probe_split(directory/'validation.jsonl',FORMAT,[1],validation_maps,base+1,queries=QUERIES)
        ev=write_probe_split(directory/'evaluation.jsonl',FORMAT,[1],evaluation_maps,base+2,exclude=val,queries=QUERIES)
        write_probe_split(directory/'train.jsonl',FORMAT,[1],train_maps,base+3,exclude=val|ev,queries=QUERIES)
        manifest['seeds'][str(seed)]={split:{'path':str((directory/f'{split}.jsonl').relative_to(root)),
            'sha256':hashlib.sha256((directory/f'{split}.jsonl').read_bytes()).hexdigest(),
            'maps':{'train':train_maps,'validation':validation_maps,'evaluation':evaluation_maps}[split]}
            for split in ('train','validation','evaluation')}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_mqar_suite(root)
    return manifest


def load_split(root,seed,split,seq_len=128):
    root=Path(root);entry=json.loads((root/'manifest.json').read_text())['seeds'][str(seed)][split]
    data=ProbeDataset(root/entry['path'],AlgorithmicTokenizer(),seq_len)
    if data.metadata['sha256']!=entry['sha256'] or len(data)!=entry['maps']:raise ValueError('Split does not match its manifest')
    return data


def validate_mqar_suite(root,seeds=None):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());out={}
    for seed in seeds or m['seeds']:
        data={split:load_split(root,seed,split) for split in ('train','validation','evaluation')}
        for split,d in data.items():
            for r in d.records:
                if r['format']!=FORMAT or r['steps']!=1 or len(r['queries'])!=QUERIES or sorted(r['queries'])!=sorted(r['successors']):
                    raise ValueError(f'seed {seed} {split}: not an all-keys one-hop MQAR record')
                if r['answers']!=[hop(r['successors'],q,1).lower() for q in r['queries']]:raise ValueError(f'seed {seed} {split}: wrong answer')
        names=list(data)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if data[a].ids&data[b].ids:raise ValueError(f'seed {seed}: map overlap between {a} and {b}')
        out[str(seed)]=data
    return out
