"""Seeded group-word suites for frozen studies (see `group_word`).

Per seed: fresh training words of lengths 1..L_train (balanced), validation words
of lengths 1..L_train, and evaluation words of one longer length. Models are
causal, so positions 1..L_train of a long evaluation word measure trained-length
behavior, and later positions measure extrapolation. Validation and evaluation
words of length >= MIN_HELD_OUT never occur in training.
"""
import hashlib,json
from pathlib import Path
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.group_word import MIN_HELD_OUT,Group,GroupWordDataset,write_group_split


def create_group_suite(root,group,seeds,train_words,train_max_len,validation_words,evaluation_words,evaluation_length,name):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty dataset directory; generated splits are immutable')
    root.mkdir(parents=True,exist_ok=True);g=Group(group);lengths=list(range(1,train_max_len+1))
    manifest={'schema_version':1,'name':name,'group':group,'group_size':len(g.elements),'train_lengths':lengths,
        'evaluation_length':evaluation_length,'min_held_out_length':MIN_HELD_OUT,'tokenizer':AlgorithmicTokenizer().snapshot(),
        'generator_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('ctm_transformer/group_suite.py','ctm_transformer/group_word.py')},
        'role':'development study data; no locked test split','seeds':{}}
    for seed in seeds:
        directory=root/f'seed{seed}';directory.mkdir()
        base=int(hashlib.sha256(f'{name}:{seed}'.encode()).hexdigest()[:8],16)
        ev=write_group_split(directory/'evaluation.jsonl',g,[evaluation_length],evaluation_words,base+2)
        val=write_group_split(directory/'validation.jsonl',g,lengths,validation_words,base+1,exclude=ev)
        write_group_split(directory/'train.jsonl',g,lengths,train_words,base+3,exclude=ev|val)
        manifest['seeds'][str(seed)]={split:{'path':f'seed{seed}/{split}.jsonl','examples':count,
            'sha256':hashlib.sha256((directory/f'{split}.jsonl').read_bytes()).hexdigest()}
            for split,count in (('train',train_words),('validation',validation_words),('evaluation',evaluation_words))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_group_suite(root)
    return manifest


def load_group_split(root,seed,split,seq_len=128):
    root=Path(root);entry=json.loads((root/'manifest.json').read_text())['seeds'][str(seed)][split]
    data=GroupWordDataset(root/entry['path'],AlgorithmicTokenizer(),seq_len)
    if data.metadata['sha256']!=entry['sha256'] or len(data)!=entry['examples']:raise ValueError('Split does not match its manifest')
    return data


def validate_group_suite(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    for seed in m['seeds']:
        data={split:load_group_split(root,seed,split) for split in ('train','validation','evaluation')}
        if data['train'].metadata['lengths']!=m['train_lengths'] or data['validation'].metadata['lengths']!=m['train_lengths']:
            raise ValueError(f'seed {seed}: training/validation lengths differ from the design')
        if data['evaluation'].metadata['lengths']!=[m['evaluation_length']]:raise ValueError(f'seed {seed}: wrong evaluation length')
        if any(r['group']!=m['group'] for d in data.values() for r in d.records):raise ValueError(f'seed {seed}: wrong group')
        if data['train'].ids&(data['validation'].ids|data['evaluation'].ids):raise ValueError(f'seed {seed}: held-out word appears in training')
    return m
