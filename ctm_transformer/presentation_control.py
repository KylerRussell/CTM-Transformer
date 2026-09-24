"""Paired edge-presentation development control; no test data are loaded."""
from collections import Counter
import hashlib,json,random
from pathlib import Path
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,load_algorithmic_split
from ctm_transformer.pointer_diagnostics import revise_query


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_presentation_control(source_root,output_root,seed=20260924):
    source_root,output_root=Path(source_root),Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):raise ValueError('Use an empty output directory')
    original={s:load_algorithmic_split(source_root,'pointer',s,128) for s in ('train','validation')}
    if original['train'].ids & original['validation'].ids:raise ValueError('Training and validation maps overlap')
    sources={s:{'path':str((source_root/'pointer'/f'{s}.jsonl').resolve()),'sha256':original[s].metadata['sha256']}
             for s in original}
    generated={}
    for index,split in enumerate(('train','validation')):
        rng=random.Random(seed+index);rows=[]
        for row in original[split].records:
            labels=sorted(row['successors'])
            if len(labels)!=8 or row['steps']!=1 or row['presentation']!=labels:
                raise ValueError('Expected canonical eight-node one-hop examples')
            order=rng.sample(labels,len(labels))
            while order==labels:order=rng.sample(labels,len(labels))
            target_split='train' if split=='train' else 'validation_shuffled'
            rows.append(revise_query(row,presentation=order,split=target_split))
        generated[target_split]=rows
    directory=output_root/'pointer';directory.mkdir(parents=True)
    # Preserve the original selection-validation bytes exactly.
    (directory/'validation.jsonl').write_bytes(Path(sources['validation']['path']).read_bytes())
    for split,rows in generated.items():
        (directory/f'{split}.jsonl').write_text(''.join(canonical_json(r)+'\n' for r in rows))
    manifest={'schema_version':1,'name':'presentation_control_v1','seed':seed,
        'role':'development training-presentation intervention on existing semantic maps; no test splits',
        'tokenizer':AlgorithmicTokenizer().snapshot(),'source_files':sources,
        'source_sha256':{p:digest(p) for p in ('ctm_transformer/presentation_control.py','ctm_transformer/pointer_diagnostics.py','ctm_transformer/algorithmic.py')},
        'presentation_policy':'one fixed noncanonical permutation per map; independent train/validation generator streams; semantic example order preserved; no epoch augmentation',
        'tasks':{'pointer':{}}}
    for split in ('train','validation','validation_shuffled'):
        path=directory/f'{split}.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        manifest['tasks']['pointer'][split]={'path':str(path.relative_to(output_root)),'sha256':digest(path),
            'examples':len(rows),'answer_counts':dict(Counter(r['answer'] for r in rows))}
    (output_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_presentation_control(output_root)
    return manifest


def validate_presentation_control(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if set(m['tasks']['pointer'])!={'train','validation','validation_shuffled'}:raise ValueError('Unexpected split inventory')
    for p,h in m['source_sha256'].items():
        if digest(p)!=h:raise ValueError('Dataset source changed: '+p)
    from ctm_transformer.algorithmic import AnswerDataset
    original={}
    for split,source in m['source_files'].items():
        if digest(source['path'])!=source['sha256']:raise ValueError('Original source data changed')
        original[split]=AnswerDataset(source['path'],AlgorithmicTokenizer(),128)
    data={s:load_algorithmic_split(root,'pointer',s,128) for s in ('train','validation','validation_shuffled')}
    if data['train'].ids & data['validation'].ids:raise ValueError('Training and validation maps overlap')
    if data['validation'].metadata['sha256']!=original['validation'].metadata['sha256']:
        raise ValueError('Ordered validation must remain byte-identical')
    for old_split,new_split in (('train','train'),('validation','validation_shuffled')):
        old,new=original[old_split],data[new_split]
        if len(old)!=len(new):raise ValueError('Example counts changed')
        for a,b in zip(old.records,new.records):
            if any(a[k]!=b[k] for k in ('id','successors','start','steps','answer','difficulty','task')):
                raise ValueError('Semantic examples or their order changed')
            if a['presentation']==b['presentation']:raise ValueError('Presentation must differ')
        if not torch.equal(old.targets,new.targets) or not torch.equal(old.lengths,new.lengths):
            raise ValueError('Supervised targets or input lengths changed')
    return data
