"""Paired enumeration of every start query on existing ordered maps."""
import hashlib
import json
from pathlib import Path
from ctm_transformer.algorithmic import AlgorithmicTokenizer, canonical_json, load_algorithmic_split
from ctm_transformer.ordered_pointer import validate_ordered_suite
from ctm_transformer.pointer_diagnostics import revise_query

GROUPS=('train_probe','validation')
NODES='ABCDEFGH'


def expanded_row(old, group, node):
    row=revise_query(old,start=node,split=f'{group}_start_{node}')
    row['original_start']=old['start']
    if group=='train_probe':
        row['query_exposure']='trained_query' if node==old['start'] else 'changed_training_map_query'
    else:
        row['query_exposure']='original_validation_query' if node==old['start'] else 'changed_validation_query'
    return row


def create_query_suite(source_root, output_root):
    source_root,output_root=Path(source_root),Path(output_root)
    original=validate_ordered_suite(source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('Use an empty all-query directory')
    manifest={'schema_version':1,'name':'pointer_all_queries_v1','mode':'all_queries',
              'tokenizer':AlgorithmicTokenizer().snapshot(),
              'source_manifest_sha256':hashlib.sha256((source_root/'manifest.json').read_bytes()).hexdigest(),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'role':'paired queries on training and checkpoint-selection validation maps; no independent test set',
              'pairing_unit':'map id; eight queries per map are correlated',
              'tasks':{'pointer':{}}}
    directory=output_root/'pointer';directory.mkdir(parents=True)
    for group in GROUPS:
        for node in NODES:
            split=f'{group}_start_{node}'
            rows=[expanded_row(old,group,node) for old in original[group].records]
            path=directory/f'{split}.jsonl'
            raw=''.join(canonical_json(row)+'\n' for row in rows).encode();path.write_bytes(raw)
            manifest['tasks']['pointer'][split]={'path':str(path.relative_to(output_root)),
                'examples':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),'source_split':group,'start':node}
    (output_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def validate_query_suite(root, source_root):
    root,source_root=Path(root),Path(source_root)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('mode')!='all_queries':raise ValueError('Expected all_queries mode')
    if manifest['source_manifest_sha256']!=hashlib.sha256((source_root/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Source manifest changed')
    expected={f'{g}_start_{n}' for g in GROUPS for n in NODES}
    if set(manifest['tasks']['pointer'])!=expected:raise ValueError('Expected exactly all eight starts for both groups')
    original=validate_ordered_suite(source_root)
    data={}
    for group in GROUPS:
        old_rows=original[group].records
        for node in NODES:
            split=f'{group}_start_{node}'
            entry=manifest['tasks']['pointer'][split]
            if entry['source_split']!=group or entry['start']!=node:raise ValueError('Incorrect split annotation')
            dataset=load_algorithmic_split(root,'pointer',split,128)
            if [r['id'] for r in dataset.records]!=[r['id'] for r in old_rows]:
                raise ValueError('Map identities/order must match the source group')
            for row,old in zip(dataset.records,old_rows):
                if row!=expanded_row(old,group,node):raise ValueError('Query grid or exposure annotation differs from source')
            data[split]=dataset
    return data
