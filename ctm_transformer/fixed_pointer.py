"""Fixed-slot copying control derived from the ordered one-hop suite."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from ctm_transformer.algorithmic import AlgorithmicTokenizer, canonical_json
from ctm_transformer.ordered_pointer import validate_ordered_suite
from ctm_transformer.pointer_diagnostics import revise_query


def create_fixed_suite(source_root, output_root):
    source_root, output_root = Path(source_root), Path(output_root)
    original = validate_ordered_suite(source_root)
    source_manifest = json.loads((source_root/'manifest.json').read_text())
    if source_manifest.get('mode') != 'ordered':
        raise ValueError('Fixed copy requires the ordered source suite')
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('Use an empty fixed-copy directory')
    rows_by_split = {split: [revise_query(row, start='A', split=split) for row in original[split].records]
                     for split in ('train','validation','test_id','train_probe','train_reordered','test_shuffled')}
    for source, split in [('train_probe','train_new_query'),('test_id','test_new_query')]:
        rows_by_split[split] = [revise_query(row, start='B', split=split) for row in rows_by_split[source]]
    directory = output_root/'pointer'
    directory.mkdir(parents=True)
    manifest = {'schema_version':1, 'name':'pointer_fixed_copy_v1', 'mode':'fixedcopy',
        'tokenizer':AlgorithmicTokenizer().snapshot(),
        'source_manifest_sha256':hashlib.sha256((source_root/'manifest.json').read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'role':'exploratory copying of successor A from the first ordered edge; query B is out of distribution',
        'split_policy':{'disjoint_maps':['train','validation','test_id'],
                        'training_map_probes':['train_probe','train_reordered','train_new_query'],
                        'paired_heldout_probes':{'test_shuffled':'test_id','test_new_query':'test_id'}},
        'tasks':{'pointer':{}}}
    for split, rows in rows_by_split.items():
        path=directory/(split+'.jsonl')
        raw=''.join(canonical_json(row)+'\n' for row in rows).encode()
        path.write_bytes(raw)
        manifest['tasks']['pointer'][split]={'path':str(path.relative_to(output_root)),
            'examples':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),
            'depth_counts':dict(Counter(row['steps'] for row in rows)),
            'answer_counts':dict(Counter(row['answer'] for row in rows))}
    (output_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def validate_fixed_suite(root, seq_len=128):
    manifest=json.loads((Path(root)/'manifest.json').read_text())
    if manifest.get('mode') != 'fixedcopy':
        raise ValueError('Expected fixedcopy mode')
    # Reuse graph-disjointness, training intervention, and held-out order checks.
    data=validate_ordered_suite(root,seq_len)
    for split in ('train','validation','test_id','train_probe','train_reordered','test_shuffled'):
        if any(r['start']!='A' or r['steps']!=1 for r in data[split].records):
            raise ValueError('Fixed-copy examples must query A for one hop')
    # The common validator already loads all manifest splits.
    for source, split in [('train_probe','train_new_query'),('test_id','test_new_query')]:
        reference={r['id']:r for r in data[source].records}
        if data[split].ids != set(reference):
            raise ValueError('New-query probe must have exactly the same maps')
        for row in data[split].records:
            old=reference[row['id']]
            if (row['start']!='B' or row['answer']==old['answer'] or
                any(row[k]!=old[k] for k in ('successors','presentation','steps'))):
                raise ValueError('New-query probe may change only start A to B and its answer')
    return data
