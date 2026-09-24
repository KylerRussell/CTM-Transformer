"""Paired ordered-edge control for the existing one-hop pointer diagnostic."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from ctm_transformer.algorithmic import AlgorithmicTokenizer, canonical_json
from ctm_transformer.pointer_diagnostics import revise_query, validate_diagnostic_suite


def create_ordered_suite(source_root, output_root):
    source_root, output_root = Path(source_root), Path(output_root)
    original = validate_diagnostic_suite(source_root)
    source_manifest = json.loads((source_root/'manifest.json').read_text())
    if source_manifest.get('mode') != 'onehop':
        raise ValueError('Ordered control requires the one-hop source suite')
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('Use an empty ordered-control directory')
    rows_by_split = {}
    for split in ('train', 'validation', 'test_id', 'train_probe', 'train_new_query'):
        rows_by_split[split] = [revise_query(row, presentation=sorted(row['successors']), split=split)
                               for row in original[split].records]
    # Retain the original random presentations as matched order-transfer probes.
    for source_split, output_split in [('train_reordered','train_reordered'), ('test_id','test_shuffled')]:
        rows = []
        for row in original[source_split].records:
            order = list(row['presentation'])
            if order == sorted(order):
                order = order[1:] + order[:1]
            rows.append(revise_query(row, presentation=order, split=output_split))
        rows_by_split[output_split] = rows
    directory = output_root/'pointer'
    directory.mkdir(parents=True)
    manifest = {'schema_version':1, 'name':'pointer_ordered_v1', 'mode':'ordered',
        'tokenizer':AlgorithmicTokenizer().snapshot(),
        'source_manifest_sha256':hashlib.sha256((source_root/'manifest.json').read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'role':'exploratory fixed-position lookup control; only edge presentation changes from onehop',
        'split_policy':{'disjoint_maps':['train','validation','test_id'],
                        'training_map_probes':['train_probe','train_reordered','train_new_query'],
                        'paired_heldout_order_probe':{'test_shuffled':'test_id'}},
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


def validate_ordered_suite(root, seq_len=128):
    data=validate_diagnostic_suite(root,seq_len)
    for split in ('train','validation','test_id','train_probe','train_new_query'):
        for row in data[split].records:
            if row['steps'] != 1 or row['presentation'] != sorted(row['successors']):
                raise ValueError('Ordered examples must use one hop and canonical edge order')
    reference={r['id']:r for r in data['test_id'].records}
    if data['test_shuffled'].ids != set(reference):
        raise ValueError('Held-out order probe must have exactly the same maps')
    for row in data['test_shuffled'].records:
        old=reference[row['id']]
        if (any(row[k]!=old[k] for k in ('successors','start','steps','answer'))
            or row['presentation']==old['presentation']):
            raise ValueError('Held-out order probe may change presentation only')
    return data
