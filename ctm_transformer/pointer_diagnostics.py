"""Versioned pointer diagnostics: tiny memorization and one-hop lookup."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from ctm_transformer.algorithmic import (AlgorithmicTokenizer, canonical_json,
    instance_identity, load_algorithmic_split, solve, validate_record)


def revise_query(row, *, steps=None, start=None, presentation=None, split=None):
    row = deepcopy(row)
    if steps is not None:
        row['steps'] = steps
    if start is not None:
        row['start'] = start
    if presentation is not None:
        row['presentation'] = list(presentation)
    if split is not None:
        row['split'] = split
    mapping = row['successors']
    row['prompt'] = 'map ' + ';'.join(f'{node}>{mapping[node]}' for node in row['presentation'])
    row['prompt'] += f";start {row['start']};steps {row['steps']}="
    row['answer'] = solve(row)
    row['difficulty'] = {'nodes': len(mapping), 'steps': row['steps']}
    row['id'] = instance_identity(row)
    validate_record(row)
    return row


def create_diagnostics(source_root, output_root, seed=20260922):
    """Keep old maps paired with the mixed-depth pilot; generate no new graphs.

    Train/validation/test have disjoint maps. Interventions deliberately reuse
    selected training maps and are marked as probes, never as held-out data.
    """
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('Use an empty diagnostic directory')
    original = {split: load_algorithmic_split(source_root, 'pointer', split, 128)
                for split in ('train', 'validation', 'test_id')}
    seen = set()
    for data in original.values():
        if seen & data.ids:
            raise ValueError('Original train/validation/test maps overlap')
        seen.update(data.ids)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for mode in ('tiny', 'onehop'):
        rng = random.Random(seed + (mode == 'onehop'))
        if mode == 'tiny':
            # Four examples for each answer label; preserve original query depths.
            candidates = deepcopy(original['train'].records)
            rng.shuffle(candidates)
            counts, training = Counter(), []
            for row in candidates:
                if counts[row['answer']] < 4:
                    training.append(row); counts[row['answer']] += 1
            if len(training) != 32:
                raise ValueError('Need four original examples per label for tiny set')
        else:
            training = [revise_query(row, steps=1) for row in original['train'].records]
        training = [revise_query(row, split='train') for row in training]
        rows_by_split = {'train': training}
        for split in ('validation', 'test_id'):
            rows_by_split[split] = [revise_query(row, steps=1 if mode == 'onehop' else None, split=split)
                                   for row in original[split].records]
        probe = training if mode == 'tiny' else rng.sample(training, 256)
        rows_by_split['train_probe'] = [revise_query(row, split='train_probe') for row in probe]
        reordered, new_queries = [], []
        for row in probe:
            # Randomly choose a different edge order; this cannot change the answer.
            order = list(row['presentation'])
            while order == row['presentation']:
                order = rng.sample(row['presentation'], len(row['presentation']))
            reordered.append(revise_query(row, presentation=order, split='train_reordered'))
            # Same map, path depth, and presentation, but a new start and endpoint.
            choices = [node for node in row['successors'] if node != row['start']]
            new_queries.append(revise_query(row, start=rng.choice(choices), split='train_new_query'))
        rows_by_split['train_reordered'] = reordered
        rows_by_split['train_new_query'] = new_queries
        root = output_root / mode
        directory = root / 'pointer'; directory.mkdir(parents=True)
        manifest = {'schema_version': 1, 'name': f'pointer_{mode}_v1', 'mode': mode,
            'seed': seed + (mode == 'onehop'), 'tokenizer': AlgorithmicTokenizer().snapshot(),
            'source_manifest_sha256': hashlib.sha256((source_root/'manifest.json').read_bytes()).hexdigest(),
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'role': 'exploratory learning diagnostic; paired training-map interventions are not held-out accuracy',
            'split_policy': {'disjoint_maps': ['train', 'validation', 'test_id'],
                             'training_map_probes': ['train_probe', 'train_reordered', 'train_new_query']},
            'tasks': {'pointer': {}}}
        for split, rows in rows_by_split.items():
            path = directory / (split + '.jsonl')
            raw = ''.join(canonical_json(row)+'\n' for row in rows).encode()
            path.write_bytes(raw)
            manifest['tasks']['pointer'][split] = {'path': str(path.relative_to(root)),
                'examples': len(rows), 'sha256': hashlib.sha256(raw).hexdigest(),
                'depth_counts': dict(Counter(row['steps'] for row in rows)),
                'answer_counts': dict(Counter(row['answer'] for row in rows))}
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        manifests[mode] = manifest
    return manifests


def validate_diagnostic_suite(root, seq_len=128):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    data = {split: load_algorithmic_split(root, 'pointer', split, seq_len)
            for split in manifest['tasks']['pointer']}
    seen = set()
    for split in ('train', 'validation', 'test_id'):
        if seen & data[split].ids:
            raise ValueError('Diagnostic train/validation/test maps overlap')
        seen.update(data[split].ids)
    probe_ids = data['train_probe'].ids
    if not probe_ids <= data['train'].ids:
        raise ValueError('Probe is not a training subset')
    if any(data[s].ids != probe_ids for s in ('train_reordered', 'train_new_query')):
        raise ValueError('Interventions must be paired with the same training maps')
    originals = {row['id']: row for row in data['train_probe'].records}
    training = {row['id']: row for row in data['train'].records}
    for key, row in originals.items():
        if any(row[field] != training[key][field] for field in ('prompt','answer','steps','start','presentation')):
            raise ValueError('Unchanged probe does not match training example')
    for split in ('train_reordered', 'train_new_query'):
        for row in data[split].records:
            original = originals[row['id']]
            if row['steps'] != original['steps']:
                raise ValueError('Intervention changed path depth')
            if split == 'train_reordered':
                if (row['presentation'] == original['presentation'] or row['start'] != original['start']
                        or row['answer'] != original['answer']):
                    raise ValueError('Invalid edge-order intervention')
            elif (row['start'] == original['start'] or row['answer'] == original['answer']
                  or row['presentation'] != original['presentation']):
                raise ValueError('Invalid start-node intervention')
    return data
