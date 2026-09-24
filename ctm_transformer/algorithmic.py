"""Deterministic algorithmic development tasks and answer-only tokenization."""
import hashlib
import json
from pathlib import Path
import random
import string

import torch


class AlgorithmicTokenizer:
    name = 'algorithmic_char_v1'
    # Reserved PAD/BOS/EOS, then a fixed character alphabet; no learned rules.
    alphabet = string.digits + string.ascii_uppercase + string.ascii_lowercase + ' +>=;:'
    pad_token = 0
    bos_token = 1
    eot_token = 2
    n_vocab = len(alphabet) + 3

    def encode(self, text, **kwargs):
        return [self.alphabet.index(char) + 3 for char in text]

    def decode(self, ids):
        return ''.join(self.alphabet[i - 3] if 3 <= i < self.n_vocab else f'<{i}>' for i in ids)

    def snapshot(self):
        return {'name': self.name, 'alphabet': self.alphabet,
                'special_tokens': {'PAD': 0, 'BOS': 1, 'EOS': 2}, 'vocab_size': self.n_vocab}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def carry_count(a, b):
    carry, count = 0, 0
    while a or b:
        carry = int(a % 10 + b % 10 + carry >= 10)
        count += carry
        a //= 10; b //= 10
    return count


def solve(record):
    if record['task'] == 'addition':
        return str(record['a'] + record['b'])
    node = record['start']
    for _ in range(record['steps']):
        node = record['successors'][node]
    return node


def instance_identity(record):
    # Commuted additions and alternative presentations/queries of a graph
    # cannot cross splits. Pointer graphs, not just graph-query pairs, are held out.
    if record['task'] == 'addition':
        return content_hash(['addition', sorted([record['a'], record['b']])])
    return content_hash(['pointer', record['successors']])


def addition_record(rng, digits, carries):
    low = 0 if digits == 1 else 10 ** (digits - 1)
    for _ in range(100000):
        a, b = rng.randrange(low, 10 ** digits), rng.randrange(low, 10 ** digits)
        n = carry_count(a, b)
        if n in carries:
            record = {'task': 'addition', 'a': a, 'b': b,
                      'prompt': f'add {a}+{b}=', 'answer': str(a + b),
                      'difficulty': {'digits': digits, 'carries': n}}
            record['id'] = instance_identity(record)
            return record
    raise RuntimeError('Unable to sample the requested carry stratum')


def pointer_record(rng, nodes, steps):
    labels = list(string.ascii_uppercase[:nodes])
    cycle = rng.sample(labels, len(labels))
    successors = {cycle[i]: cycle[(i + 1) % nodes] for i in range(nodes)}
    start = rng.choice(labels)
    presentation = rng.sample(labels, len(labels))
    prompt = 'map ' + ';'.join(f'{node}>{successors[node]}' for node in presentation)
    prompt += f';start {start};steps {steps}='
    record = {'task': 'pointer', 'successors': successors, 'start': start, 'steps': steps,
              'presentation': presentation, 'prompt': prompt,
              'difficulty': {'nodes': nodes, 'steps': steps}}
    record['answer'] = solve(record)
    record['id'] = instance_identity(record)
    return record


def validate_record(record):
    if record.get('task') == 'addition':
        a, b = record['a'], record['b']
        if type(a) is not int or type(b) is not int or min(a, b) < 0:
            raise ValueError('Invalid addition operands')
        if len(str(a)) != len(str(b)):
            raise ValueError('Operands must have equal digit lengths')
        prompt = f'add {a}+{b}='
        difficulty = {'digits': len(str(a)), 'carries': carry_count(a, b)}
    elif record.get('task') == 'pointer':
        mapping = record['successors']
        nodes = len(mapping)
        labels = list(string.ascii_uppercase[:nodes])
        if not 2 <= nodes <= 26 or set(mapping) != set(labels) or set(mapping.values()) != set(labels):
            raise ValueError('Invalid node permutation')
        if type(record['steps']) is not int or not 1 <= record['steps'] < nodes or record['start'] not in mapping:
            raise ValueError('Invalid pointer query')
        seen, node = set(), labels[0]
        while node not in seen:
            seen.add(node); node = mapping[node]
        if len(seen) != nodes:
            raise ValueError('Pointer mapping must be one cycle')
        order = record['presentation']
        if sorted(order) != labels:
            raise ValueError('Invalid edge presentation')
        prompt = 'map ' + ';'.join(f'{node}>{mapping[node]}' for node in order)
        prompt += f";start {record['start']};steps {record['steps']}="
        difficulty = {'nodes': nodes, 'steps': record['steps']}
    else:
        raise ValueError('Unknown task')
    if (record.get('prompt') != prompt or record.get('answer') != solve(record)
            or record.get('id') != instance_identity(record) or record.get('difficulty') != difficulty):
        raise ValueError('Record prompt, solution, identity, or difficulty is inconsistent')


def generate_suite(root, seed=20260921):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use an empty dataset directory; generated splits are immutable')
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'name': 'algorithmic_v1', 'seed': seed,
                'tokenizer': AlgorithmicTokenizer().snapshot(),
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'role': 'development pilot; these evaluation splits are not a locked paper test set', 'tasks': {}}
    # 1-digit additions have only 55 unordered pairs, so their quotas are small.
    addition = {
        'train': [(32, 1, [0, 1]), (2048, 2, [0, 1])],
        'validation': [(8, 1, [0, 1]), (128, 2, [0, 1])],
        'test_id': [(8, 1, [0, 1]), (128, 2, [0, 1])],
        'ood_carry': [(256, 2, [2])],
        'ood_length': [(128, 3, [0, 1]), (128, 4, [0, 1])],
        'ood_joint': [(128, 3, [2, 3]), (128, 4, [2, 3, 4])],
    }
    pointer = {
        'train': [(2048, 8, [1, 2, 3])],
        'validation': [(128, 8, [1, 2, 3])],
        'test_id': [(128, 8, [1, 2, 3])],
        'ood_depth': [(256, 8, [4, 5, 6])],
        'ood_length': [(256, 12, [1, 2, 3])],
        'ood_joint': [(256, 12, [4, 5, 6])],
    }
    for task_index, (task, spec) in enumerate((('addition', addition), ('pointer', pointer))):
        seen = set()
        directory = root / task; directory.mkdir()
        manifest['tasks'][task] = {}
        for split_index, (split, strata) in enumerate(spec.items()):
            split_seed = seed + 1000 * task_index + split_index
            rng = random.Random(split_seed)
            rows = []
            for count, size, difficulties in strata:
                accepted = 0
                for _ in range(count * 1000):
                    # Pointer depth counts differ by at most one within a stratum.
                    row = (addition_record(rng, size, difficulties) if task == 'addition' else
                           pointer_record(rng, size, difficulties[accepted % len(difficulties)]))
                    if row['id'] in seen:
                        continue
                    seen.add(row['id']); row['split'] = split
                    validate_record(row)
                    rows.append(row); accepted += 1
                    if accepted == count:
                        break
                if accepted != count:
                    raise RuntimeError(f'Not enough distinct instances for {task}/{split}')
            rng.shuffle(rows)
            path = directory / (split + '.jsonl')
            data = ''.join(canonical_json(row) + '\n' for row in rows).encode()
            path.write_bytes(data)
            manifest['tasks'][task][split] = {'path': str(path.relative_to(root)),
                'examples': len(rows), 'seed': split_seed, 'sha256': hashlib.sha256(data).hexdigest(),
                'strata': strata}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class AnswerDataset:
    def __init__(self, path, tokenizer, seq_len):
        self.tokenizer = tokenizer
        raw = Path(path).read_bytes()
        self.records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not self.records:
            raise ValueError('Empty answer dataset')
        self.ids = set()
        xs, ys, lengths = [], [], []
        for record in self.records:
            validate_record(record)
            if record['id'] in self.ids:
                raise ValueError('Duplicate semantic instance')
            self.ids.add(record['id'])
            prefix = [tokenizer.bos_token] + tokenizer.encode(record['prompt'])
            answer = tokenizer.encode(record['answer']) + [tokenizer.eot_token]
            combined = prefix + answer
            if len(combined) - 1 > seq_len:
                raise ValueError('Example exceeds seq_len; answer truncation is not allowed')
            x, y = combined[:-1], [-100] * (len(prefix) - 1) + answer
            lengths.append(len(x))
            xs.append(x + [tokenizer.pad_token] * (seq_len - len(x)))
            ys.append(y + [-100] * (seq_len - len(y)))
        if len({r['task'] for r in self.records}) != 1:
            raise ValueError('Use one task family per dataset')
        self.inputs, self.targets = torch.tensor(xs), torch.tensor(ys)
        self.lengths = torch.tensor(lengths)
        self.metadata = {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                         'examples': len(self.records), 'max_input_length': max(lengths),
                         'supervised_tokens': int((self.targets != -100).sum()),
                         'task': self.records[0]['task'], 'splits': sorted({r.get('split') for r in self.records})}

    def __len__(self):
        return len(self.records)

    def batch(self, index):
        lengths = self.lengths[index]
        width = int(lengths.max())
        return self.inputs[index, :width], self.targets[index, :width], int(lengths.sum())


def load_algorithmic_split(root, task, split, seq_len):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    entry = manifest['tasks'][task][split]
    path = root / entry['path']
    dataset = AnswerDataset(path, AlgorithmicTokenizer(), seq_len)
    if dataset.metadata['sha256'] != entry['sha256'] or len(dataset) != entry['examples']:
        raise ValueError('Dataset does not match its manifest')
    if dataset.metadata['task'] != task or dataset.metadata['splits'] != [split]:
        raise ValueError('Task/split label does not match manifest')
    return dataset
