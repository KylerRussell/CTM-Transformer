"""Unconstrained greedy exact-answer evaluation for controlled tasks."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from ctm_transformer.algorithmic import AlgorithmicTokenizer, load_algorithmic_split
from ctm_transformer.experiment import evaluate_loss, file_hash
from ctm_transformer.research import model_family


@torch.inference_mode()
def greedy_answers(model, records, config, device, depth):
    tokenizer = AlgorithmicTokenizer()
    prompts = [[tokenizer.bos_token] + tokenizer.encode(r['prompt']) for r in records]
    # Global task policy, never the particular gold-answer length.
    budgets = {'addition': 6, 'pointer': 2}
    budget = budgets[records[0]['task']]
    if any(r['task'] != records[0]['task'] for r in records):
        raise ValueError('Mixed task batch')
    if max(map(len, prompts)) + budget - 1 > config.max_seq_len:
        raise ValueError('Prompt and generation budget exceed model window')
    generated = [[] for _ in records]
    finished = [False] * len(records)
    for _ in range(budget):
        width = max(map(len, prompts))
        ids = torch.full((len(records), width), tokenizer.pad_token, dtype=torch.long, device=device)
        lengths = torch.tensor([len(p) for p in prompts], device=device)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)] = torch.tensor(prompt, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.dtype == 'bfloat16'):
            logits = model(ids, max_thought_steps=depth)['logits']
        next_ids = logits[torch.arange(len(records), device=device), lengths - 1].argmax(-1).tolist()
        for row, token in enumerate(next_ids):
            if finished[row]:
                continue
            if token == tokenizer.eot_token:
                finished[row] = True
            else:
                generated[row].append(token)
                prompts[row].append(token)
        if all(finished):
            break
    return [{'id': r['id'], 'prediction': tokenizer.decode(tokens), 'answer': r['answer'],
             'terminated': ended, 'correct': ended and tokens == tokenizer.encode(r['answer']),
             'difficulty': r['difficulty']}
            for r, tokens, ended in zip(records, generated, finished)]


def summarize_predictions(predictions):
    def counts(rows):
        correct = sum(row['correct'] for row in rows)
        return {'examples': len(rows), 'correct': correct, 'exact_match': correct / len(rows),
                'terminated': sum(row['terminated'] for row in rows)}
    groups = defaultdict(list)
    for row in predictions:
        key = ','.join(f'{k}={v}' for k, v in sorted(row['difficulty'].items()))
        groups[key].append(row)
    return counts(predictions) | {'by_difficulty': {key: counts(rows) for key, rows in sorted(groups.items())}}


def evaluate_checkpoint(checkpoint, dataset_root, task, device, output, depths=None, splits=None,
                        selection='checkpoint supplied by caller; pilot uses best validation answer-token CE'):
    from scripts.eval_harness import _load_checkpoint
    model, config = _load_checkpoint(checkpoint)
    if config.tokenizer != AlgorithmicTokenizer.name or config.vocab_size != AlgorithmicTokenizer.n_vocab:
        raise ValueError('Checkpoint does not use the fixed algorithmic tokenizer')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    model.to(device).eval()
    depths = depths or ([1] if model_family(config) == 'transformer' else [config.max_thought_steps])
    if not depths or any(type(t) is not int or t <= 0 for t in depths) or len(set(depths)) != len(depths):
        raise ValueError('Depths must be positive and unique')
    if model_family(config) == 'transformer' and depths != [1]:
        raise ValueError('Standard Transformer supports T=1 only')
    manifest_path = Path(dataset_root) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    splits = splits or [s for s in manifest['tasks'][task] if s != 'train']
    output = Path(output)
    if output.exists():
        raise ValueError('Evaluation output already exists')
    report = {'schema_version': 1, 'complete': False, 'checkpoint': model._checkpoint_metadata,
        'task': task, 'dataset_manifest_sha256': file_hash(manifest_path),
        'model_family': model_family(config), 'device': str(device),
        'precision': config.dtype, 'depths': depths,
        'scoring': 'unconstrained greedy generation; exact answer tokens and EOS required; no gold prefixes',
        'generation_budgets': {'addition': 6, 'pointer': 2},
        'selection': selection,
        'source_sha256': {p: file_hash(p) for p in ('ctm_transformer/algorithmic.py',
             'ctm_transformer/algorithmic_eval.py', 'ctm_transformer/experiment.py',
             'ctm_transformer/model.py', 'ctm_transformer/baselines.py')},
        'results': {}}
    started = time.perf_counter()
    for split in splits:
        dataset = load_algorithmic_split(dataset_root, task, split, config.seq_len)
        report['results'][split] = {'dataset': dataset.metadata, 'depths': {}}
        for depth in depths:
            validation = evaluate_loss(model, dataset, replace(config, max_thought_steps=depth), device)
            predictions = []
            for offset in range(0, len(dataset), config.batch_size):
                predictions.extend(greedy_answers(model, dataset.records[offset:offset + config.batch_size],
                                                  config, device, depth))
            report['results'][split]['depths'][str(depth)] = {
                **summarize_predictions(predictions), 'answer_token_ce': validation['loss'],
                'supervised_tokens': validation['target_tokens'], 'predictions': predictions}
    report['complete'] = True
    report['wall_seconds'] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output)
    return report
