"""Shared single-GPU development training for CTM and Transformer baselines."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ctm_transformer.algorithmic import AlgorithmicTokenizer, AnswerDataset
from ctm_transformer.research import build_model, model_family, parameter_counts, block_applications


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_config(config):
    if config.optimizer != 'adamw' or config.use_8bit_adam:
        raise ValueError('Shared research runner supports standard AdamW only')
    if config.dtype not in {'float32', 'bfloat16'} or (config.dtype == 'bfloat16' and not config.bf16_autocast):
        raise ValueError('Use FP32 or BF16 autocast with FP32 parameters')
    for name in ('batch_size', 'seq_len', 'gradient_accumulation_steps', 'max_steps', 'eval_interval', 'log_interval'):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not 0 <= config.warmup_steps < config.max_steps:
        raise ValueError('max_steps must exceed nonnegative warmup_steps')
    if config.learning_rate <= 0 or config.grad_clip <= 0 or config.weight_decay < 0:
        raise ValueError('Invalid optimizer settings')
    if not 0 <= config.adam_beta1 < 1 or not 0 <= config.adam_beta2 < 1:
        raise ValueError('AdamW betas must lie in [0, 1)')
    if model_family(config) == 'ctm':
        if config.temporal_loss_type not in {'final_ce', 'ramp_mono', 'dynamic_aggregate'}:
            raise ValueError('Unsupported temporal objective')
        if not math.isfinite(config.mono_penalty_weight) or config.mono_penalty_weight < 0:
            raise ValueError('mono_penalty_weight must be finite and nonnegative')
        if not 0 <= config.mono_penalty_min_frac <= 1 or not 0 <= config.mono_penalty_decay_until_frac <= 1:
            raise ValueError('Monotonic penalty schedule fractions must lie in [0, 1]')
        if config.temporal_loss_type == 'ramp_mono':
            endpoints = (config.tick_ramp_start, config.tick_ramp_end)
            if any(not math.isfinite(v) or v < 0 for v in endpoints):
                raise ValueError('Temporal CE weights must be finite and nonnegative')
            raw = torch.linspace(*endpoints, config.max_thought_steps)
            if raw.sum() <= 0:
                raise ValueError('Temporal CE weights must have positive total weight')
        if any(getattr(config, flag) for flag in ('use_distillation', 'use_cached_teacher',
            'use_two_phase_curriculum', 't_curriculum', 'use_hyperloop', 'use_feec',
            'use_cuda_graphs', 'tiled_schedule')):
            raise ValueError('Shared runner does not implement the requested CTM training extension')



def objective_metadata(config):
    """Record direct CE weights; validation always measures the final logits."""
    depth = config.max_thought_steps
    if model_family(config) != 'ctm' or config.temporal_loss_type == 'final_ce':
        weights = [0.0] * (depth - 1) + [1.0]
    elif config.temporal_loss_type == 'dynamic_aggregate':
        weights = None  # Per-token selection: minimum CE and maximum certainty.
    else:
        raw = torch.linspace(config.tick_ramp_start, config.tick_ramp_end, depth)
        weights = (raw / raw.sum()).tolist()
    return {'type': getattr(config, 'temporal_loss_type', 'final_ce'),
            'direct_ce_tick_weights_at_max_depth': weights,
            'validation_metric': 'final-tick supervised-token CE, independent of training objective',
            'dynamic_rule': 'mean over supervised tokens of (minimum-tick CE + maximum-certainty-tick CE)/2'
                if getattr(config, 'temporal_loss_type', '') == 'dynamic_aggregate' else None}

def load_text_blocks(path, tokenizer, seq_len):
    """Non-overlapping input windows; each has the following token as target.

    Adjacent windows share only their boundary target/input token. No wrapping
    or padding. The final incomplete window is explicitly counted and dropped.
    """
    raw = Path(path).read_bytes()
    tokens = torch.tensor(tokenizer.encode(raw.decode('utf-8'), allowed_special=set()), dtype=torch.long)
    count = (len(tokens) - 1) // seq_len
    if count < 1:
        raise ValueError(f'{path}: need at least seq_len+1 tokens')
    return tokens.unfold(0, seq_len + 1, seq_len), {
        'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
        'tokens': len(tokens), 'blocks': count,
        'unused_input_tokens': len(tokens) - 1 - count * seq_len,
    }


def training_depth(config, generator):
    if model_family(config) != 'recurrent_depth':
        return config.max_thought_steps
    # One uniformly sampled budget per optimizer update, shared by microbatches.
    return int(torch.randint(config.train_depth_min, config.train_depth_max + 1, (), generator=generator))


def get_batch(dataset, index, device):
    if isinstance(dataset, AnswerDataset):
        inputs, targets, real_tokens = dataset.batch(index)
        return inputs.to(device), targets.to(device), real_tokens
    batch = dataset[index].to(device)
    return batch[:, :-1], batch[:, 1:], batch[:, :-1].numel()


@torch.no_grad()
def evaluate_loss(model, blocks, config, device):
    """Final-output token-weighted CE, including the final partial batch."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for offset in range(0, len(blocks), config.batch_size):
            inputs, targets, _ = get_batch(blocks, slice(offset, offset + config.batch_size), device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.dtype == 'bfloat16'):
                logits = model(inputs, max_thought_steps=config.max_thought_steps)['logits']
            total += F.cross_entropy(logits.float().reshape(-1, config.vocab_size),
                                     targets.reshape(-1), reduction='sum').item()
            count += int((targets != -100).sum())
    finally:
        model.train(was_training)
    if not math.isfinite(total):
        raise RuntimeError('Non-finite validation loss')
    return {'loss': total / count, 'target_tokens': count, 'thought_steps': config.max_thought_steps}


def train_experiment(config, identity, seed, data_format='text', selection_readouts=None, save_validation_checkpoints=False, recurrent_objective=None):
    validate_training_config(config)
    if recurrent_objective is not None:
        from ctm_transformer.recurrent_temporal import validate_recurrent_objective, recurrent_objective_metadata, RecurrentTemporalTransformer
        validate_recurrent_objective(config, recurrent_objective)
    if selection_readouts is not None:
        if data_format != 'algorithmic' or not selection_readouts or len(set(selection_readouts)) != len(selection_readouts) or any(p not in ('final', 'confidence') for p in selection_readouts):
            raise ValueError('Explicit selection readouts require unique final/confidence policies on algorithmic data')
    device = torch.device(config.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('A working CUDA device is required')
    torch.cuda.set_device(device)
    if config.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16')
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Data ordering and depth budgets cannot depend on architecture RNG usage.
    data_rng = torch.Generator().manual_seed(seed + 1)
    depth_rng = torch.Generator().manual_seed(seed + 2)
    if data_format == 'algorithmic':
        tokenizer = AlgorithmicTokenizer()
        if config.tokenizer != tokenizer.name:
            raise ValueError('Algorithmic tasks require algorithmic_char_v1')
        train_blocks = AnswerDataset(config.data_path, tokenizer, config.seq_len)
        eval_blocks = AnswerDataset(config.eval_data_path, tokenizer, config.seq_len)
        train_data, eval_data = train_blocks.metadata, eval_blocks.metadata
        if train_blocks.ids & eval_blocks.ids:
            raise ValueError('Training and evaluation contain overlapping semantic instances')
        if train_data['task'] != eval_data['task']:
            raise ValueError('Training and evaluation task families differ')
        policy = 'independent examples; BOS and EOS; answer and EOS supervision only; right padding masked; train shuffled each epoch; final partial training batch dropped'
    elif data_format == 'text':
        import tiktoken
        tokenizer = tiktoken.get_encoding(config.tokenizer)
        train_blocks, train_data = load_text_blocks(config.data_path, tokenizer, config.seq_len)
        eval_blocks, eval_data = load_text_blocks(config.eval_data_path, tokenizer, config.seq_len)
        policy = 'separate UTF-8 files; contiguous windows; no EOS insertion; incomplete window dropped; train windows shuffled each epoch; final partial training batch dropped'
    else:
        raise ValueError('Unknown data_format')
    if tokenizer.n_vocab != config.vocab_size:
        raise ValueError('Tokenizer and frozen vocabulary size differ')
    if train_data['sha256'] == eval_data['sha256']:
        raise ValueError('Training and evaluation files have identical content')
    if len(train_blocks) < config.batch_size:
        raise ValueError('Training data must contain at least one full batch')
    out = Path(config.checkpoint_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty checkpoint_dir; resuming is not implemented')
    out.mkdir(parents=True, exist_ok=True)
    model = (build_model(config) if recurrent_objective is None else
             RecurrentTemporalTransformer(config, recurrent_objective)).to(device).train()
    training_objective = (objective_metadata(config) if recurrent_objective is None else
                          recurrent_objective_metadata(config, recurrent_objective))

    def training_applications(depth):
        extra = ((depth - 1) * config.coda_layers if recurrent_objective in ('uniform', 'dynamic_aggregate') else 0)
        return block_applications(config, depth) + extra
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)
    sources = ['ctm_transformer/experiment.py', 'ctm_transformer/research.py',
               'ctm_transformer/baselines.py', 'ctm_transformer/model.py',
               'ctm_transformer/config.py', 'ctm_transformer/algorithmic.py', 'scripts/train_research.py']
    if recurrent_objective is not None:
        sources += ['ctm_transformer/recurrent_temporal.py']
    if selection_readouts is not None:
        sources += ['ctm_transformer/readout_selection.py', 'ctm_transformer/confidence_readout.py', 'ctm_transformer/algorithmic_eval.py']
    from scripts.eval_harness import _tokenizer_snapshot
    snapshot = json.dumps(tokenizer.snapshot() if data_format == 'algorithmic' else _tokenizer_snapshot(tokenizer), sort_keys=True)
    (out / 'tokenizer.json').write_text(snapshot)
    record = {
        'schema_version': 2, 'runner': 'shared_research_v2', 'data_format': data_format,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'model_family': model_family(config),
        'config_identity': identity, 'effective_config': asdict(config), 'seed': seed,
        'data_seed': seed + 1, 'depth_seed': seed + 2,
        'data': {'training': train_data, 'evaluation': eval_data,
                 'policy': policy},
        'tokenizer_sha256': hashlib.sha256(snapshot.encode()).hexdigest(),
        'code_sha256': {p: file_hash(p) for p in sources},
        'packages': {p: version(p) for p in ('torch', 'numpy', 'tiktoken')},
        'gpu': torch.cuda.get_device_name(device), 'cuda_runtime': torch.version.cuda,
        'precision': 'FP32 parameters and AdamW moments; BF16 autocast' if config.dtype == 'bfloat16' else 'FP32',
        'parameters': parameter_counts(model),
        'training_objective': training_objective,
        'checkpoint_selection': {'readouts': selection_readouts or ['final'], 'criterion': 'minimum supervised-token validation CE across declared readouts and checkpoints; policy list breaks ties; earlier checkpoint retained on ties', 'save_every_validation': save_validation_checkpoints},
        'learning_rate_schedule': 'linear warmup then cosine to 10% of peak; indexed by completed update count',
        'depth_sampling': 'uniform inclusive range per update' if model_family(config) == 'recurrent_depth' else 'fixed',
    }
    (out / 'research_run.json').write_text(json.dumps(record, indent=2) + '\n')
    order = torch.randperm(len(train_blocks), generator=data_rng)
    cursor, tokens_seen, applications_seen, supervised_seen, examples_seen, padded_seen = 0, 0, 0, 0, 0, 0
    depth_counts = Counter()
    train_seconds, best_loss = 0.0, float('inf')
    best_readout = 'final'
    best_by_policy = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def save(name, step, readout='final'):
        payload = {'model_family': model_family(config), 'config': asdict(config),
            'model_state_dict': model.state_dict(), 'optimizer_state_dicts': [optimizer.state_dict()],
            'step': step, 'tokens_seen': tokens_seen, 'config_identity': identity, 'readout_policy': readout,
            'runner': 'shared_research_v2', 'data_format': data_format, 'supervised_tokens_seen': supervised_seen, 'examples_seen': examples_seen, 'depth_counts': dict(depth_counts)}
        if recurrent_objective is not None:
            payload['training_objective'] = training_objective
        temporary = out / (name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(out / name)

    with (out / 'metrics.jsonl').open('w') as metrics:
        for step in range(1, config.max_steps + 1):
            if config.warmup_steps and step <= config.warmup_steps:
                factor = step / config.warmup_steps
            else:
                progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
                factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = config.learning_rate * factor
            depth = training_depth(config, depth_rng)
            depth_counts[depth] += 1
            if hasattr(model, '_train_step'):
                model._train_step.fill_(step - 1)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_tick_loss = None
            microbatches = []
            for _ in range(config.gradient_accumulation_steps):
                if cursor + config.batch_size > len(order):
                    order = torch.randperm(len(train_blocks), generator=data_rng)
                    cursor = 0
                microbatches.append(get_batch(train_blocks, order[cursor:cursor + config.batch_size], device))
                cursor += config.batch_size
            supervised_in_update = sum(int((targets != -100).sum()) for _, targets, _ in microbatches)
            for inputs, targets, real_tokens in microbatches:
                target_count = int((targets != -100).sum())
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.dtype == 'bfloat16'):
                    model_output = model(inputs, targets=targets, max_thought_steps=depth)
                    loss = model_output['loss']
                weight = target_count / supervised_in_update
                (loss * weight).backward()
                accumulated_loss += float(loss.detach()) * weight
                if 'per_tick_loss' in model_output:
                    tick_loss = model_output['per_tick_loss'].detach() * weight
                    accumulated_tick_loss = tick_loss if accumulated_tick_loss is None else accumulated_tick_loss + tick_loss
                tokens_seen += real_tokens
                padded_seen += inputs.numel()
                supervised_seen += target_count
                examples_seen += inputs.shape[0]
                applications_seen += training_applications(depth) * inputs.numel()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            train_seconds += elapsed
            row = {'step': step, 'tokens_seen': tokens_seen, 'supervised_tokens_seen': supervised_seen,
                   'examples_seen': examples_seen, 'padded_token_positions': padded_seen, 'loss': accumulated_loss,
                   'thought_steps': depth, 'block_applications_per_sequence': training_applications(depth),
                   'lr': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm), 'update_seconds': elapsed}
            if accumulated_tick_loss is not None:
                row['per_tick_supervised_ce'] = accumulated_tick_loss.float().tolist()
            if step % config.eval_interval == 0 or step == config.max_steps:
                if selection_readouts is None:
                    row['validation'] = evaluate_loss(model, eval_blocks, config, device)
                    selected_readout = 'final'
                else:
                    from ctm_transformer.readout_selection import evaluate_readouts
                    row['validation_readouts'] = evaluate_readouts(model, eval_blocks, config, device, selection_readouts)
                    selected_readout = min(selection_readouts, key=lambda p: row['validation_readouts'][p]['loss'])
                    row['validation'] = {**row['validation_readouts'][selected_readout], 'readout_policy': selected_readout}
                    for policy in selection_readouts:
                        score = row['validation_readouts'][policy]['loss']
                        if score < best_by_policy.get(policy, float('inf')):
                            best_by_policy[policy] = score
                            save('best_' + policy + '.pt', step, policy)
                    if save_validation_checkpoints:
                        save(f'step_{step:06d}.pt', step, selected_readout)
                if row['validation']['loss'] < best_loss:
                    best_loss = row['validation']['loss']
                    best_readout = selected_readout
                    save('best.pt', step, best_readout)
            metrics.write(json.dumps(row) + '\n')
            metrics.flush()
            if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
                print(json.dumps(row), flush=True)
        save('final.pt', config.max_steps)
    summary = {'complete': True, 'model_family': model_family(config), 'steps': config.max_steps,
        'tokens_seen': tokens_seen, 'supervised_tokens_seen': supervised_seen, 'examples_seen': examples_seen,
        'padded_token_positions': padded_seen, 'token_block_applications': applications_seen,
        'depth_counts': dict(depth_counts), 'training_seconds': train_seconds,
        'training_tokens_per_second': tokens_seen / train_seconds,
        'wall_seconds': time.perf_counter() - started, 'best_validation_loss': best_loss,
        'best_readout_policy': best_readout, 'best_validation_loss_by_readout': best_by_policy,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'parameters': parameter_counts(model), 'config_identity': identity}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
