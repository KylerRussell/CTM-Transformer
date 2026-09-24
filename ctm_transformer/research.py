"""Load fully specified, versioned research configurations."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from ctm_transformer.config import CTMConfig
from ctm_transformer.baselines import BaselineConfig


def model_family(config):
    return getattr(config, 'model_family', 'ctm')


def config_from_dict(values, family='ctm', require_all=False):
    if family not in {'ctm', 'transformer', 'recurrent_depth'}:
        raise ValueError(f'Unknown model family: {family}')
    cls = CTMConfig if family == 'ctm' else BaselineConfig
    known = {field.name for field in fields(cls)}
    missing, unknown = known - values.keys(), values.keys() - known
    if unknown or (require_all and missing):
        raise ValueError(f'Configuration must be fully specified with known fields; missing={sorted(missing)}, unknown={sorted(unknown)}')
    config = cls(**deepcopy(values))
    if model_family(config) != family:
        raise ValueError('Manifest/checkpoint family does not match config.model_family')
    return config


def build_model(config):
    if model_family(config) == 'ctm':
        from ctm_transformer.model import CTMTransformer
        return CTMTransformer(config)
    from ctm_transformer.baselines import BaselineTransformer
    return BaselineTransformer(config)


def parameter_counts(model):
    counts = {'total': 0, 'token_embedding': 0, 'position_embedding': 0, 'readout': 0, 'other': 0}
    for name, parameter in model.named_parameters():
        part = ('token_embedding' if name.startswith('token_embedding.') else
                'position_embedding' if name.startswith('pos_embedding.') else
                'readout' if name.startswith(('output_proj.', 'lm_head.')) else 'other')
        counts[part] += parameter.numel()
        counts['total'] += parameter.numel()
    return counts


def block_applications(config, depth):
    family = model_family(config)
    if family == 'transformer':
        return config.n_layers
    if family == 'recurrent_depth':
        return config.prelude_layers + config.core_layers * depth + config.coda_layers
    if config.use_hyperloop:
        raise ValueError('Research block accounting does not support Hyperloop')
    return config.n_layers * depth


def load_research_config(path):
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('schema_version') != 1 or not manifest.get('name'):
        raise ValueError('Research config requires schema_version=1 and a name')
    values = manifest.get('config', {})
    family = manifest.get('model_family', 'ctm')
    config = config_from_dict(values, family, require_all=True)
    if family == 'ctm':
        for name in ('vocab_size', 'd_model', 'd_latent', 'n_layers', 'n_heads', 'nlm_groups',
                     'history_len', 'max_thought_steps', 'seq_len', 'max_seq_len', 'batch_size'):
            value = getattr(config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if config.d_model % config.n_heads or config.d_latent % config.nlm_groups:
            raise ValueError('Head and temporal MLP group counts must divide their widths')
        if config.seq_len > config.max_seq_len:
            raise ValueError('seq_len exceeds max_seq_len')
    identity = {'name': manifest['name'], 'schema_version': 1,
                'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    return config, identity
