"""Small decoder and Geiping-style recurrent-depth research baselines.

Independent implementation; see research/BASELINES.md for the reference and
explicit adaptations. Inputs/targets are already shifted by the data loader.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class BaselineConfig:
    model_family: str = 'transformer'
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_hidden_dim: int = 768
    prelude_layers: int = 0
    core_layers: int = 0
    coda_layers: int = 0
    max_thought_steps: int = 1
    train_depth_min: int = 1
    train_depth_max: int = 1
    norm_eps: float = 1e-5
    init_std: float = 0.02
    dropout: float = 0.0
    use_positional_encoding: bool = True
    max_seq_len: int = 128
    tie_embeddings: bool = False
    gradient_checkpointing: bool = False
    batch_size: int = 4
    seq_len: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    optimizer: str = 'adamw'
    use_8bit_adam: bool = False
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    max_steps: int = 2000
    warmup_steps: int = 100
    eval_interval: int = 250
    log_interval: int = 25
    dtype: str = 'bfloat16'
    bf16_autocast: bool = True
    tokenizer: str = 'gpt2'
    dataset: str = ''
    data_path: str | None = None
    eval_data_path: str | None = None
    checkpoint_dir: str = 'checkpoints/baseline'
    device: str = 'cuda'

    def __post_init__(self):
        if self.model_family not in {'transformer', 'recurrent_depth'}:
            raise ValueError('Unknown baseline model_family')
        for name in ('vocab_size', 'd_model', 'n_heads', 'ffn_hidden_dim', 'max_thought_steps',
                     'train_depth_min', 'train_depth_max', 'max_seq_len', 'seq_len', 'batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('n_layers', 'prelude_layers', 'core_layers', 'coda_layers'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.d_model % self.n_heads or self.seq_len > self.max_seq_len:
            raise ValueError('Invalid head width or sequence window')
        if not 0 <= self.dropout < 1 or self.norm_eps <= 0 or self.init_std <= 0:
            raise ValueError('Invalid dropout, norm epsilon, or initialization scale')
        if self.train_depth_min > self.train_depth_max:
            raise ValueError('Invalid training depth range')
        if self.model_family == 'transformer':
            if self.n_layers < 1 or any((self.prelude_layers, self.core_layers, self.coda_layers)):
                raise ValueError('A standard Transformer uses only n_layers')
            if (self.max_thought_steps, self.train_depth_min, self.train_depth_max) != (1, 1, 1):
                raise ValueError('A standard Transformer has no thought-depth sweep; use T=1')
        elif self.n_layers != 0 or min(self.prelude_layers, self.core_layers, self.coda_layers) < 1:
            raise ValueError('Recurrent depth requires prelude/core/coda layers and n_layers=0')


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.dropout = config.dropout
        self.q = nn.Linear(config.d_model, config.d_model, bias=True)
        self.k = nn.Linear(config.d_model, config.d_model, bias=True)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        def split(projection):
            return projection(x).view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(split(self.q), split(self.k), split(self.v),
            is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, config, sandwich=False):
        super().__init__()
        norm = lambda: nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn_norm, self.ffn_norm = norm(), norm()
        self.attn_post = norm() if sandwich else nn.Identity()
        self.ffn_post = norm() if sandwich else nn.Identity()
        self.attn = CausalAttention(config)
        self.gate = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.attn_post(x + self.dropout(self.attn(self.attn_norm(x))))
        y = self.ffn_norm(x)
        return self.ffn_post(x + self.dropout(self.down(F.silu(self.gate(y)) * self.up(y))))


class BaselineTransformer(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = (nn.Embedding(config.max_seq_len, config.d_model)
                              if config.use_positional_encoding else None)
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        recurrent = config.model_family == 'recurrent_depth'
        def blocks(count):
            return nn.ModuleList(DecoderBlock(config, sandwich=recurrent) for _ in range(count))
        self.layers = blocks(config.n_layers)
        self.prelude = blocks(config.prelude_layers)
        self.core = blocks(config.core_layers)
        self.coda = blocks(config.coda_layers)
        self.injection = nn.Linear(2 * config.d_model, config.d_model, bias=False) if recurrent else None
        # As in the reference design, the recurrent output and coda share a norm.
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _blocks(self, blocks, x):
        for block in blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, input_ids, targets=None, max_thought_steps=None, return_all_logits=False):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError('Expected [batch, sequence] input within max_seq_len')
        depth = self.config.max_thought_steps if max_thought_steps is None else max_thought_steps
        if type(depth) is not int or depth < 1:
            raise ValueError('max_thought_steps must be a positive integer')
        if self.config.model_family == 'transformer' and depth != 1:
            raise ValueError('Standard Transformer depth is fixed; use T=1')
        x = self.token_embedding(input_ids)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(torch.arange(input_ids.shape[1], device=input_ids.device))
        tick_logits = []
        if self.injection is None:
            x = self._blocks(self.layers, x)
            applications = len(self.layers)
        else:
            embedded = self._blocks(self.prelude, x)
            # Deterministic zero state is an explicit small-scale adaptation.
            x = torch.zeros_like(embedded)
            for _ in range(depth):
                x = self.injection(torch.cat((x, embedded), dim=-1))
                x = self._blocks(self.core, x)
                if return_all_logits:
                    decoded = self._blocks(self.coda, self.final_norm(x))
                    tick_logits.append(self.lm_head(self.final_norm(decoded)))
            x = self.final_norm(x)
            x = self._blocks(self.coda, x)
            applications = len(self.prelude) + depth * len(self.core) + len(self.coda)
        logits = self.lm_head(self.final_norm(x))
        result = {'logits': logits, 'thought_steps': depth, 'block_applications': applications}
        if return_all_logits:
            result['all_logits'] = tick_logits if tick_logits else [logits]
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError('Targets must be shifted and have the same shape as input_ids')
            result['loss'] = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                             targets.reshape(-1), ignore_index=-100)
        return result
