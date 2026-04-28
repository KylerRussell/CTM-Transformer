"""
CTM-Transformer Training Script

Supports two data modes:
  1. Local text file:    --data_path path/to/file.txt
  2. FineWeb-Edu (HF):   --dataset fineweb-edu  (streams from HuggingFace)

Uses tiktoken GPT-2 BPE tokenizer. Supports AMD GPU (ROCm), gradient
checkpointing, temporal loss aggregation, and checkpoint resume.

Usage:
    # Pretrain on FineWeb-Edu (streaming, no disk needed)
    python -m ctm_transformer.train --dataset fineweb-edu --device cuda

    # Pretrain on FineWeb-Edu 10BT sample
    python -m ctm_transformer.train --dataset fineweb-edu --dataset_subset sample-10BT

    # Train on local text file
    python -m ctm_transformer.train --data_path path/to/text.txt --device cuda
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.adamuon import AdaMuon, build_param_groups
from ctm_transformer.engram import EngramTable


# ── Optimizer Construction ──────────────────────────────────────────────

def _parse_curriculum_stages(spec: str) -> list[tuple[int, float]]:
    """Parse a curriculum stages string like '2:0.30,4:0.60,8:1.00' into
    [(T, end_fraction), ...].

    Format: comma-separated pairs of `T:end_fraction`. Whitespace around
    items is ignored. Blank string returns [], which means "no curriculum
    even if --t_curriculum is set" (CTMConfig falls back to its default
    schedule via field default_factory).

    Validates that:
      - Each T is a positive integer.
      - Each end_fraction is in (0, 1].
      - end_fractions are strictly increasing (a curriculum that goes
        backward in fraction would be a config bug).
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    stages: list[tuple[int, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Bad curriculum stage '{chunk}' — expected 'T:end_fraction', "
                f"e.g. '2:0.30'."
            )
        t_str, frac_str = chunk.split(":", 1)
        try:
            T = int(t_str.strip())
            frac = float(frac_str.strip())
        except ValueError:
            raise ValueError(
                f"Bad curriculum stage '{chunk}' — T must be int, end_fraction "
                f"must be float."
            ) from None
        if T <= 0:
            raise ValueError(f"Curriculum stage T={T} must be positive.")
        if not (0 < frac <= 1):
            raise ValueError(
                f"Curriculum stage end_fraction={frac} must be in (0, 1]."
            )
        stages.append((T, frac))

    # Validate strictly increasing fractions
    for i in range(1, len(stages)):
        if stages[i][1] <= stages[i-1][1]:
            raise ValueError(
                f"Curriculum stage fractions must be strictly increasing: "
                f"got {stages[i-1][1]} then {stages[i][1]}."
            )

    return stages


def _engram_table_param_ids(model: torch.nn.Module) -> set[int]:
    """Return IDs of parameters owned (directly) by EngramTable modules.

    These are the giant lookup tables that get the 5× LR / no-weight-decay
    treatment per the Engram paper. Using id() identity rather than name
    matching so wrapping/renaming the engram_table attribute doesn't break.
    """
    ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, EngramTable):
            for p in m.parameters(recurse=True):
                ids.add(id(p))
    return ids


def _make_adamw(config: CTMConfig, params, **overrides):
    """Create an AdamW optimizer (or its 8-bit equivalent) from config.

    When `config.use_8bit_adam` is True, uses bitsandbytes' AdamW8bit
    (block-wise 8-bit quantization of the m and v moment buffers).
    Memory: ~1 byte per param vs 8 bytes for fp32 m+v. On a 200M model
    that's ~2.8 GB of VRAM recovered. Per the bitsandbytes paper and
    follow-up LLaMA-scale training work, no measurable accuracy
    degradation when used as a drop-in replacement.

    Falls back to torch.optim.AdamW if bitsandbytes is requested but
    not importable, with a loud warning — better to train at full
    precision than to silently fail.

    Args:
        config: CTMConfig (reads use_8bit_adam, betas, lr, wd defaults).
        params: parameter list or list of param-group dicts.
        **overrides: any AdamW kwarg to override (e.g. lr, weight_decay).

    Returns:
        Configured AdamW (or AdamW8bit) optimizer.
    """
    kwargs = dict(
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    kwargs.update(overrides)

    if not config.use_8bit_adam:
        return torch.optim.AdamW(params, **kwargs)

    # 8-bit AdamW path
    try:
        import bitsandbytes as bnb
    except ImportError:
        # Warn once per process — calling build_optimizers multiple times
        # (e.g. in test loops or interactive sessions) shouldn't spam.
        if not getattr(_make_adamw, "_bnb_warned", False):
            print(
                "WARNING: --use_8bit_adam set but `bitsandbytes` is not installed. "
                "Falling back to standard fp32 AdamW. Install with:\n"
                "    pip install bitsandbytes",
                file=sys.stderr,
            )
            _make_adamw._bnb_warned = True
        return torch.optim.AdamW(params, **kwargs)

    # bitsandbytes' AdamW8bit signature matches torch.optim.AdamW exactly,
    # so the same kwargs flow through. The block-wise quantization is
    # automatic; no extra config needed.
    return bnb.optim.AdamW8bit(params, **kwargs)


def build_optimizers(model: torch.nn.Module, config: CTMConfig) -> list[torch.optim.Optimizer]:
    """
    Build the optimizer list for training.

    Returns a list of one or two optimizers:
      * "adamw"   → [AdamW] over all parameters. When Engram is on, the
                    Engram tables get their own param group with 5× LR
                    (`engram_lr_mult`) and no weight decay.
      * "adamuon" → [AdaMuon, AdamW] split via adamuon.build_param_groups
                    (2D hidden weights to AdaMuon; 1D / embeddings / LM-head
                    / >2D NLM stacks / Engram tables to AdamW). The AdamW
                    optimizer holds two internal param groups when Engram
                    is on: a "general" group at base LR, and an "engram"
                    group at base LR × engram_lr_mult with weight_decay=0.

    Per-group LR scheduling: each param_group carries a custom `lr_mult`
    field. The training loop multiplies the base LR by this when applying
    the schedule (see train()).

    When `config.use_8bit_adam` is True, the AdamW path(s) above use
    bitsandbytes' AdamW8bit. AdaMuon stays in fp32 — it's already cheap
    on memory (its V_t buffer mirrors the param shape only, no fp32 Adam
    moments to compress) and bitsandbytes doesn't ship a Muon-shaped
    8-bit variant.
    """
    engram_ids = _engram_table_param_ids(model) if config.use_engram else set()

    if config.optimizer == "adamw":
        if engram_ids:
            general_p = []
            engram_p = []
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                (engram_p if id(p) in engram_ids else general_p).append(p)
            groups: list[dict] = [
                {"params": general_p, "lr_mult": 1.0},
            ]
            if engram_p:
                groups.append({
                    "params": engram_p,
                    "lr_mult": config.engram_lr_mult,
                    "weight_decay": config.engram_weight_decay,
                })
            opt = _make_adamw(config, groups)
        else:
            opt = _make_adamw(config, model.parameters())
        return [opt]

    if config.optimizer == "adamuon":
        muon_params, adamw_params = build_param_groups(model, verbose=False)

        # Split the AdamW pool into general + engram-table groups so we can
        # apply the paper's 5× LR / no-WD rule to the lookup tables only.
        if engram_ids:
            general_p = [p for p in adamw_params if id(p) not in engram_ids]
            engram_p = [p for p in adamw_params if id(p) in engram_ids]
        else:
            general_p, engram_p = adamw_params, []

        opts: list[torch.optim.Optimizer] = []
        if muon_params:
            opts.append(AdaMuon(
                muon_params,
                lr=config.learning_rate,
                weight_decay=config.adamuon_weight_decay,
                beta=config.adamuon_beta,
                eps=config.adamuon_eps,
                ns_steps=config.adamuon_ns_steps,
                rms_target=config.adamuon_rms_target,
            ))
            # AdaMuon's single param-group wants an lr_mult so the scheduler
            # treats it uniformly with everything else.
            opts[-1].param_groups[0]["lr_mult"] = 1.0

        adamw_groups: list[dict] = []
        if general_p:
            adamw_groups.append({"params": general_p, "lr_mult": 1.0})
        if engram_p:
            adamw_groups.append({
                "params": engram_p,
                "lr_mult": config.engram_lr_mult,
                "weight_decay": config.engram_weight_decay,
            })
        if adamw_groups:
            # Note the weight_decay override: when running with --optimizer adamuon,
            # the global default WD comes from `adamuon_weight_decay` (paper-spec
            # 0.1), not the standalone-AdamW `weight_decay` field.
            opts.append(_make_adamw(config, adamw_groups,
                                    weight_decay=config.adamuon_weight_decay))

        if not opts:
            raise RuntimeError("build_param_groups returned no parameters.")
        return opts

    raise ValueError(f"Unknown optimizer: {config.optimizer!r} (use 'adamw' or 'adamuon')")


# ── Tokenizer ───────────────────────────────────────────────────────────

def get_tokenizer(config: CTMConfig):
    """
    Load a tiktoken BPE tokenizer.

    Returns:
        tokenizer: tiktoken.Encoding object with encode/decode methods.
    """
    import tiktoken
    enc = tiktoken.get_encoding(config.tokenizer)
    return enc


def tokenize_text(text: str, tokenizer) -> np.ndarray:
    """Tokenize a string into a numpy array of token IDs."""
    tokens = tokenizer.encode(text, allowed_special=set())
    return np.array(tokens, dtype=np.int32)


# ── Datasets ────────────────────────────────────────────────────────────

class TokenizedTextDataset(Dataset):
    """
    Tokenized text dataset from a local file.

    Produces fixed-length chunks of token sequences for causal LM training.
    Each sample is a contiguous chunk of (seq_len + 1) tokens:
    - input:  tokens[i : i + seq_len]
    - target: tokens[i + 1 : i + seq_len + 1]
    """

    def __init__(self, token_ids: np.ndarray, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


class FineWebEduDataset(IterableDataset):
    """
    Streaming dataset from HuggingFace FineWeb-Edu.

    Streams documents from the dataset, tokenizes them on the fly, and
    packs them into fixed-length sequences. Supports multi-worker data
    loading via automatic sharding.

    Uses the same proven pattern as OpenMythos training scripts.

    Args:
        tokenizer: tiktoken encoding for BPE tokenization.
        seq_len: Context window size.
        subset: FineWeb-Edu subset name (e.g. "sample-10BT", "sample-100BT",
                "sample-350BT", or "default" for the full ~1.4T token dataset).
        rank: Process rank for distributed training (0 for single GPU).
        world_size: Total number of processes (1 for single GPU).
    """

    def __init__(self, tokenizer, seq_len: int, subset: str = "sample-10BT",
                 rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        from datasets import load_dataset

        # Multi-worker sharding: each DataLoader worker gets a unique shard
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        # Pack documents into fixed-length chunks
        buf = []
        for sample in ds:
            tokens = self.tokenizer.encode(sample["text"], allowed_special=set())
            buf.extend(tokens)

            # Yield complete chunks as they fill up
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ── Learning Rate Schedule ──────────────────────────────────────────────

def get_lr(step: int, config: CTMConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)

    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpoint Management ──────────────────────────────────────────────

def save_checkpoint(model, optimizers, step, config, path, keep_last=3):
    """
    Save model checkpoint with rotation.

    `optimizers` is a list — supports both single-optimizer (AdamW) and
    dual-optimizer (AdaMuon + AdamW) setups. State dicts are stored under
    `optimizer_state_dicts` (a list); the legacy single-optimizer key
    `optimizer_state_dict` is also written for backward-compatibility with
    older checkpoints loaded by external scripts.
    """
    ckpt_dir = path.parent
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [o.state_dict() for o in optimizers],
        "step": step,
        "config": vars(config),
    }
    # Backward-compat: also write the single-opt key when there's only one.
    if len(optimizers) == 1:
        payload["optimizer_state_dict"] = optimizers[0].state_dict()
    torch.save(payload, path)

    # Rotate old step checkpoints (keep last N)
    step_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in step_ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def load_checkpoint(model, optimizers, path, device):
    """
    Load model checkpoint, return the step number.

    `optimizers` is a list. Loads in order from `optimizer_state_dicts` if
    present; otherwise falls back to the legacy single-`optimizer_state_dict`
    key (which only works when len(optimizers) == 1).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizers is not None:
        if "optimizer_state_dicts" in ckpt:
            states = ckpt["optimizer_state_dicts"]
            if len(states) != len(optimizers):
                raise RuntimeError(
                    f"Checkpoint has {len(states)} optimizer state(s) but "
                    f"current run has {len(optimizers)}. Did you switch "
                    f"--optimizer between runs?"
                )
            for o, s in zip(optimizers, states):
                o.load_state_dict(s)
        elif "optimizer_state_dict" in ckpt:
            # Legacy single-optimizer checkpoint
            if len(optimizers) != 1:
                raise RuntimeError(
                    "Legacy single-optimizer checkpoint cannot be loaded "
                    "into a multi-optimizer (AdaMuon) run."
                )
            optimizers[0].load_state_dict(ckpt["optimizer_state_dict"])

    return int(ckpt.get("step", 0))


# ── Training Loop ───────────────────────────────────────────────────────

def train(config: CTMConfig):
    """Main training loop."""
    device = config.resolve_device()
    print(f"Device: {device}")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = get_tokenizer(config)
    print(f"Tokenizer: {config.tokenizer} (vocab_size={tokenizer.n_vocab:,})")

    # Sync vocab size with tokenizer
    if config.vocab_size != tokenizer.n_vocab:
        print(f"  Updating config.vocab_size: {config.vocab_size} → {tokenizer.n_vocab}")
        config.vocab_size = tokenizer.n_vocab

    # ── Data Loading ────────────────────────────────────────────────────
    use_fineweb = config.dataset == "fineweb-edu"

    if use_fineweb:
        # Streaming from HuggingFace — no disk space needed
        print(f"Dataset: HuggingFaceFW/fineweb-edu ({config.dataset_subset})")
        print(f"  Streaming mode — data loaded on the fly from HuggingFace Hub")

        train_dataset = FineWebEduDataset(
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            subset=config.dataset_subset,
            rank=0,
            world_size=1,
        )
        eval_dataset = FineWebEduDataset(
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            subset=config.dataset_subset,
            rank=0,
            world_size=1,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=(device != "cpu"),
            persistent_workers=True,
        )
        # For eval, use a small separate stream
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            num_workers=1,
            pin_memory=(device != "cpu"),
        )
        raw_text = None  # No raw text buffer for generation prompts

    elif config.data_path is not None:
        # Load from local text file
        data_path = Path(config.data_path)
        if not data_path.exists():
            print(f"Error: data file not found: {config.data_path}")
            sys.exit(1)
        raw_text = data_path.read_text(encoding="utf-8", errors="replace")
        print(f"Loaded {len(raw_text):,} characters from {config.data_path}")

        all_tokens = tokenize_text(raw_text, tokenizer)
        print(f"Tokenized: {len(all_tokens):,} tokens "
              f"(compression ratio: {len(raw_text)/len(all_tokens):.1f}x)")

        if config.eval_data_path:
            eval_text = Path(config.eval_data_path).read_text(encoding="utf-8", errors="replace")
            eval_tokens = tokenize_text(eval_text, tokenizer)
            train_tokens = all_tokens
        else:
            split = int(len(all_tokens) * 0.9)
            train_tokens = all_tokens[:split]
            eval_tokens = all_tokens[split:]

        train_dataset = TokenizedTextDataset(train_tokens, config.seq_len)
        eval_dataset = TokenizedTextDataset(eval_tokens, config.seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device != "cpu"),
            drop_last=True,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        print(f"Train: {len(train_dataset):,} samples, Eval: {len(eval_dataset):,} samples")

    else:
        # Synthetic data for smoke testing
        print("No data source specified. Using synthetic data for smoke test.")
        raw_text = "The quick brown fox jumps over the lazy dog. " * 1000
        all_tokens = tokenize_text(raw_text, tokenizer)
        split = int(len(all_tokens) * 0.9)

        train_dataset = TokenizedTextDataset(all_tokens[:split], config.seq_len)
        eval_dataset = TokenizedTextDataset(all_tokens[split:], config.seq_len)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size,
                                 shuffle=False, drop_last=True)

    # ── Model ───────────────────────────────────────────────────────────
    model = CTMTransformer(config).to(device)

    # Mixed precision strategy
    #
    # We use the "manual cast" pattern (cast the entire model to the target
    # low-precision dtype, run the forward/backward natively in that dtype)
    # rather than the "autocast" pattern (keep the model in fp32 and let
    # torch.amp.autocast pick which ops run in bf16/fp16).
    #
    # Why manual cast over autocast:
    #   1. Activation memory is half-precision throughout — autocast with
    #      an fp32 model still keeps the param-shaped tensors in fp32, so
    #      gradient checkpointing's recomputation re-allocates fp32 buffers.
    #   2. Autocast forces certain ops (RMSNorm, LayerNorm, softmax) back
    #      to fp32 even when the surrounding tensors are bf16. That's the
    #      source of the "Mismatch dtype between input and weight" warning
    #      seen in mixed-mode runs: autocast upcasts the RMSNorm input to
    #      fp32 while the weight is still the manually-cast bf16 from
    #      `model.to(torch.bfloat16)` — the fused kernel can't dispatch
    #      and falls back to a slower path.
    #   3. The CTM thought loop is the dominant compute; bf16 throughout
    #      is stable in practice for transformer-shaped models.
    #
    # Setting amp_dtype = torch.float32 means the autocast context further
    # down is enabled=False (a no-op), avoiding the fused-kernel warning.
    if config.dtype == "bfloat16" and device != "cpu":
        model = model.to(torch.bfloat16)
        amp_dtype = torch.float32   # autocast disabled; model is already bf16
    elif config.dtype == "float16" and device != "cpu":
        model = model.to(torch.float16)
        amp_dtype = torch.float32   # autocast disabled; model is already fp16
    else:
        amp_dtype = torch.float32

    n_params = model.get_num_params()
    print(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"Config: d_model={config.d_model}, d_latent={config.d_latent}, "
          f"n_layers={config.n_layers}, n_heads={config.n_heads}, "
          f"thought_steps={config.max_thought_steps}, history_len={config.history_len}, "
          f"nlm_hidden={config.nlm_hidden_dim}, nlm_groups={config.nlm_groups}")

    # Temporal-loss schedule summary. Helps verify the decay plan matches
    # expectations before kicking off a multi-day run.
    base_mono = config.mono_penalty_weight
    if config.mono_penalty_decay_until_frac > 0 and base_mono > 0:
        decay_step = int(config.mono_penalty_decay_until_frac * config.max_steps)
        floor_mono = base_mono * config.mono_penalty_min_frac
        print(f"Temporal loss: ramp[{config.tick_ramp_start}→{config.tick_ramp_end}], "
              f"mono_penalty {base_mono} → {floor_mono:.3f} over first "
              f"{decay_step:,} steps ({config.mono_penalty_decay_until_frac:.0%} of training)")
    else:
        print(f"Temporal loss: ramp[{config.tick_ramp_start}→{config.tick_ramp_end}], "
              f"mono_penalty {base_mono} (no decay)")

    # Curriculum schedule summary. Catches misconfiguration before launch.
    if config.t_curriculum:
        if not config.t_curriculum_stages:
            raise ValueError(
                "--t_curriculum is set but the stages list is empty. "
                "Either disable the flag or provide --t_curriculum_stages."
            )
        final_T = config.t_curriculum_stages[-1][0]
        if final_T != config.max_thought_steps:
            raise ValueError(
                f"Curriculum's final stage T={final_T} doesn't match "
                f"--max_thought_steps={config.max_thought_steps}. The curriculum "
                f"should end with the model trained at its target depth, and the "
                f"per-tick adapter list is sized at construction time. Either "
                f"set --max_thought_steps {final_T} or change the final stage."
            )
        print(f"Thought-step curriculum:")
        prev_frac = 0.0
        for T_val, end_frac in config.t_curriculum_stages:
            start_step = int(prev_frac * config.max_steps)
            end_step = int(end_frac * config.max_steps)
            ckpt_active = (
                config.gradient_checkpointing
                and T_val >= config.gradient_checkpointing_min_T
            )
            ckpt_label = "checkpointed" if ckpt_active else "uncheckpointed (faster)"
            print(f"  T={T_val} from step {start_step:>10,} → {end_step:>10,} "
                  f"({prev_frac:.0%} → {end_frac:.0%}) — {ckpt_label}")
            prev_frac = end_frac
    else:
        ckpt_active = (
            config.gradient_checkpointing
            and config.max_thought_steps >= config.gradient_checkpointing_min_T
        )
        ckpt_label = "checkpointed" if ckpt_active else "uncheckpointed"
        print(f"Thought-step curriculum: disabled "
              f"(T={config.max_thought_steps} fixed, {ckpt_label})")

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizers = build_optimizers(model, config)

    def _count_group(g):
        return sum(p.numel() for p in g["params"])

    def _adam_label() -> str:
        """Show which AdamW variant is actually in use."""
        if not config.use_8bit_adam:
            return "AdamW (fp32)"
        # Detect whether the requested 8-bit version actually loaded
        # (could have fallen back to fp32 in _make_adamw on ImportError).
        # opts[-1] is the AdamW companion when present, opts[0] otherwise.
        last = optimizers[-1] if config.optimizer == "adamuon" else optimizers[0]
        cls = type(last).__name__
        return "AdamW8bit (bnb)" if cls == "AdamW8bit" else "AdamW (fp32, 8bit fallback)"

    if len(optimizers) == 1:
        print(f"Optimizer: {config.optimizer} → {_adam_label()} "
              f"({len(optimizers[0].param_groups)} param group(s))")
        for i, g in enumerate(optimizers[0].param_groups):
            mult = g.get("lr_mult", 1.0)
            tag = "engram" if mult != 1.0 else "general"
            print(f"  [{tag}] {_count_group(g)/1e6:.1f}M params, lr_mult={mult}")
    else:
        muon_n = sum(_count_group(g) for g in optimizers[0].param_groups)
        adam_n = sum(_count_group(g) for g in optimizers[1].param_groups)
        print(f"Optimizer: adamuon — {muon_n/1e6:.1f}M params on AdaMuon, "
              f"{adam_n/1e6:.1f}M params on {_adam_label()} (embeddings/LM-head/1D/Engram)")
        for i, g in enumerate(optimizers[1].param_groups):
            mult = g.get("lr_mult", 1.0)
            tag = "engram" if mult != 1.0 else "general"
            print(f"  AdamW[{tag}] {_count_group(g)/1e6:.1f}M params, lr_mult={mult}")

    # ── Resume from checkpoint ──────────────────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    start_step = 0

    existing_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if existing_ckpts:
        print(f"  Resuming from {existing_ckpts[-1]}")
        start_step = load_checkpoint(model, optimizers, existing_ckpts[-1], device)
        print(f"  Resumed at step {start_step}")
    elif (ckpt_dir / "best.pt").exists():
        print(f"  Loading best checkpoint")
        start_step = load_checkpoint(model, optimizers, ckpt_dir / "best.pt", device)

    # ── Training ────────────────────────────────────────────────────────
    step = start_step
    best_eval_loss = float("inf")
    log_losses = []
    tokens_seen = step * config.batch_size * config.seq_len

    print(f"\n{'='*70}")
    print(f"Starting training for {config.max_steps:,} steps")
    if use_fineweb:
        target_tokens = config.max_steps * config.batch_size * config.seq_len
        print(f"Target: ~{target_tokens/1e9:.1f}B tokens from FineWeb-Edu")
    print(f"{'='*70}\n")

    model.train()
    train_iter = iter(train_loader)
    t_start = time.time()

    # Gradient accumulation: we run `accum_steps` micro-batches per optimizer
    # step, scaling each micro-batch's loss by 1/accum_steps so the gradient
    # we eventually apply is the *mean* over the effective batch (matching
    # the semantics of a single forward at batch_size = batch_size·accum_steps).
    # `step` remains the OPTIMIZER step counter — max_steps, LR schedule, eval
    # and log intervals are all calibrated against it. Throughput is reported
    # as effective tokens/sec (counting all micro-batches).
    accum_steps = max(1, config.gradient_accumulation_steps)

    # Track the previous T for curriculum-transition detection. We print
    # a clear banner when T changes so transitions are easy to spot in
    # long log files.
    prev_T = None

    while step < config.max_steps:
        # Learning rate schedule — applied once per OPTIMIZER step (not per
        # micro-batch). Each param_group carries an `lr_mult` (default 1.0)
        # which the scheduler multiplies by the base LR — used to give the
        # Engram embedding tables their 5× LR per the Engram paper.
        lr = get_lr(step, config)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = lr * param_group.get("lr_mult", 1.0)

        # Sync the model's _train_step buffer so loss-side schedules
        # (mono_penalty decay) can be progress-aware. Cheap (one int
        # write per optimizer step) and keeps schedules deterministic
        # under checkpoint resumes — `step` itself is already restored
        # from the checkpoint via load_checkpoint.
        model._train_step.fill_(step)

        # Resolve curriculum T for this step. When t_curriculum is off,
        # this returns config.max_thought_steps every time — same as the
        # pre-curriculum behavior. When on, walks the stages list and
        # picks the appropriate T.
        current_T = config.resolve_thought_steps(step)

        # Announce curriculum transitions. Easy to grep in long logs.
        if config.t_curriculum and prev_T is not None and current_T != prev_T:
            print(
                f"\n{'='*70}\n"
                f"  CURRICULUM TRANSITION at step {step}: T = {prev_T} → {current_T}\n"
                f"  (the next {current_T - prev_T} per-tick adapter slot(s) start "
                f"learning now)\n"
                f"{'='*70}\n",
                flush=True,
            )
        prev_T = current_T

        # Zero grads once per optimizer step, before the accumulation loop.
        for o in optimizers:
            o.zero_grad(set_to_none=True)

        t0 = time.time()
        device_type = device.split(":")[0] if ":" in device else device

        # Accumulate gradients over `accum_steps` micro-batches.
        accum_loss = 0.0
        last_result = None
        for micro in range(accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type=device_type,
                                    dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                result = model(x, targets=y, max_thought_steps=current_T)
                # Scale the loss so the accumulated gradient is the MEAN over
                # the effective batch — matches what a single forward pass at
                # batch_size = batch_size·accum_steps would compute.
                loss = result["loss"] / accum_steps

            loss.backward()
            accum_loss += loss.item() * accum_steps   # un-scale for logging
            last_result = result   # keep last for cert/tick logging

        # Average loss across the accumulation window for logging
        loss_for_log = accum_loss / accum_steps

        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        for o in optimizers:
            o.step()

        dt = time.time() - t0
        # Effective batch tokens accounts for accumulation
        batch_tokens = config.batch_size * config.seq_len * accum_steps
        tokens_seen += batch_tokens
        log_losses.append(loss_for_log)
        # Use last_result for tick / certainty logging (representative of the
        # final micro-batch — over the accumulation window these are close
        # enough that picking one is fine for monitoring)
        result = last_result

        # ── Logging ─────────────────────────────────────────────────────
        if step % config.log_interval == 0:
            window = log_losses[-config.log_interval:]
            avg_loss = sum(window) / len(window)
            tick_losses = result["per_tick_loss"].tolist()

            # Δticks = last_tick_loss - first_tick_loss
            # Negative = the model improves across thought steps (good — iterative
            #            refinement is working). Magnitude is the per-token CE
            #            improvement from spending more thought.
            # ~0      = thought loop produces identical output at every step
            #            (degenerate — architecture isn't using its iterative
            #            capacity).
            # Positive = later ticks are WORSE than earlier ticks (regression —
            #            either training instability or model is "thinking
            #            itself wrong" past some optimal step).
            tick_delta = tick_losses[-1] - tick_losses[0] if len(tick_losses) > 1 else 0.0

            cert = result["certainties"]
            cert_first = cert[0].mean().item()
            cert_last = cert[-1].mean().item()
            tok_per_sec = batch_tokens / max(dt, 1e-6)

            elapsed = time.time() - t_start
            eta_sec = (config.max_steps - step) * (elapsed / max(step - start_step, 1))

            # T= field is only useful when curriculum is varying it.
            # Otherwise it's redundant — the ticks list length is the same
            # info — and adds visual noise to the log.
            t_field = f"T={current_T} | " if config.t_curriculum else ""

            print(
                f"Step {step:6d} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{tok_per_sec/1e3:.1f}k tok/s | "
                f"{t_field}"
                f"cert {cert_first:.2f}→{cert_last:.2f} | "
                f"ticks [{' '.join(f'{l:.3f}' for l in tick_losses)}] | "
                f"Δticks {tick_delta:+.3f} | "
                f"{tokens_seen/1e6:.0f}M tok | "
                f"ETA {eta_sec/3600:.1f}h"
            )

        # ── Evaluation ──────────────────────────────────────────────────
        if step > 0 and step % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device, config, amp_dtype)
            print(f"\n  >>> Eval loss: {eval_loss:.4f} (best: {best_eval_loss:.4f})")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(model, optimizers, step, config, ckpt_dir / "best.pt")
                print(f"  >>> Saved best checkpoint at step {step}")

            # Save step checkpoint
            save_checkpoint(model, optimizers, step, config,
                          ckpt_dir / f"step_{step:07d}.pt")

            # Generate a sample
            generate_sample(model, tokenizer, device, config, raw_text)
            print()
            model.train()

        step += 1

    # Final save
    save_checkpoint(model, optimizers, step, config, ckpt_dir / "final.pt")
    print(f"\nTraining complete. {tokens_seen/1e9:.2f}B tokens processed.")


@torch.no_grad()
def evaluate(model, eval_loader, device, config, amp_dtype):
    """Run evaluation and return average loss.

    Uses the curriculum-resolved T (from the model's current `_train_step`)
    rather than the full `max_thought_steps`. This is the operationally
    honest "what can the model do right now" — at early curriculum phases
    the late per-tick adapters are random init and would produce garbage
    if invoked. As training progresses past the final stage, eval uses
    full T anyway because resolve_thought_steps returns max_thought_steps.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    max_eval_batches = 50
    device_type = device.split(":")[0] if ":" in device else device

    # Read current curriculum T. For non-curriculum runs this is just
    # max_thought_steps. The model was already _train_step.fill_'d by
    # the caller before evaluate was invoked.
    current_T = config.resolve_thought_steps(int(model._train_step.item()))

    for x, y in eval_loader:
        if n_batches >= max_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast(device_type=device_type,
                                dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            result = model(x, targets=y, max_thought_steps=current_T)

        total_loss += result["loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def generate_sample(model, tokenizer, device, config, raw_text=None):
    """Generate and print a short sample from the model."""
    model.eval()

    # Build a prompt — either from training data or a fixed string
    if raw_text and len(raw_text) > 200:
        start = random.randint(0, len(raw_text) - 200)
        prompt_text = raw_text[start : start + 80]
    else:
        prompt_text = "The most important thing to understand about science is"

    prompt_tokens = tokenizer.encode(prompt_text, allowed_special=set())
    prompt_tokens = prompt_tokens[:32]  # Keep prompt short
    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    generated = model.generate(
        prompt_ids,
        max_new_tokens=48,
        temperature=0.8,
        top_k=40,
    )

    gen_tokens = generated[0].tolist()
    prompt_decoded = tokenizer.decode(prompt_tokens)
    gen_decoded = tokenizer.decode(gen_tokens[len(prompt_tokens):])

    print(f"  >>> Sample: '{prompt_decoded}' → '{gen_decoded}'")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CTM-Transformer Training")

    # Data source (mutually exclusive: local file vs HF dataset)
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_path", type=str, default=None,
                           help="Path to local training text file")
    data_group.add_argument("--eval_data_path", type=str, default=None,
                           help="Path to local eval text file")
    data_group.add_argument("--dataset", type=str, default=None,
                           choices=["fineweb-edu"],
                           help="HuggingFace dataset to stream (e.g. 'fineweb-edu')")
    data_group.add_argument("--dataset_subset", type=str, default="sample-10BT",
                           help="FineWeb-Edu subset: sample-10BT, sample-100BT, sample-350BT, default")
    data_group.add_argument("--tokenizer", type=str, default="gpt2",
                           help="Tiktoken encoding name (gpt2, r50k_base, cl100k_base)")

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--d_model", type=int, default=512)
    model_group.add_argument("--d_latent", type=int, default=512)
    model_group.add_argument("--n_heads", type=int, default=8)
    model_group.add_argument("--n_layers", type=int, default=4)
    model_group.add_argument("--nlm_hidden_dim", type=int, default=32)
    model_group.add_argument("--nlm_groups", type=int, default=1)
    model_group.add_argument("--history_len", type=int, default=8)
    model_group.add_argument("--max_thought_steps", type=int, default=8)
    model_group.add_argument("--seq_len", type=int, default=512)
    model_group.add_argument("--sync_method", type=str, default="diag_summary",
                            choices=["full", "diag_summary", "low_rank"])
    model_group.add_argument("--per_tick_heads", action="store_true",
                            help="Give each thought tick its own output adapter feeding into a "
                                 "shared LM head. Removes gradient interference between ticks "
                                 "and empirically prevents the iterative-refinement collapse "
                                 "where the model learns to produce identical output at every "
                                 "tick. Cost: ~1%% extra params (T copies of a small adapter).")

    # Thought-step curriculum
    curr_group = parser.add_argument_group("Thought-step curriculum")
    curr_group.add_argument("--t_curriculum", action="store_true",
                            help="Enable curriculum on thought-step depth: gradually increase T "
                                 "from low values early in training to max_thought_steps near "
                                 "the end. Concentrates gradient signal early so the thought "
                                 "loop learns meaningful refinement before being asked to do "
                                 "deep multi-step reasoning. Empirically helps avoid the "
                                 "'all ticks identical' collapse mode.")
    curr_group.add_argument("--t_curriculum_stages", type=str, default="2:0.30,4:0.60,8:1.00",
                            help="Curriculum stage definitions as 'T:end_frac,T:end_frac,...'. "
                                 "Each stage: T value to use until the given fraction of "
                                 "max_steps is reached. Default: '2:0.30,4:0.60,8:1.00' = "
                                 "T=2 for first 30%%, T=4 for next 30%%, T=8 for final 40%%. "
                                 "Final stage's T must equal --max_thought_steps.")

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--learning_rate", type=float, default=3e-4)
    train_group.add_argument("--max_steps", type=int, default=100000)
    train_group.add_argument("--warmup_steps", type=int, default=1000)
    train_group.add_argument("--device", type=str, default="auto")
    train_group.add_argument("--dtype", type=str, default="bfloat16",
                            choices=["float32", "float16", "bfloat16"])
    train_group.add_argument("--gradient_checkpointing", action="store_true", default=True)
    train_group.add_argument("--no_gradient_checkpointing", action="store_true")
    train_group.add_argument("--gradient_checkpointing_min_T", type=int, default=5,
                             help="Minimum thought-step T at which checkpointing actually "
                                  "activates. T below this threshold skips checkpointing "
                                  "for the speedup, since the activation graph fits in VRAM "
                                  "without recomputation. Default 5 means T=2 and T=4 "
                                  "phases run uncheckpointed, T=5+ activates checkpointing. "
                                  "Lower this to 3 if you OOM at T=4 uncheckpointed; raise "
                                  "to 9 if you have VRAM headroom at T=8 and want max speed.")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                             help="Number of micro-batches per optimizer step. Effective batch = "
                                  "batch_size × this. Use to fit larger effective batches in "
                                  "limited VRAM (each micro-batch's forward/backward is one "
                                  "batch_size, but gradients accumulate before stepping).")
    train_group.add_argument("--eval_interval", type=int, default=500)
    train_group.add_argument("--log_interval", type=int, default=50)
    train_group.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Optimizer
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument("--optimizer", type=str, default="adamw",
                           choices=["adamw", "adamuon"],
                           help="Optimizer: 'adamw' (default) or 'adamuon' (sign-stabilized "
                                "Muon with element-wise V_t and RMS alignment).")
    opt_group.add_argument("--adamuon_beta", type=float, default=0.95,
                           help="Shared β for AdaMuon's first/second momentum (paper default 0.95).")
    opt_group.add_argument("--adamuon_eps", type=float, default=1e-8)
    opt_group.add_argument("--adamuon_ns_steps", type=int, default=5,
                           help="Newton-Schulz iterations for the polar factor.")
    opt_group.add_argument("--adamuon_rms_target", type=float, default=0.2,
                           help="Target update RMS after alignment (matches Adam's empirical norm).")
    opt_group.add_argument("--adamuon_weight_decay", type=float, default=0.1,
                           help="Weight decay for both AdaMuon and its AdamW companion (paper uses 0.1).")
    opt_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay for plain --optimizer adamw (unused under adamuon).")
    opt_group.add_argument("--use_8bit_adam", action="store_true",
                           help="Use bitsandbytes' AdamW8bit for the AdamW optimizer(s). "
                                "Cuts optimizer state memory ~8x with no measurable accuracy "
                                "loss. Affects AdamW only; AdaMuon stays fp32. Requires "
                                "`pip install bitsandbytes`.")

    # Engram (conditional memory)
    engram_group = parser.add_argument_group("Engram")
    engram_group.add_argument("--use_engram", action="store_true",
                              help="Enable Engram conditional memory (hashed N-gram lookup).")
    engram_group.add_argument("--engram_ngram_orders", type=int, nargs="+", default=[2, 3],
                              help="N-gram orders to track (paper recommends [2, 3]).")
    engram_group.add_argument("--engram_n_heads", type=int, default=8,
                              help="K independent hash heads per order.")
    engram_group.add_argument("--engram_slots_per_table", type=int, default=65521,
                              help="M, slots per (order, head) table. Prime preferred. "
                                   "Default 65521 = largest prime ≤ 2^16. "
                                   "Total table params = len(orders) × n_heads × slots × d_head.")
    engram_group.add_argument("--engram_d_head", type=int, default=64,
                              help="Embedding dim per head (d_mem = orders × heads × d_head).")
    engram_group.add_argument("--engram_layers", type=int, nargs="*", default=[],
                              help="Layer indices where Engram fuses. Empty = auto: "
                                   "{1, n_layers // 2} for n_layers ≥ 4, else {0}.")
    engram_group.add_argument("--engram_no_conv", action="store_true",
                              help="Skip the depthwise causal conv refinement (slight loss "
                                   "per Fig 5 ablation; ~30%% fewer Engram fusion params).")
    engram_group.add_argument("--engram_conv_kernel", type=int, default=4)
    engram_group.add_argument("--engram_conv_dilation", type=int, default=3)
    engram_group.add_argument("--engram_lr_mult", type=float, default=5.0,
                              help="LR multiplier for Engram embedding tables (paper: 5×).")
    engram_group.add_argument("--engram_weight_decay", type=float, default=0.0,
                              help="Weight decay for Engram tables (paper: 0).")

    # Ternary weight quantization (TWN)
    tern_group = parser.add_argument_group("Ternary")
    tern_group.add_argument("--use_ternary", action="store_true",
                            help="Replace backbone nn.Linears with TernaryLinear "
                                 "(weights ∈ {-α, 0, +α} during forward, fp32 latent "
                                 "weight + STE backward). Excludes token_embedding, "
                                 "LM head, NLM stacks. Training is ~10-30%% slower; "
                                 "inference can pack to 2-bit for ~8× weight memory cut.")
    tern_group.add_argument("--ternary_only_modules", type=str, nargs="*", default=[],
                            help="Optional whitelist of subtree names to quantize "
                                 "(e.g. 'synapse' or 'k_proj v_proj attn_out_proj'). "
                                 "Empty = all eligible Linears.")

    # ── CTM-v2 Features ─────────────────────────────────────────────────
    v2_group = parser.add_argument_group("CTM-v2 Features")

    # FEEC Integrator
    v2_group.add_argument("--use_feec", action="store_true",
                          help="Enable FEEC integrator for structure-preserving "
                               "thought loop dynamics. Provides bounded gradients "
                               "as T scales via symplectic-like integration.")
    v2_group.add_argument("--feec_dt_init", type=float, default=0.1,
                          help="Initial learnable step size per layer.")
    v2_group.add_argument("--feec_damping_init", type=float, default=0.1,
                          help="Initial damping coefficient γ.")
    v2_group.add_argument("--feec_clamp_dt", type=float, default=1.0,
                          help="Upper bound on dt for stability.")
    v2_group.add_argument("--feec_energy_penalty_weight", type=float, default=0.01,
                          help="Weight of energy growth penalty in loss.")

    # Matrix-Valued Residual Streams
    v2_group.add_argument("--use_matrix_streams", action="store_true",
                          help="Replace NLM FIFO buffers + O(D²) sync with "
                               "Hyperloop-style parallel residual streams.")
    v2_group.add_argument("--n_streams", type=int, default=4,
                          help="Number of parallel residual streams.")
    v2_group.add_argument("--stream_gating", type=str, default="diagonal",
                          choices=["diagonal", "sigmoid"],
                          help="Gating parameterization for matrix streams.")

    # DSSA
    v2_group.add_argument("--use_dssa", action="store_true",
                          help="Replace O(N²) cross-attention with Dual-Space "
                               "Sparse Attention (SSE + MoBA hybrid).")
    v2_group.add_argument("--dssa_n_partitions", type=int, default=32)
    v2_group.add_argument("--dssa_top_k", type=int, default=8)
    v2_group.add_argument("--dssa_block_size", type=int, default=64)
    v2_group.add_argument("--dssa_top_k_blocks", type=int, default=4)

    # Hyperloop
    v2_group.add_argument("--use_hyperloop", action="store_true",
                          help="Weight-share middle layers via looping. "
                               "Preserves depth while cutting unique params.")
    v2_group.add_argument("--hyperloop_n_begin", type=int, default=2)
    v2_group.add_argument("--hyperloop_n_middle", type=int, default=4)
    v2_group.add_argument("--hyperloop_n_end", type=int, default=2)
    v2_group.add_argument("--hyperloop_middle_loops", type=int, default=2)

    # Loop Position Embeddings
    v2_group.add_argument("--use_loop_pos_emb", action="store_true",
                          help="Add learned per-thought-step embeddings to "
                               "distinguish iterations in the thought loop.")

    # Triton Acceleration
    v2_group.add_argument("--use_triton_attention", action="store_true",
                          help="Use Triton-accelerated tiled attention kernel.")
    v2_group.add_argument("--use_cuda_graphs", action="store_true",
                          help="Wrap thought loop in CUDA Graph for kernel "
                               "launch elimination.")
    v2_group.add_argument("--tiled_schedule", action="store_true",
                          help="Use N·log(N) tiled schedule for thought steps.")

    return parser.parse_args()


def main():
    args = parse_args()

    config = CTMConfig(
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        dataset=args.dataset or "",
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        nlm_hidden_dim=args.nlm_hidden_dim,
        nlm_groups=args.nlm_groups,
        per_tick_heads=args.per_tick_heads,
        t_curriculum=args.t_curriculum,
        t_curriculum_stages=_parse_curriculum_stages(args.t_curriculum_stages),
        history_len=args.history_len,
        max_thought_steps=args.max_thought_steps,
        seq_len=args.seq_len,
        sync_method=args.sync_method,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        device=args.device,
        dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_min_T=args.gradient_checkpointing_min_T,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        optimizer=args.optimizer,
        adam_beta1=args.adam_beta1 if hasattr(args, 'adam_beta1') else 0.9,
        adam_beta2=args.adam_beta2 if hasattr(args, 'adam_beta2') else 0.95,
        use_8bit_adam=args.use_8bit_adam,
        adamuon_beta=args.adamuon_beta,
        adamuon_eps=args.adamuon_eps,
        adamuon_ns_steps=args.adamuon_ns_steps,
        adamuon_rms_target=args.adamuon_rms_target,
        adamuon_weight_decay=args.adamuon_weight_decay,
        use_engram=args.use_engram,
        engram_ngram_orders=args.engram_ngram_orders,
        engram_n_heads=args.engram_n_heads,
        engram_slots_per_table=args.engram_slots_per_table,
        engram_d_head=args.engram_d_head,
        engram_layers=args.engram_layers,
        engram_use_conv=not args.engram_no_conv,
        engram_conv_kernel=args.engram_conv_kernel,
        engram_conv_dilation=args.engram_conv_dilation,
        engram_lr_mult=args.engram_lr_mult,
        engram_weight_decay=args.engram_weight_decay,
        use_ternary=args.use_ternary,
        ternary_only_modules=args.ternary_only_modules,
        # CTM-v2 features
        use_feec=args.use_feec,
        feec_dt_init=args.feec_dt_init,
        feec_damping_init=args.feec_damping_init,
        feec_clamp_dt=args.feec_clamp_dt,
        feec_energy_penalty_weight=args.feec_energy_penalty_weight,
        use_matrix_streams=args.use_matrix_streams,
        n_streams=args.n_streams,
        stream_gating=args.stream_gating,
        use_dssa=args.use_dssa,
        dssa_n_partitions=args.dssa_n_partitions,
        dssa_top_k=args.dssa_top_k,
        dssa_block_size=args.dssa_block_size,
        dssa_top_k_blocks=args.dssa_top_k_blocks,
        use_hyperloop=args.use_hyperloop,
        hyperloop_n_begin=args.hyperloop_n_begin,
        hyperloop_n_middle=args.hyperloop_n_middle,
        hyperloop_n_end=args.hyperloop_n_end,
        hyperloop_middle_loops=args.hyperloop_middle_loops,
        use_loop_pos_emb=args.use_loop_pos_emb,
        use_triton_attention=args.use_triton_attention,
        use_cuda_graphs=args.use_cuda_graphs,
        tiled_schedule=args.tiled_schedule,
    )

    train(config)


if __name__ == "__main__":
    main()