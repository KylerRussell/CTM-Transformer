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

def save_checkpoint(model, optimizer, step, config, path, keep_last=3):
    """Save model checkpoint with rotation."""
    ckpt_dir = path.parent
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": vars(config),
    }, path)

    # Rotate old step checkpoints (keep last N)
    step_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in step_ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint, return the step number."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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

    # Mixed precision
    if config.dtype == "bfloat16" and device != "cpu":
        model = model.to(torch.bfloat16)
        amp_dtype = torch.bfloat16
    elif config.dtype == "float16" and device != "cpu":
        model = model.to(torch.float16)
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    n_params = model.get_num_params()
    print(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"Config: d_model={config.d_model}, d_latent={config.d_latent}, "
          f"n_layers={config.n_layers}, n_heads={config.n_heads}, "
          f"thought_steps={config.max_thought_steps}, history_len={config.history_len}, "
          f"nlm_hidden={config.nlm_hidden_dim}, nlm_groups={config.nlm_groups}")

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Resume from checkpoint ──────────────────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    start_step = 0

    existing_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if existing_ckpts:
        print(f"  Resuming from {existing_ckpts[-1]}")
        start_step = load_checkpoint(model, optimizer, existing_ckpts[-1], device)
        print(f"  Resumed at step {start_step}")
    elif (ckpt_dir / "best.pt").exists():
        print(f"  Loading best checkpoint")
        start_step = load_checkpoint(model, optimizer, ckpt_dir / "best.pt", device)

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

    while step < config.max_steps:
        # Get next batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        # Learning rate schedule
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass with mixed precision
        t0 = time.time()
        device_type = device.split(":")[0] if ":" in device else device

        with torch.amp.autocast(device_type=device_type,
                                dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            result = model(x, targets=y)
            loss = result["loss"]

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()

        dt = time.time() - t0
        batch_tokens = config.batch_size * config.seq_len
        tokens_seen += batch_tokens
        log_losses.append(loss.item())

        # ── Logging ─────────────────────────────────────────────────────
        if step % config.log_interval == 0:
            window = log_losses[-config.log_interval:]
            avg_loss = sum(window) / len(window)
            tick_losses = result["per_tick_loss"].tolist()

            cert = result["certainties"]
            cert_first = cert[0].mean().item()
            cert_last = cert[-1].mean().item()
            tok_per_sec = batch_tokens / max(dt, 1e-6)

            elapsed = time.time() - t_start
            eta_sec = (config.max_steps - step) * (elapsed / max(step - start_step, 1))

            print(
                f"Step {step:6d} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{tok_per_sec/1e3:.1f}k tok/s | "
                f"cert {cert_first:.2f}→{cert_last:.2f} | "
                f"ticks [{' '.join(f'{l:.3f}' for l in tick_losses)}] | "
                f"{tokens_seen/1e6:.0f}M tok | "
                f"ETA {eta_sec/3600:.1f}h"
            )

        # ── Evaluation ──────────────────────────────────────────────────
        if step > 0 and step % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device, config, amp_dtype)
            print(f"\n  >>> Eval loss: {eval_loss:.4f} (best: {best_eval_loss:.4f})")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(model, optimizer, step, config, ckpt_dir / "best.pt")
                print(f"  >>> Saved best checkpoint at step {step}")

            # Save step checkpoint
            save_checkpoint(model, optimizer, step, config,
                          ckpt_dir / f"step_{step:07d}.pt")

            # Generate a sample
            generate_sample(model, tokenizer, device, config, raw_text)
            print()
            model.train()

        step += 1

    # Final save
    save_checkpoint(model, optimizer, step, config, ckpt_dir / "final.pt")
    print(f"\nTraining complete. {tokens_seen/1e9:.2f}B tokens processed.")


@torch.no_grad()
def evaluate(model, eval_loader, device, config, amp_dtype):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    max_eval_batches = 50
    device_type = device.split(":")[0] if ":" in device else device

    for x, y in eval_loader:
        if n_batches >= max_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast(device_type=device_type,
                                dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            result = model(x, targets=y)

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
    train_group.add_argument("--eval_interval", type=int, default=500)
    train_group.add_argument("--log_interval", type=int, default=50)
    train_group.add_argument("--checkpoint_dir", type=str, default="checkpoints")

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
        history_len=args.history_len,
        max_thought_steps=args.max_thought_steps,
        seq_len=args.seq_len,
        sync_method=args.sync_method,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        device=args.device,
        dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    train(config)


if __name__ == "__main__":
    main()
