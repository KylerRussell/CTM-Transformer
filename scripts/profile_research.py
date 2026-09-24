"""Measure a small CTM or baseline training workload on one GPU; no data downloads.

Run as a module from the repository root. Independent processes can profile
different configurations on cuda:0 and cuda:1. This is a synthetic timing
probe, not evidence about model quality or full data-pipeline throughput.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from ctm_transformer.config import CTMConfig
from ctm_transformer.experiment import file_hash
from ctm_transformer.research import load_research_config, build_model, model_family, parameter_counts, block_applications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, help="Fully specified research config; architecture overrides are rejected")
    parser.add_argument("--thought-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--nlm-groups", type=int)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpointing", action="store_true", default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = None
    if args.config:
        if any(getattr(args, name) is not None for name in (
            "thought_steps", "batch_size", "seq_len", "d_model", "n_layers", "nlm_groups", "checkpointing", "dtype"
        )):
            parser.error("Use a versioned config variant rather than overriding the supplied config")
        config, identity = load_research_config(args.config)
        config.device = args.device
        if config.dtype == "bfloat16" and not config.bf16_autocast:
            parser.error("BF16 research profiling requires bf16_autocast=true to match the trainer")
    else:
        defaults = dict(thought_steps=4, batch_size=4, seq_len=128, d_model=256,
                        n_layers=4, nlm_groups=1, checkpointing=False, dtype="bfloat16")
        for name, default in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
        config = CTMConfig(
            vocab_size=4096, d_model=args.d_model, d_latent=args.d_model,
            n_heads=8, n_layers=args.n_layers, history_len=8, nlm_hidden_dim=16,
            nlm_groups=args.nlm_groups, max_thought_steps=args.thought_steps,
            max_seq_len=args.seq_len, seq_len=args.seq_len, batch_size=args.batch_size,
            use_positional_encoding=True, dropout=0.0,
            gradient_checkpointing=args.checkpointing, gradient_checkpointing_min_T=1,
            dtype=args.dtype, device=args.device, bf16_autocast=args.dtype == "bfloat16",
        )
    if min(args.steps, args.warmup, config.batch_size, config.seq_len, config.max_thought_steps,
           config.d_model, (config.n_layers if model_family(config) != 'recurrent_depth' else config.core_layers)) <= 0:
        parser.error("model dimensions, budgets, steps, and warmup must be positive")
    if config.d_model % config.n_heads or (model_family(config) == 'ctm' and config.d_latent % config.nlm_groups):
        parser.error("head and temporal MLP counts must divide their widths")
    if config.optimizer != "adamw" or config.use_8bit_adam:
        parser.error("This probe supports standard AdamW only")
    if config.dtype not in {"float32", "bfloat16"}:
        parser.error("This probe supports float32 or bfloat16")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("a working CUDA device is required; no CPU fallback")
    torch.cuda.set_device(device)
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bfloat16")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model = build_model(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), device=device)
    ids, targets = tokens[:, :-1], tokens[:, 1:]

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.dtype == "bfloat16"):
            loss = model(ids, targets=targets)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    losses = []
    for _ in range(args.steps):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        losses.append(float(loss))
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("Non-finite loss in timed training steps")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None
    report = {
        "purpose": "synthetic GPU training timing; excludes data loading and compilation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "commit": commit, "worktree_status": status,
        "source_sha256": {p: file_hash(p) for p in (
            "scripts/profile_research.py", "ctm_transformer/research.py",
            "ctm_transformer/model.py", "ctm_transformer/config.py", "ctm_transformer/baselines.py")},
        "seed": args.seed, "config": asdict(config), "config_identity": identity,
        "precision": "FP32 parameters and AdamW state; BF16 autocast" if config.bf16_autocast else "FP32",
        "model_family": model_family(config),
        "parameters": parameter_counts(model),
        "profile_thought_steps": config.max_thought_steps,
        "block_applications_per_sequence": block_applications(config, config.max_thought_steps),
        "depth_policy": "fixed configured evaluation depth, including for variable-depth training presets",
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "warmup_steps": args.warmup, "timed_steps": args.steps,
        "median_step_seconds": statistics.median(durations),
        "p95_step_seconds": sorted(durations)[max(0, int(0.95 * len(durations) + 0.999) - 1)],
        "tokens_per_second": config.batch_size * config.seq_len * args.steps / sum(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "step_seconds": durations, "training_losses": losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "device", "parameters_total", "tokens_per_second", "median_step_seconds", "peak_allocated_bytes"
    )}, indent=2))


if __name__ == "__main__":
    main()
