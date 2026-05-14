#!/usr/bin/env python
"""
Extend existing teacher cache shards with `teacher_z` (last-layer hidden
states) so the Predictive-Coding auxiliary loss has targets to learn.

This script takes a directory of `shard_*.npz` files produced by
`scripts/cache_teacher_logits.py` and, for each shard, runs the teacher
model in inference mode over the cached input_ids to capture the
final-layer hidden state per token. The hidden state is appended to the
existing shard under the key `teacher_z` and the shard is rewritten
in place (atomically — we write to .npz.tmp first, then rename).

Why a separate script rather than modifying the original cache builder:
  • Most users already spent compute building the existing cache. Forcing
    them to re-tokenize and re-run the teacher from scratch is wasteful
    when only the hidden state needs to be added.
  • The hidden state roughly doubles shard size on disk. Users who don't
    care about Predictive Coding shouldn't pay that storage cost.
  • Hidden-state dimension (teacher_d_model) varies by teacher model.
    Keeping it optional sidesteps a schema break.

Usage:
    python -m ctm_transformer.scripts.extend_teacher_cache_with_z \\
        --cache_dir ./teacher_cache \\
        --teacher_model_name "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" \\
        --device cuda:0 \\
        --batch_size 4

By default the script skips shards that already contain a `teacher_z`
key (idempotent). Pass `--force` to overwrite.

CAVEATS:
  • Disk usage: with seq_len=512 and teacher_d_model=3136, each row is
    ~3 MB of fp16 hidden state. A shard of 1000 rows grows by ~3 GB.
    Make sure the host filesystem has headroom.
  • The teacher hidden state stored is from the LAST transformer layer
    (before the LM head projection). Different teachers expose this
    differently; we follow the HF convention of
    `model(..., output_hidden_states=True).hidden_states[-1]`.
  • Mixed precision: we cast hidden states to fp16 on disk to keep the
    shards small. The student-side LayerNorm absorbs the precision loss.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Directory containing shard_*.npz files.")
    p.add_argument("--teacher_model_name", type=str, required=True,
                   help="HF model name (must match what was used to "
                        "build the original cache).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Forward-pass batch size. Lower if OOM.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--force", action="store_true",
                   help="Overwrite teacher_z even if already present.")
    p.add_argument("--limit_shards", type=int, default=0,
                   help="Process only the first N shards. 0 → all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading teacher: {args.teacher_model_name} → {args.device} ({args.dtype})",
          flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=dtype,
    ).to(args.device)
    teacher.eval()

    cache_dir = Path(args.cache_dir)
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz under {cache_dir}")
    if args.limit_shards > 0:
        shards = shards[:args.limit_shards]
    print(f"Found {len(shards)} shards under {cache_dir}", flush=True)

    for shard_idx, shard_path in enumerate(shards):
        with np.load(shard_path) as src:
            keys = set(src.files)
            if "teacher_z" in keys and not args.force:
                print(f"[{shard_idx+1}/{len(shards)}] {shard_path.name}: "
                      f"teacher_z already present, skipping (use --force "
                      f"to overwrite)",
                      flush=True)
                continue
            data = {k: src[k] for k in src.files}

        input_ids = data["input_ids"]                # int32 [N, S]
        N, S = input_ids.shape

        # Run teacher in mini-batches to capture hidden states.
        hidden_states_chunks = []
        with torch.no_grad():
            for start in range(0, N, args.batch_size):
                end = min(start + args.batch_size, N)
                ids = torch.from_numpy(
                    input_ids[start:end].astype(np.int64)
                ).to(args.device)
                # autocast not used; we set dtype on the model directly.
                out = teacher(ids, output_hidden_states=True)
                last_hidden = out.hidden_states[-1]   # [B, S, D_teacher]
                # Cast to fp16 for storage (smaller shards, less I/O).
                hidden_states_chunks.append(
                    last_hidden.to(torch.float16).cpu().numpy()
                )
        teacher_z = np.concatenate(hidden_states_chunks, axis=0)  # [N, S, D]
        assert teacher_z.shape[0] == N and teacher_z.shape[1] == S, \
            f"shape mismatch: {teacher_z.shape} vs ({N}, {S}, *)"

        data["teacher_z"] = teacher_z

        # Atomic rewrite: write to .tmp, then rename. Avoids leaving
        # a corrupt half-written shard if the script is interrupted.
        tmp_path = shard_path.with_suffix(".npz.tmp")
        np.savez_compressed(tmp_path, **data)
        os.replace(tmp_path, shard_path)

        size_mb = shard_path.stat().st_size / (1024 * 1024)
        print(f"[{shard_idx+1}/{len(shards)}] {shard_path.name}: "
              f"added teacher_z shape={teacher_z.shape} dtype=fp16 "
              f"→ shard now {size_mb:.1f} MB",
              flush=True)

    print("Done. CachedTeacherDataset will now yield teacher_z, and "
          "training with --use_predictive_coding will receive PC targets.",
          flush=True)


if __name__ == "__main__":
    main()
