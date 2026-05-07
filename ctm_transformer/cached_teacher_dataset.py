"""
cached_teacher_dataset.py — Consumer for offline teacher logit caches.

Reads shards produced by scripts/cache_teacher_logits.py and yields tuples
of (input_ids, targets, top_indices, top_values, residual_log_mass) for
the training loop. Drop-in replacement for FineWebEduDataset when running
with --use_cached_teacher.

Yielded tensor shapes (per sample):
    input_ids        : long  [seq_len]
    targets          : long  [seq_len]
    top_indices      : long  [seq_len, K]
    top_values       : float [seq_len, K]   (in fp32; kept as raw logits)
    residual_log_mass: float [seq_len]      (fp32; not consumed by current loss)

The DataLoader's default collate_fn stacks each into a batch.

Sharding behavior:
    - Multi-worker DataLoader: shards are distributed round-robin across
      workers (worker i gets shards i, i+W, i+2W, …).
    - Distributed training: when world_size > 1, each rank is given a
      disjoint range of shards (rank 0 gets shards 0..N/W-1, rank 1 gets
      N/W..2N/W-1, etc). The DataLoader workers further subdivide.
    - Loop epochs: when all shards are exhausted, the dataset reshuffles
      its shard order (with a deterministic seed per epoch) and continues.

Memory:
    Each shard is loaded into memory in full (~800 MB for samples_per_shard
    = 1024 at K=256, S=512). At any time only one shard per worker is live.
    With num_workers=2 that's ~1.6 GB of RAM — trivial on a homelab.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class CachedTeacherDataset(IterableDataset):
    """Streams cached teacher logits + input ids from a directory of shards.

    Args:
        cache_dir: directory containing shard_*.npz files written by
            cache_teacher_logits.py.
        rank, world_size: distributed-training hooks. Each rank receives
            an interleaved subset of shards. When DataParallel is the
            launcher (single process, multiple devices), pass rank=0,
            world_size=1.
        shuffle_shards: if True, shard order is shuffled per epoch
            (seeded by epoch index for reproducibility).
        loop: if True, dataset cycles through shards forever; if False,
            stops after one pass.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike,
        rank: int = 0,
        world_size: int = 1,
        shuffle_shards: bool = True,
        loop: bool = True,
        is_eval: bool = False,
        eval_fraction: float = 0.05,
    ):
        self.cache_dir = Path(cache_dir)
        self.rank = rank
        self.world_size = world_size
        self.shuffle_shards = shuffle_shards
        self.loop = loop

        all_shards = sorted(self.cache_dir.glob("shard_*.npz"))
        if not all_shards:
            raise FileNotFoundError(
                f"No shard_*.npz files found under {self.cache_dir}. "
                f"Run scripts/cache_teacher_logits.py first."
            )
        # Reserve the last `eval_fraction` of shards for eval (mirrors
        # the train.py convention of reserving the last 5% of parquet
        # files for evaluation).
        n_eval = max(1, int(len(all_shards) * eval_fraction))
        if is_eval:
            self._all_shards = all_shards[-n_eval:]
        else:
            self._all_shards = all_shards[:-n_eval] if n_eval > 0 else all_shards
        # Inspect first shard to record cache dimensions for sanity logging.
        with np.load(all_shards[0]) as z:
            self.seq_len = int(z["input_ids"].shape[1])
            self.top_k = int(z["top_indices"].shape[2])
        split = "eval" if is_eval else "train"
        print(
            f"[CachedTeacherDataset/{split}] {len(self._all_shards)} shards "
            f"(of {len(all_shards)} total) under {self.cache_dir} | "
            f"seq_len={self.seq_len} top_k={self.top_k}",
            flush=True,
        )

    # ────────────────────────────────────────────────────────────────
    # Shard partitioning across rank × worker
    # ────────────────────────────────────────────────────────────────
    def _partition_shards(self) -> list[Path]:
        """Pick the subset of shards this rank+worker is responsible for."""
        worker = get_worker_info()
        n_workers = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        total_shards_groups = self.world_size * n_workers
        my_group = self.rank * n_workers + wid
        # Round-robin slice
        return self._all_shards[my_group::total_shards_groups]

    # ────────────────────────────────────────────────────────────────
    # Iteration
    # ────────────────────────────────────────────────────────────────
    def __iter__(self):
        my_shards = self._partition_shards()
        if not my_shards:
            return

        epoch = 0
        while True:
            order = list(my_shards)
            if self.shuffle_shards:
                # Seed by epoch + my-shard-fingerprint so different ranks
                # see different shuffles even within the same epoch — but
                # the same rank reproduces its order across runs.
                seed = 1000003 * epoch + hash(tuple(p.name for p in order))
                rng = random.Random(seed & 0xFFFFFFFF)
                rng.shuffle(order)

            for shard_path in order:
                with np.load(shard_path) as z:
                    input_ids = z["input_ids"]              # int32 [N, S]
                    targets = z["targets"]                  # int32 [N, S]
                    top_indices = z["top_indices"]          # int32 [N, S, K]
                    top_values = z["top_values"]            # f16   [N, S, K]
                    residual = z["residual_log_mass"]       # f16   [N, S]

                    N = input_ids.shape[0]
                    # Optional in-shard shuffle so consecutive samples
                    # don't all come from the same parquet file ordering.
                    if self.shuffle_shards:
                        perm = np.random.RandomState(
                            (epoch * 7919 + N) & 0xFFFFFFFF
                        ).permutation(N)
                    else:
                        perm = np.arange(N)

                    for i in perm:
                        yield (
                            torch.from_numpy(input_ids[i].astype(np.int64)),
                            torch.from_numpy(targets[i].astype(np.int64)),
                            torch.from_numpy(top_indices[i].astype(np.int64)),
                            torch.from_numpy(top_values[i].astype(np.float32)),
                            torch.from_numpy(residual[i].astype(np.float32)),
                        )

            if not self.loop:
                return
            epoch += 1


def cached_teacher_collate(batch):
    """Default collate_fn works fine, but defining it explicitly here lets
    train.py be unambiguous about the order it expects.
    Each element of `batch` is a 5-tuple from CachedTeacherDataset.__iter__.
    """
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    top_indices = torch.stack([b[2] for b in batch], dim=0)
    top_values = torch.stack([b[3] for b in batch], dim=0)
    residual = torch.stack([b[4] for b in batch], dim=0)
    return input_ids, targets, top_indices, top_values, residual
