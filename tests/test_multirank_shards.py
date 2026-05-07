"""
test_multirank_shards.py — Test the shard-partitioning and resume logic
for multi-rank (multi-GPU) cache producers.

Verifies:
  1. Two ranks writing to the same dir produce disjoint shard indices.
  2. Resume detection finds the correct next-shard for each rank.
  3. samples_to_skip computation is per-rank, not global.
  4. shard increment correctly advances by world_size.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np


def make_fake_shard(out_dir: Path, idx: int):
    """Write a placeholder shard file for resume-detection testing."""
    arr = np.zeros((2, 4), dtype=np.int32)
    np.savez(
        out_dir / f"shard_{idx:06d}.npz",
        input_ids=arr, targets=arr,
        top_indices=np.zeros((2, 4, 4), dtype=np.int32),
        top_values=np.zeros((2, 4, 4), dtype=np.float16),
        residual_log_mass=np.zeros((2, 4), dtype=np.float16),
    )


def test_rank_partitioning_no_overlap():
    """Two ranks writing to disjoint slots — verify the shard
    partitioning math (used in main()) is correct."""
    print("[test 1] Rank-aware shard partitioning is disjoint...", end=" ")

    # Simulate two ranks each picking their next-shard from a partial
    # state: rank 0 has written shards 0,2; rank 1 has written shards 1,3,5.
    # Their next shards should be 4 (rank 0) and 7 (rank 1) — both
    # disjoint, both following the rank+world_size pattern.
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for idx in [0, 1, 2, 3, 5]:
            make_fake_shard(out_dir, idx)

        all_existing = sorted(out_dir.glob("shard_*.npz"))

        # Rank 0 logic
        rank = 0
        my_existing_0 = [
            p for p in all_existing
            if (int(p.stem.split("_")[1]) % world_size) == rank
        ]
        if my_existing_0:
            last_mine = int(my_existing_0[-1].stem.split("_")[1])
            next_shard_0 = last_mine + world_size
        else:
            next_shard_0 = rank
        assert next_shard_0 == 4, f"rank 0 next: expected 4, got {next_shard_0}"

        # Rank 1 logic
        rank = 1
        my_existing_1 = [
            p for p in all_existing
            if (int(p.stem.split("_")[1]) % world_size) == rank
        ]
        if my_existing_1:
            last_mine = int(my_existing_1[-1].stem.split("_")[1])
            next_shard_1 = last_mine + world_size
        else:
            next_shard_1 = rank
        assert next_shard_1 == 7, f"rank 1 next: expected 7, got {next_shard_1}"

        # No overlap when each progresses
        rank0_future = [next_shard_0 + i * world_size for i in range(5)]  # 4,6,8,10,12
        rank1_future = [next_shard_1 + i * world_size for i in range(5)]  # 7,9,11,13,15
        assert not (set(rank0_future) & set(rank1_future))
    print(f"OK (rank 0 → 4, rank 1 → 7, no overlap)")


def test_samples_to_skip_per_rank():
    """samples_to_skip should be per-rank (count of THIS rank's shards),
    not global. With rank 0 having written 3 shards, samples_to_skip
    should be 3 * samples_per_shard regardless of how many shards rank 1
    has written."""
    print("[test 2] samples_to_skip is per-rank...", end=" ")

    world_size = 2
    samples_per_shard = 1024

    # Scenario: rank 0 wrote shards 0,2,4 (3 of its own).
    #           rank 1 wrote shards 1,3,5,7 (4 of its own).
    # rank 0's samples_to_skip should be 3 * 1024 = 3072.
    # rank 1's samples_to_skip should be 4 * 1024 = 4096.
    # These come from independent streams, so they're not 7*1024.

    # Replicate the formula from main():
    #   samples_to_skip = (next_shard - rank) // world_size * samples_per_shard
    rank, last_mine = 0, 4
    next_shard_0 = last_mine + world_size  # 6
    sts_0 = (next_shard_0 - rank) // world_size * samples_per_shard
    assert sts_0 == 3 * samples_per_shard, f"rank 0: {sts_0}"

    rank, last_mine = 1, 7
    next_shard_1 = last_mine + world_size  # 9
    sts_1 = (next_shard_1 - rank) // world_size * samples_per_shard
    assert sts_1 == 4 * samples_per_shard, f"rank 1: {sts_1}"

    # Edge case: rank with no shards yet → next_shard = rank → samples_to_skip = 0
    rank = 0
    next_shard_fresh = rank
    sts_fresh = (next_shard_fresh - rank) // world_size * samples_per_shard
    assert sts_fresh == 0, f"fresh rank: {sts_fresh}"
    print(f"OK (rank 0 skip=3072, rank 1 skip=4096, fresh skip=0)")


def test_increment_advances_by_world_size():
    """Walking a fresh rank-0 producer should produce shard indices
    0, 2, 4, 6, ... (advancing by world_size, not by 1)."""
    print("[test 3] shard_idx advances by world_size...", end=" ")

    world_size = 2
    rank = 0
    shard_idx = rank  # initial
    written = []
    for _ in range(5):
        written.append(shard_idx)
        shard_idx += world_size
    assert written == [0, 2, 4, 6, 8], f"got {written}"

    rank = 1
    shard_idx = rank
    written = []
    for _ in range(5):
        written.append(shard_idx)
        shard_idx += world_size
    assert written == [1, 3, 5, 7, 9], f"got {written}"
    print("OK")


def test_rank_seed_independence():
    """Different ranks must seed their curriculum streamer differently
    so they don't produce identical sample streams."""
    print("[test 4] Rank-mixed seeds are distinct...", end=" ")

    base_seed = 42
    rank_seeds = [
        (base_seed + r * 1_000_003) & 0xFFFFFFFF
        for r in range(8)
    ]
    # All 8 should be unique
    assert len(set(rank_seeds)) == 8, (
        f"only {len(set(rank_seeds))} unique seeds out of 8"
    )
    # And rank 0's seed should still be the base seed
    assert rank_seeds[0] == base_seed, (
        f"rank 0 seed = {rank_seeds[0]}, expected {base_seed}"
    )
    print("OK")


def test_world_size_1_unchanged_behavior():
    """When world_size=1, the new code should behave identically to
    the old single-process code (rank=0, increment by 1)."""
    print("[test 5] world_size=1 preserves single-process behavior...", end=" ")

    world_size = 1
    rank = 0

    # Resume from a 3-shard cache
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for idx in [0, 1, 2]:
            make_fake_shard(out_dir, idx)

        all_existing = sorted(out_dir.glob("shard_*.npz"))
        my_existing = [
            p for p in all_existing
            if (int(p.stem.split("_")[1]) % world_size) == rank
        ]
        last_mine = int(my_existing[-1].stem.split("_")[1])
        next_shard = last_mine + world_size
        assert next_shard == 3, f"single-process resume: {next_shard}"

        samples_to_skip = (next_shard - rank) // world_size * 1024
        assert samples_to_skip == 3 * 1024, f"single-process skip: {samples_to_skip}"

    # Increment behavior: 0,1,2,3,4 (every shard, not every other)
    shard_idx = 0
    written = []
    for _ in range(5):
        written.append(shard_idx)
        shard_idx += world_size
    assert written == [0, 1, 2, 3, 4], f"single-process increment: {written}"
    print("OK")


if __name__ == "__main__":
    test_rank_partitioning_no_overlap()
    test_samples_to_skip_per_rank()
    test_increment_advances_by_world_size()
    test_rank_seed_independence()
    test_world_size_1_unchanged_behavior()
    print("\nAll multi-rank smoke tests passed.")
