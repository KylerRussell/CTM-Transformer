"""
cache_teacher_logits.py — Offline producer for teacher top-K logits.

Runs the teacher (NemotronH-4B in BF16, NOT the FP8 variant — see notes
below) over packed sequences from the SAME curriculum the student trains
on, and writes per-token top-K logit caches to disk. Training then reads
from this cache and the teacher is dropped from the training loop entirely.

Why offline:
  - Teacher forward is currently 413 ms/step (23% of step time) and runs
    *serially* with student forward despite living on cuda:1 — HF's
    Python-side dispatch holds the GIL across module forwards.
  - cuda:1 sits at 4% utilization; we want it free for the student via
    DDP after this change.
  - Standard production pattern: NeMo-Aligner blog, MiniPLM (arXiv
    2410.17215), "Pre-training Distillation Design Space" (arXiv
    2410.16215) all do offline KD logit caching.

Why BF16 teacher (not FP8):
  - RTX 3090 is SM 8.6. FP8 tensor-core MMA requires SM 8.9 (Ada) or
    SM 9.0 (Hopper). Loading the FP8 checkpoint on Ampere produces a
    per-layer dequant-to-bf16 software path with no Marlin fast kernel.
  - The BF16 variant `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` runs at
    native tensor-core speed and is what should be used here.

Curriculum:
  By default this script reads the SAME 12-source NVIDIA Nemotron
  pretraining curriculum the student uses (Nemotron-Pretraining-RQA,
  Math-Textbooks, STEM-SFT, Formal-Logic, Code-Concepts, Multiple-Choice,
  Wiki-Rewrite, MIND, etc.), with the per-source mixing weights defined
  in ctm_transformer/config.py. The data path is identical to what
  CurriculumDataset would have used at training time.

  Use --phase {1,2} to select which phase's mixing weights to cache.
  The two phases must be cached separately into different output dirs;
  the training loop then switches between them at phase1_tokens.

Cache format (each shard is an uncompressed .npz):
    input_ids        : int32   [N, S]
    targets          : int32   [N, S]
    top_indices      : int32   [N, S, K]
    top_values       : float16 [N, S, K]   (RAW logits)
    residual_log_mass: float16 [N, S]      (log(1 - sum_top_K softmax))

  Storage: ~773 KB / sample at S=512, K=256.

Usage:
    # Phase 1
    python -m scripts.cache_teacher_logits \\
        --phase 1 \\
        --output_dir /data/teacher_cache/phase1 \\
        --data_cache_dir ./data_cache \\
        --teacher_batch_size 8 \\
        --max_samples 1000000 \\
        --device cuda:0

    # Phase 2
    python -m scripts.cache_teacher_logits \\
        --phase 2 \\
        --output_dir /data/teacher_cache/phase2 \\
        --data_cache_dir ./data_cache \\
        --teacher_batch_size 8 \\
        --max_samples 4000000 \\
        --device cuda:0

Resume: shards already on disk are skipped on restart.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Pull config defaults (curriculum mix + weights). The producer is a
# sibling of train.py, not a child — we deliberately don't `from
# ctm_transformer.train import ...` because train.py's import side
# effects (pyarrow, hf_hub, datasets, the whole training stack) are
# heavy and we want this script to be runnable in a minimal env.
from ctm_transformer.config import CTMConfig

# Note: `transformers` is imported lazily inside main() so this module
# can be imported (e.g. for unit tests of the curriculum streamer) in
# environments that don't have transformers installed.


# ─────────────────────────────────────────────────────────────────────────
# Parquet streamer (mirror of train.py's sliding_window_generator).
# Inlined here so the producer doesn't drag in train.py's full import
# chain. Behavior is identical — same eval-tail reservation, same ordering.
# ─────────────────────────────────────────────────────────────────────────

def sliding_window_generator(
    ds_name: str,
    subset: str,
    cache_dir: str,
    total_shards: int = 1,
    shard_index: int = 0,
    is_eval: bool = False,
):
    """Pulls parquet files from HF Hub (or local cache), yields {'text': ...}.
    Mirrors train.py's sliding_window_generator so the cached samples are
    statistically equivalent to the live-teacher path."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    path = (f"datasets/{ds_name}/{subset}"
            if subset != "default" else f"datasets/{ds_name}")
    try:
        all_files = sorted([
            f for f in fs.ls(path, detail=False) if f.endswith(".parquet")
        ])
    except Exception as e:
        print(f"Error listing files for {path}: {e}", flush=True)
        return

    # Reserve the last 5% for eval (matching train.py).
    n_eval = max(1, len(all_files) // 20)
    if is_eval:
        my_files = all_files[-n_eval:]
    else:
        train_files = all_files[:-n_eval]
        my_files = train_files[shard_index::total_shards]

    os.makedirs(cache_dir, exist_ok=True)
    repo_prefix = f"datasets/{ds_name}/"

    for file_path in my_files:
        path_in_repo = (file_path[len(repo_prefix):]
                        if file_path.startswith(repo_prefix)
                        else file_path.split("/")[-1])
        local_dest = os.path.join(cache_dir, path_in_repo)
        try:
            if not os.path.exists(local_dest):
                print(f"  Downloading {path_in_repo}...", flush=True)
                local_dest = hf_hub_download(
                    repo_id=ds_name,
                    filename=path_in_repo,
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
        except Exception as e:
            print(f"  Skipping (download error) {file_path}: {e}", flush=True)
            continue

        try:
            pf = pq.ParquetFile(local_dest)
            for batch in pf.iter_batches(batch_size=512):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    text = row.get("text", "")
                    if text:
                        yield {"text": str(text)}
        except Exception as e:
            print(f"  Skipping (read error) {local_dest}: {e}", flush=True)


def stratified_window_generator(
    ds_name: str,
    subset: str,
    cache_dir: str,
    seed: int = 0,
    is_eval: bool = False,
):
    """Stratified-sampling alternative to sliding_window_generator.

    Same parquet/HF discovery as the sequential generator, but visits
    parquet files in a SHUFFLED order and shuffles row order within each
    file's row-group batches. The yielded text-document distribution is
    a representative slice of the source's full content rather than its
    first N parquet files in directory order.

    Why this matters: the local data_cache is partial — many sources
    have just a fraction of their full upstream Hub content downloaded.
    Even when fully populated, sequential reading concentrates the cache
    on early-index parquet files, which (for large sources like RQA at
    ~43 B tokens) means the cached 190 M tokens come from the first
    0.4% of available data. Stratified shuffling spreads coverage across
    the entire source.

    Implementation notes:
      - File-level shuffle is seeded so resume-after-kill yields the
        SAME sample sequence (critical for deterministic resume; the
        consumer's shard-index → sample mapping must remain stable).
      - Within-file shuffle uses RecordBatch row-permutations which
        pyarrow handles cheaply — no full file load required.
      - For sources whose local cache is only one or two parquet files
        (e.g. Formal-Logic at 70 MB), file shuffle is a no-op; row
        shuffle still helps.
      - This generator does NOT cap how much it yields. Per-source
        token budgeting is handled at the caller (the curriculum
        streamer), which uses mixing weights to decide when to stop
        pulling from each source.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    path = (f"datasets/{ds_name}/{subset}"
            if subset != "default" else f"datasets/{ds_name}")
    try:
        all_files = sorted([
            f for f in fs.ls(path, detail=False) if f.endswith(".parquet")
        ])
    except Exception as e:
        print(f"Error listing files for {path}: {e}", flush=True)
        return

    # The producer does NOT reserve an eval tail of parquet files.
    # Reasoning: this script writes a cache; the consumer
    # (CachedTeacherDataset) does its own eval split by reserving the
    # last 5% of CACHE SHARDS, not by reserving source files. Double-
    # reserving here was a bug — it made small sources (1-2 parquet
    # files) silently produce no training data: at len(all_files)=2,
    # n_eval=1, train_files=all_files[:-1] left only 1 file, and the
    # original cycle worked but a re-listed-from-scratch second cycle
    # could see the file count change and produce zero. We let the
    # consumer be the only place that reserves eval, and the producer
    # uses every parquet it sees.
    #
    # The `is_eval` parameter is kept for API compatibility but is now
    # a no-op in the producer's stratified path. This producer is
    # offline-only; eval data for the student is built into the cache's
    # last 5% of shards by CachedTeacherDataset.
    train_files = list(all_files)

    # Source-specific seed: same `seed` arg from caller, but different
    # per-source so the file-shuffle for "RQA" isn't correlated with
    # "Wiki-Rewrite" (which they would be if both used the bare `seed`).
    src_seed = (hash((ds_name, subset, seed)) & 0xFFFFFFFF)
    rng = random.Random(src_seed)
    shuffled_files = list(train_files)
    rng.shuffle(shuffled_files)

    os.makedirs(cache_dir, exist_ok=True)
    repo_prefix = f"datasets/{ds_name}/"

    for file_path in shuffled_files:
        path_in_repo = (file_path[len(repo_prefix):]
                        if file_path.startswith(repo_prefix)
                        else file_path.split("/")[-1])
        local_dest = os.path.join(cache_dir, path_in_repo)
        try:
            if not os.path.exists(local_dest):
                print(f"  Downloading {path_in_repo}...", flush=True)
                local_dest = hf_hub_download(
                    repo_id=ds_name,
                    filename=path_in_repo,
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
        except Exception as e:
            print(f"  Skipping (download error) {file_path}: {e}", flush=True)
            continue

        try:
            pf = pq.ParquetFile(local_dest)
            for batch in pf.iter_batches(batch_size=512):
                df = batch.to_pandas()
                # Shuffle rows within this batch so document order
                # doesn't leak parquet row-group locality.
                shuffled = df.sample(frac=1.0, random_state=src_seed)
                for _, row in shuffled.iterrows():
                    text = row.get("text", "")
                    if text:
                        yield {"text": str(text)}
        except Exception as e:
            print(f"  Skipping (read error) {local_dest}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# Curriculum-aware text streaming (mirrors CurriculumDataset.__iter__)
# ─────────────────────────────────────────────────────────────────────────

def stream_curriculum_text(
    datasets: list[str],
    subsets: list[str],
    weights: list[float],
    data_cache_dir: str,
    seed: int = 42,
    stratified: bool = True,
    cycle_exhausted: bool = True,
):
    """Yield {'text': str} samples by weighted-interleaving curriculum sources.
    Mirrors CurriculumDataset's iter logic exactly. Producer is single-process
    (rank=0, world=1); the training-time consumer parallelizes via its own
    DataLoader workers.

    Args:
        stratified: If True (default), use stratified_window_generator —
            shuffle parquet files within each source and shuffle rows
            within each file. Recommended for cache building because
            sequential reading of partial local data concentrates the
            cache on the first few parquet files of each source. If
            False, falls back to sliding_window_generator (matches the
            live training data path bit-for-bit, but cache coverage of
            partially-downloaded sources is poor).
        cycle_exhausted: If True (default), when a source's generator
            runs out of documents, restart it with a different shuffle
            seed and keep yielding. This preserves the configured
            mixing weights regardless of how much underlying data each
            source has — a 10%-weight source stays at 10% even if it
            only has 18M unique tokens. Documents repeat (with shuffled
            order across cycles), but since teacher logits are
            deterministic this is just oversampling.

            If False (legacy behavior), exhausted sources are dropped
            from the active pool and remaining sources' weights are
            renormalized. This concentrates the cache on whichever
            sources have the most data, regardless of the configured
            weights.
    """
    if not (len(datasets) == len(subsets) == len(weights)):
        raise ValueError(
            f"Curriculum lists must be the same length: "
            f"datasets={len(datasets)} subsets={len(subsets)} weights={len(weights)}"
        )
    print(f"[curriculum] {len(datasets)} sources "
          f"(stratified={stratified}, cycle={cycle_exhausted}):", flush=True)
    for d, s, w in zip(datasets, subsets, weights):
        print(f"  {w:>5.2f}  {d}/{s}", flush=True)

    # Helper: construct a fresh generator for source i with a per-cycle
    # seed so cycle 2's document order differs from cycle 1's.
    def _make_generator(i: int, cycle: int):
        # Composite seed: caller seed × source index × cycle counter.
        # Hash to bound it to 32 bits.
        gen_seed = hash((seed, datasets[i], subsets[i], cycle)) & 0xFFFFFFFF
        if stratified:
            return stratified_window_generator(
                ds_name=datasets[i], subset=subsets[i],
                cache_dir=data_cache_dir, seed=gen_seed, is_eval=False,
            )
        else:
            # The sequential generator doesn't take a seed; cycle 2+
            # will produce the same document order as cycle 1. Documents
            # still repeat, just without re-shuffling. Stratified is
            # strongly recommended when cycling is on.
            return sliding_window_generator(
                ds_name=datasets[i], subset=subsets[i],
                cache_dir=data_cache_dir,
                total_shards=1, shard_index=0, is_eval=False,
            )

    generators = [_make_generator(i, cycle=0) for i in range(len(datasets))]
    cycle_counts = [0] * len(datasets)

    rng = random.Random(seed)
    active_indices = list(range(len(generators)))
    current_weights = list(weights)
    # Track per-source yield counts so we can detect "this source returned
    # zero docs on this cycle" reliably. A source that returned > 0 docs
    # in cycle N MUST return > 0 in cycle N+1 (data didn't change between
    # cycles). If it doesn't, we treat that as a transient issue and try
    # one more time before giving up — networked filesystems (HfFileSystem)
    # can rarely return empty listings on retry.
    cycle_yield_counts: list[list[int]] = [[0] for _ in range(len(generators))]
    MAX_CONSECUTIVE_EMPTY_CYCLES = 2

    while active_indices:
        active_weights = [current_weights[i] for i in active_indices]
        if sum(active_weights) <= 0:
            break
        idx_in_active = rng.choices(
            range(len(active_indices)), weights=active_weights,
        )[0]
        idx = active_indices[idx_in_active]
        try:
            doc = next(generators[idx])
            cycle_yield_counts[idx][-1] += 1
            yield doc
        except StopIteration:
            if cycle_exhausted:
                cycle_counts[idx] += 1
                docs_this_cycle = cycle_yield_counts[idx][-1]
                cycle_yield_counts[idx].append(0)

                # If this is the FIRST cycle and it yielded ZERO docs,
                # the source has no readable data at all — drop it.
                if cycle_counts[idx] == 1 and docs_this_cycle == 0:
                    print(f"[curriculum] source {idx} "
                          f"({datasets[idx]}/{subsets[idx]}) yielded zero "
                          f"docs on initial pass; dropping permanently.",
                          flush=True)
                    active_indices.pop(idx_in_active)
                    continue

                # Check the last few cycles for a streak of zero-yields.
                # A source whose data has just become unavailable (network
                # blip, file deletion) shouldn't be retried forever.
                recent = cycle_yield_counts[idx][-MAX_CONSECUTIVE_EMPTY_CYCLES-1:-1]
                if (len(recent) >= MAX_CONSECUTIVE_EMPTY_CYCLES
                        and all(c == 0 for c in recent)):
                    print(f"[curriculum] source {idx} "
                          f"({datasets[idx]}/{subsets[idx]}) yielded zero "
                          f"docs for {MAX_CONSECUTIVE_EMPTY_CYCLES} cycles "
                          f"in a row; dropping permanently.", flush=True)
                    active_indices.pop(idx_in_active)
                    continue

                print(f"[curriculum] source {idx} "
                      f"({datasets[idx]}/{subsets[idx]}) exhausted after "
                      f"{docs_this_cycle} docs; cycling "
                      f"(pass #{cycle_counts[idx] + 1}).", flush=True)
                generators[idx] = _make_generator(idx, cycle=cycle_counts[idx])
                # Don't yield here — let the next outer-loop iteration
                # pull from the new generator naturally. This keeps the
                # "yielded zero on this cycle" bookkeeping simple.
            else:
                # Legacy behavior: drop the exhausted source.
                print(f"[curriculum] source {idx} "
                      f"({datasets[idx]}/{subsets[idx]}) exhausted; "
                      f"remaining sources continue.", flush=True)
                active_indices.pop(idx_in_active)


# ─────────────────────────────────────────────────────────────────────────
# Sequence packer
# ─────────────────────────────────────────────────────────────────────────

class SequencePacker:
    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else (tokenizer.bos_token_id or 0)
        )
        self.buffer: list[int] = []

    def feed(self, text: str):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.buffer.extend(ids)
        self.buffer.append(self.eos)

    def drain(self):
        chunk_len = self.seq_len + 1
        while len(self.buffer) >= chunk_len:
            chunk = self.buffer[:chunk_len]
            self.buffer = self.buffer[chunk_len:]
            yield chunk


# ─────────────────────────────────────────────────────────────────────────
# Teacher forward → top-K
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def teacher_forward_topk(
    teacher,
    input_ids: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (top_indices [int32], top_values [fp16], residual_log_mass [fp16])."""
    out = teacher(input_ids)
    logits = out.logits

    # Top-K of logits == top-K of softmax(logits/T) for any T > 0,
    # so storing raw logits leaves temperature free at training time.
    top_v, top_i = logits.topk(top_k, dim=-1)

    # Stable log(1 - P_top) using the log1mexp identity.
    full_lse = torch.logsumexp(logits.float(), dim=-1)
    top_lse = torch.logsumexp(top_v.float(), dim=-1)
    log_p_top = (top_lse - full_lse).clamp(max=-1e-7)
    threshold = -float(np.log(2.0))
    residual_log_mass = torch.where(
        log_p_top > threshold,
        torch.log(-torch.expm1(log_p_top)),
        torch.log1p(-torch.exp(log_p_top)),
    )

    return (
        top_i.to(torch.int32),
        top_v.to(torch.float16),
        residual_log_mass.to(torch.float16),
    )


# ─────────────────────────────────────────────────────────────────────────
# Shard writer (atomic via .tmp + rename)
# ─────────────────────────────────────────────────────────────────────────

def write_shard(
    out_dir: Path,
    shard_idx: int,
    input_ids_full: np.ndarray,
    top_indices: np.ndarray,
    top_values: np.ndarray,
    residual_log_mass: np.ndarray,
):
    """Atomic shard write.

    np.savez auto-appends '.npz' if the path doesn't already end in '.npz',
    so the tmp filename MUST end in '.npz' or os.replace can't find it.
    We use '<name>.tmp.npz' for the in-flight write and rename to
    '<name>.npz' on success.
    """
    final = out_dir / f"shard_{shard_idx:06d}.npz"
    tmp = out_dir / f".shard_{shard_idx:06d}.tmp.npz"
    inputs = input_ids_full[:, :-1].astype(np.int32)
    targets = input_ids_full[:, 1:].astype(np.int32)
    np.savez(
        tmp,
        input_ids=inputs,
        targets=targets,
        top_indices=top_indices,
        top_values=top_values,
        residual_log_mass=residual_log_mass,
    )
    os.replace(tmp, final)


# ─────────────────────────────────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Curriculum
    p.add_argument("--phase", type=int, choices=[1, 2], required=True,
                   help="Which curriculum phase to cache. Phase 1 = logic-"
                        "priming mix; Phase 2 = full-competence mix. The "
                        "two phases must be cached into different --output_dir's.")
    p.add_argument("--datasets", type=str, nargs="*", default=None,
                   help="Override curriculum datasets list. If omitted, defaults "
                        "are pulled from CTMConfig (same source as training).")
    p.add_argument("--subsets", type=str, nargs="*", default=None)
    p.add_argument("--weights", type=float, nargs="*", default=None)

    # Model / tokenizer
    p.add_argument("--teacher_model", type=str,
                   default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                   help="HF model ID for the teacher. Default is the BF16 variant. "
                        "FP8 should NOT be used on Ampere — see file docstring.")
    p.add_argument("--tokenizer", type=str,
                   default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                   help="HF tokenizer. Defaults to the teacher's. 'hf:' prefix "
                        "is accepted for compatibility with train.py.")

    # Paths
    p.add_argument("--data_cache_dir", type=str, default="./data_cache",
                   help="Directory for downloaded parquet files. Defaults to "
                        "./data_cache, matching CurriculumDataset's default — "
                        "if you've trained before, parquets are already here "
                        "and no re-download happens.")
    p.add_argument("--output_dir", type=str, required=True)

    # Cache shape
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--top_k", type=int, default=128,
                   help="Top-K logits per token. K=128 keeps gradient cosine "
                        ">0.95 vs full-vocab KL (per the throughput report) "
                        "while halving storage vs K=256. K=256 is the safer "
                        "choice if disk allows.")
    p.add_argument("--samples_per_shard", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=1_000_000,
                   help="Stop after this many samples cached for THIS phase. "
                        "At seq_len=512: 1M samples ≈ 512M tokens. Used in "
                        "addition to --max_cache_gb; whichever limit is hit "
                        "first wins.")
    p.add_argument("--max_cache_gb", type=float, default=0.0,
                   help="Hard cap on the cache directory size in GB. The "
                        "producer checks the directory size after each shard "
                        "and exits cleanly when this is reached. 0 disables "
                        "(only --max_samples gates the run).")
    p.add_argument("--no_stratified", action="store_true",
                   help="Disable stratified sampling. Falls back to sequential "
                        "parquet reads (matches the live training data path "
                        "exactly but caches the first slice of each source "
                        "rather than a representative sample).")
    p.add_argument("--no_cycle", action="store_true",
                   help="Disable cycling exhausted sources. By default, when "
                        "a small source (e.g. Formal-Logic at 18M tokens) "
                        "runs out, it's restarted with a fresh shuffle so "
                        "the configured mixing weights stay accurate. With "
                        "--no_cycle, exhausted sources are dropped from the "
                        "pool, concentrating the cache on whichever sources "
                        "have the most data regardless of weights.")

    # Compute
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--teacher_batch_size", type=int, default=8,
                   help="Teacher forward batch size. No backward, no grad checkpoint, "
                        "no student to compete with — can be MUCH larger than "
                        "training batch size. B=8..16 fits NemotronH-4B BF16 on a 3090.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])

    # Multi-GPU data-parallel caching. To run two producers in parallel:
    #   GPU 0:  python -m scripts.cache_teacher_logits ... --rank 0 --world_size 2 --device cuda:0
    #   GPU 1:  python -m scripts.cache_teacher_logits ... --rank 1 --world_size 2 --device cuda:1
    # The two processes write disjoint shard indices (rank N takes
    # shards N, N+W, N+2W, ...) into the SAME --output_dir. Each rank
    # uses an independent random stream over the curriculum so the
    # cached samples are statistically distinct (not duplicates).
    # Disk-cap is a shared global check across both ranks — whichever
    # rank notices the cap is hit first will exit, and the other rank
    # will exit on its next shard boundary.
    p.add_argument("--rank", type=int, default=0,
                   help="This producer's rank in a multi-GPU caching run. "
                        "Use 0 (default) for single-process. For two GPUs, "
                        "launch two processes with --rank 0 and --rank 1.")
    p.add_argument("--world_size", type=int, default=1,
                   help="Total number of producer processes. Each rank "
                        "writes shards N, N+W, N+2W, ... into the shared "
                        "--output_dir. Default 1 (single-process).")

    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Curriculum interleave seed.")
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def resolve_curriculum(args) -> tuple[list[str], list[str], list[float]]:
    if args.datasets or args.subsets or args.weights:
        if not (args.datasets and args.subsets and args.weights):
            print("ERROR: --datasets, --subsets, --weights must all be supplied "
                  "together if any one is.", flush=True)
            sys.exit(1)
        return args.datasets, args.subsets, args.weights

    cfg = CTMConfig()
    if args.phase == 1:
        return (list(cfg.phase1_datasets),
                list(cfg.phase1_dataset_subsets),
                list(cfg.phase1_dataset_weights))
    else:
        return (list(cfg.phase2_datasets),
                list(cfg.phase2_dataset_subsets),
                list(cfg.phase2_dataset_weights))


def main():
    args = parse_args()
    datasets, subsets, weights = resolve_curriculum(args)

    # Lazy import — keeps this module loadable in minimal environments
    # (e.g. test runners that don't have transformers installed).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Tokenizer
    tok_id = (args.tokenizer[3:] if args.tokenizer.startswith("hf:")
              else args.tokenizer)
    print(f"Loading tokenizer: {tok_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    # Teacher
    print(f"Loading teacher: {args.teacher_model} on {args.device}", flush=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        device_map={"": args.device},
    )
    teacher.eval()
    teacher.requires_grad_(False)
    vocab_size = teacher.config.vocab_size
    if args.top_k >= vocab_size:
        raise ValueError(f"top_k={args.top_k} must be < vocab_size={vocab_size}")
    print(f"Teacher vocab: {vocab_size}, caching top-{args.top_k}", flush=True)

    # ── Multi-rank validation ─────────────────────────────────────────
    rank = args.rank
    world_size = args.world_size
    if not (0 <= rank < world_size):
        raise ValueError(f"--rank ({rank}) must be in [0, --world_size={world_size})")
    if world_size > 1:
        print(f"[multi-rank] this process is rank {rank} of {world_size}. "
              f"Will write shards {rank}, {rank + world_size}, "
              f"{rank + 2*world_size}, ... into {args.output_dir}.",
              flush=True)

    # ── Output dir + rank-aware resume detection ───────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Look at ALL existing shards (so we can report total progress
    # across both ranks), but compute next_shard for THIS rank only.
    all_existing = sorted(out_dir.glob("shard_*.npz"))
    my_existing = [
        p for p in all_existing
        if (int(p.stem.split("_")[1]) % world_size) == rank
    ]
    if my_existing:
        last_mine = int(my_existing[-1].stem.split("_")[1])
        # next_shard for this rank is the smallest index > last_mine
        # that satisfies idx % world_size == rank.
        next_shard = last_mine + world_size
        print(f"Resume: rank {rank} sees {len(my_existing)} of its own shards "
              f"(global total: {len(all_existing)}); next index = {next_shard}",
              flush=True)
    else:
        next_shard = rank
        if all_existing:
            print(f"Resume: rank {rank} has no shards yet; "
                  f"global total: {len(all_existing)}. Starting at {next_shard}.",
                  flush=True)

    # samples_to_skip is per-RANK: how many samples this rank's stream
    # should skip to align with what it has already written.
    samples_to_skip = (next_shard - rank) // world_size * args.samples_per_shard
    samples_done = samples_to_skip

    # Validate disk-cap configuration. The cap is GLOBAL across all
    # ranks (computed against the whole output_dir), so two ranks
    # writing into the same dir share the budget.
    max_cache_bytes = (
        int(args.max_cache_gb * (1024 ** 3)) if args.max_cache_gb > 0 else 0
    )
    if max_cache_bytes > 0:
        print(f"[disk-cap] cache size limit: {args.max_cache_gb:.1f} GB "
              f"({max_cache_bytes:,} bytes), shared across all ranks. "
              f"Will exit cleanly when reached.",
              flush=True)

    def _cache_dir_size_bytes(d: Path) -> int:
        """Sum of regular-file sizes under d. Cheap; called once per shard."""
        total = 0
        for f in d.iterdir():
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    # Curriculum seed is rank-mixed so each rank sees an INDEPENDENT
    # stream over the data. Without this, both ranks would generate the
    # same documents in the same order and produce duplicate shards
    # (just stored under different indices). With this, the cached
    # samples across all ranks are statistically diverse.
    rank_seed = (args.seed + rank * 1_000_003) & 0xFFFFFFFF
    text_iter = stream_curriculum_text(
        datasets=datasets, subsets=subsets, weights=weights,
        data_cache_dir=args.data_cache_dir, seed=rank_seed,
        stratified=(not args.no_stratified),
        cycle_exhausted=(not args.no_cycle),
    )
    packer = SequencePacker(tokenizer, args.seq_len)

    # Skip-ahead so resume aligns
    if samples_to_skip > 0:
        print(f"Skipping {samples_to_skip} already-cached samples (this rank)...",
              flush=True)
        skipped = 0
        for doc in text_iter:
            packer.feed(doc["text"])
            for _chunk in packer.drain():
                skipped += 1
                if skipped >= samples_to_skip:
                    break
            if skipped >= samples_to_skip:
                break
        print(f"  ...resume aligned at sample {skipped}.", flush=True)

    # Producer loop. shard_idx increments by world_size so this rank
    # only writes to its assigned shard slots; the other rank fills the
    # gaps.
    pending: list[list[int]] = []
    shard_idx = next_shard
    t0 = time.time()
    tokens_processed = 0
    disk_full = False

    for doc in text_iter:
        if samples_done >= args.max_samples or disk_full:
            break
        packer.feed(doc["text"])
        for chunk in packer.drain():
            pending.append(chunk)
            if len(pending) >= args.samples_per_shard:
                arr = np.array(pending[:args.samples_per_shard], dtype=np.int64)
                pending = pending[args.samples_per_shard:]

                N, Sp1 = arr.shape
                S = Sp1 - 1
                K = args.top_k
                inputs_only = arr[:, :S]

                top_i_buf = np.zeros((N, S, K), dtype=np.int32)
                top_v_buf = np.zeros((N, S, K), dtype=np.float16)
                resid_buf = np.zeros((N, S), dtype=np.float16)

                for bs in range(0, N, args.teacher_batch_size):
                    be = min(bs + args.teacher_batch_size, N)
                    inp = torch.from_numpy(inputs_only[bs:be]).to(args.device)
                    ti, tv, rm = teacher_forward_topk(teacher, inp, K)
                    top_i_buf[bs:be] = ti.cpu().numpy()
                    top_v_buf[bs:be] = tv.cpu().numpy()
                    resid_buf[bs:be] = rm.cpu().numpy()

                write_shard(out_dir, shard_idx, arr,
                            top_i_buf, top_v_buf, resid_buf)

                tokens_processed += N * S
                samples_done += N

                # Disk-cap check, AFTER the shard is on disk so the
                # number we report is honest. Costs one stat() per file
                # which on NVMe is sub-millisecond per shard.
                cache_bytes = 0
                if max_cache_bytes > 0:
                    cache_bytes = _cache_dir_size_bytes(out_dir)
                    if cache_bytes >= max_cache_bytes:
                        disk_full = True

                if shard_idx % args.log_interval == 0 or disk_full:
                    elapsed = time.time() - t0
                    rate = tokens_processed / max(elapsed, 1e-6)
                    remaining_samples = (
                        max(args.max_samples - samples_done, 0)
                        if max_cache_bytes == 0
                        else max(
                            (max_cache_bytes - cache_bytes)
                            // max(cache_bytes // max(samples_done - samples_to_skip, 1), 1),
                            0,
                        )
                    )
                    eta_h = remaining_samples * S / max(rate, 1.0) / 3600
                    cap_str = (
                        f"cache={cache_bytes/(1024**3):.1f}GB "
                        if max_cache_bytes > 0 else ""
                    )
                    rank_str = (
                        f"rank={rank}/{world_size} "
                        if world_size > 1 else ""
                    )
                    print(f"[shard {shard_idx:>5d}] {rank_str}phase={args.phase} "
                          f"samples={samples_done:>9d} {cap_str}"
                          f"tok/s={rate:>8.0f} eta={eta_h:>5.1f}h",
                          flush=True)
                # Increment by world_size so this rank only writes to its
                # assigned shard slots. With world_size=1 this collapses
                # to the single-process behavior.
                shard_idx += world_size
                if samples_done >= args.max_samples or disk_full:
                    break

    if disk_full:
        print(f"Disk cap reached ({args.max_cache_gb:.1f} GB). "
              f"Stopping cleanly.", flush=True)
    new_shards_this_run = (shard_idx - next_shard) // world_size
    print(f"Done. Rank {rank} wrote {new_shards_this_run} new shards "
          f"({samples_done} samples total for phase {args.phase}).",
          flush=True)


if __name__ == "__main__":
    main()