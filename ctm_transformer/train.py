"""
CTM-Transformer Training Script

Single-file training entry point. Combines what used to be train.py +
train_offload.py + the small ancillary modules (PhaseTimer, the cached-
teacher dataset). The original module structure is preserved as section
banners below.

Two run modes are exposed:
  - Single-GPU / torchrun:  python -m ctm_transformer.train [args]
  - Dual-GPU mp.spawn:      python -m ctm_transformer.train --multi_gpu [args]

The dispatch happens at the bottom of the file: when --multi_gpu is set,
main() forwards to main_multi_gpu(), which spawns one process per GPU and
runs worker_multi_gpu() on each. Otherwise the standard single-process
train() runs (with optional torchrun-driven DDP).
"""

from __future__ import annotations

import time
import statistics
from contextlib import contextmanager
from collections import defaultdict
import torch
import os
import random
from pathlib import Path
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

# After consolidation, the model components and the AdaMuon optimizer live
# in sibling modules. CTMConfig stays in config.py (unchanged).
from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer, EngramTable
from ctm_transformer.extras import AdaMuon, build_param_groups
from ctm_transformer.validation import (
    compute_validation_metrics,
    compute_validation_metrics_t_sweep,
    format_validation_report,
    format_t_sweep_report,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import math
import sys
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
import pyarrow.parquet as pq
os.environ["TORCHINDUCTOR_SPLIT_REDUCTIONS"] = "1"
os.environ["TORCHINDUCTOR_ONLINE_SOFTMAX"] = "1"
import warnings

warnings.filterwarnings("ignore", message=".*Online softmax is disabled on the fly.*")
# ═════════════════════════════════════════════════════════════════════════
# phase_timer.py
# ═════════════════════════════════════════════════════════════════════════

class PhaseTimer:
    def __init__(self, enabled: bool = True, warmup_steps: int = 5,
                 track_memory: bool = True):
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.step_count = 0
        self._is_cuda = torch.cuda.is_available()
        self._in_warmup = True

        # Memory tracking
        self.track_memory = track_memory and self._is_cuda
        self.mem_peak: dict[str, list[int]] = defaultdict(list)   # bytes, peak DURING phase
        self.mem_delta: dict[str, list[int]] = defaultdict(list)  # bytes, end - start (signed)
        self.step_peak_alloc: list[int] = []                      # peak allocated in step
        self.step_end_reserved: list[int] = []                    # caching-allocator pool size at step end
        self._step_phase_peaks: list[int] | None = None

    def _sync(self):
        if self._is_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def step(self):
        """Wrap each training step."""
        if not self.enabled:
            yield
            return
        self.step_count += 1
        self._in_warmup = self.step_count <= self.warmup_steps
        record_mem = self.track_memory and not self._in_warmup
        if record_mem:
            self._step_phase_peaks = []
        try:
            yield
        finally:
            if record_mem:
                self._sync()
                if self._step_phase_peaks:
                    self.step_peak_alloc.append(max(self._step_phase_peaks))
                else:
                    self.step_peak_alloc.append(torch.cuda.memory_allocated())
                self.step_end_reserved.append(torch.cuda.memory_reserved())
                self._step_phase_peaks = None

    @contextmanager
    def __call__(self, name: str):
        """Time + memory-track a phase. No-ops in warmup."""
        if not self.enabled or self._in_warmup:
            yield
            return
        self._sync()
        track_mem = self.track_memory
        if track_mem:
            # Note: peak counter is per-device global. Phases are sequential
            # in this codebase; nesting would clobber the inner phase's peak.
            torch.cuda.reset_peak_memory_stats()
            start_alloc = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[name].append(time.perf_counter() - t0)
            if track_mem:
                end_alloc = torch.cuda.memory_allocated()
                peak = torch.cuda.max_memory_allocated()
                self.mem_peak[name].append(peak)
                self.mem_delta[name].append(end_alloc - start_alloc)
                if self._step_phase_peaks is not None:
                    self._step_phase_peaks.append(peak)

    def report(self, last_n: int | None = None, sort_by: str = "mean"):
        if not self.timings:
            print("[PhaseTimer] no samples yet")
            return

        steps_recorded = max(1, self.step_count - self.warmup_steps)
        rows = []
        total_mean_per_step = 0.0

        def _avg(seq):
            return sum(seq) / len(seq) if seq else 0.0

        for name, samples in self.timings.items():
            true_n = len(self.timings[name])
            if last_n is not None:
                samples = samples[-last_n:]
            n = len(samples)
            if n == 0:
                continue
            mean_ms = statistics.mean(samples) * 1000
            med_ms = statistics.median(samples) * 1000
            sorted_s = sorted(samples)
            p95_ms = sorted_s[min(int(n * 0.95), n - 1)] * 1000
            mean_per_step = mean_ms * (true_n / steps_recorded)

            # Memory: peak-during-phase and net delta
            peak_samples = self.mem_peak.get(name, [])
            delta_samples = self.mem_delta.get(name, [])
            if last_n is not None:
                peak_samples = peak_samples[-last_n:]
                delta_samples = delta_samples[-last_n:]
            peak_gb = _avg(peak_samples) / (1024**3)
            delta_mb = _avg(delta_samples) / (1024**2)

            rows.append((name, mean_ms, med_ms, p95_ms, n, mean_per_step,
                         peak_gb, delta_mb))
            total_mean_per_step += mean_per_step

        if sort_by == "mean":
            rows.sort(key=lambda r: -r[1])
        elif sort_by == "median":
            rows.sort(key=lambda r: -r[2])
        else:
            rows.sort(key=lambda r: r[0])

        bar = "─" * 96
        print()
        print(bar)
        print(f"PhaseTimer  (step {self.step_count}, warmup {self.warmup_steps})")
        print(bar)
        print(f"{'phase':<22s} {'mean(ms)':>9s} {'median':>8s} {'p95':>8s} "
              f"{'n':>4s} {'%total':>7s} {'peak(GB)':>9s} {'Δ(MB)':>9s}")
        print(bar)
        for name, mean_ms, med_ms, p95_ms, n, mean_per_step, peak_gb, delta_mb in rows:
            pct = 100 * mean_per_step / total_mean_per_step if total_mean_per_step > 0 else 0
            print(f"{name:<22s} {mean_ms:9.2f} {med_ms:8.2f} {p95_ms:8.2f} "
                  f"{n:4d} {pct:6.1f}% {peak_gb:9.2f} {delta_mb:+9.1f}")
        print(bar)
        print(f"{'TOTAL (mean per step)':<22s} {total_mean_per_step:9.2f} ms")

        # Step-level memory summary
        if self.track_memory and self.step_peak_alloc:
            sp = self.step_peak_alloc[-last_n:] if last_n else self.step_peak_alloc
            sr = self.step_end_reserved[-last_n:] if last_n else self.step_end_reserved
            print(f"{'step peak allocated':<22s} {_avg(sp)/1024**3:.2f} GB "
                  f"(max {max(sp)/1024**3:.2f} GB)")
            print(f"{'step end reserved':<22s} {_avg(sr)/1024**3:.2f} GB "
                  f"(max {max(sr)/1024**3:.2f} GB)")
        print(bar)
        print()

# ═════════════════════════════════════════════════════════════════════════
# cached_teacher_dataset.py
# ═════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════
# train_clean.py
# ═════════════════════════════════════════════════════════════════════════

def is_torchrun_launch() -> bool:
    """True if we were launched via torchrun (RANK + LOCAL_RANK + WORLD_SIZE
    in env). torchrun sets all three; presence of any one is sufficient
    in practice but we check the canonical RANK + LOCAL_RANK pair."""
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ


def setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed if running under torchrun.

    Returns (rank, world_size, local_rank). When not running under
    torchrun, returns (0, 1, 0) and is a no-op.

    Side effect: on non-rank-0 processes, replaces the built-in `print`
    with a no-op so verbose startup banners and per-step logs only
    appear once. Errors and explicit `sys.stderr.write` calls are NOT
    silenced.
    """
    if not is_torchrun_launch():
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # NCCL is the right backend for CUDA — supports tensor reductions
    # directly between GPU memory regions. The Gloo CPU backend works
    # everywhere but is significantly slower for our workload.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    # Silence non-rank-0 stdout. We do this BEFORE returning so the
    # rest of training emits a single rank's view in the log file.
    # Errors should still flow: stderr is unaffected, and assertion/
    # exception traces don't go through `print`.
    if rank != 0:
        import builtins
        _real_print = builtins.print
        def _silenced_print(*args, **kwargs):
            # Allow `flush=True` calls to flush stderr if anyone wrote
            # to it via print(file=sys.stderr); other calls are dropped.
            return None
        builtins.print = _silenced_print

    return rank, world_size, local_rank


def cleanup_distributed():
    """Tear down the process group at the end of training, if it was set up."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Rank-0 check — used to gate prints, checkpoint saves, eval."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def unwrap_model(m):
    """Return the underlying nn.Module if `m` is a DDP or OptimizedModule wrapper, else `m`.

    Use this anywhere we need to access non-Parameter attributes
    (e.g. `model._train_step`, `model.config`) — DDP and compile forward through
    `.forward()` but don't proxy arbitrary attribute lookups."""
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if hasattr(m, "module"):
        m = m.module
    return m

from huggingface_hub import HfFileSystem, hf_hub_download
from datasets import IterableDataset as HFIterableDataset, Features, Value

def sliding_window_generator(ds_name: str, subset: str, cache_dir: str, 
                             total_shards: int = 1, shard_index: int = 0,
                             is_eval: bool = False):
    """
    Custom generator that manually pulls parquet files from HF, 
    processes them, and deletes them to save disk space.
    """
    fs = HfFileSystem()
    path = f"datasets/{ds_name}/{subset}" if subset != "default" else f"datasets/{ds_name}"
    
    try:
        # Resolve all parquet files for this subset
        all_files = sorted([f for f in fs.ls(path, detail=False) if f.endswith('.parquet')])
    except Exception as e:
        print(f"Error listing files for {path}: {e}")
        return

    # Simple disjoint split: reserve the last 5% of files for evaluation
    n_eval = max(1, len(all_files) // 20)
    if is_eval:
        # Eval only sees the reserved tail
        my_files = all_files[-n_eval:]
    else:
        # Train sees everything except the reserved tail
        train_files = all_files[:-n_eval]
        # Shard the remaining file list across workers
        my_files = train_files[shard_index::total_shards]
    
    os.makedirs(cache_dir, exist_ok=True)

    for file_path in my_files:
        # Resolve the relative path in the repo
        repo_prefix = f"datasets/{ds_name}/"
        if file_path.startswith(repo_prefix):
            path_in_repo = file_path[len(repo_prefix):]
        else:
            path_in_repo = file_path.split("/")[-1]

        # Check if already in cache to avoid cluttering logs
        local_dest = os.path.join(cache_dir, path_in_repo)
        
        try:
            if not os.path.exists(local_dest):
                print(f"  [Worker {shard_index}] Downloading {path_in_repo}...", flush=True)
                # Use hf_hub_download for robust, atomic downloading with LFS support
                local_dest = hf_hub_download(
                    repo_id=ds_name,
                    filename=path_in_repo,
                    repo_type="dataset",
                    local_dir=cache_dir
                )
        except Exception as e:
            print(f"Error downloading {file_path}: {e}", flush=True)
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
            print(f"Error processing {local_dest}: {e}")
        finally:
            # We no longer delete the file here so that pre-downloaded 
            # data is preserved for future training runs.
            pass


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

class HFTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.n_vocab = len(tokenizer)
        
    def encode(self, text, allowed_special=None):
        return self.tokenizer.encode(text, add_special_tokens=False)
        
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

def get_tokenizer(config: CTMConfig):
    """
    Load a tiktoken BPE tokenizer or HuggingFace tokenizer.

    Returns:
        tokenizer: object with encode/decode methods and n_vocab attribute.
    """
    if config.tokenizer.startswith("hf:"):
        from transformers import AutoTokenizer
        tokenizer_name = config.tokenizer[3:]
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return HFTokenizerWrapper(hf_tokenizer)
    else:
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
        ).select_columns(["text"]).shard(num_shards=total_shards, index=shard_index)

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


class CurriculumDataset(IterableDataset):
    """
    Streaming interleaved dataset from HuggingFace for a specific curriculum phase.
    """

    def __init__(self, tokenizer, seq_len: int, datasets: list[str], subsets: list[str], 
                 weights: list[float], rank: int = 0, world_size: int = 1, is_eval: bool = False):
        self.is_eval = is_eval
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.datasets = datasets
        self.subsets = subsets
        self.weights = weights
        self.rank = rank
        self.world_size = world_size
        # Use a local directory on disk instead of /tmp (which is often RAM-backed)
        self.cache_dir = os.path.abspath("./data_cache")

    def __iter__(self):
        import random
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id
        
        print(f"  [Worker {shard_index}] Initialized - Processing curriculum dataset mix.", flush=True)

        # Initialize all generators
        generators = []
        for ds_name, subset in zip(self.datasets, self.subsets):
            gen = sliding_window_generator(
                ds_name=ds_name,
                subset=subset,
                cache_dir=self.cache_dir,
                total_shards=total_shards,
                shard_index=shard_index,
                is_eval=self.is_eval
            )
            generators.append(gen)

        # Custom weighted interleaving (all_exhausted strategy)
        rng = random.Random(42 + shard_index)
        active_indices = list(range(len(generators)))
        current_weights = list(self.weights)

        def get_sample():
            while active_indices:
                # Normalize weights for active indices
                active_weights = [current_weights[i] for i in active_indices]
                if sum(active_weights) == 0:
                    break
                
                idx_in_active = rng.choices(range(len(active_indices)), weights=active_weights)[0]
                idx = active_indices[idx_in_active]
                
                try:
                    return next(generators[idx])
                except StopIteration:
                    active_indices.pop(idx_in_active)
            return None

        # Pack documents into fixed-length chunks
        buf = []
        while True:
            sample = get_sample()
            if sample is None:
                break
                
            text_val = sample.get("text", "")
            tokens = self.tokenizer.encode(text_val, allowed_special=set())
            buf.extend(tokens)

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

    DDP-aware: the underlying model's state_dict is saved (no `module.`
    prefix), so checkpoints round-trip cleanly between single-GPU and
    DDP runs.
    """
    # Only rank 0 saves. Other ranks would either overwrite (race) or
    # waste I/O writing identical state.
    if not is_main_process():
        return
    underlying = unwrap_model(model)
    ckpt_dir = path.parent
    payload = {
        "model_state_dict": underlying.state_dict(),
        "optimizer_state_dicts": [o.state_dict() for o in optimizers],
        "step": step,
        "config": vars(config),
    }
    # Backward-compat: also write the single-opt key when there's only one.
    if len(optimizers) == 1:
        payload["optimizer_state_dict"] = optimizers[0].state_dict()
    torch.save(payload, path)

    # Rotate old step checkpoints (keep last N).
    # IMPORTANT: sort by step number, NOT lexicographically. Default sort
    # would put "step_100000.pt" before "step_85000.pt" because '1' < '8',
    # which causes the rotation logic to delete the most recent checkpoint
    # (thinking it's the oldest) once step counts cross the next digit.
    def _step_num(p):
        try:
            return int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    step_ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=_step_num)
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

    DDP-aware: state is loaded into the underlying module so the
    `module.`-prefix accidentally written by older DDP-naive code paths
    is not produced.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    underlying = unwrap_model(model)
    # Strip possible prefixes from older/compiled checkpoints
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    underlying.load_state_dict(sd)

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

def train(
    config: CTMConfig,
    profile: bool = False,
    profile_warmup: int = 10,
    profile_interval: int = 50,
):
    """Main training loop.

    Args:
        config: Model and training configuration.
        profile: If True, time each per-step phase (data, teacher_fwd,
            student_fwd, backward, grad_clip, optimizer) and print a
            summary every ``profile_interval`` steps. Adds a CUDA
            synchronization at each phase boundary, so leave off for
            production runs.
        profile_warmup: Skip this many initial steps before recording.
        profile_interval: How often (in steps) to print the report.
    """
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
    use_cached_teacher = config.use_cached_teacher
    use_fineweb = config.dataset == "fineweb-edu"
    use_curriculum = config.use_two_phase_curriculum
    first_batch = None

    if use_cached_teacher:
        # ── Offline-cached teacher path ──────────────────────────────────
        # Highest precedence: when this flag is set, all other data
        # source choices are overridden — the cache fully determines
        # what the student sees, and the live teacher is dropped from
        # the training process.
        #
        # Layout:
        #   - With --use_two_phase_curriculum: expects two subdirectories
        #     named phase1/ and phase2/ under teacher_cache_dir, each
        #     populated by `scripts/cache_teacher_logits.py --phase {1,2}`.
        #   - Without curriculum: shards live directly under teacher_cache_dir.
        if not config.teacher_cache_dir:
            print("ERROR: --use_cached_teacher requires --teacher_cache_dir")
            sys.exit(1)
        if not config.use_distillation:
            print("[cached-teacher] auto-enabling use_distillation "
                  "(implied by --use_cached_teacher)")
            config.use_distillation = True

        from pathlib import Path as _Path
        cache_root = _Path(config.teacher_cache_dir)
        phase1_dir = cache_root / "phase1"
        phase2_dir = cache_root / "phase2"

        # Read distributed sharding info. When DDP is on, the train
        # dataset round-robins shards across ranks so every rank sees
        # a disjoint subset; the eval dataset always uses rank=0/world=1
        # because eval runs on rank 0 only.
        ddp_rank = dist.get_rank() if dist.is_initialized() else 0
        ddp_world = dist.get_world_size() if dist.is_initialized() else 1

        def _make_loader(cache_dir, *, is_eval=False, shuffle=True, loop=True,
                         num_workers=2, prefetch_factor=4):
            ds = CachedTeacherDataset(
                cache_dir=str(cache_dir),
                # Eval runs on rank 0 only; train shards round-robin
                # across ranks so each sees a unique slice.
                rank=0 if is_eval else ddp_rank,
                world_size=1 if is_eval else ddp_world,
                shuffle_shards=shuffle,
                loop=loop,
                is_eval=is_eval,
            )
            if ds.seq_len != config.seq_len:
                print(f"ERROR: cache at {cache_dir} has seq_len={ds.seq_len} "
                      f"but config.seq_len={config.seq_len}. "
                      f"Re-run the cache producer with --seq_len {config.seq_len}.")
                sys.exit(1)
            loader = DataLoader(
                ds,
                batch_size=config.batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                pin_memory=(device != "cpu"),
                persistent_workers=(num_workers > 0),
                collate_fn=cached_teacher_collate,
            )
            return ds, loader

        if use_curriculum:
            # Curriculum cache: phase1/ and phase2/ subdirectories.
            if not phase1_dir.exists():
                print(f"ERROR: --use_two_phase_curriculum + --use_cached_teacher "
                      f"requires {phase1_dir} to exist. "
                      f"Run scripts/cache_teacher_logits.py --phase 1 first.")
                sys.exit(1)

            phase2_missing = not phase2_dir.exists()
            if phase2_missing:
                # Allow validation runs / Phase 2 cache build in parallel
                # with training. Phase 2 won't actually be consumed until
                # tokens_seen >= phase1_tokens (default 500M), so it's
                # safe to bootstrap with phase 1 data here. The training
                # loop will need a real phase 2 cache by the time the
                # curriculum switch fires; if it doesn't, the switch
                # will use phase 1 again, which trains correctly but
                # without the curriculum's intended distribution shift.
                print(f"WARNING: {phase2_dir} does not exist. "
                      f"Using {phase1_dir} as the Phase 2 fallback. "
                      f"Build the real Phase 2 cache before training "
                      f"reaches phase1_tokens={config.phase1_tokens:,}.")

            print(f"Dataset: Cached Teacher Logits (curriculum mode)")
            print(f"  Phase 1: {phase1_dir}")
            print(f"  Phase 2: {phase2_dir if not phase2_missing else f'{phase1_dir} (fallback)'}")

            _p1_ds, phase1_loader = _make_loader(phase1_dir, is_eval=False)
            _p2_dir_actual = phase1_dir if phase2_missing else phase2_dir
            _p2_ds, phase2_loader = _make_loader(_p2_dir_actual, is_eval=False)
            # Eval pulls from phase 2 (the more recent / harder mix); using
            # is_eval=True reserves the last 5% of phase-2 shards. When
            # phase 2 is missing, eval falls back to the phase 1 tail.
            _eval_ds, eval_loader = _make_loader(
                _p2_dir_actual, is_eval=True, shuffle=False, loop=False,
                num_workers=0,
            )

            # Match the live-teacher curriculum control state.
            train_loader = phase1_loader
            current_phase = 1
            raw_text = None
            print("  Waiting for initial data pre-fetch...", end="", flush=True)
            train_iter = iter(train_loader)
            try:
                first_batch = next(train_iter)
                print(" Done.")
            except StopIteration:
                print(" FAILED: phase1 cache directory is empty!")
                sys.exit(1)
        else:
            # Single-cache (no curriculum) path.
            print(f"Dataset: Cached Teacher Logits ({cache_root})")
            _train_ds, train_loader = _make_loader(cache_root, is_eval=False)
            _eval_ds, eval_loader = _make_loader(
                cache_root, is_eval=True, shuffle=False, loop=False,
                num_workers=0,
            )
            raw_text = None
            print("  Waiting for initial data pre-fetch...", end="", flush=True)
            train_iter = iter(train_loader)
            try:
                first_batch = next(train_iter)
                print(" Done.")
            except StopIteration:
                print(" FAILED: cache directory is empty!")
                sys.exit(1)

    elif use_curriculum:
        print(f"Dataset: Curriculum Mode (Phase 1 & Phase 2 Interleaved)")
        print(f"  Streaming mode — data loaded on the fly from HuggingFace Hub")
        
        phase1_dataset = CurriculumDataset(
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            datasets=config.phase1_datasets,
            subsets=config.phase1_dataset_subsets,
            weights=config.phase1_dataset_weights,
            rank=0,
            world_size=1,
            is_eval=False,
        )
        phase2_dataset = CurriculumDataset(
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            datasets=config.phase2_datasets,
            subsets=config.phase2_dataset_subsets,
            weights=config.phase2_dataset_weights,
            rank=0,
            world_size=1,
            is_eval=False,
        )
        # Create a dedicated evaluation dataset instance using the eval split
        eval_dataset = CurriculumDataset(
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            datasets=config.phase2_datasets,
            subsets=config.phase2_dataset_subsets,
            weights=config.phase2_dataset_weights,
            rank=0,
            world_size=1,
            is_eval=True,
        )
        
        phase1_loader = DataLoader(
            phase1_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            prefetch_factor=8,
            pin_memory=(device != "cpu"),
            persistent_workers=True,
        )
        phase2_loader = DataLoader(
            phase2_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            prefetch_factor=8,
            pin_memory=(device != "cpu"),
            persistent_workers=True,
        )
        
        # Eval uses dedicated eval split
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=(device != "cpu"),
        )
        raw_text = None
        
        # Start with Phase 1 loader
        train_loader = phase1_loader
        current_phase = 1

        print("  Waiting for initial data pre-fetch...", end="", flush=True)
        train_iter = iter(train_loader)
        try:
            first_batch = next(train_iter)
            print(" Done.")
        except StopIteration:
            print(" FAILED: Dataset is empty!")
            sys.exit(1)
        
    elif use_fineweb:
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
            num_workers=4,
            prefetch_factor=8,
            pin_memory=(device != "cpu"),
            persistent_workers=True,
        )
        # For eval, use a small separate stream
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=(device != "cpu"),
        )
        raw_text = None  # No raw text buffer for generation prompts

        print("  Waiting for initial data pre-fetch...", end="", flush=True)
        train_iter = iter(train_loader)
        try:
            first_batch = next(train_iter)
            print(" Done.")
        except StopIteration:
            print(" FAILED: Dataset is empty!")
            sys.exit(1)

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
            num_workers=1,
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
        train_iter = iter(train_loader)
        raw_text = None  # Not needed for local file mode usually, but keep for consistency
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
    # Legacy configurations use the "manual cast" pattern (cast the entire model to the target
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
    if config.bf16_autocast:
        if config.dtype != "bfloat16" or not device.startswith("cuda"):
            raise ValueError("bf16_autocast requires dtype=bfloat16 and a CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected GPU does not support BF16 autocast")
        # Research mode retains FP32 parameters and optimizer moments.
        amp_dtype = torch.bfloat16
    elif config.dtype == "bfloat16" and device != "cpu":
        model = model.to(torch.bfloat16)
        amp_dtype = torch.float32   # autocast disabled; model is already bf16
    elif config.dtype == "float16" and device != "cpu":
        model = model.to(torch.float16)
        amp_dtype = torch.float32   # autocast disabled; model is already fp16
    else:
        amp_dtype = torch.float32

    n_params = model.get_num_params(non_embedding=False)
    if is_main_process():
        print(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
        print(f"Config: d_model={config.d_model}, d_latent={config.d_latent}, "
              f"n_layers={config.n_layers}, n_heads={config.n_heads}, "
              f"thought_steps={config.max_thought_steps}, history_len={config.history_len}, "
              f"nlm_hidden={config.nlm_hidden_dim}, nlm_groups={config.nlm_groups}")
        if config.use_hyperloop:
            print(f"Hyperloop: Enabled (begin={config.hyperloop_n_begin}, "
                  f"middle={config.hyperloop_n_middle}x{config.hyperloop_middle_loops}, "
                  f"end={config.hyperloop_n_end})")

    # ── DDP wrap ────────────────────────────────────────────────────────
    # If running under torchrun, wrap the model so backward triggers
    # all-reduce of gradients across ranks. DDP must come AFTER:
    #   - .to(device)         (every rank's params live on the right card)
    #   - .to(dtype)          (parameter dtypes are final)
    # and BEFORE:
    #   - optimizer construction (so the optimizer sees the wrapped params)
    #
    # find_unused_parameters=True is REQUIRED here because our model has
    # tick-conditional execution (FiLM picks 1 of T gamma/beta vectors
    # per forward, the other T-1 receive no gradient that step). Without
    # this flag, DDP's all-reduce hangs forever waiting for grads on the
    # unused params.
    #
    # static_graph=True is a *post-init* speedup we set after one warm-up
    # step (PyTorch checks for it once and caches the unused-param set
    # going forward). We can't set both flags at construction; the order
    # is: construct with find_unused=True → run one step → set static.
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(
            model,
            device_ids=[local_rank] if device.startswith("cuda") else None,
            output_device=local_rank if device.startswith("cuda") else None,
            find_unused_parameters=True,
        )
        if is_main_process():
            print(f"[DDP] Wrapped model. world_size={dist.get_world_size()}, "
                  f"this rank={dist.get_rank()}, local_rank={local_rank}")

    # ── Teacher Model ───────────────────────────────────────────────────
    teacher_model = None
    teacher_device = None  # resolved below for use in the training loop
    if config.use_cached_teacher:
        # Cache-based distillation: nothing to load, nothing to put on
        # cuda:1. The student loop reads top-K logits straight from disk.
        print("[cached-teacher] Skipping teacher model load — "
              "top-K logits will be streamed from "
              f"{config.teacher_cache_dir}")
    elif config.use_distillation:
        # Resolve teacher device. "auto" colocates with the student;
        # an explicit "cuda:N" places the teacher on a separate GPU.
        # Validate that the requested device is actually available so
        # we fail loudly at startup, not 200 steps in with a CUDA error.
        if config.teacher_device == "auto":
            teacher_device = device
        else:
            teacher_device = config.teacher_device
            if teacher_device.startswith("cuda:"):
                idx = int(teacher_device.split(":")[1])
                if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"--teacher_device={teacher_device} requested but only "
                        f"{torch.cuda.device_count()} CUDA device(s) visible. "
                        f"Check CUDA_VISIBLE_DEVICES."
                    )

        try:
            from transformers import AutoModelForCausalLM
            colocated = (torch.device(teacher_device) == torch.device(device))
            placement_msg = (
                f"colocated with student on {device}"
                if colocated
                else f"on separate device {teacher_device} (student on {device})"
            )
            print(f"Loading teacher model: {config.teacher_model_name} ({placement_msg})...")
            teacher_model = AutoModelForCausalLM.from_pretrained(
                config.teacher_model_name,
                torch_dtype="auto",
                device_map={"": teacher_device},
                trust_remote_code=True
            )
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            print("Teacher model loaded successfully.")
            if not colocated:
                print(
                    f"  Teacher inputs will be transferred to {teacher_device} "
                    f"and outputs back to {device} via non_blocking copies, "
                    f"overlapping with student compute."
                )
        except Exception as e:
            print(f"WARNING: Failed to load teacher model: {e}")
            print("Proceeding without distillation. If this is unexpected, please verify model name and access.")
            config.use_distillation = False
            teacher_model = None
            teacher_device = None

    # Temporal-loss schedule summary. Helps verify the decay plan matches
    # expectations before kicking off a multi-day run.
    base_mono = config.mono_penalty_weight
    loss_description = (
        f"ramp[{config.tick_ramp_start}→{config.tick_ramp_end}]"
        if config.temporal_loss_type == "ramp_mono" else config.temporal_loss_type
    )
    if config.mono_penalty_decay_until_frac > 0 and base_mono > 0:
        decay_step = int(config.mono_penalty_decay_until_frac * config.max_steps)
        floor_mono = base_mono * config.mono_penalty_min_frac
        print(f"Temporal loss: {loss_description}, "
              f"mono_penalty {base_mono} → {floor_mono:.3f} over first "
              f"{decay_step:,} steps ({config.mono_penalty_decay_until_frac:.0%} of training)")
    else:
        print(f"Temporal loss: {loss_description}, "
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
    ddp_world = dist.get_world_size() if dist.is_initialized() else 1
    tokens_seen = step * config.batch_size * config.seq_len * ddp_world

    print(f"\n{'='*70}")
    print(f"Starting training for {config.max_steps:,} steps")
    target_tokens = config.max_steps * config.batch_size * config.seq_len * ddp_world
    if use_curriculum:
        print(f"Target: ~{target_tokens/1e9:.1f}B tokens from Curriculum Mode (NVIDIA Nemotron Datasets)")
    elif use_fineweb:
        print(f"Target: ~{target_tokens/1e9:.1f}B tokens from FineWeb-Edu")
    else:
        print(f"Target: Local dataset ({config.data_path})")
    print(f"{'='*70}\n")

    model.train()
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

    # Phase profiler — disabled by default. When enabled, wraps each
    # per-step phase with a timed context that calls torch.cuda.synchronize
    # at the boundaries and aggregates wall-clock samples.
    timer = PhaseTimer(enabled=profile, warmup_steps=profile_warmup)
    if profile:
        print(f"Phase profiling enabled (warmup={profile_warmup} steps, "
              f"report every {profile_interval} steps)")

    while step < config.max_steps:
        # ── Two-Phase Curriculum Logic ──────────────────────────────────
        if use_curriculum:
            if current_phase == 1 and tokens_seen >= config.phase1_tokens:
                print(f"\n{'='*70}\n"
                      f"  CURRICULUM PHASE SWITCH at step {step} ({tokens_seen/1e6:.0f}M tokens)\n"
                      f"  Entering Phase 2: Full Competence\n"
                      f"  - Switching to Hybrid Mix Loader\n"
                      f"  - Switching Temporal Loss to Dynamic Aggregation\n"
                      f"{'='*70}\n",
                      flush=True)
                current_phase = 2
                train_loader = phase2_loader
                print("  Waiting for Phase 2 data pre-fetch...", end="", flush=True)
                train_iter = iter(train_loader)
                try:
                    first_batch = next(train_iter)
                    print(" Done.")
                except StopIteration:
                    print(" FAILED: Phase 2 dataset is empty!")
                    sys.exit(1)
                config.temporal_loss_type = "dynamic_aggregate"

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
        unwrap_model(model)._train_step.fill_(step)

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

        # Wrap the entire step (excluding logging and eval) in timer.step()
        # so the warmup counter advances. Inner scopes time individual phases.
        with timer.step():
            # Accumulate gradients over `accum_steps` micro-batches.
            accum_loss = 0.0
            last_result = None
            for micro in range(accum_steps):
                with timer("data"):
                    if first_batch is not None:
                        batch = first_batch
                        first_batch = None
                    else:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            batch = next(train_iter)

                    # Two batch shapes are supported:
                    #   (x, y)                                          — live-teacher / no-distill path
                    #   (x, y, top_indices, top_values, residual)       — cached-teacher path
                    cached_top_indices = None
                    cached_top_values = None
                    if config.use_cached_teacher:
                        x, y, cached_top_indices, cached_top_values, _residual = batch
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        cached_top_indices = cached_top_indices.to(device, non_blocking=True)
                        cached_top_values = cached_top_values.to(device, non_blocking=True)
                    else:
                        x, y = batch
                        x = x.to(device)
                        y = y.to(device)

                # ── Teacher Forward Pass ────────────────────────────────
                # When the teacher lives on a separate device, we issue
                # its work without waiting for it to finish. The teacher
                # output is only consumed in the student's loss block
                # (after the full student forward pass), so the teacher's
                # ~88 ms of compute overlaps with the student's ~454 ms
                # of compute on the other GPU. The cross-device copy of
                # t_logits uses non_blocking=True; the actual sync
                # happens implicitly when the student's KL kernel reads
                # the tensor, by which point the copy is already done.
                #
                # Note about the timer: with the colocated-teacher path,
                # the surrounding `timer("teacher_fwd")` scope syncs at
                # the boundary, which is fine since teacher and student
                # share the device anyway. With the separate-device path
                # we deliberately *don't* time the teacher with a sync
                # boundary — that would destroy the overlap. We log the
                # time-to-issue under "teacher_issue" instead, which
                # measures the cost of launching teacher kernels but not
                # waiting for them.
                #
                # When use_cached_teacher is True, this whole block is
                # skipped — the top-K logits already arrived with the
                # batch, so there's nothing to do here.
                t_logits, t_z = None, None
                if teacher_model is not None:
                    want_hidden = config.distill_feature_weight > 0
                    # Normalize device strings: "cuda" and "cuda:0" both
                    # refer to device 0, but compare unequal as strings.
                    # Use torch.device for canonical comparison.
                    teacher_colocated = (
                        torch.device(teacher_device) == torch.device(device)
                    )

                    if teacher_colocated:
                        with timer("teacher_fwd"):
                            with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                                teacher_out = teacher_model(x, output_hidden_states=want_hidden)
                                t_logits = teacher_out.logits
                                if want_hidden:
                                    t_z = teacher_out.hidden_states[-1]
                    else:
                        # Separate-device teacher: launch and don't sync.
                        with timer("teacher_issue"):
                            x_for_teacher = x.to(teacher_device, non_blocking=True)
                            teacher_device_type = (
                                teacher_device.split(":")[0]
                                if ":" in teacher_device
                                else teacher_device
                            )
                            with torch.no_grad(), torch.amp.autocast(
                                device_type=teacher_device_type,
                                dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32),
                            ):
                                teacher_out = teacher_model(
                                    x_for_teacher,
                                    output_hidden_states=want_hidden,
                                )
                                # Initiate non-blocking copies back to
                                # student device. These return tensors
                                # that will be ready by the time the
                                # KL kernel reads them.
                                t_logits = teacher_out.logits.to(
                                    device, non_blocking=True
                                )
                                if want_hidden:
                                    t_z = teacher_out.hidden_states[-1].to(
                                        device, non_blocking=True
                                    )

                with timer("student_fwd"):
                    with torch.amp.autocast(device_type=device_type,
                                            dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                        result = model(
                            x,
                            targets=y,
                            max_thought_steps=current_T,
                            teacher_logits=t_logits,
                            teacher_z=t_z,
                            cached_top_indices=cached_top_indices,
                            cached_top_values=cached_top_values,
                        )
                        # Scale the loss so the accumulated gradient is the MEAN
                        # over the effective batch — matches what a single forward
                        # pass at batch_size = batch_size·accum_steps would
                        # compute.
                        loss = result["loss"] / accum_steps

                with timer("backward"):
                    loss.backward()
                accum_loss += loss.item() * accum_steps   # un-scale for logging
                last_result = result   # keep last for cert/tick logging

            # Average loss across the accumulation window for logging
            loss_for_log = accum_loss / accum_steps

            with timer("grad_clip"):
                if config.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                else:
                    grad_norm = torch.tensor(0.0)

            with timer("optimizer"):
                for o in optimizers:
                    o.step()

        dt = time.time() - t0
        # Effective batch tokens accounts for accumulation AND, under DDP,
        # the world size — every rank processes its own batch each step,
        # so the effective tokens/step is multiplied by world_size.
        ddp_world = dist.get_world_size() if dist.is_initialized() else 1
        batch_tokens = config.batch_size * config.seq_len * accum_steps * ddp_world
        tokens_seen += batch_tokens
        log_losses.append(loss_for_log)
        # Use last_result for tick / certainty logging (representative of the
        # final micro-batch — over the accumulation window these are close
        # enough that picking one is fine for monitoring)
        result = last_result

        # ── Logging ─────────────────────────────────────────────────────
        if step % config.log_interval == 0 and is_main_process():
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

        # ── Phase Profiler Report ────────────────────────────────────────
        # Independent of the main log cadence so the report can be denser
        # or sparser depending on what's being investigated.
        if profile and step > profile_warmup and step % profile_interval == 0:
            timer.report(last_n=profile_interval)

        # ── Evaluation ──────────────────────────────────────────────────
        if step > 0 and step % config.eval_interval == 0:
            # Eval runs on rank 0 only; other ranks wait at the barrier
            # below so all ranks stay aligned for the next training step.
            if is_main_process():
                eval_loss = evaluate(model, eval_loader, device, config, amp_dtype)
                print(f"\n  >>> Eval loss: {eval_loss:.4f} (best: {best_eval_loss:.4f})")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(model, optimizers, step, config, ckpt_dir / "best.pt")
                    print(f"  >>> Saved best checkpoint at step {step}")

                # Save step checkpoint
                save_checkpoint(model, optimizers, step, config,
                              ckpt_dir / f"step_{step:07d}.pt")

                # Generate a sample using real data from the eval set
                generate_sample(model, tokenizer, device, config, eval_loader)
                print()
            # All ranks must rejoin here before the next training step,
            # otherwise rank 0 (still doing eval) and rank 1 (already
            # iterating) drift apart and the next backward all-reduce
            # hangs waiting for rank 0's grads.
            if dist.is_initialized():
                dist.barrier()
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
    current_T = config.resolve_thought_steps(int(unwrap_model(model)._train_step.item()))

    for batch in eval_loader:
        if n_batches >= max_eval_batches:
            break
        # Eval just needs (x, y); cached batches carry extra teacher tensors
        # that we ignore here (we're measuring LM loss, not distillation).
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast(device_type=device_type,
                                dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            result = model(x, targets=y, max_thought_steps=current_T)

        total_loss += result["loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def generate_sample(model, tokenizer, device, config, eval_loader_or_text=None):
    """Generate and print a short sample from the model."""
    model.eval()
    prompt_text = None
    target_text = None
    prompt_ids = None

    # Option A: Pull a real sample from the evaluation loader (preferred for streaming)
    if isinstance(eval_loader_or_text, DataLoader):
        try:
            # Grab one batch from eval
            batch = next(iter(eval_loader_or_text))
            # Cached batches are 5-tuples (input_ids, targets, top_indices,
            # top_values, residual); plain batches are 2-tuples. We only
            # need x for prompt sampling.
            x = batch[0]
            # Take the first sequence in the batch
            # We take a piece of x as the prompt
            prompt_len = min(32, x.size(1) // 2)
            prompt_ids = x[0:1, :prompt_len].to(device)
            # The rest of the sequence is the "ground truth" we want to see
            target_ids = x[0, prompt_len:].tolist()
            
            prompt_text = tokenizer.decode(prompt_ids[0].tolist())
            target_text = tokenizer.decode(target_ids[:48]) # Only show first 48 tokens of target
        except Exception as e:
            prompt_text = "The most important thing to understand about science is"
    
    # Option B: Use provided raw text buffer (local file mode)
    if prompt_text is None and isinstance(eval_loader_or_text, str) and len(eval_loader_or_text) > 200:
        start = random.randint(0, len(eval_loader_or_text) - 200)
        prompt_text = eval_loader_or_text[start : start + 80]
    
    # Option C: Hard fallback
    if prompt_text is None:
        prompt_text = "The most important thing to understand about science is"

    if prompt_ids is None:
        prompt_tokens = tokenizer.encode(prompt_text, allowed_special=set())
        prompt_tokens = prompt_tokens[:32]
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    generated = model.generate(
        prompt_ids,
        max_new_tokens=48,
        temperature=0.8,
        top_k=40,
    )

    gen_tokens = generated[0].tolist()
    gen_decoded = tokenizer.decode(gen_tokens[prompt_ids.size(1):])

    print("  >>> Prompt: '{}'".format(prompt_text.replace('\n', ' ')))
    print("  >>> Model:  '{}'".format(gen_decoded.replace('\n', ' ')))
    if target_text:
        print("  >>> Target: '{}'".format(target_text.replace('\n', ' ')))


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

    # Distillation Strategy
    distill_group = parser.add_argument_group("Distillation")
    distill_group.add_argument("--use_distillation", action="store_true",
                               help="Enable teacher-student distillation.")
    distill_group.add_argument("--teacher_model_name", type=str, default="nvidia/Nemotron-3-4B-Base",
                               help="HuggingFace model ID for the teacher model.")
    distill_group.add_argument("--teacher_device", type=str, default="auto",
                               help="Device placement for the teacher. 'auto' (default) "
                                    "colocates the teacher with the student. On multi-GPU "
                                    "systems, set to e.g. 'cuda:1' to run the teacher on a "
                                    "separate GPU — the student gets full memory on cuda:0, "
                                    "and the cross-device transfers overlap with student "
                                    "compute via non_blocking copies.")
    distill_group.add_argument("--distill_logit_weight", type=float, default=1.0,
                               help="Weight for soft-target KL divergence loss.")
    distill_group.add_argument("--distill_feature_weight", type=float, default=0.0,
                               help="Weight for feature-alignment loss. Default 0 (off). "
                                    "Cross-architecture feature distillation is fragile; "
                                    "enable only with --distill_feature_method cosine or mse_normed.")
    distill_group.add_argument("--distill_temperature", type=float, default=4.0,
                               help="Temperature T for soft-target distillation. The KL is "
                                    "scaled by T^2 (Hinton 2015) so gradient magnitude is "
                                    "T-invariant. Use 1.0 to disable softening.")
    distill_group.add_argument("--distill_tick_aggregation", type=str, default="all",
                               choices=["all", "first", "last", "decay_ramp", "lm_aligned"],
                               help="Which thought ticks receive KD signal. 'all' applies KD "
                                    "uniformly across every tick; 'first' supervises only the "
                                    "model's initial-guess tick (recommended when distillation "
                                    "is overpowering the thought-loop dynamics); 'last' is the "
                                    "legacy behavior — known unstable when combined with "
                                    "mono_penalty because it can cause the per-tick loss to "
                                    "go [10, 9, 8, 7, 6, 9] (jumps up at the final tick); "
                                    "'decay_ramp' uses linearly-decaying weights from tick 0 "
                                    "to tick T-1 (front-loaded teacher signal); 'lm_aligned' "
                                    "mirrors the temporal loss weighting.")
    distill_group.add_argument("--distill_feature_method", type=str, default="cosine",
                               choices=["cosine", "mse_normed", "mse"],
                               help="Feature-distillation method (used only when "
                                    "distill_feature_weight > 0). 'cosine' is recommended "
                                    "for cross-architecture distillation.")
    distill_group.add_argument("--distill_top_k", type=int, default=256,
                               help="Top-K soft-target distillation. KL is computed only "
                                    "over the teacher's top-K most-likely tokens at each "
                                    "position, dramatically reducing fp32 softmax memory "
                                    "traffic at large vocabularies. 0 disables (full-vocab "
                                    "KD). Reasonable values are 128-512.")
    distill_group.add_argument("--use_cached_teacher", action="store_true",
                               help="Read pre-computed teacher top-K logits from disk "
                                    "(produced by scripts/cache_teacher_logits.py) instead "
                                    "of running a live teacher each step. Drops the teacher "
                                    "model from the training process entirely; --teacher_device "
                                    "is ignored. Implies --use_distillation.")
    distill_group.add_argument("--teacher_cache_dir", type=str, default="",
                               help="Directory containing shard_*.npz files written by "
                                    "scripts/cache_teacher_logits.py. Required when "
                                    "--use_cached_teacher is set.")

    # Model Architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--d_model", type=int, default=512)
    model_group.add_argument("--d_latent", type=int, default=512)
    model_group.add_argument("--n_heads", type=int, default=8)
    model_group.add_argument("--n_layers", type=int, default=4)
    model_group.add_argument("--nlm_hidden_dim", type=int, default=32)
    model_group.add_argument("--nlm_groups", type=int, default=1,
                            help="Number of temporal MLPs: 1 shares across all neurons; "
                                 "d_latent gives independent neuron MLPs.")
    model_group.add_argument("--use_positional_encoding", action="store_true",
                            help="Encode token positions; recommended for order-sensitive text experiments.")
    model_group.add_argument("--history_len", type=int, default=8)
    model_group.add_argument("--max_thought_steps", type=int, default=8)
    model_group.add_argument("--seq_len", type=int, default=512)
    model_group.add_argument("--sync_method", type=str, default="diag_summary",
                            choices=["full", "diag_summary", "low_rank", "sparse_decay"])
    model_group.add_argument("--sync_sparse_pairs", type=int, default=256)
    model_group.add_argument("--synapse_type", type=str, default="mlp", choices=["mlp", "unet"])
    model_group.add_argument("--temporal_loss_type", type=str, default="ramp_mono", choices=["final_ce", "ramp_mono", "dynamic_aggregate"])
    model_group.add_argument("--no_attention_residuals", action="store_true",
                            help="Use sequential gated thought layers without attention residual mixing.")
    model_group.add_argument("--mono_penalty_weight", type=float, default=0.5,
                            help="Temporal monotonicity penalty; must be 0 for final_ce.")
    model_group.add_argument("--use_feature_encoder", action="store_true")
    model_group.add_argument("--per_tick_heads", action="store_true",
                            help="Give each thought tick its own output adapter feeding into a "
                                 "shared LM head. Removes gradient interference between ticks "
                                 "and empirically prevents the iterative-refinement collapse "
                                 "where the model learns to produce identical output at every "
                                 "tick. Cost: ~1%% extra params (T copies of a small adapter).")
    model_group.add_argument("--use_shared_head_film", action="store_true",
                            help="Replace per-tick heads (1.07B params at V=131072) with a single "
                                 "shared head + per-tick FiLM modulation (135M + 16K params). "
                                 "Matches published CTM architecture and reduces head FLOPs ~85%% "
                                 "in forward and backward. Mutually exclusive with --per_tick_heads.")
    model_group.add_argument("--tie_embeddings", action="store_true",
                            help="Tie the LM head weight matrix to the token embedding (Press & "
                                 "Wolf 2017). Saves vocab_size*d_model parameters (134M at "
                                 "V=131072, d_model=1024) plus the corresponding optimizer state. "
                                 "Requires --use_shared_head_film.")

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

    # Two-Phase Data Curriculum
    data_curr_group = parser.add_argument_group("Two-Phase Data Curriculum")
    data_curr_group.add_argument("--use_two_phase_curriculum", action="store_true",
                                 help="Enable Phase 1 (Logic Priming) and Phase 2 (Hybrid Mix) "
                                      "interleaved datasets curriculum.")
    data_curr_group.add_argument("--phase1_tokens", type=int, default=500000000,
                                 help="Tokens to train in Phase 1 before switching to Phase 2.")

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
    train_group.add_argument("--profile", action="store_true",
                             help="Enable per-phase wall-clock profiling. Reports mean / "
                                  "median / p95 of each training-step phase (data, "
                                  "teacher_fwd, student_fwd, backward, grad_clip, optimizer) "
                                  "every --profile_interval steps. Adds a torch.cuda.synchronize "
                                  "at each phase boundary, so it has small (single-digit %%) "
                                  "overhead — use it for diagnosis, not in production runs.")
    train_group.add_argument("--profile_warmup", type=int, default=10,
                             help="Number of initial steps to skip before recording timing "
                                  "samples (avoids polluting stats with allocator/JIT warmup).")
    train_group.add_argument("--profile_interval", type=int, default=50,
                             help="Print phase-timing report every N steps (only when "
                                  "--profile is set).")
    train_group.add_argument("--val_interval", type=int, default=200,
                             help="Run held-out validation (PPL + distill top-k alignment) "
                                  "every N steps. Rank 0 only. ~50 fwd passes, cheap. "
                                  "Set to 0 to disable.")
    train_group.add_argument("--t_sweep_interval", type=int, default=2000,
                             help="Run T-ablation sweep (T=1,4,8) every N steps. Rank 0 only. "
                                  "~3× the cost of a single validation run. Set to 0 to disable.")

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

    train_group.add_argument("--compile", action="store_true",
                             help="Use torch.compile to optimize the model.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Set up DDP (must happen BEFORE any CUDA allocations) ───────────
    # If we were launched by torchrun, this initializes the process
    # group, sets the right CUDA device per rank, and returns the rank
    # info. Otherwise it's a no-op and returns (0, 1, 0).
    rank, world_size, local_rank = setup_distributed()
    is_ddp = world_size > 1

    if is_ddp:
        # CUDA Graphs and DDP don't compose cleanly: graph capture
        # records a fixed kernel sequence, but DDP's NCCL all-reduces
        # are dispatched dynamically by the autograd engine. Disable
        # CUDA Graphs automatically and warn (rather than failing
        # mid-training with a confusing error).
        if args.use_cuda_graphs:
            if rank == 0:
                print("[DDP] Disabling --use_cuda_graphs (incompatible "
                      "with DDP's dynamic all-reduce dispatch).")
            args.use_cuda_graphs = False

        # Override --device to the local rank's GPU. The user's
        # `--device cuda` becomes `cuda:LOCAL_RANK`. Important: this
        # has to happen after setup_distributed sets the default
        # device, but config.device is still used in train() below
        # to .to(device) the model.
        if args.device.startswith("cuda"):
            args.device = f"cuda:{local_rank}"

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
        use_positional_encoding=args.use_positional_encoding,
        per_tick_heads=args.per_tick_heads,
        use_shared_head_film=args.use_shared_head_film,
        tie_embeddings=args.tie_embeddings,
        use_distillation=args.use_distillation,
        teacher_model_name=args.teacher_model_name,
        teacher_device=args.teacher_device,
        distill_logit_weight=args.distill_logit_weight,
        distill_feature_weight=args.distill_feature_weight,
        distill_temperature=args.distill_temperature,
        distill_tick_aggregation=args.distill_tick_aggregation,
        distill_feature_method=args.distill_feature_method,
        distill_top_k=args.distill_top_k,
        use_cached_teacher=args.use_cached_teacher,
        teacher_cache_dir=args.teacher_cache_dir,
        t_curriculum=args.t_curriculum,
        t_curriculum_stages=_parse_curriculum_stages(args.t_curriculum_stages),
        use_two_phase_curriculum=args.use_two_phase_curriculum,
        phase1_tokens=args.phase1_tokens,
        history_len=args.history_len,
        max_thought_steps=args.max_thought_steps,
        seq_len=args.seq_len,
        sync_method=args.sync_method,
        sync_sparse_pairs=args.sync_sparse_pairs,
        synapse_type=args.synapse_type,
        temporal_loss_type=args.temporal_loss_type,
        mono_penalty_weight=args.mono_penalty_weight,
        use_attention_residuals=not args.no_attention_residuals,
        use_feature_encoder=args.use_feature_encoder,
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

    try:
        train(
            config,
            profile=args.profile,
            profile_warmup=args.profile_warmup,
            profile_interval=args.profile_interval,
        )
    finally:
        cleanup_distributed()

# ═════════════════════════════════════════════════════════════════════════
# train_offload_clean.py
# ═════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", message=".*Online softmax is disabled on the fly.*")

import torch
if hasattr(torch, "_inductor"):
    # Optional compiler tuning flags differ between PyTorch releases.
    for option in ("split_reductions", "online_softmax"):
        if hasattr(torch._inductor.config, option):
            setattr(torch._inductor.config, option, True)

import torch.distributed as dist
import torch.multiprocessing as mp



def worker_multi_gpu(rank, world_size, config_dict, runtime_kwargs=None):
    """Training worker for one GPU.

    runtime_kwargs are non-CTMConfig settings the worker needs:
      - profile: bool — enable PhaseTimer instrumentation
      - profile_warmup: int — steps to skip before recording timings
      - profile_interval: int — how often (in steps) to print the report
    """
    if runtime_kwargs is None:
        runtime_kwargs = {}
    # ── Setup process group ────────────────────────────────────────────
    # Try NCCL first (much faster for GPU tensors), fall back to gloo.
    # The user's original DDP issue was backward-integrated allreduce,
    # not NCCL itself. Our explicit post-backward allreduce avoids that.
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['NCCL_DEBUG'] = 'INFO'
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    backend = 'nccl' if torch.cuda.is_available() and dist.is_nccl_available() else 'gloo'
    # Specify device_id to mute the UserWarning
    if backend == 'nccl':
        dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=device)
    else:
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    config = CTMConfig(**config_dict)
    dtype = getattr(torch, config.dtype) if isinstance(config.dtype, str) else config.dtype

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = get_tokenizer(config)
    if config.vocab_size != tokenizer.n_vocab:
        config.vocab_size = tokenizer.n_vocab

    if rank == 0:
        print(f"[worker {rank}] Building model on {device}...")

    # ── Build model ──────────────────────────────────────────────────
    model = CTMTransformer(config).to(device, dtype)
    if runtime_kwargs.get('compile', False):
        if rank == 0: print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if rank == 0:
        n_params = model.get_num_params()
        print(f"Model: {n_params / 1e6:.1f}M params × {world_size} workers")

    # Sync initial weights (broadcast from rank 0)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    # ── Optimizer ────────────────────────────────────────────────────
    # Use the same _make_adamw helper as the single-process train.py so
    # --use_8bit_adam actually takes effect. Vanilla torch.optim.AdamW
    # would burn ~3.6 GB extra VRAM per rank for fp32 m+v moments and
    # be a few ms slower per step. The bitsandbytes path is a drop-in
    # replacement — same API, same kwargs.
    optimizer = _make_adamw(config, model.parameters())
    if rank == 0:
        opt_name = type(optimizer).__name__
        print(f"  Optimizer: {opt_name}"
              + (" (8-bit moments)" if "8bit" in opt_name else ""))

    # ── Data ─────────────────────────────────────────────────────────

    cache_root = Path(config.teacher_cache_dir)
    phase_dir = (cache_root / "phase1") if config.use_two_phase_curriculum else cache_root

    # Each worker gets different shards
    train_ds = CachedTeacherDataset(
        cache_dir=str(phase_dir),
        rank=rank, world_size=world_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size,
        collate_fn=cached_teacher_collate,
        num_workers=1, pin_memory=True,
    )
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    # ── Eval loader (rank 0 only) ────────────────────────────────────
    # Held-out slice of the cached teacher data via is_eval=True.
    # Only rank 0 evaluates — weights stay synced across ranks via the
    # flat allreduce on grads, so single-rank eval is correct.
    # Factory pattern lets each call (and each T in the T-sweep) get a
    # fresh deterministic loader.
    eval_loader_factory = None
    if rank == 0:
        def _make_eval_loader():
            eval_ds = CachedTeacherDataset(
                cache_dir=str(phase_dir),
                rank=0, world_size=1,
                is_eval=True,
                eval_fraction=0.05,
                shuffle_shards=False,
            )
            return torch.utils.data.DataLoader(
                eval_ds, batch_size=config.batch_size,
                collate_fn=cached_teacher_collate,
                num_workers=1, pin_memory=True,
            )
        eval_loader_factory = _make_eval_loader

    # ── Checkpoint ───────────────────────────────────────────────────
    start_step = 0
    ckpt_dir = Path(config.checkpoint_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(device_ids=[device.index] if backend == 'nccl' else None)

    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        start_step = load_checkpoint(model, [optimizer], resume_path, device)
        if rank == 0:
            print(f"Resumed at step {start_step}")

    # ── Training Loop ────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"CTM-Transformer Dual-GPU Training ({backend}, no DDP)")
        print(f"  Steps: {start_step} → {config.max_steps}")
        print(f"  Batch: {world_size} × {config.seq_len} tokens")
        print(f"  Thought ticks: {config.max_thought_steps}")
        print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
        print(f"{'='*60}\n")

    model.train()
    step_times = []
    save_interval = getattr(config, 'save_interval', 5000)
    t_start = time.time()
    last_log_time = t_start
    last_log_step = start_step
    last_grad_norm = 0.0

    # ── Phase profiling (rank 0 only) ─────────────────────────────────
    # Both ranks do identical work, so measuring on rank 0 is
    # representative without doubling the sync overhead. The timer
    # issues torch.cuda.synchronize() at every phase boundary; running
    # it on both ranks would force two syncs per phase for the same
    # information.
    profile_enabled = runtime_kwargs.get('profile', False) and rank == 0
    profile_warmup = runtime_kwargs.get('profile_warmup', 20)
    profile_interval = runtime_kwargs.get('profile_interval', 100)
    timer = PhaseTimer(enabled=profile_enabled, warmup_steps=profile_warmup)
    if profile_enabled:
        print(f"Phase profiling enabled (warmup={profile_warmup} steps, "
              f"report every {profile_interval} steps)")

    optimizer.zero_grad()
    prev_T = None
    for step in range(start_step, config.max_steps):
        with timer.step():
            t0 = time.perf_counter()

            # Sync _train_step BEFORE forward so loss-side schedules
            # (mono_penalty decay, etc.) read the current step.
            model._train_step.fill_(step)

            # Resolve curriculum T for this step. When t_curriculum is
            # off this returns config.max_thought_steps. When on, walks
            # the stages list and picks the appropriate T.
            current_T = config.resolve_thought_steps(step)

            # Announce curriculum transitions on rank 0 only.
            if rank == 0 and config.t_curriculum and prev_T is not None and current_T != prev_T:
                print(
                    f"\n{'='*70}\n"
                    f"  CURRICULUM TRANSITION at step {step}: T = {prev_T} → {current_T}\n"
                    f"{'='*70}\n",
                    flush=True,
                )
            prev_T = current_T

            with timer("data"):
                batch = get_batch()
                ids, tgt, top_idx, top_val, _res = batch
                ids = ids.to(device)
                tgt = tgt.to(device)
                top_idx = top_idx.to(device)
                top_val = top_val.to(device)

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # ── Forward + Backward (each GPU independently) ──────────────
            with timer("student_fwd"):
                result = model(
                    ids, targets=tgt,
                    cached_top_indices=top_idx,
                    cached_top_values=top_val,
                    max_thought_steps=current_T,
                )
                # Scale loss so the gradients accumulate correctly
                loss = result['loss'] / config.gradient_accumulation_steps

            with timer("backward"):
                loss.backward()

            grad_norm = 0.0
            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == config.max_steps:

                # ── All-reduce gradients (single flattened call) ────────────────
                # PyTorch's native C++ flattening minimizes kernel launch overhead.
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if grads:
                    with timer("sync_wait"):
                        # Barrier isolates compute desync from actual communication time
                        if dist.is_initialized():
                            dist.barrier(device_ids=[device.index] if backend == 'nccl' else None)

                    with timer("allreduce"):
                        flat_grads = torch._utils._flatten_dense_tensors(grads)
                        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                        flat_grads.div_(world_size)

                        # Unflatten back via views and copy
                        for g, u in zip(grads, torch._utils._unflatten_dense_tensors(flat_grads, grads)):
                            g.copy_(u)

                # ── Gradient clipping ────────────────────────────────────────
                with timer("grad_clip"):
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )

                # ── Optimizer step ───────────────────────────────────────────
                with timer("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

            t1 = time.perf_counter()
            step_time = t1 - t0
            step_times.append(step_time)

        # ── Periodic profiler report (rank 0 only) ───────────────────
        if profile_enabled and step > 0 and step % profile_interval == 0:
            timer.report(last_n=profile_interval)

        # ── Logging (rank 0 only) ────────────────────────────────────
        if rank == 0 and step % config.log_interval == 0:
            current_time = time.time()
            recent_time = current_time - last_log_time
            recent_steps = max(step - last_log_step, 1) if step > start_step else 1
            
            recent_tokens = recent_steps * world_size * config.seq_len
            tps = recent_tokens / recent_time if recent_time > 0 else 0
            
            elapsed = current_time - t_start
            total_tokens_seen = max(step - start_step, 1) * world_size * config.seq_len
            avg_tps = total_tokens_seen / elapsed if elapsed > 0 else 0
            
            last_log_time = current_time
            last_log_step = step

            elapsed = time.time() - t_start
            eta_sec = (config.max_steps - step) * (elapsed / max(step - start_step, 1))

            per_tick = result.get('per_tick_loss')
            tick_str = ""
            if per_tick is not None:
                tick_list = per_tick.tolist()
                tick_delta = tick_list[-1] - tick_list[0] if len(tick_list) > 1 else 0
                tick_str = (
                    f" | ticks [{' '.join(f'{t:.3f}' for t in tick_list)}] | "
                    f"Δticks {tick_delta:+.3f}"
                )

            cert_str = ""
            if "certainties" in result:
                cert = result["certainties"]
                cert_first = cert[0].mean().item()
                cert_last = cert[-1].mean().item()
                cert_str = f" | cert {cert_first:.2f}→{cert_last:.2f}"

            t_field = f"T={current_T} | " if config.t_curriculum else ""
            print(
                f"step {step:>6d} | {t_field}loss {loss.item() * config.gradient_accumulation_steps:.4f} | "
                f"lr {lr:.2e} | grad {last_grad_norm:.2f} | "
                f"{tps/1000:.1f}k tok/s (avg {avg_tps/1000:.1f}k)"
                f"{cert_str}{tick_str} | "
                f"ETA {eta_sec/3600:.1f}h"
            )

        # ── Periodic validation (rank 0 only) ────────────────────────
        # Held-out PPL + distillation top-k alignment every val_interval
        # steps; pricier T-ablation every t_sweep_interval steps.
        # Both are CLI-tunable; set to 0 to disable.
        # Wrapped in try/except: a hiccup in a periodic probe should
        # never take down a multi-day training run. We log and continue.
        VAL_INTERVAL = runtime_kwargs.get('val_interval', 200)
        T_SWEEP_INTERVAL = runtime_kwargs.get('t_sweep_interval', 2000)

        if (rank == 0 and eval_loader_factory is not None
                and VAL_INTERVAL > 0
                and step > 0 and step % VAL_INTERVAL == 0):
            try:
                unwrap_model(model)._train_step.fill_(step)
                metrics = compute_validation_metrics(
                    model=model,
                    eval_loader=eval_loader_factory(),
                    device=device, config=config,
                    amp_dtype=dtype, max_batches=50,
                )
                print(f"  >>> val @ step {step}:")
                print(format_validation_report(metrics))
            except Exception as e:
                print(f"  [val @ step {step}] FAILED: {type(e).__name__}: {e}",
                      flush=True)
                # Restore train mode in case the failure left model.eval() set
                model.train()

        if (rank == 0 and eval_loader_factory is not None
                and T_SWEEP_INTERVAL > 0
                and step > 0 and step % T_SWEEP_INTERVAL == 0):
            try:
                unwrap_model(model)._train_step.fill_(step)
                t_sweep = compute_validation_metrics_t_sweep(
                    model=model,
                    eval_loader_factory=eval_loader_factory,
                    device=device, config=config,
                    amp_dtype=dtype,
                    T_values=(1, 4, 8),
                    max_batches=30,
                )
                print(f"  >>> T-ablation @ step {step}:")
                print(format_t_sweep_report(t_sweep))
            except Exception as e:
                print(f"  [T-sweep @ step {step}] FAILED: {type(e).__name__}: {e}",
                      flush=True)
                model.train()

        # ── Checkpoint ───────────────────────────────────────────────
        if step > 0 and step % save_interval == 0:
            if rank == 0:
                save_checkpoint(model, [optimizer], step, config,
                                ckpt_dir / f"step_{step}.pt")
                save_checkpoint(model, [optimizer], step, config, resume_path)
                print(f"  Saved checkpoint at step {step}")
            dist.barrier(device_ids=[device.index] if backend == 'nccl' else None)

    if rank == 0:
        print("\nTraining complete!")
    dist.destroy_process_group()


def main_multi_gpu():
    args = parse_args()

    for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        os.environ.pop(key, None)

    config = CTMConfig(
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        dataset=args.dataset or "",
        dataset_subset=args.dataset_subset,
        tokenizer=args.tokenizer,
        d_model=args.d_model, d_latent=args.d_latent,
        n_heads=args.n_heads, n_layers=args.n_layers,
        nlm_hidden_dim=args.nlm_hidden_dim, nlm_groups=args.nlm_groups,
        use_positional_encoding=args.use_positional_encoding,
        per_tick_heads=args.per_tick_heads,
        use_shared_head_film=args.use_shared_head_film,
        tie_embeddings=args.tie_embeddings,
        use_distillation=args.use_distillation,
        teacher_model_name=args.teacher_model_name,
        teacher_device=args.teacher_device,
        distill_logit_weight=args.distill_logit_weight,
        distill_feature_weight=args.distill_feature_weight,
        distill_temperature=args.distill_temperature,
        distill_tick_aggregation=args.distill_tick_aggregation,
        distill_feature_method=args.distill_feature_method,
        distill_top_k=args.distill_top_k,
        use_cached_teacher=args.use_cached_teacher,
        teacher_cache_dir=args.teacher_cache_dir,
        t_curriculum=args.t_curriculum,
        t_curriculum_stages=_parse_curriculum_stages(args.t_curriculum_stages),
        use_two_phase_curriculum=args.use_two_phase_curriculum,
        phase1_tokens=args.phase1_tokens,
        history_len=args.history_len,
        max_thought_steps=args.max_thought_steps,
        seq_len=args.seq_len,
        sync_method=args.sync_method,
        sync_sparse_pairs=args.sync_sparse_pairs,
        synapse_type=args.synapse_type,
        temporal_loss_type=args.temporal_loss_type,
        mono_penalty_weight=args.mono_penalty_weight,
        use_attention_residuals=not args.no_attention_residuals,
        use_feature_encoder=args.use_feature_encoder,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        device=args.device, dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_min_T=args.gradient_checkpointing_min_T,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_interval=args.eval_interval, log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        optimizer=args.optimizer,
        adam_beta1=getattr(args, 'adam_beta1', 0.9),
        adam_beta2=getattr(args, 'adam_beta2', 0.95),
        use_8bit_adam=args.use_8bit_adam,
        adamuon_beta=args.adamuon_beta, adamuon_eps=args.adamuon_eps,
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
        use_feec=args.use_feec,
        feec_dt_init=args.feec_dt_init,
        feec_damping_init=args.feec_damping_init,
        feec_clamp_dt=args.feec_clamp_dt,
        feec_energy_penalty_weight=args.feec_energy_penalty_weight,
        use_matrix_streams=args.use_matrix_streams,
        n_streams=args.n_streams, stream_gating=args.stream_gating,
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

    # Serialize config to dict for multiprocessing
    config_dict = {f.name: getattr(config, f.name)
                   for f in config.__dataclass_fields__.values()}

    world_size = min(torch.cuda.device_count(), 2)
    print(f"Spawning {world_size} workers...")

    # Profile args travel separately from config_dict because PhaseTimer
    # is a runtime concern, not a model-architecture concern (and CTMConfig
    # doesn't have fields for it). Workers read these via a kwargs dict.
    runtime_kwargs = dict(
        profile=args.profile,
        profile_warmup=args.profile_warmup,
        profile_interval=args.profile_interval,
        compile=getattr(args, 'compile', False),
        val_interval=args.val_interval,
        t_sweep_interval=args.t_sweep_interval,
    )

    mp.spawn(worker_multi_gpu, args=(world_size, config_dict, runtime_kwargs),
             nprocs=world_size, join=True)


# ═════════════════════════════════════════════════════════════════════════
# Entry-point dispatch
# ═════════════════════════════════════════════════════════════════════════
#
# Two run modes share this file:
#   - Single-process / torchrun-driven DDP: ``main()`` (the original
#     train.py entry).
#   - Dual-GPU mp.spawn (no DDP): ``main_multi_gpu()`` (the original
#     train_offload.py entry).
#
# Pre-parsing dispatch keeps each main()'s argparse contract unchanged:
# we strip the ``--multi_gpu`` flag from ``sys.argv`` before delegating,
# so neither main() needs to know the other exists.

if __name__ == "__main__":
    import sys as _sys
    if "--multi_gpu" in _sys.argv:
        _sys.argv.remove("--multi_gpu")
        main_multi_gpu()
    else:
        main()