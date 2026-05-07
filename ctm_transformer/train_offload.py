"""
Dual-GPU Multi-Process Data Parallel Training for CTM-Transformer.

Uses torch.multiprocessing to run 2 independent training processes.
Each GPU processes a different micro-batch. Gradients are averaged
via gloo backend (CPU-based, no NCCL needed).

NO DDP wrapping — just manual all_reduce on gradients after backward.
This avoids DDP's compute-communication overlap which causes PCIe stalls.

Expected throughput: ~0.7k tok/s total (2 × 0.4k minus ~15% sync overhead).
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.phase_timer import PhaseTimer
from ctm_transformer.train import (
    parse_args, get_tokenizer, get_lr,
    save_checkpoint, load_checkpoint,
    _parse_curriculum_stages,
    _make_adamw,
)


def worker(rank, world_size, config_dict, runtime_kwargs=None):
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
    from ctm_transformer.cached_teacher_dataset import (
        CachedTeacherDataset, cached_teacher_collate,
    )

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
    for step in range(start_step, config.max_steps):
        with timer.step():
            t0 = time.perf_counter()

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

            model._train_step.fill_(step + 1)

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

            print(
                f"step {step:>6d} | loss {loss.item() * config.gradient_accumulation_steps:.4f} | "
                f"lr {lr:.2e} | grad {last_grad_norm:.2f} | "
                f"{tps/1000:.1f}k tok/s (avg {avg_tps/1000:.1f}k)"
                f"{cert_str}{tick_str} | "
                f"ETA {eta_sec/3600:.1f}h"
            )

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


def main():
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
        use_cuda_graphs=False,
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
    )

    mp.spawn(worker, args=(world_size, config_dict, runtime_kwargs),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()