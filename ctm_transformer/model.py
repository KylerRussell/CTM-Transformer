"""
CTM-Transformer — All Architectural Components

This file holds the entire model architecture for the CTM-Transformer (v2),
consolidated from what used to be one-class-per-file. The original module
structure is preserved as section banners below.

Section order respects dependencies — leaves first, composite modules later:
  1. triton_kernels.py     — Triton/CUDA-graph attention helpers
  2. dssa.py               — Dual-Space Sparse Attention
  3. memory.py             — FIFO history + synchronization
  4. nlm.py                — Per-neuron MLPs
  5. matrix_stream.py      — Hyperloop-style residual streams (NLM/sync replacement)
  6. feec_integrator.py    — Symplectic-like integrator for the thought loop
  7. thought_layer.py      — One iteration block of the thought loop
  8. model.py              — CTMTransformer assembly + forward + generate
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional
import torch.utils.checkpoint as torch_checkpoint
from contextlib import nullcontext

from ctm_transformer.biological import (
    HebbianSynapse,
    CerebellarReadout,
    EpisodicDND,
    critical_symmetric_init_,
)
from ctm_transformer.predictive_coding import PCLayer, PCStateManager



# ═════════════════════════════════════════════════════════════════════════
# triton_kernels.py
# ═════════════════════════════════════════════════════════════════════════

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────
# Triton Kernel: Fused Flash-style Causal Attention
# ──────────────────────────────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def _fused_attention_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Fused Q·K^T with causal masking, online softmax, and multiplication by V.
        
        Grid: (batch * n_heads, ceil(S / BLOCK_S))
        """
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)

        # Query rows this tile handles
        q_start = pid_s * BLOCK_S
        q_offsets = q_start + tl.arange(0, BLOCK_S)
        q_mask = q_offsets < S
        safe_q = tl.where(q_mask, q_offsets, 0)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        safe_d = tl.where(d_mask, d_offsets, 0)

        # Accumulators in float32 for precision
        out_acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)
        m_i = tl.full([BLOCK_S], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_S], dtype=tl.float32)

        # Load Q tile: [BLOCK_S, BLOCK_D]
        q_ptrs = Q_ptr + pid_bh * stride_qh + safe_q[:, None] * stride_qs + safe_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)

        # Loop over key/value tiles
        for k_start in range(0, S, BLOCK_S):
            k_offsets = k_start + tl.arange(0, BLOCK_S)
            k_mask = k_offsets < S
            safe_k = tl.where(k_mask, k_offsets, 0)

            # Load K tile: [BLOCK_S, BLOCK_D]
            k_ptrs = K_ptr + pid_bh * stride_kh + safe_k[:, None] * stride_ks + safe_d[None, :] * stride_kd
            k_tile = tl.load(k_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)

            # Q·K^T: [BLOCK_S, BLOCK_S]
            qk = tl.dot(q_tile, tl.trans(k_tile)) * scale

            # Causal mask: position q can only attend to position k <= q
            causal = q_offsets[:, None] >= k_offsets[None, :]
            qk = tl.where(causal & q_mask[:, None] & k_mask[None, :],
                          qk, float('-inf'))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            
            # alpha = exp(m_i - m_new), zero if m_i is -inf
            alpha = tl.where(m_i == float('-inf'), 0.0, tl.exp(m_i - m_new))
            
            # p = exp(qk - m_new), zero if m_new is -inf
            p = tl.where(m_new[:, None] == float('-inf'), 0.0, tl.exp(qk - m_new[:, None]))
            
            # Load V tile: [BLOCK_S, BLOCK_D]
            v_ptrs = V_ptr + pid_bh * stride_vh + safe_k[:, None] * stride_vs + safe_d[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)

            # Update output accumulator: Out = Out * alpha + Softmax(QK) * V
            out_acc = out_acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        # Final normalization
        out_acc = out_acc / l_i[:, None]

        # Store output: [BLOCK_S, BLOCK_D]
        out_ptrs = Out_ptr + pid_bh * stride_oh + safe_q[:, None] * stride_os + safe_d[None, :] * stride_od
        tl.store(out_ptrs, out_acc.to(Out_ptr.dtype.element_ty), mask=(q_mask[:, None] & d_mask[None, :]))


def triton_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Triton-accelerated fused causal attention.

    Args:
        q: [B*H, S, D] queries
        k: [B*H, S, D] keys
        v: [B*H, S, D] values
        scale: 1/sqrt(d_head)

    Returns:
        output: [B*H, S, D]
    """
    BH, S, D = q.shape

    # Power of 2 larger than or equal to D
    BLOCK_D = 1 << (D - 1).bit_length()
    
    # Adjust BLOCK_S to fit in shared memory (standard 96KB-100KB limit)
    # Fused kernel needs 4 tiles (Q, K, V, Acc). 
    # With D=220 (BLOCK_D=256), each float32 tile is 64KB if BLOCK_S=64.
    # 4 * 16 * 256 * 4 bytes = 16KB.
    # 4 * 32 * 256 * 4 bytes = 32KB.
    # On most GPUs, 32 would work, but 16 is safer and avoids fragmentation.
    if BLOCK_D >= 256:
        BLOCK_S = 16
    elif BLOCK_D >= 128:
        BLOCK_S = 32
    else:
        BLOCK_S = 64

    # Output buffer
    output = torch.empty_like(q)

    grid = (BH, (S + BLOCK_S - 1) // BLOCK_S)

    _fused_attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(0),
        output.stride(1), output.stride(2),
        S=S, D=D,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        scale=scale,
    )

    return output


# ──────────────────────────────────────────────────────────────────────────
# CUDA Graph Wrapper for the Thought Loop
# ──────────────────────────────────────────────────────────────────────────

class CUDAGraphThoughtLoop:
    """Wraps the thought loop in a CUDA Graph for kernel launch elimination.

    The thought loop consists of T sequential steps, each launching
    multiple small CUDA kernels (attention, synapse, NLM/stream, norms).
    At T=8 with 8 layers, that's ~200+ kernel launches per forward pass.
    CUDA Graphs capture the entire sequence of launches into a single
    graph and replay it in one GPU operation.

    Lifecycle:
      1. warmup() — runs one forward with recording (initializes graph)
      2. replay() — efficient replay on subsequent calls (same shapes)
      3. invalidate() — called when shapes change (new B, S, T)

    Important constraints:
      - All tensor shapes must be identical between capture and replay.
      - No CPU-side branching during captured execution.
      - No dynamic memory allocation during replay.

    Args:
        enabled: Whether to use CUDA Graphs (disabled on CPU or AMD).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: dict = {}
        self._static_outputs: dict = {}
        self._captured_shapes: Optional[tuple] = None

    def _should_recapture(self, B: int, S: int, T: int) -> bool:
        """Check if we need to re-capture the graph (shape change)."""
        shapes = (B, S, T)
        if self._captured_shapes != shapes:
            return True
        return False

    @contextmanager
    def capture_context(self, B: int, S: int, T: int):
        """Context manager for CUDA graph capture.

        Usage:
            with cuda_graph.capture_context(B, S, T):
                # Run the thought loop — this is being captured
                result = model._thought_loop(...)

        Yields control to the caller, who should run exactly one
        forward pass inside the context.
        """
        if not self.enabled:
            yield
            return

        # Warmup (non-captured run to allocate all tensors)
        torch.cuda.synchronize()

        if self._graph is not None:
            self._graph = None

        self._graph = torch.cuda.CUDAGraph()
        self._captured_shapes = (B, S, T)

        # Capture
        with torch.cuda.graph(self._graph):
            yield

        torch.cuda.synchronize()

    def replay(self):
        """Replay the captured graph."""
        if self._graph is not None:
            self._graph.replay()
        else:
            raise RuntimeError(
                "No CUDA Graph captured. Call capture_context() first."
            )

    def invalidate(self):
        """Invalidate the captured graph (e.g., on shape change)."""
        self._graph = None
        self._captured_shapes = None
        self._static_inputs.clear()
        self._static_outputs.clear()

    @property
    def is_captured(self) -> bool:
        return self._graph is not None


# ──────────────────────────────────────────────────────────────────────────
# N·log(N) Tiled Schedule for Thought Step Reuse
# ──────────────────────────────────────────────────────────────────────────

def compute_tiled_schedule(T: int, n_layers: int, tile_size: int = 4) -> list[list[tuple[int, int]]]:
    """Compute a tiled execution schedule for the thought loop.

    Instead of processing thought steps 1, 2, ..., T sequentially where
    each step independently reads its history from HBM, this schedule
    groups steps into tiles so that intermediate results stay in SRAM
    across multiple iterations.

    The schedule produces groups of (step, layer) pairs that can share
    temporary data. Within each group, steps are processed sequentially
    but their intermediate results are kept in fast memory.

    Reference: Recurrent Transformer IO-aware tiling algorithm.

    Algorithm:
      - Divide T steps into tiles of size `tile_size`
      - Within each tile, all layer outputs are kept in SRAM
      - Between tiles, only the boundary states need HBM round-trips
      - Total HBM traffic: O(T/tile_size * n_layers * D) = O(T·log(T)·D)
        when tile_size = O(log(T))

    Args:
        T: Number of thought steps.
        n_layers: Number of layers per step.
        tile_size: Steps per tile (auto-tuned to ~log(T) if 0).

    Returns:
        List of tiles, where each tile is a list of (step, layer) pairs.
    """
    if tile_size <= 0:
        # Auto-tune: use log2(T) rounded up, minimum 2
        tile_size = max(2, int(math.ceil(math.log2(max(T, 2)))))

    schedule = []
    for tile_start in range(0, T, tile_size):
        tile_end = min(tile_start + tile_size, T)
        tile = []
        for t in range(tile_start, tile_end):
            for l in range(n_layers):
                tile.append((t, l))
        schedule.append(tile)

    return schedule


# ──────────────────────────────────────────────────────────────────────────
# Pure PyTorch Fallback: Flash-style Causal Attention
# ──────────────────────────────────────────────────────────────────────────

def pytorch_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """PyTorch fallback for causal attention (used when Triton unavailable).

    Uses torch.nn.functional.scaled_dot_product_attention when available
    (PyTorch 2.0+), otherwise manual implementation.

    Args:
        q: [B, H, S, D] queries
        k: [B, H, S, D] keys
        v: [B, H, S, D] values
        scale: scaling factor (1/sqrt(d_head))
        dropout_p: dropout probability
        training: whether in training mode

    Returns:
        output: [B, H, S, D]
    """
    if hasattr(F, 'scaled_dot_product_attention'):
        # PyTorch 2.0+ — uses FlashAttention or memory-efficient backend
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout_p if training else 0.0,
            scale=scale,
        )

    # Manual fallback
    S = q.size(-2)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    causal_mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
    )
    attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

    attn_weights = F.softmax(attn_weights, dim=-1)
    if training and dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    return torch.matmul(attn_weights, v)


# ──────────────────────────────────────────────────────────────────────────
# Accelerated Attention Dispatcher
# ──────────────────────────────────────────────────────────────────────────

def accelerated_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    dropout_p: float = 0.0,
    training: bool = False,
    use_triton: bool = True,
) -> torch.Tensor:
    """Dispatch to the fastest available causal attention implementation.

    Priority: Triton tiled kernel > PyTorch SDPA > Manual.

    Args:
        q: [B, S, n_heads, d_head] queries (will be transposed internally)
        k: [B, S, n_heads, d_head] keys
        v: [B, S, n_heads, d_head] values
        n_heads: number of attention heads
        dropout_p: dropout probability
        training: whether in training mode
        use_triton: try Triton first

    Returns:
        output: [B, S, n_heads * d_head]
    """
    B, S, H, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Transpose to [B, H, S, D] for attention
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if use_triton and _HAS_TRITON and q.is_cuda:
        # Triton path: reshape to [B*H, S, D]
        q_flat = q.reshape(B * H, S, D).contiguous()
        k_flat = k.reshape(B * H, S, D).contiguous()
        v_flat = v.reshape(B * H, S, D).contiguous()
        out = triton_causal_attention(q_flat, k_flat, v_flat, scale)
        out = out.view(B, H, S, D).transpose(1, 2).reshape(B, S, H * D)
        return out

    # PyTorch SDPA / manual fallback
    out = pytorch_causal_attention(q, k, v, scale, dropout_p, training)
    return out.transpose(1, 2).reshape(B, S, H * D)


# ═════════════════════════════════════════════════════════════════════════
# dssa.py
# ═════════════════════════════════════════════════════════════════════════

class SparseStateExpansion(nn.Module):
    """Linear attention with sparse state expansion.

    Maintains m state partitions S_i ∈ R^{d_key × d_value}. On each token,
    an input-dependent gating vector selects the top-k partitions to update:

        S_i ← S_i + e_i · k^T v    if i ∈ top-k(gating)
        S_i ← S_i                   otherwise

    Output is computed as: o = Σ_i (q^T S_i) · selection_weight_i

    Args:
        d_model: Model dimension (total, split across heads).
        n_heads: Number of attention heads.
        n_partitions: Number of state partitions m.
        top_k: Number of partitions to update per step.
        d_head: Dimension per head (default: d_model // n_heads).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_partitions: int = 32,
        top_k: int = 8,
        d_head: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_partitions = n_partitions
        self.top_k = min(top_k, n_partitions)
        self.d_head = d_head or (d_model // n_heads)

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        # Gating: produces partition selection logits
        self.gate_proj = nn.Linear(d_model, n_partitions, bias=True)

        # Layer norms for stability
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass with sparse state expansion (vectorized).

        Mathematically equivalent to the per-position recurrent scan in
        ``_forward_reference``, but expressed as four standard tensor
        ops (matmul, scatter, matmul, softmax) instead of a Python loop
        over S positions × k partitions × B batch elements.

        Derivation. The recurrent state is
            state[t, h, p, d_k, d_v] = Σ_{t' ≤ t} pw[t', p] · k[t', h, d_k] · v[t', h, d_v]
        where pw is the dense partition-weight tensor (gate_weights at
        the chosen partitions, zero elsewhere). The readout is then
            readout[t, h, p, d_v]
                = Σ_{d_k} q[t, h, d_k] · state[t, h, p, d_k, d_v]
                = Σ_{t' ≤ t} pw[t', p] · (q[t,h] · k[t',h]) · v[t', h, d_v]
        i.e. linear attention with per-partition gated values. We never
        materialize the [B, H, m, dh, dh] state — it cancels out.

        Why the original was slow. The previous implementation had a
        ``for t in range(S)`` Python loop with nested ``for i in range(k)``
        and ``for b in range(B)`` plus a ``part_idx[b].item()`` call that
        forced a CUDA→CPU sync on every iteration: 4096 syncs per call
        at S=512, k=8, B=1. Vectorizing eliminates the sync entirely
        and lets cuBLAS handle the heavy lifting.

        Memory. Peak intermediate is ``weighted_v`` of shape
        [B, S, H, m, dh] (~32 MB at default config in bf16) and the
        attention scores [B, H, S, S] (~8 MB). Total well under 100 MB
        for typical configs; scales linearly with S and quadratically
        with H·dh, so very long contexts (S>4k) may want chunking.
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        m = self.n_partitions
        k = self.top_k

        # Project Q from queries, K/V from text
        q = self.q_proj(x).view(B, S, H, dh)
        kk = self.k_proj(text_kv).view(B, S, H, dh)
        v = self.v_proj(text_kv).view(B, S, H, dh)

        # Normalize Q and K for stable dot products (preserves SSE design)
        q = self.q_norm(q)
        kk = self.k_norm(kk)

        # Top-k partition selection. We use the values returned by topk
        # directly instead of an extra gather (the original did
        # ``gate_logits.gather(-1, topk(...).indices)`` which recomputes
        # the same values).
        gate_logits = self.gate_proj(text_kv)            # [B, S, m]
        top_values, top_indices = gate_logits.topk(k, dim=-1)
        gate_weights = F.softmax(top_values, dim=-1)     # [B, S, k]

        # Scatter the per-position k weights into a dense [B, S, m]
        # tensor. Non-selected partitions stay at zero, so they
        # contribute nothing to the readout — equivalent to "only the
        # top-k partitions get updated" but in a form the rest of the
        # pipeline can consume in vectorized fashion.
        # We use scatter_add_ rather than scatter_ so that the (very
        # rare) case of duplicate top-k indices accumulates exactly
        # like the sequential reference would.
        pw = torch.zeros(B, S, m, device=x.device, dtype=x.dtype)
        pw.scatter_add_(2, top_indices, gate_weights.to(pw.dtype))

        # Linear-attention scores (no softmax). Use matmul (cuBLAS GEMM)
        # rather than einsum for the hot path.
        q_h = q.permute(0, 2, 1, 3)     # [B, H, S_t, dh]
        k_h = kk.permute(0, 2, 3, 1)    # [B, H, dh,  S_u]
        scores = torch.matmul(q_h, k_h)  # [B, H, S_t, S_u]

        # Causal mask — fill with 0 (linear attention has no softmax,
        # so future positions just shouldn't contribute).
        if causal:
            causal_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, 0.0)

        # Per-partition weighted values: gate v by pw, broadcasting over
        # heads and value dimension.
        #   weighted_v[b, u, h, p, d] = pw[b, u, p] · v[b, u, h, d]
        weighted_v = v.unsqueeze(3) * pw.view(B, S, 1, m, 1)  # [B, S, H, m, dh]

        # Per-partition readout. Reshape for a single batched matmul:
        #   scores [B, H, S_t, S_u] @ weighted_v [B, H, S_u, m·dh]
        #     → readout_flat [B, H, S_t, m·dh]
        # then split m·dh back out and permute S_t to dim 1.
        wv_h = weighted_v.permute(0, 2, 1, 3, 4).reshape(B, H, S, m * dh)
        readout_flat = torch.matmul(scores, wv_h)            # [B, H, S, m·dh]
        readout = readout_flat.view(B, H, S, m, dh).permute(0, 2, 1, 3, 4)
        # readout: [B, S, H, m, dh]

        # Per-partition norms → softmax over partitions, with a tiny eps
        # so positions with no contribution yet (e.g. early positions
        # whose top-k partitions never received content) don't NaN.
        partition_norms = readout.norm(dim=-1) + 1e-8        # [B, S, H, m]
        attn_weights = F.softmax(partition_norms, dim=-1)    # [B, S, H, m]

        # Combine partitions. Equivalent to einsum('bthm,bthmd->bthd')
        # but written as a broadcast-and-sum so it's clearly a
        # weighted-mean op rather than a matmul.
        out = (attn_weights.unsqueeze(-1) * readout).sum(dim=3)  # [B, S, H, dh]
        out = out.reshape(B, S, H * dh)
        return self.out_proj(out)

    def _forward_reference(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Sequential reference implementation, kept for numerical
        regression testing. Do NOT use in training — this is the original
        Python-loop version that hits ~4096 CUDA syncs per call. The
        public ``forward`` produces numerically identical output (to bf16
        rounding) at a fraction of the cost.
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        m = self.n_partitions
        k = self.top_k

        q = self.q_proj(x).view(B, S, H, dh)
        kk = self.k_proj(text_kv).view(B, S, H, dh)
        v = self.v_proj(text_kv).view(B, S, H, dh)
        q = self.q_norm(q)
        kk = self.k_norm(kk)

        gate_logits = self.gate_proj(text_kv)
        _, top_indices = gate_logits.topk(k, dim=-1)
        gate_weights = F.softmax(
            gate_logits.gather(-1, top_indices), dim=-1
        )

        state = torch.zeros(B, H, m, dh, dh, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(S):
            kt = kk[:, t]
            vt = v[:, t]
            qt = q[:, t]
            gt = gate_weights[:, t]
            idx_t = top_indices[:, t]

            kv_outer = torch.einsum('bhd,bhe->bhde', kt, vt)

            for i in range(k):
                part_idx = idx_t[:, i]
                weight = gt[:, i]
                for b in range(B):
                    pi = part_idx[b].item()
                    state[b, :, pi] = state[b, :, pi] + weight[b] * kv_outer[b]

            readout = torch.einsum('bhd,bhmde->bhme', qt, state)
            partition_norms = readout.norm(dim=-1) + 1e-8
            attn_weights = F.softmax(partition_norms, dim=-1)
            out_t = torch.einsum('bhm,bhmd->bhd', attn_weights, readout)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, S, H * dh)
        return self.out_proj(output)


class MixtureOfBlockAttention(nn.Module):
    """Block-wise sparse attention (MoBA).

    Divides the KV sequence into blocks of fixed size. For each query,
    scores blocks by their relevance (mean key similarity) and attends
    only to the top-k blocks. This gives O(N·k/B) compute instead of O(N²).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        block_size: Size of each KV block.
        top_k_blocks: Number of blocks to attend to per query.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 64,
        top_k_blocks: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass with block-sparse attention.

        Args:
            x: [B, S, d_model] — query source.
            text_kv: [B, S, d_model] — KV source.
            causal: Whether to apply causal masking.

        Returns:
            output: [B, S, d_model]
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        bs = self.block_size

        q = self.q_proj(x).view(B, S, H, dh).transpose(1, 2)     # [B, H, S, dh]
        k = self.k_proj(text_kv).view(B, S, H, dh).transpose(1, 2)  # [B, H, S, dh]
        v = self.v_proj(text_kv).view(B, S, H, dh).transpose(1, 2)  # [B, H, S, dh]

        # Number of blocks (pad last block if needed)
        n_blocks = (S + bs - 1) // bs
        pad_len = n_blocks * bs - S

        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        # Reshape K, V into blocks: [B, H, n_blocks, block_size, dh]
        k_blocks = k.view(B, H, n_blocks, bs, dh)
        v_blocks = v.view(B, H, n_blocks, bs, dh)

        # Block-level scoring: mean key per block
        k_block_means = k_blocks.mean(dim=3)  # [B, H, n_blocks, dh]

        # Score each query against block means
        # [B, H, S, dh] @ [B, H, dh, n_blocks] → [B, H, S, n_blocks]
        block_scores = torch.matmul(q, k_block_means.transpose(-2, -1))
        block_scores = block_scores / math.sqrt(dh)

        # Causal masking: block b is valid for query at position t
        # only if block_start <= t (i.e., at least part of the block is visible)
        if causal:
            block_starts = torch.arange(n_blocks, device=q.device) * bs  # [n_blocks]
            query_pos = torch.arange(S, device=q.device)  # [S]
            # [S, n_blocks]: True if block start > query position (future block)
            causal_block_mask = block_starts.unsqueeze(0) > query_pos.unsqueeze(1)
            block_scores = block_scores.masked_fill(
                causal_block_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )

        # Select top-k blocks per query
        k_effective = min(self.top_k_blocks, n_blocks)
        _, top_block_idx = block_scores.topk(k_effective, dim=-1)  # [B, H, S, k]

        # Gather the selected blocks' KV
        # Expand indices for gathering: [B, H, S, k] → gather from [B, H, n_blocks, bs, dh]
        # We need to attend within the selected blocks
        scale = math.sqrt(dh)
        outputs = torch.zeros(B, H, S, dh, device=q.device, dtype=q.dtype)

        for i in range(k_effective):
            block_idx = top_block_idx[:, :, :, i]  # [B, H, S]

            # Gather K and V for this block selection
            # block_idx → index into k_blocks: [B, H, n_blocks, bs, dh]
            bi_expanded = block_idx.unsqueeze(-1).unsqueeze(-1).expand(
                B, H, S, bs, dh
            )
            ki = k_blocks.gather(2, bi_expanded)  # [B, H, S, bs, dh]
            vi = v_blocks.gather(2, bi_expanded)  # [B, H, S, bs, dh]

            # Attention within block
            # [B, H, S, 1, dh] @ [B, H, S, dh, bs] → [B, H, S, 1, bs]
            attn = torch.matmul(
                q.unsqueeze(3), ki.transpose(-2, -1)
            ).squeeze(3) / scale  # [B, H, S, bs]

            # Causal mask within block
            if causal:
                block_starts_sel = block_idx * bs  # [B, H, S]
                positions = torch.arange(bs, device=q.device)  # [bs]
                abs_positions = block_starts_sel.unsqueeze(-1) + positions  # [B, H, S, bs]
                query_positions = torch.arange(S, device=q.device).view(1, 1, S, 1)
                future_mask = abs_positions > query_positions
                attn = attn.masked_fill(future_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            # Handle all-masked rows (position 0 with future-only blocks)
            attn = torch.nan_to_num(attn, nan=0.0)

            # Weighted sum
            # [B, H, S, 1, bs] @ [B, H, S, bs, dh] → [B, H, S, dh]
            block_out = torch.matmul(attn.unsqueeze(3), vi).squeeze(3)
            outputs = outputs + block_out

        # Average over blocks (simple; could use block-score-weighted)
        outputs = outputs / max(k_effective, 1)

        # Reshape and project output
        outputs = outputs.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(outputs)


class DualSpaceSparseAttention(nn.Module):
    """Hybrid attention: SSE for compressed retrieval + MoBA for exact blocks.

    Combines the constant-memory linear attention of Sparse State Expansion
    with periodic exact block attention from Mixture of Block Attention.
    A learnable router determines the mixing ratio per position.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_partitions: SSE state partitions.
        sse_top_k: Active SSE partitions per step.
        block_size: MoBA block size.
        moba_top_k_blocks: MoBA blocks per query.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_partitions: int = 32,
        sse_top_k: int = 8,
        block_size: int = 64,
        moba_top_k_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.sse = SparseStateExpansion(
            d_model=d_model,
            n_heads=n_heads,
            n_partitions=n_partitions,
            top_k=sse_top_k,
        )
        self.moba = MixtureOfBlockAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            top_k_blocks=moba_top_k_blocks,
        )

        # Router: determines mixing ratio between SSE and MoBA
        # Output: 2 logits (SSE weight, MoBA weight) per position
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Hybrid SSE + MoBA attention.

        Args:
            x: [B, S, d_model] — query source.
            text_kv: [B, S, d_model] — KV source.
            causal: Whether to apply causal masking.

        Returns:
            output: [B, S, d_model]
        """
        # Route: decide SSE vs MoBA mixing per position
        route_logits = self.router(x)  # [B, S, 2]
        route_weights = F.softmax(route_logits, dim=-1)  # [B, S, 2]

        # SSE branch (compressed, constant memory)
        sse_out = self.sse(x, text_kv, causal=causal)  # [B, S, D]

        # MoBA branch (exact, block-sparse)
        moba_out = self.moba(x, text_kv, causal=causal)  # [B, S, D]

        # Weighted combination
        w_sse = route_weights[:, :, 0:1]   # [B, S, 1]
        w_moba = route_weights[:, :, 1:2]  # [B, S, 1]

        output = w_sse * sse_out + w_moba * moba_out
        output = self.out_norm(output)
        output = self.dropout(output)

        return output

# ═════════════════════════════════════════════════════════════════════════
# memory.py
# ═════════════════════════════════════════════════════════════════════════

class TemporalMemory(nn.Module):
    """
    FIFO history buffers for pre-activations and post-activations.

    Maintains two rolling buffers of shape [batch, history_len, d_latent].
    Each call to push() shifts the buffer forward (oldest entry dropped)
    and appends the new values at the end.

    Args:
        d_latent: Neuron count (width of each activation snapshot).
        history_len: Depth of the FIFO buffer.
    """

    def __init__(self, d_latent: int, history_len: int):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len

        # Learnable initial states for history buffers
        self.pre_history_init = nn.Parameter(torch.zeros(1, self.history_len, self.d_latent))
        self.post_history_init = nn.Parameter(torch.zeros(1, self.history_len, self.d_latent))

        # Buffers are registered as non-persistent state (not saved with model params)
        # They are initialized lazily on first push() to match batch size and device.
        self.register_buffer("pre_history", None, persistent=False)
        self.register_buffer("post_history", None, persistent=False)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Initialize both history buffers using the learnable initial states.

        Args:
            batch_size: Current batch size.
            device: Target device.
            dtype: Data type for buffers.
        """
        self.pre_history = self.pre_history_init.to(device=device, dtype=dtype).expand(
            batch_size, -1, -1
        ).clone()
        self.post_history = self.post_history_init.to(device=device, dtype=dtype).expand(
            batch_size, -1, -1
        ).clone()

    def push_pre(self, pre_activations: torch.Tensor):
        """
        Push new pre-activations into the FIFO buffer.

        Uses roll + indexed assignment instead of cat. Memory cost is
        equivalent (both allocate a fresh [B, H, D] tensor per call), but
        the autograd graph is cleaner: roll has one parent (the buffer),
        cat has two (the slice + the new entry), and the roll variant
        plays better with the periodic .detach() applied at thought-step
        boundaries to break cross-step graph chaining.

        Args:
            pre_activations: [batch, d_latent] — new pre-activation values.
        """
        # roll returns a NEW tensor (not in-place), so the subsequent
        # index_put modifies that new tensor — never the original leaf.
        # Safe under autograd even when the buffer is a leaf with
        # requires_grad=True.
        self.pre_history = self.pre_history.roll(-1, dims=1)
        self.pre_history[:, -1, :] = pre_activations

    def push_post(self, post_activations: torch.Tensor):
        """
        Push new post-activations into the FIFO buffer.

        Args:
            post_activations: [batch, d_latent] — new post-activation values.
        """
        self.post_history = self.post_history.roll(-1, dims=1)
        self.post_history[:, -1, :] = post_activations

    def get_pre_history(self) -> torch.Tensor:
        """Returns the full pre-activation history buffer [batch, history_len, d_latent]."""
        return self.pre_history

    def get_post_history(self) -> torch.Tensor:
        """Returns the full post-activation history buffer [batch, history_len, d_latent]."""
        return self.post_history


class SynchronizationComputer(nn.Module):
    """
    Computes neural synchronization representations from post-activation history.

    The synchronization matrix S_t = Z_t @ Z_t^T captures the coupling
    structure between neurons based on their recent activation patterns.
    This replaces token-position-based queries with state-based queries.

    Three modes are supported (controlled by `method`):
    - "full": Full S_t flattened → [d_latent^2] (expensive, most expressive)
    - "diag_summary": diag(S_t) + row_means + col_means → [3 * d_latent] (practical)
    - "low_rank": Top-k SVD approximation → [rank * d_latent] (balanced)
    - "sparse_decay": Sparse pairing with learnable exponential decay → [sync_sparse_pairs]

    Args:
        d_latent: Neuron count.
        history_len: FIFO buffer depth.
        method: Synchronization computation method.
        rank: Rank for low_rank method.
        sync_sparse_pairs: Number of sparse pairs for sparse_decay method.
    """

    def __init__(
        self,
        d_latent: int,
        history_len: int,
        method: str = "diag_summary",
        rank: int = 32,
        sync_sparse_pairs: int = 256,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len
        self.method = method
        self.rank = rank
        self.sync_sparse_pairs = sync_sparse_pairs

        if method == "sparse_decay":
            # Pre-choose D_chosen neuron pairs from D total neurons
            idxs_left = torch.randint(low=0, high=d_latent, size=(sync_sparse_pairs,))
            idxs_right = torch.randint(low=0, high=d_latent, size=(sync_sparse_pairs,))
            self.register_buffer("idxs_left", idxs_left)
            self.register_buffer("idxs_right", idxs_right)
            # Define learnable exponential decay scaling factors per neuron pair
            self.r = nn.Parameter(torch.zeros(1, sync_sparse_pairs, 1))

    def compute(self, post_history: torch.Tensor) -> torch.Tensor:
        """
        Compute synchronization representation from post-activation history.

        Args:
            post_history: [batch, history_len, d_latent] — the Z_t matrix.

        Returns:
            sync_repr: [batch, sync_dim] — flattened synchronization representation.
        """
        B, H, D = post_history.shape
        H_safe = max(H, 1)

        # ── Fast path for diag_summary ──────────────────────────────────
        # The naive path materializes S = Zᵀ Z / H of shape [B, D, D],
        # which at D=768, B=512, bf16 is 576 MiB *per layer per thought
        # step* — enough to OOM a 200M model on consumer hardware.
        # Since we only need diag(S), row_means(S), col_means(S), we can
        # compute all three directly from Z in O(B·H·D) memory, never
        # touching the [D, D] matrix.
        #
        # Identities (all exact, no approximation):
        #   diag(S)[b,i]      = (1/H) · Σ_h Z[b,h,i]²
        #   row_means(S)[b,i] = (1/(D·H)) · Σ_h Z[b,h,i] · (Σ_j Z[b,h,j])
        #   col_means(S)[b,j] = row_means(S)[b,j]   (S is symmetric)
        if self.method == "diag_summary":
            # diag: sum of squares along the history axis.
            # einsum fuses the elementwise square with the reduction,
            # avoiding the [B, H, D] intermediate that (post*post) would
            # allocate before .sum(dim=1).
            diag = torch.einsum('bhd,bhd->bd', post_history, post_history) / H_safe  # [B, D]

            # row_means: via bmm against the per-history-step row sums
            # Zsum: [B, H, 1] — cheap, no D×D tensor ever exists
            Zsum = post_history.sum(dim=2, keepdim=True)                # [B, H, 1]
            row_means = torch.bmm(
                post_history.transpose(1, 2),                            # [B, D, H]
                Zsum,                                                    # [B, H, 1]
            ).squeeze(2) / (H_safe * D)                                  # [B, D]

            # col_means == row_means by symmetry of Zᵀ Z
            col_means = row_means

            return torch.cat([diag, row_means, col_means], dim=1)       # [B, 3D]

        # ── Fast path for sparse_decay ──────────────────────────────────
        if self.method == "sparse_decay":
            S_post = post_history # [B, T=H, D]
            # decay BACK in time
            t_back = torch.arange(H - 1, -1, -1, device=post_history.device, dtype=post_history.dtype)
            t_back = t_back.view(1, H, 1) # [1, H, 1]
            
            # Compute per NEURON PAIR exponential decays
            # self.r is [1, D_chosen, 1]
            # r permuted to [1, 1, D_chosen] to match t_back broadcast [1, H, 1] -> [1, H, D_chosen]
            exp_decay = torch.exp(-t_back * self.r.view(1, 1, -1)) # [1, H, D_chosen]
            
            # Subsampled S
            # S[:,:,idxs_left] * exp_decay * S[:,:,idxs_right]
            S_left = S_post[:, :, self.idxs_left] # [B, H, D_chosen]
            S_right = S_post[:, :, self.idxs_right] # [B, H, D_chosen]
            S_multiplied = S_left * exp_decay * S_right # [B, H, D_chosen]
            
            # Sum over the free T (H) dimension and normalise by sqrt of AUC of decays
            synch_representation = S_multiplied.sum(dim=1) / torch.sqrt(exp_decay.sum(dim=1)) # [B, D_chosen]
            return synch_representation

        # ── Full-matrix path (required for "full" and "low_rank") ───────
        # S_t = Z_tᵀ @ Z_t → [batch, d_latent, d_latent]
        # Note: We compute Zᵀ @ Z (not Z @ Zᵀ) to get neuron-neuron coupling
        S = torch.bmm(post_history.transpose(1, 2), post_history)
        S = S / H_safe

        if self.method == "full":
            # Flatten entire matrix
            return S.reshape(B, D * D)

        elif self.method == "low_rank":
            # Top-k eigenvalue approximation via SVD
            # Only compute the top `rank` singular values/vectors
            U, s, _ = torch.linalg.svd(S, full_matrices=False)
            # Take top-rank components: U[:, :, :rank] * s[:, :rank]
            k = min(self.rank, D)
            low_rank = U[:, :, :k] * s[:, :k].unsqueeze(1)  # [batch, d_latent, rank]
            return low_rank.reshape(B, D * k)

        else:
            raise ValueError(f"Unknown sync method: {self.method}")

    @property
    def output_dim(self) -> int:
        """Dimension of the synchronization representation vector."""
        if self.method == "full":
            return self.d_latent * self.d_latent
        elif self.method == "diag_summary":
            return 3 * self.d_latent
        elif self.method == "low_rank":
            return min(self.rank, self.d_latent) * self.d_latent
        elif self.method == "sparse_decay":
            return self.sync_sparse_pairs
        else:
            raise ValueError(f"Unknown sync method: {self.method}")

# ═════════════════════════════════════════════════════════════════════════
# nlm.py
# ═════════════════════════════════════════════════════════════════════════

class NeuronLevelModels(nn.Module):
    """
    Per-neuron MLPs that process temporal history of pre-activations.

    Each of the d_latent neurons has its own small MLP:
        history [batch, history_len] → hidden [batch, nlm_hidden] → output [batch, 1]

    When nlm_groups > 1, neurons are grouped and share MLP parameters
    within each group (reducing parameter count by the group factor).

    Args:
        d_latent: Number of neurons in the latent space.
        history_len: Depth of the FIFO buffer (temporal window).
        nlm_hidden_dim: Hidden dimension of each neuron's MLP.
        nlm_groups: Number of neuron groups (1 = true per-neuron, >1 = grouped).
        dropout: Dropout rate applied after hidden layer.
    """

    def __init__(
        self,
        d_latent: int,
        history_len: int,
        nlm_hidden_dim: int,
        nlm_groups: int = 1,
        dropout: float = 0.1,
        use_ip: bool = False,
        ip_lr: float = 0.01,
        ip_target: float = 0.1,
        ip_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len
        self.nlm_hidden_dim = nlm_hidden_dim
        self.nlm_groups = nlm_groups
        self.use_ip = use_ip
        self.ip_lr = ip_lr
        self.ip_target = ip_target
        self.ip_ema_decay = ip_ema_decay

        assert d_latent % nlm_groups == 0, \
            f"d_latent ({d_latent}) must be divisible by nlm_groups ({nlm_groups})"
        self.neurons_per_group = d_latent // nlm_groups

        # Each group has its own 2-layer MLP:
        #   Layer 1: [history_len] → [nlm_hidden_dim]
        #   Layer 2: [nlm_hidden_dim] → [1]
        # Packed as [nlm_groups, ...] tensors for batched computation.

        # Layer 1 weights and biases: [nlm_groups, history_len, nlm_hidden_dim]
        self.w1 = nn.Parameter(
            torch.empty(nlm_groups, history_len, nlm_hidden_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(nlm_groups, nlm_hidden_dim))

        # Layer 2 weights and biases: [nlm_groups, nlm_hidden_dim, 1]
        self.w2 = nn.Parameter(
            torch.empty(nlm_groups, nlm_hidden_dim, 1)
        )
        self.b2 = nn.Parameter(torch.zeros(nlm_groups, 1))

        # Learnable output gate — controls how much the NLM output
        # contributes vs. a simple pass-through of the latest pre-activation.
        # Initialized near 0.5 so both the learned dynamics and direct signal
        # contribute equally at initialization.
        self.gate = nn.Parameter(torch.full((d_latent,), 0.0))

        self.dropout = nn.Dropout(dropout)

        # Intrinsic Plasticity: per-hidden-unit adaptive threshold.
        # Both are non-trainable buffers updated in-place during forward.
        if use_ip:
            self.register_buffer(
                "_act_ema", torch.zeros(nlm_groups, nlm_hidden_dim)
            )
            self.register_buffer(
                "ip_bias", torch.zeros(nlm_groups, nlm_hidden_dim)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize NLM weights for stable training.

        Layer 1: Kaiming init (fan_in = history_len) so that the pre-activation
        history maps into a well-scaled hidden space.
        Layer 2: Small init so initial NLM output ≈ 0, meaning the gated
        output is dominated by the pass-through at the start of training.
        """
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in = self.history_len
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)

        # Small init for layer 2 so NLM starts near identity
        nn.init.normal_(self.w2, std=0.01)
        nn.init.zeros_(self.b2)

    def forward(self, pre_activation_history: torch.Tensor) -> torch.Tensor:
        """
        Process temporal history through per-neuron MLPs.

        Args:
            pre_activation_history: [batch, history_len, d_latent]
                The FIFO buffer of pre-activations, oldest first.

        Returns:
            post_activations: [batch, d_latent]
                The new neuron states after temporal processing.
        """
        B, H, D = pre_activation_history.shape
        assert H == self.history_len and D == self.d_latent

        # Reshape for grouped computation:
        # [batch, history_len, d_latent] → [batch, history_len, nlm_groups, neurons_per_group]
        x = pre_activation_history.view(B, H, self.nlm_groups, self.neurons_per_group)
        # → [batch, nlm_groups, neurons_per_group, history_len]
        x = x.permute(0, 2, 3, 1)

        # Layer 1: [batch, groups, neurons_per_group, history_len] @ [groups, history_len, hidden]
        # Using einsum for the grouped matmul:
        # b=batch, g=groups, n=neurons_per_group, h=history_len, d=hidden
        h = torch.einsum("bgnh,ghd->bgnd", x, self.w1)
        # Add bias: [groups, hidden] broadcasts over [batch, groups, neurons_per_group, hidden]
        h = h + self.b1.unsqueeze(0).unsqueeze(2)

        # Intrinsic Plasticity: shift activation threshold before nonlinearity.
        # ip_bias shifts each hidden unit's working point toward ip_target mean firing.
        if self.use_ip:
            h = h + self.ip_bias.unsqueeze(0).unsqueeze(2)

        # Activation: GELU (smooth, avoids dead neurons unlike ReLU)
        h = F.gelu(h)

        # IP online update: β ← β + η_IP·(target − ĥ), no grad.
        # Only runs during training; evaluation sees fixed ip_bias.
        if self.use_ip and self.training:
            with torch.no_grad():
                # mean over batch (dim 0) and neurons_per_group (dim 2) → [G, D_hidden]
                batch_mean = h.detach().mean(dim=(0, 2))
                self._act_ema.mul_(self.ip_ema_decay).add_(
                    batch_mean * (1.0 - self.ip_ema_decay)
                )
                self.ip_bias.add_(self.ip_lr * (self.ip_target - self._act_ema))

        h = self.dropout(h)

        # Layer 2: [batch, groups, neurons_per_group, hidden] @ [groups, hidden, 1]
        out = torch.einsum("bgnd,gdo->bgno", h, self.w2)
        out = out + self.b2.unsqueeze(0).unsqueeze(2)
        # → [batch, groups, neurons_per_group, 1] → [batch, groups, neurons_per_group]
        out = out.squeeze(-1)

        # Reshape back: [batch, groups, neurons_per_group] → [batch, d_latent]
        nlm_output = out.reshape(B, self.d_latent)

        # Gated output: blend NLM dynamics with direct pass-through of latest input.
        # gate ∈ (0, 1) via sigmoid; starts at 0.5 so both contribute equally.
        alpha = torch.sigmoid(self.gate)  # [d_latent]
        latest_input = pre_activation_history[:, -1, :]  # [batch, d_latent]

        post_activations = alpha * torch.tanh(nlm_output) + (1 - alpha) * latest_input

        return post_activations

# ═════════════════════════════════════════════════════════════════════════
# intrinsic_plasticity.py
# ═════════════════════════════════════════════════════════════════════════

class IntrinsicPlasticityLayer(nn.Module):
    """Per-neuron adaptive threshold implementing Intrinsic Plasticity.

    Adds a learnable per-neuron bias (ip_bias) to any [..., d] activation
    tensor and updates it online during training to keep the mean activation
    close to ip_target:

        ema  ← decay · ema + (1 − decay) · mean_batch(x)
        bias ← bias + η · (target − ema)

    No gradient flows through the update — it is a pure homeostatic rule.
    At eval time the bias is fixed (converged).  Checkpoint-compatible: both
    ip_bias and _act_ema are registered buffers.

    Works with any flat or sequence-shaped input; reshapes to [*, d]
    internally to compute the cross-sample mean.
    """

    def __init__(
        self,
        d: int,
        ip_lr: float = 0.01,
        ip_target: float = 0.0,
        ip_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.ip_lr = ip_lr
        self.ip_target = ip_target
        self.ip_ema_decay = ip_ema_decay
        self.register_buffer("ip_bias", torch.zeros(d))
        self.register_buffer("_act_ema", torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ip_bias
        if self.training:
            with torch.no_grad():
                flat = x.detach().reshape(-1, x.shape[-1])   # [N, d]
                batch_mean = flat.mean(dim=0)                  # [d]
                self._act_ema.mul_(self.ip_ema_decay).add_(
                    batch_mean * (1.0 - self.ip_ema_decay)
                )
                self.ip_bias.add_(self.ip_lr * (self.ip_target - self._act_ema))
        return x

# ═════════════════════════════════════════════════════════════════════════
# matrix_stream.py
# ═════════════════════════════════════════════════════════════════════════

class SchemaRouter(nn.Module):
    """Cosine-similarity schema router for MatrixResidualStream.

    Maintains per-stream EMA prototype vectors.  On each forward pass it
    routes the incoming collapsed latent toward the most-similar schema
    stream, enabling rapid schema-consistent assimilation while protecting
    dissimilar streams from interference.

    Routing weights are passed to MatrixResidualStream.mix_post so that
    high-match streams receive a proportionally larger fraction of the new
    layer output.  The returned max_similarity scalar can also boost the
    Hebbian/BTSP lr_modulator in ThoughtLayer for the current tick.

    Args:
        n_streams:           Number of parallel residual streams.
        d_latent:            Stream dimension.
        ema_decay:           EMA decay for prototype update (per-batch).
        temperature:         Softmax temperature (lower = winner-take-all).
        novelty_threshold:   Max cosine-sim below this → caller may wake a
                             dormant stream.
    """

    def __init__(
        self,
        n_streams: int,
        d_latent: int,
        ema_decay: float = 0.99,
        temperature: float = 0.1,
        novelty_threshold: float = 0.1,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.d_latent = d_latent
        self.ema_decay = ema_decay
        self.temperature = max(float(temperature), 1e-6)
        self.novelty_threshold = novelty_threshold
        # Per-stream prototype (EMA of routed inputs).  Buffer so it's in
        # state_dict and checkpoint-restored; not a Parameter (no grad).
        self.register_buffer("prototypes", torch.zeros(n_streams, d_latent))

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:           [B, S, d_latent] — collapsed stream state (detached).
            active_mask: [N] bool         — which streams are active.
        Returns:
            routing_weights: [B, S, N]  — softmax weights per stream.
            max_similarity:  [B, S]     — max cosine-sim to any active stream.
        """
        B, S, D = x.shape
        x_norm = F.normalize(x.float(), dim=-1)             # [B, S, D]
        proto_norm = F.normalize(self.prototypes.float(), dim=-1)  # [N, D]
        sims = torch.einsum("bsd,nd->bsn", x_norm, proto_norm)     # [B, S, N]

        # Mask dormant streams: drive their similarity to -∞ so they get
        # ≈0 routing weight but don't disturb the softmax of active ones.
        mask_f = active_mask.float().view(1, 1, -1)                 # [1, 1, N]
        masked_sims = sims + (1.0 - mask_f) * (-1e9)
        routing_weights = F.softmax(
            masked_sims / self.temperature, dim=-1
        ).to(x.dtype)                                               # [B, S, N]

        # Max similarity over active streams only
        max_similarity = (sims * mask_f).max(dim=-1).values.to(x.dtype)  # [B, S]

        # EMA prototype update (no grad, only during training)
        if self.training:
            with torch.no_grad():
                rw = routing_weights.float().detach()
                xd = x.float().detach()
                total_weight = rw.sum(dim=(0, 1))                  # [N]
                weighted_x = torch.einsum("bsn,bsd->nd", rw, xd)  # [N, D]
                safe_count = total_weight.clamp(min=1e-8).unsqueeze(1)
                batch_proto = weighted_x / safe_count              # [N, D]
                self.prototypes.mul_(self.ema_decay).add_(
                    batch_proto * (1.0 - self.ema_decay)
                )

        return routing_weights, max_similarity


class ThalamicGate(nn.Module):
    """Thalamic Multiplicative Gating: entropy-based attention routing.

    Intercepts cross-attention output and scales it per-position by a gain
    derived from the Shannon entropy of the softmax attention distribution.

    Low entropy (focused) → gain near 1 → signal passes through.
    High entropy (diffuse/noisy) → gain near 0 → latent state protected.

    gain = sigmoid(w * H_norm + b)  where w, b are learnable scalars and
    H_norm ∈ [0, 1] is mean-over-heads normalized entropy.
    Default init: w=-4, b=3 → gain ≈ 0.95 at H=0, ≈ 0.27 at H=1.
    """

    def __init__(self, ema_decay: float = 0.99):
        super().__init__()
        self.gain_w = nn.Parameter(torch.tensor(-4.0))
        self.gain_b = nn.Parameter(torch.tensor(3.0))
        self._ema_decay = ema_decay
        self.register_buffer("_gain_ema", torch.tensor(1.0))

    def forward(self, attn_out: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_out:     [B, S, d_model]
            attn_weights: [B, n_heads, S, S] — softmax attention probs (causal-masked)
        Returns:
            gated: [B, S, d_model]
        """
        eps = 1e-8
        # Shannon entropy per head per query position
        H = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # [B, n_heads, S]
        # Normalize by log(S) → [0, 1] (S = max key len, causal upper bound)
        S = attn_weights.shape[-1]
        H_norm = H / (math.log(S) + eps)
        H_mean = H_norm.mean(dim=1)  # [B, S]
        gain = torch.sigmoid(self.gain_w * H_mean + self.gain_b)  # [B, S]
        if self.training:
            with torch.no_grad():
                self._gain_ema.mul_(self._ema_decay).add_(
                    gain.detach().mean() * (1.0 - self._ema_decay)
                )
        return attn_out * gain.unsqueeze(-1)


class MatrixResidualStream(nn.Module):
    """Matrix-valued residual stream with manifold-constrained mixing.

    Maintains n parallel streams of dimension d_latent. Each thought layer
    receives the mixed streams, processes them, and returns updated streams
    via post-mixing with residual gating.

    Args:
        d_latent: Width of each stream (matches the model's latent dimension).
        n_streams: Number of parallel streams. More streams = more capacity
            for tracking different aspects of the thought process, but also
            more memory. 4-8 is a good range for 200M-scale models.
        d_model: Text embedding dimension (for query projection output).
        gating: Gating parameterization:
            "diagonal" — each stream has a scalar gate (cheapest, works well)
            "sigmoid"  — sigmoid-gated diagonal (slightly more expressive)
    """

    def __init__(
        self,
        d_latent: int,
        n_streams: int = 4,
        d_model: int = 512,
        gating: str = "diagonal",
        use_schema_routing: bool = False,
        schema_routing_ema_decay: float = 0.99,
        schema_routing_temperature: float = 0.1,
        schema_routing_novelty_threshold: float = 0.1,
        use_multi_rate: bool = False,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_streams = n_streams
        self.d_model = d_model
        self.gating = gating

        # ── Pre-mixing: mix streams before the layer ────────────────────
        # Input-dependent mixing weights: a small MLP that takes the
        # collapsed stream vector and produces mixing logits.
        self.pre_mix_proj = nn.Linear(d_latent, n_streams * n_streams)
        nn.init.zeros_(self.pre_mix_proj.weight)
        nn.init.zeros_(self.pre_mix_proj.bias)
        # Zero init → initial pre-mixing is identity (softmax of zeros
        # gives uniform weights, but with the residual connection the
        # stream stays at its input value).

        # ── Post-mixing: mix streams after the layer ────────────────────
        self.post_mix_proj = nn.Linear(d_latent, n_streams * n_streams)
        nn.init.zeros_(self.post_mix_proj.weight)
        nn.init.zeros_(self.post_mix_proj.bias)

        # ── Residual gating: per-stream scalar gates ────────────────────
        # H_res: diagonal gating on the residual path.
        # Initialized at 0 → sigmoid(0) = 0.5 so residual and new
        # content contribute equally at init.
        self.res_gate = nn.Parameter(torch.zeros(n_streams))

        # ── Query collapse: project n streams → single query vector ─────
        # Replaces SynchronizationComputer. Collapses the multi-stream
        # state into a single query vector for cross-attention.
        # Attention over streams: query-dependent weighting of which
        # stream(s) contribute to the attention query.
        self.stream_query_weights = nn.Linear(d_latent, n_streams)
        self.query_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Stream initialization ───────────────────────────────────────
        # Learnable per-stream initial state (broadcast to batch/seq).
        self.stream_init = nn.Parameter(
            torch.zeros(n_streams, d_latent)
        )

        # ── Collapse: multi-stream → single latent for synapse/output ──
        # Attention-weighted collapse of streams into a single vector.
        self.collapse_weights = nn.Linear(d_latent, n_streams)
        self.collapse_norm = nn.LayerNorm(d_latent)

        # ── Structural plasticity: binary activity mask ──────────────────
        # True = stream active, False = dormant (zeroed out each step).
        # Stored as a buffer so it's included in state_dict and restored
        # on checkpoint load. Default: all streams active.
        # Grows/prunes are driven externally by StructuralPlasticityController.
        self.register_buffer("active_mask", torch.ones(n_streams, dtype=torch.bool))

        # ── Multi-rate update cadence (optional) ─────────────────────────
        # Cross-frequency-coupling analogue: streams refresh at different
        # timescales. Stream i is updated only on ticks where t % period_i == 0;
        # otherwise it holds its value (a slow stream). Periods follow a
        # power-of-2 schedule [1, 1, 2, 4, 8, ...] so the first two streams
        # track fast dynamics while later streams integrate over longer windows.
        # Stored as a buffer (checkpoint-safe). Only meaningful when enabled.
        self.use_multi_rate = use_multi_rate
        if use_multi_rate:
            periods = torch.tensor(
                [1 << max(0, i - 1) for i in range(n_streams)], dtype=torch.long
            )
            self.register_buffer("stream_update_period", periods)

        # ── Dynamic Schema Router (optional) ───────────────────────────
        self.schema_router = (
            SchemaRouter(
                n_streams=n_streams,
                d_latent=d_latent,
                ema_decay=schema_routing_ema_decay,
                temperature=schema_routing_temperature,
                novelty_threshold=schema_routing_novelty_threshold,
            ) if use_schema_routing else None
        )
        self._novelty_threshold = schema_routing_novelty_threshold

    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize the stream state for a new forward pass.

        Returns:
            state: [B, S, n_streams, d_latent]
        """
        # Broadcast learnable init to [B, S, n, D]. Respect the requested
        # device: under CPU-offload, stream_init lives on CPU but the thought
        # loop runs on GPU — the .to(device) here is a differentiable copy, so
        # gradients still flow back to the (CPU-resident) stream_init param.
        state = self.stream_init.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1, -1
        ).clone().to(device=device, dtype=dtype)
        # Dormant streams start zeroed (CUDA-graph safe multiply)
        state = state * self.active_mask.to(device=device, dtype=dtype).view(1, 1, -1, 1)
        return state

    def mix_pre(
        self,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-layer mixing of streams.

        Takes the current multi-stream state and mixes across streams
        using input-dependent weights. This allows different thought
        layers to "see" different combinations of the stream content.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            mixed: [B, S, n_streams, d_latent]
        """
        B, S, N, D = stream.shape

        # Compute mixing weights from the mean stream state
        mean_state = stream.mean(dim=2)  # [B, S, D]
        mix_logits = self.pre_mix_proj(mean_state)  # [B, S, N*N]
        mix_logits = mix_logits.view(B, S, N, N)  # [B, S, N_out, N_in]

        # Softmax over input streams (convex combination ↔ Birkhoff)
        mix_weights = F.softmax(mix_logits, dim=-1)  # [B, S, N, N]

        # Apply mixing: each output stream is a weighted sum of input streams
        # [B, S, N_out, N_in] @ [B, S, N_in, D] → [B, S, N_out, D]
        mixed = torch.matmul(mix_weights, stream)

        return mixed

    def mix_post(
        self,
        stream: torch.Tensor,
        layer_output: torch.Tensor,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Post-layer mixing with residual gating.

        After the thought layer processes the collapsed stream into a
        single-vector output, this method updates the multi-stream state
        by combining the layer output with the existing streams via
        learned gating.

        Args:
            stream:          [B, S, n_streams, d_latent] — pre-layer stream state
            layer_output:    [B, S, d_latent]            — output from the thought layer
            routing_weights: [B, S, n_streams] or None   — schema-routing weights.
                If provided, each stream receives a proportional share of
                layer_output rather than the learned post_mix_proj mixing.

        Returns:
            updated: [B, S, n_streams, d_latent]
        """
        B, S, N, D = stream.shape

        # Expand layer output to all streams
        # [B, S, D] → [B, S, 1, D] → broadcast to [B, S, N, D]
        expanded = layer_output.unsqueeze(2).expand_as(stream)

        if routing_weights is not None:
            # Schema routing: stream i gets routing_weights[..., i] fraction of
            # the layer output.  High-match stream → nearly full update;
            # low-match streams → protected from overwriting.
            # routing_weights: [B, S, N] → [B, S, N, 1]
            new_content = routing_weights.unsqueeze(-1) * expanded
        else:
            # Standard learned post-mixing
            mean_state = stream.mean(dim=2)              # [B, S, D]
            mix_logits = self.post_mix_proj(mean_state)  # [B, S, N*N]
            mix_logits = mix_logits.view(B, S, N, N)
            mix_weights = F.softmax(mix_logits, dim=-1)
            new_content = torch.matmul(mix_weights, expanded)  # [B, S, N, D]

        # Residual gating: blend old stream with new content
        # gate ∈ (0, 1) via sigmoid; 0.5 at init (both contribute equally)
        gate = torch.sigmoid(self.res_gate)  # [N]
        gate = gate.view(1, 1, N, 1)  # broadcast shape

        updated = gate * new_content + (1 - gate) * stream

        # Zero dormant streams (CUDA-graph safe: mask multiply, no control flow)
        updated = updated * self.active_mask.to(dtype=updated.dtype).view(1, 1, -1, 1)

        return updated

    def apply_update_cadence(
        self, t: int, old: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor:
        """Multi-rate freeze: keep `old` for slow streams, `new` for fast ones.

        A stream with period p is refreshed only when t % p == 0; otherwise it
        retains its previous value, giving the stream a slower effective
        timescale. Implemented as a CUDA-graph-safe mask multiply (no control
        flow). No-op unless multi-rate is enabled.

        Args:
            t:   current thought-tick index.
            old: [B, S, n_streams, d_latent] — stream state entering this tick.
            new: [B, S, n_streams, d_latent] — freshly mixed stream state.
        """
        if not self.use_multi_rate:
            return new
        update = (t % self.stream_update_period == 0)            # [n_streams] bool
        mask = update.view(1, 1, -1, 1).to(device=new.device, dtype=new.dtype)
        return mask * new + (1.0 - mask) * old

    def maybe_activate_novel_stream(self, mean_max_similarity: float) -> int | None:
        """Activate a dormant stream if mean_max_similarity is below the novelty threshold.

        Called during training when the schema router reports that the current
        batch does not match any existing schema.  Wakes the first dormant
        stream slot so it can specialize on the novel concept.

        Args:
            mean_max_similarity: Scalar — mean over (B, S) of max cosine-sim.
        Returns:
            Index of the newly activated stream, or None if nothing was done.
        """
        if mean_max_similarity >= self._novelty_threshold:
            return None
        dormant = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(dormant) == 0:
            return None
        idx = int(dormant[0].item())
        self.activate_stream(idx)
        return idx

    # ── Structural plasticity interface ─────────────────────────────────

    def apply_mask(self, stream: torch.Tensor) -> torch.Tensor:
        """Zero out inactive streams in a stream state tensor.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            Masked stream with inactive slots zeroed.
        """
        return stream * self.active_mask.to(dtype=stream.dtype).view(1, 1, -1, 1)

    def activate_stream(self, idx: int) -> None:
        """Re-activate a dormant stream slot with near-zero initialization.

        Re-zeroes the slot's learnable init state and resets its gate to
        sigmoid(-3) ≈ 0.05, so the new stream integrates gradually without
        disrupting the current computation.
        """
        self.active_mask[idx] = True
        with torch.no_grad():
            self.stream_init.data[idx].zero_()
            self.res_gate.data[idx] = -3.0   # very low gate → grows organically

    def deactivate_stream(self, idx: int) -> None:
        """Deactivate a stream slot (mask it out).

        Content will be zeroed on the next mix_post call.
        """
        self.active_mask[idx] = False

    def to_query(
        self,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse multi-stream state into attention query.

        Replaces SynchronizationComputer.compute(). Uses attention-weighted
        collapse over streams to produce a single query vector per position.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            query: [B, S, d_model]
        """
        B, S, N, D = stream.shape

        # Compute stream attention weights
        mean_state = stream.mean(dim=2)  # [B, S, D]
        weights = self.stream_query_weights(mean_state)  # [B, S, N]
        weights = F.softmax(weights, dim=-1)  # [B, S, N]

        # Weighted sum over streams
        collapsed = torch.einsum('bsn,bsnd->bsd', weights, stream)  # [B, S, D]

        # Project to query space
        query = self.query_proj(collapsed)  # [B, S, d_model]
        return query

    def collapse(
        self,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse multi-stream state to single latent vector.

        Used for the synapse model input and output head. Attention-weighted
        combination of streams, normalized.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            collapsed: [B, S, d_latent]
        """
        B, S, N, D = stream.shape

        mean_state = stream.mean(dim=2)
        weights = self.collapse_weights(mean_state)  # [B, S, N]
        weights = F.softmax(weights, dim=-1)

        collapsed = torch.einsum('bsn,bsnd->bsd', weights, stream)
        return self.collapse_norm(collapsed)

    def extra_repr(self) -> str:
        return (
            f"d_latent={self.d_latent}, n_streams={self.n_streams}, "
            f"d_model={self.d_model}, gating={self.gating}"
        )

# ═════════════════════════════════════════════════════════════════════════
# feec_integrator.py
# ═════════════════════════════════════════════════════════════════════════

class FEECIntegrator(nn.Module):
    """Mixed finite element integrator for the CTM thought loop.

    Maintains a dual state (position z, velocity J) and evolves them
    via a symplectic-like step that preserves discrete energy.

    Args:
        d_latent: Dimension of the latent state.
        n_layers: Number of thought layers (one dt per layer).
        dt_init: Initial step size (small for stability at init).
        damping_init: Initial damping coefficient γ.
            Higher damping → more energy dissipation → faster convergence
            but less expressive dynamics. The model learns to balance.
        clamp_dt: Upper bound on learnable dt to prevent instability.
    """

    def __init__(
        self,
        d_latent: int,
        n_layers: int,
        dt_init: float = 0.1,
        damping_init: float = 0.1,
        clamp_dt: float = 1.0,
        state_clip: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_layers = n_layers
        self.clamp_dt = clamp_dt
        # Trust-region ceiling on the per-position L2 norm of the relaxation
        # state. Absolute value = state_clip · √d_latent (0 → disabled). See
        # CTMConfig.feec_state_clip: bounds transient amplification so the
        # iterated relaxation can't diverge to NaN even when ρ momentarily > 1.
        self.max_state_norm = (
            float(state_clip) * (d_latent ** 0.5) if state_clip and state_clip > 0 else 0.0
        )

        # Per-layer learnable step sizes. Initialized small so the
        # integrator starts near the identity (z_{t+1} ≈ z_t).
        # Using log-parameterization: dt = softplus(raw_dt) to keep dt > 0.
        self.raw_dt = nn.Parameter(
            torch.full((n_layers,), self._inv_softplus(dt_init))
        )

        # Global learnable damping coefficient γ.
        # Parameterized via sigmoid: γ = sigmoid(raw_damping) ∈ (0, 1).
        # This ensures γ stays in a reasonable range — too large and the
        # velocity collapses to zero (no dynamics), too small and energy
        # isn't dissipated (potential oscillation).
        self.raw_damping = nn.Parameter(
            torch.tensor(self._inv_sigmoid(damping_init))
        )

        # Force scaling: learnable per-layer scaling of the force field
        # to match the magnitude of the integrator's natural dynamics.
        # Initialized to 1.0 (identity scaling).
        self.force_scale = nn.Parameter(torch.ones(n_layers))

    @staticmethod
    def _inv_softplus(y: float) -> float:
        """Inverse of softplus: x such that softplus(x) = y."""
        import math
        if y > 20:
            return y
        return math.log(math.exp(y) - 1)

    @staticmethod
    def _inv_sigmoid(y: float) -> float:
        """Inverse of sigmoid: x such that sigmoid(x) = y."""
        import math
        return math.log(y / (1 - y))

    def get_dt(self, layer_idx: int) -> torch.Tensor:
        """Get the step size for a given layer, clamped for stability."""
        dt = F.softplus(self.raw_dt[layer_idx])
        return dt.clamp(max=self.clamp_dt)

    @property
    def damping(self) -> torch.Tensor:
        """Current damping coefficient γ ∈ (0, 1)."""
        return torch.sigmoid(self.raw_damping)

    def step(
        self,
        z: torch.Tensor,
        velocity: torch.Tensor,
        force: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one symplectic integration step.

        Symplectic Euler (velocity-first variant):
            J_{t+1} = (1 - γ·Δt) · J_t + Δt · s_l · F
            u_{t+1} = u_t + Δt · J_{t+1}

        The velocity-first variant has better energy preservation than
        position-first for dissipative systems (our case with γ > 0).

        Args:
            z: [B, S, d_latent] — current position (latent state).
            velocity: [B, S, d_latent] — current velocity.
            force: [B, S, d_latent] — force field from thought layer.
            layer_idx: which layer this step corresponds to.

        Returns:
            z_new: [B, S, d_latent] — updated position.
            velocity_new: [B, S, d_latent] — updated velocity.
        """
        dt = self.get_dt(layer_idx)
        gamma = self.damping
        scale = self.force_scale[layer_idx]

        # Velocity update (with damping)
        velocity_new = (1 - gamma * dt) * velocity + dt * scale * force

        # Position update (using NEW velocity — symplectic)
        z_new = z + dt * velocity_new

        # Trust-region clip: cap the per-position L2 norm of state and velocity.
        # Differentiable (scale ≤ 1, shrink-only) so the learning-pass step is
        # unaffected; only engages when growth is pathological. This is what
        # makes the iterated relaxation robust to a transiently supercritical ρ.
        if self.max_state_norm > 0.0:
            z_new = self._clip_norm(z_new, self.max_state_norm)
            velocity_new = self._clip_norm(velocity_new, self.max_state_norm)

        return z_new, velocity_new

    @staticmethod
    def _clip_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Scale rows whose per-position L2 norm exceeds ``max_norm`` back down
        to it; leave smaller rows untouched. Shrink-only and differentiable."""
        norm = x.norm(dim=-1, keepdim=True)
        scale = (max_norm / norm.clamp_min(1e-6)).clamp(max=1.0)
        return x * scale

    def energy(
        self,
        z: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discrete energy E = ½|J|² + ½|z|².

        Used for monitoring/logging, NOT for the loss function directly.
        A well-behaved thought loop should show non-increasing energy
        after the initial transient.

        Args:
            z: [B, S, d_latent]
            velocity: [B, S, d_latent]

        Returns:
            Scalar energy, averaged over batch and sequence.
        """
        kinetic = 0.5 * (velocity ** 2).sum(dim=-1).mean()
        potential = 0.5 * (z ** 2).sum(dim=-1).mean()
        return kinetic + potential

    def energy_penalty(
        self,
        z: torch.Tensor,
        velocity: torch.Tensor,
        z_prev: torch.Tensor,
        velocity_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Penalty for energy growth between steps.

        Returns ReLU(E_new - E_old) — zero when energy decreases (good),
        positive when energy increases (penalized). Analogous to the
        mono_penalty in the temporal loss but for the dynamical system.

        Args:
            z, velocity: current state
            z_prev, velocity_prev: previous state

        Returns:
            Scalar penalty (≥ 0).
        """
        e_new = self.energy(z, velocity)
        e_old = self.energy(z_prev, velocity_prev)
        return F.relu(e_new - e_old)

    def extra_repr(self) -> str:
        dts = [self.get_dt(i).item() for i in range(self.n_layers)]
        return (
            f"n_layers={self.n_layers}, "
            f"damping={self.damping.item():.4f}, "
            f"dt=[{', '.join(f'{d:.3f}' for d in dts)}]"
        )

class AmortizedInferenceNet(nn.Module):
    """Single-pass prediction of the prospective configuration equilibrium state.

    Maps text representations directly to an approximate z*, bypassing the
    iterative inference loop.  After this warm-start, only a few relaxation
    ticks are needed to reach the true fixed point instead of up to 1000.

    Trained via an auxiliary MSE loss: MSE(z_hat, z*_detached), so the
    network learns to match wherever the inference loop converges.
    """

    def __init__(self, d_model: int, d_latent: int, hidden_dim: int = 0):
        super().__init__()
        hidden = hidden_dim or (2 * d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_emb: [B, S, d_model]
        Returns:
            z_hat:    [B, S, d_latent] predicted equilibrium latents
        """
        return self.net(text_emb)


class EFEThoughtController(nn.Module):
    """R1 — Expected-Free-Energy halting policy over the thought loop.

    A PonderNet-style controller that, at each thought tick t, maps a small
    set of EFE features to a halting probability λ_t ∈ (0, 1):

        ambiguity_t = 1 − mean certainty_t      (predictive entropy proxy)
        epistemic_t = normalized ‖z_t − z_{t-1}‖ (latent information gain)
        progress_t  = t / (T − 1)                (loop position)

    The per-tick halting distribution is the PonderNet construction
        p_t = λ_t · Π_{t'<t} (1 − λ_{t'}),
    with the final tick absorbing the remaining mass so Σ_t p_t = 1.

    The controller is consumed by ``CTMTransformer.forward``:
      - the reconstruction loss Σ_t p_t · CE_t replaces the uniform per-tick
        mean, rewarding confident early emission;
      - a KL(p ‖ Geometric(halt_prior)) term keeps the loop from collapsing to
        "halt immediately";
      - an EFE-shaping term Σ_t p_t · (ambiguity_t − epistemic_t) pulls halting
        mass toward low-EFE ticks (confident + info-gain exhausted).
    At eval / generate, the loop early-exits once the cumulative halt
    probability crosses a threshold — the wall-clock payoff.

    Operates on scalar (sequence-mean) features so the loss it shapes is built
    from the per-tick scalar CE that ``_thought_step`` already returns — which
    keeps it correct under gradient checkpointing (side-effect tensors stored
    on ``self`` would not be).
    """

    N_FEATURES = 3

    def __init__(self, hidden_dim: int, halt_prior: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Bias the initial halting logit so λ ≈ halt_prior at the start, before
        # the controller has learned anything (matches the geometric prior).
        p = float(min(max(halt_prior, 1e-4), 1 - 1e-4))
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            self.mlp[-1].bias.fill_(math.log(p / (1.0 - p)))

    def halt_logit(
        self,
        ambiguity: torch.Tensor,
        epistemic: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        """Map scalar EFE features to a scalar halting logit for one tick."""
        feats = torch.stack([ambiguity, epistemic, progress]).to(
            self.mlp[0].weight.dtype
        )
        return self.mlp(feats).squeeze(-1)


class _GradReverse(torch.autograd.Function):
    """Gradient-reversal layer (Ganin & Lempitsky 2015).

    Identity on the forward pass; multiplies the gradient by −λ on the
    backward pass. Lets an adversarial objective be optimised with a single
    backward / single optimizer step (distributed-safe).
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class REMDreamer(nn.Module):
    """R10 — REM-style adversarial dreaming over pooled latents.

    A latent-space GAN. The generator maps noise → a "dream" latent; the
    discriminator scores latents as real (the backbone's own pooled latent from
    data) vs dreamed. A gradient-reversal layer on the generator path makes the
    whole adversarial objective a single loss, so generator, discriminator, and
    the backbone's real-latent encoder all train in ONE backward / ONE optimizer
    step — keeping DDP / flat-allreduce replicas in sync (a per-rank internal
    GAN optimizer would diverge across ranks). The loss is

        L = BCE(D(real), 1) + BCE(D(GRL_λ(G(noise))), 0).

    D minimises it (real→1, dream→0); the backbone's real latents are pulled to
    the "real" side; GRL flips the generator's gradient so G learns to produce
    latents D rates as real. This is adversarial dreaming (Deperrois et al. 2022)
    adapted to be distributed-safe and to reuse the main optimizer.

    ──────────────────────────────────────────────────────────────────────────
    KNOWN ISSUE — DEFAULT-DISABLED (2026-05-23). `use_rem_dreaming` defaults OFF
    and should stay off; it causes a sudden, irreversible CERTAINTY COLLAPSE.

    Symptom: the next-token distribution flattens toward uniform — output
    certainty (1 − H(softmax)/log V) drops from a healthy ~0.3 to a pinned
    ~0.02 within one logging interval and never recovers — while cross-entropy,
    perplexity and the PC loss keep improving normally. The collapse is a sharp
    cliff triggered by a sleep micro-cycle (observed reproducibly at the 3rd
    sleep, ~step 1200), accompanied by a gradient spike.

    Root cause: this loss includes BCE(D(real), 1) on the backbone's OWN pooled
    latent, and via the GRL the adversarial objective shapes that same latent —
    the one that feeds the readout. The pressure drives real latents toward the
    noise-like "dream" distribution, so the output becomes input-insensitive and
    high-entropy. The effect accumulates across sleeps until it tips. This is
    INHERENT to the design (REM deliberately trains the output-determining
    latent), not a tuning bug — lowering rem_loss_weight only delays it.

    Confirmed by ablation (2026-05-23): disabling REM removes the cliff entirely
    (certainty develops healthily to ~0.3–0.45); re-enabling it reproduces the
    collapse. Ruled OUT as causes via isolation + a stable-rank probe: the
    Hebbian sleep carry (cliff persists with carry≈0) and latent rank collapse
    (srank stays ~1.5 throughout, healthy and collapsed alike).

    If ever revisited: the only way to keep REM AND sharp outputs is to make D
    score a DETACHED latent / a separate head so the adversarial gradient never
    reaches the readout-determining latent — which largely defeats its purpose.
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, d_latent: int, noise_dim: int = 0, hidden_dim: int = 0):
        super().__init__()
        self.noise_dim = noise_dim or d_latent
        hidden = hidden_dim or max(64, d_latent // 4)
        self.generator = nn.Sequential(
            nn.Linear(self.noise_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_latent),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(d_latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def loss(
        self, real_latents: torch.Tensor, grl_lambda: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single adversarial REM loss + discriminator accuracy (for logging).

        Args:
            real_latents: [n, d_latent] grad-carrying pooled latents from the
                backbone (the "real" samples).
        """
        n = real_latents.shape[0]
        noise = torch.randn(
            n, self.noise_dim, device=real_latents.device, dtype=real_latents.dtype
        )
        dream = grad_reverse(self.generator(noise), grl_lambda)
        d_real = self.discriminator(real_latents).float().squeeze(-1)
        d_dream = self.discriminator(dream).float().squeeze(-1)
        loss = (
            F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            + F.binary_cross_entropy_with_logits(d_dream, torch.zeros_like(d_dream))
        )
        with torch.no_grad():
            acc = 0.5 * ((d_real > 0).float().mean() + (d_dream < 0).float().mean())
        return loss, acc.detach()


# ═════════════════════════════════════════════════════════════════════════
# thought_layer.py
# ═════════════════════════════════════════════════════════════════════════

class UNetSynapse(nn.Module):
    """
    1D U-Net applied across the latent feature dimension to allow 
    complex spatial information sharing between neurons.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        # Input shape: [BS, 1, in_features]
        self.down1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.GELU())
        self.pool1 = nn.MaxPool1d(2)
        
        self.down2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.GELU())
        self.pool2 = nn.MaxPool1d(2)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv_up1 = nn.Sequential(nn.Conv1d(32 + 16, 16, kernel_size=3, padding=1), nn.GELU())
        
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv_up2 = nn.Sequential(nn.Conv1d(16 + 1, 1, kernel_size=3, padding=1), nn.GELU())
        
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BS, in_features]
        x_in = x.unsqueeze(1) # [BS, 1, in_features]
        
        d1 = self.down1(x_in)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        u1 = self.up1(p2)
        if u1.size(2) != d1.size(2):
            u1 = F.interpolate(u1, size=d1.size(2))
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up1(u1)
        if u2.size(2) != x_in.size(2):
            u2 = F.interpolate(u2, size=x_in.size(2))
        u2 = torch.cat([u2, x_in], dim=1)
        u2 = self.conv_up2(u2)
        
        out = u2.squeeze(1) # [BS, in_features]
        return self.proj(out)


class DendriticSynapse(nn.Module):
    """
    Biologically-inspired dendritic nonlinear integration layer.

    Biological pyramidal neurons integrate inputs non-linearly across
    spatially-distinct dendritic compartments, each capable of generating
    an independent local spike. This yields a bilinear (quadratic) response
    rule that captures feature co-occurrences within a single layer rather
    than requiring multiple stacked linear ops.

    Architecture
    ------------
    somatic (linear) path:
        x → W_soma → [d_out]

    dendritic (bilinear) path:
        - Divide output into n_branches compartments of size d_branch = d_out // n_branches
        - u = W_u(x), v = W_v(x)  →  [BS, n_branches, d_branch]  (proximal / distal projections)
        - quadratic term: u * v                                     (compartment-local bilinear product)
        - per-branch gate: g = sigmoid(W_gate(x)) → [BS, n_branches]  (dendritic spike threshold)
        - gated output: Σ_i g_i · (u_i * v_i)  →  [BS, d_out]

    output = Dropout(soma + dendrite)

    The W_u / W_v dendritic weights are small-init so the module starts
    near the somatic linear path and learns bilinear structure gradually.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_branches: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_out % n_branches != 0:
            raise ValueError(
                f"DendriticSynapse: d_out ({d_out}) must be divisible by "
                f"n_branches ({n_branches})"
            )
        self.n_branches = n_branches
        self.d_branch = d_out // n_branches

        # Somatic (linear) path — initialised with Kaiming for stable gradients
        self.soma = nn.Linear(d_in, d_out)

        # Dendritic bilinear path: two independent projections
        # Small init → bilinear term ≈ 0 at start, learns gradually
        self.W_u = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)

        # Per-branch spike gate: zero init → sigmoid(0) = 0.5 (balanced start)
        self.branch_gate = nn.Linear(d_in, n_branches)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.soma.weight, a=math.sqrt(5))
        for lin in (self.W_u, self.W_v):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BS, d_in]
        BS = x.shape[0]

        # Somatic path
        soma = self.soma(x)  # [BS, d_out]

        # Dendritic bilinear path
        u = self.W_u(x).view(BS, self.n_branches, self.d_branch)  # [BS, n, d_b]
        v = self.W_v(x).view(BS, self.n_branches, self.d_branch)  # [BS, n, d_b]
        quadratic = u * v                                          # [BS, n, d_b]

        # Per-branch dendritic spike gate
        gates = torch.sigmoid(self.branch_gate(x)).unsqueeze(-1)  # [BS, n, 1]
        dendrite = (quadratic * gates).view(BS, -1)               # [BS, d_out]

        return self.dropout(soma + dendrite)


class ThoughtLayer(nn.Module):
    """
    A single thought processing block with per-position latent states
    and causally-masked cross-attention.

    Args:
        d_latent: Internal neuron count (state dimension).
        d_model: Text embedding dimension (cross-attention KV dimension).
        n_heads: Number of attention heads.
        nlm_hidden_dim: Hidden dimension per neuron MLP.
        history_len: FIFO buffer depth.
        nlm_groups: Number of neuron groups for NLMs.
        sync_method: Synchronization computation method.
        sync_rank: Rank for low-rank sync.
        dropout: Dropout rate.

        use_matrix_streams: If True, use MatrixResidualStream instead of NLM+Sync.
        n_streams: Number of parallel streams (when use_matrix_streams=True).
        stream_gating: Gating parameterization for matrix streams.
        use_dssa: If True, use Dual-Space Sparse Attention.
        dssa_n_partitions: SSE state partitions.
        dssa_top_k: Active SSE partitions.
        dssa_block_size: MoBA block size.
        dssa_top_k_blocks: MoBA blocks per query.
        use_triton_attention: Use Triton-accelerated attention.
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        n_heads: int,
        nlm_hidden_dim: int,
        history_len: int,
        nlm_groups: int = 1,
        sync_method: str = "diag_summary",
        sync_rank: int = 32,
        dropout: float = 0.1,

        # v2 features
        use_matrix_streams: bool = False,
        n_streams: int = 4,
        stream_gating: str = "diagonal",
        use_dssa: bool = False,
        dssa_n_partitions: int = 32,
        dssa_top_k: int = 8,
        dssa_block_size: int = 64,
        dssa_top_k_blocks: int = 4,
        use_triton_attention: bool = False,
        synapse_type: str = "mlp",
        dendritic_n_branches: int = 4,
        # ── Biological extensions (opt-in) ──────────────────────────
        use_hebbian: bool = False,
        hebbian_bottleneck_dim: int = 64,
        hebbian_decay_init: float = 0.9,
        hebbian_lr_init: float = 0.1,
        hebbian_gate_init: float = -3.0,
        hebbian_force_gate: float | None = None,
        hebbian_update_rule: str = "outer_product",
        use_btsp: bool = False,
        btsp_lr_init: float = 0.05,
        btsp_kernel_decay_init: float = 0.9,
        btsp_salience_threshold: float = 0.0,
        btsp_weight_dependent: bool = False,
        btsp_w_max: float = 1.0,
        btsp_consolidate_prob: float = 0.3,
        btsp_delay_shape: float = 0.0,
        use_ip: bool = False,
        ip_lr: float = 0.01,
        ip_target: float = 0.1,
        ip_ema_decay: float = 0.99,
        use_thalamic_gating: bool = False,
        use_schema_routing: bool = False,
        schema_routing_ema_decay: float = 0.99,
        schema_routing_temperature: float = 0.1,
        schema_routing_novelty_threshold: float = 0.1,
        schema_routing_btsp_scale: float = 2.0,
        hebbian_n_compartments: int = 1,
        use_stc: bool = False,
        stc_tag_decay: float = 0.5,
        stc_threshold_tag: float = 0.05,
        use_critical_init: bool = False,
        critical_init_sym_frac: float = 0.6,
        critical_init_spectral_radius: float = 0.999,
        use_multi_rate_streams: bool = False,
        use_burstprop: bool = False,
        burstprop_tau: float = 100.0,
        burstprop_scale_init: float = 1.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.synapse_type = synapse_type
        self.use_critical_init = bool(use_critical_init)
        self.critical_init_sym_frac = critical_init_sym_frac
        self.critical_init_spectral_radius = critical_init_spectral_radius
        self.n_heads = n_heads
        self.use_matrix_streams = use_matrix_streams
        self.use_dssa = use_dssa
        self.use_triton_attention = use_triton_attention
        self.schema_btsp_scale = schema_routing_btsp_scale if use_schema_routing else 0.0

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads

        # ── v2: Matrix-Valued Residual Stream (replaces Sync+NLM) ────────
        if use_matrix_streams:
            self.stream = MatrixResidualStream(
                d_latent=d_latent,
                n_streams=n_streams,
                d_model=d_model,
                gating=stream_gating,
                use_schema_routing=use_schema_routing,
                schema_routing_ema_decay=schema_routing_ema_decay,
                schema_routing_temperature=schema_routing_temperature,
                schema_routing_novelty_threshold=schema_routing_novelty_threshold,
                use_multi_rate=use_multi_rate_streams,
            )
            # No sync_computer, nlm, or memory needed
            self.sync_computer = None
            self.nlm = None
            self.memory = None
        else:
            self.stream = None
            # ── Synchronization Computer (v1 path) ──────────────────────
            self.sync_computer = SynchronizationComputer(
                d_latent=d_latent,
                history_len=history_len,
                method=sync_method,
                rank=sync_rank,
            )
            sync_dim = self.sync_computer.output_dim

            # ── Query Generation from Synchronization State (v1) ────────
            self.query_proj = nn.Sequential(
                nn.Linear(sync_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

            # ── Neuron-Level Models (v1 path) ───────────────────────────
            self.nlm = NeuronLevelModels(
                d_latent=d_latent,
                history_len=history_len,
                nlm_hidden_dim=nlm_hidden_dim,
                nlm_groups=nlm_groups,
                dropout=dropout,
                use_ip=use_ip,
                ip_lr=ip_lr,
                ip_target=ip_target,
                ip_ema_decay=ip_ema_decay,
            )

            # ── Memory Buffers (v1 path) ────────────────────────────────
            self.memory = TemporalMemory(d_latent=d_latent, history_len=history_len)

        # ── v2: DSSA Cross-Attention (replaces manual Q/K/V) ────────────
        if use_dssa:
            self.dssa = DualSpaceSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_partitions=dssa_n_partitions,
                sse_top_k=dssa_top_k,
                block_size=dssa_block_size,
                moba_top_k_blocks=dssa_top_k_blocks,
                dropout=dropout,
            )
            # Don't need the manual attention components
            self.q_norm = None
            self.k_proj = None
            self.v_proj = None
            self.attn_out_proj = None
            self.attn_dropout = None
        else:
            self.dssa = None
            # ── Standard Cross-Attention ────────────────────────────────
            self.q_norm = nn.LayerNorm(d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.attn_out_proj = nn.Linear(d_model, d_model)
            self.attn_dropout = nn.Dropout(dropout)

        # ── Synapse Model ───────────────────────────────────────────────
        self.synapse_norm = nn.LayerNorm(d_model + d_latent)
        if synapse_type == "unet":
            self.synapse = UNetSynapse(d_model + d_latent, d_latent, dropout=dropout)
        elif synapse_type == "dendritic":
            self.synapse = DendriticSynapse(
                d_model + d_latent, d_latent,
                n_branches=dendritic_n_branches,
                dropout=dropout,
            )
        else:  # "mlp"
            self.synapse = nn.Sequential(
                nn.Linear(d_model + d_latent, d_latent * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_latent * 2, d_latent),
            )
        self.synapse_gate = nn.Linear(d_model + d_latent, d_latent)

        # ── Post-NLM Layer Norm ─────────────────────────────────────────
        self.post_norm = nn.LayerNorm(d_latent)

        # ── Hebbian Fast-Weight Synapse (optional) ──────────────────────
        # Augments the synapse output with a content-addressable readout
        # from a per-(B, S) fast-weight matrix updated online during the
        # thought loop. See biological.HebbianSynapse for details.
        self.use_hebbian = use_hebbian
        if use_hebbian:
            self.hebbian = HebbianSynapse(
                d_latent=d_latent,
                d_model=d_model,
                bottleneck_dim=hebbian_bottleneck_dim,
                decay_init=hebbian_decay_init,
                lr_init=hebbian_lr_init,
                gate_init=hebbian_gate_init,
                force_gate=hebbian_force_gate,
                update_rule=hebbian_update_rule,
                use_btsp=use_btsp,
                btsp_lr_init=btsp_lr_init,
                btsp_kernel_decay_init=btsp_kernel_decay_init,
                btsp_salience_threshold=btsp_salience_threshold,
                btsp_weight_dependent=btsp_weight_dependent,
                btsp_w_max=btsp_w_max,
                btsp_consolidate_prob=btsp_consolidate_prob,
                btsp_delay_shape=btsp_delay_shape,
                n_compartments=hebbian_n_compartments,
                use_stc=use_stc,
                stc_tag_decay=stc_tag_decay,
                stc_threshold_tag=stc_threshold_tag,
                use_critical_init=use_critical_init,
                critical_init_sym_frac=critical_init_sym_frac,
                critical_init_spectral_radius=critical_init_spectral_radius,
                use_burstprop=use_burstprop,
                burstprop_tau=burstprop_tau,
                burstprop_scale_init=burstprop_scale_init,
            )
        else:
            self.hebbian = None

        # ── Adaptive Intrinsic Plasticity (optional) ────────────────────
        # Applied to the synapse output (pre_activations) before post_norm
        # in both the v1 (NLM) and v2 (MatrixStream) forward paths.
        # Lives here rather than inside NeuronLevelModels so it activates
        # regardless of which path is selected.
        self.ip_layer = (
            IntrinsicPlasticityLayer(
                d=d_latent,
                ip_lr=ip_lr,
                ip_target=ip_target,
                ip_ema_decay=ip_ema_decay,
            )
            if use_ip else None
        )

        # ── Thalamic Multiplicative Gate (optional) ─────────────────────
        # Applied to cross-attention output before the synapse.
        # Suppresses updates when attention is diffuse (high entropy),
        # protecting the latent state from noisy or uncertain signals.
        # Inactive when use_dssa=True (DSSA doesn't expose weight matrices).
        self.thalamic_gate = ThalamicGate() if (use_thalamic_gating and not use_dssa) else None

    def _synapse_recurrent_weight(self) -> torch.Tensor:
        """The synapse projection whose z-columns form the recurrent matrix A.

        dendritic → soma; unet → final proj linear; mlp → first linear. The
        d_model columns are feed-forward (attention) input; columns [d_model:]
        read the previous latent z and constitute the recurrent dynamics.
        """
        if self.synapse_type == "dendritic":
            return self.synapse.soma.weight        # [d_latent, d_model + d_latent]
        elif self.synapse_type == "unet":
            return self.synapse.proj[1].weight     # [d_latent, d_model + d_latent]
        return self.synapse[0].weight              # mlp: [2·d_latent, d_model + d_latent]

    def _synapse_recurrent_slice(self) -> torch.Tensor:
        """View of the z-recurrence columns — the dynamics matrix A.

        Square ([d_latent, d_latent]) for dendritic/unet; non-square
        ([2·d_latent, d_latent]) for mlp.
        """
        return self._synapse_recurrent_weight()[:, self.d_model:]

    def _reset_special_inits(self):
        """Re-apply critical-symmetric init to the synapse z-recurrence slice.

        Invoked by CTMTransformer.__init__ *after* the global ``_init_weights``
        pass (which would otherwise overwrite it with normal_(std=0.02)).
        For the mlp synapse the z-slice is non-square, so only the spectral
        scaling applies (symmetry undefined).
        """
        if not self.use_critical_init:
            return
        critical_symmetric_init_(
            self._synapse_recurrent_slice(),
            self.critical_init_sym_frac,
            self.critical_init_spectral_radius,
        )

    @torch.no_grad()
    def renormalize_synapse_spectral_radius(self, target: float = 1.0) -> tuple[float, float]:
        """Project the synapse z-recurrence matrix back to spectral radius ≤ target.

        Synaptic-homeostasis (SHY) analogue: when training pushes the recurrent
        dynamics matrix supercritical, downscale it back toward criticality.
        Square slice → leading |eigenvalue|; non-square (mlp) → top singular
        value. Only ever shrinks (never amplifies). Returns (rho_before, rho_after).
        """
        W = self._synapse_recurrent_slice()
        A = W.detach().float()
        if A.shape[0] == A.shape[1]:
            rho = torch.linalg.eigvals(A).abs().max().real.item()
        else:
            rho = torch.linalg.svdvals(A).max().item()
        if rho > target and rho > 1e-12:
            W.mul_(target / rho)
            return rho, float(target)
        return rho, rho

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize memory buffers for a new sequence/batch."""
        if self.memory is not None:
            self.memory.reset(batch_size, device, dtype)

    def _standard_attention(
        self,
        queries: torch.Tensor,
        text_keys: torch.Tensor,
        text_values: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standard O(N²) cross-attention with causal masking.

        Args:
            queries: [B, S, d_model]
            text_keys: [B, S, d_model]
            text_values: [B, S, d_model]
            key_padding_mask: [B, S] or None
            return_weights: If True, also return softmax weights [B, n_heads, S, S]
                for entropy-based thalamic gating. On the Triton path weights are
                computed from QK^T only (no V matmul overhead).

        Returns:
            attn_out: [B, S, d_model]
            attn_weights: [B, n_heads, S, S] or None
        """
        B, S, D = queries.shape

        q = self.q_norm(queries)
        k = self.k_proj(text_keys)
        v = self.v_proj(text_values)

        # Check if we should use accelerated attention
        if self.use_triton_attention:
            weights_for_gate = None
            if return_weights:
                # Compute entropy weights via standard QKT (no V matmul)
                q_h = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
                k_h = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
                causal_mask = torch.triu(
                    torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
                )
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                weights_for_gate = F.softmax(scores, dim=-1)
            q_reshaped = q.view(B, S, self.n_heads, self.head_dim)
            k_reshaped = k.view(B, S, self.n_heads, self.head_dim)
            v_reshaped = v.view(B, S, self.n_heads, self.head_dim)
            attn_out = accelerated_causal_attention(
                q_reshaped, k_reshaped, v_reshaped,
                n_heads=self.n_heads,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                training=self.training,
                use_triton=True,
            )
            attn_out = self.attn_out_proj(attn_out)
            return attn_out, weights_for_gate

        # Standard path
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.attn_out_proj(attn_out)

        return attn_out, (attn_weights if return_weights else None)

    def forward(
        self,
        text_keys: torch.Tensor,
        text_values: torch.Tensor,
        prev_state: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,

        stream_state: torch.Tensor | None = None,
        hebbian_state: torch.Tensor | None = None,
        hebbian_lr_modulator: torch.Tensor | None = None,
        hebbian_top_down: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Execute one thought step with per-position states and causal masking.

        Args:
            text_keys:   [batch, seq_len, d_model] — text embeddings for K projection.
            text_values: [batch, seq_len, d_model] — text embeddings for V projection.
            prev_state:  [batch, seq_len, d_latent] — per-position neuron states z_{t-1}.
            key_padding_mask: [batch, seq_len] — True for padded positions.

            stream_state: [batch, seq_len, n_streams, d_latent] — matrix stream state (v2).
            hebbian_state: [batch, seq_len, m, n] — Hebbian fast-weight matrix, or None.
            hebbian_lr_modulator: [batch, seq_len] or None — per-position
                multiplier applied to Hebbian outer-product update. Supplied
                by the caller as (1 + alpha * uncertainty) from the
                previous tick's certainty. None → uniform lr.

        Returns:
            new_state:   [batch, seq_len, d_latent] — updated per-position states z_t.
            sync_repr:   [batch*seq_len, sync_dim] — sync repr (v1) or dummy (v2).
            new_stream_state: [batch, seq_len, n_streams, d_latent] or None.
            new_hebbian_state: [batch, seq_len, m, n] or None.
            hebbian_lr_eff: scalar tensor (mean effective lr post-modulation),
                or None when Hebbian path not active.
        """
        B, S, D = text_keys.shape
        BS = B * S

        # ── Step 1: Query Generation ────────────────────────────────────
        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            # v2 path: query from matrix stream
            mixed_stream = self.stream.mix_pre(stream_state)
            queries = self.stream.to_query(mixed_stream)  # [B, S, d_model]
            sync_repr = torch.zeros(BS, 1, device=text_keys.device, dtype=text_keys.dtype)
        else:
            # v1 path: query from synchronization
            prev_flat = prev_state.reshape(BS, -1)
            post_history = self.memory.get_post_history()
            sync_repr = self.sync_computer.compute(post_history)
            queries_flat = self.query_proj(sync_repr)
            queries = queries_flat.reshape(B, S, D)

        # ── Step 2: Cross-Attention ─────────────────────────────────────
        if self.use_dssa and self.dssa is not None:
            # v2 DSSA path — weights not exposed, thalamic gate disabled
            attn_out = self.dssa(queries, text_keys, causal=True)
        else:
            # Standard attention (possibly Triton-accelerated)
            attn_out, attn_weights_for_gate = self._standard_attention(
                queries, text_keys, text_values, key_padding_mask,
                return_weights=(self.thalamic_gate is not None),
            )
            if self.thalamic_gate is not None and attn_weights_for_gate is not None:
                attn_out = self.thalamic_gate(attn_out, attn_weights_for_gate)

        # Flatten attention output
        attn_flat = attn_out.reshape(BS, D)

        # ── Step 3: Synapse Model ───────────────────────────────────────
        routing_weights = None
        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            collapsed = self.stream.collapse(stream_state)  # [B, S, d_latent]
            prev_flat = collapsed.reshape(BS, -1)

            # ── Step 3.5: Schema routing ─────────────────────────────────
            # Compute cosine-similarity routing before Hebbian so the match
            # score can boost the current tick's lr_modulator.
            if self.stream.schema_router is not None:
                routing_weights, schema_max_sim = self.stream.schema_router(
                    collapsed.detach(), self.stream.active_mask,
                )
                # Store sideband for novelty check in _thought_step
                self.stream._last_max_similarity = (
                    schema_max_sim.detach().mean().item()
                )
                # High match → boost Hebbian/BTSP LR for rapid assimilation
                if self.schema_btsp_scale > 0:
                    boost = 1.0 + self.schema_btsp_scale * schema_max_sim
                    hebbian_lr_modulator = (
                        hebbian_lr_modulator * boost
                        if hebbian_lr_modulator is not None
                        else boost
                    )
        else:
            prev_flat = prev_state.reshape(BS, -1)

        synapse_input = torch.cat([attn_flat, prev_flat], dim=1)
        synapse_input = self.synapse_norm(synapse_input)

        gate = torch.sigmoid(self.synapse_gate(synapse_input))
        candidate = self.synapse(synapse_input)
        pre_activations = gate * candidate + (1 - gate) * prev_flat

        # ── Hebbian Fast-Weight Readout + Update ────────────────────────
        # Reads from M·a, adds (gated) to the synapse pre_activations,
        # then updates M with the outer product of the latent state and
        # the attention output. M persists across ticks within a forward
        # pass; the caller threads it via `hebbian_state`.
        new_hebbian_state = hebbian_state  # passthrough by default
        hebbian_lr_eff = None
        if self.use_hebbian and self.hebbian is not None and hebbian_state is not None:
            # Use post-synapse latents as the "z" of the outer product so
            # the Hebbian trace encodes the current refined state, not
            # the stale prev_state. This is the working-memory write rule:
            # bind the current attention key (a) to the current latent (z).
            pre_act_2d = pre_activations.reshape(B, S, -1)
            heb_readout, new_hebbian_state, hebbian_lr_eff = self.hebbian(
                pre_act_2d, attn_out, hebbian_state,
                lr_modulator=hebbian_lr_modulator,
                top_down_signal=hebbian_top_down,
            )
            pre_activations = pre_activations + heb_readout.reshape(BS, -1)

        # ── Step 4: Post-processing ─────────────────────────────────────
        new_stream_state = None

        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            # v2 path: update matrix streams (replaces NLM)
            # Apply IP before post_norm: shifts each neuron's working point
            # toward ip_target, correcting cross-sample systematic biases.
            if self.ip_layer is not None:
                pre_activations = self.ip_layer(pre_activations)
            post_activations = self.post_norm(pre_activations)
            new_state_flat = post_activations
            # Update stream state; pass routing_weights for schema-biased mixing
            layer_output = post_activations.reshape(B, S, -1)
            new_stream_state = self.stream.mix_post(
                stream_state, layer_output, routing_weights=routing_weights
            )
        else:
            # v1 path: NLM processing
            self.memory.push_pre(pre_activations)
            pre_history = self.memory.get_pre_history()
            post_activations = self.nlm(pre_history)
            self.memory.push_post(post_activations)
            # Apply IP to NLM output before post_norm (same rule, same hook point)
            if self.ip_layer is not None:
                post_activations = self.ip_layer(post_activations)
            new_state_flat = self.post_norm(post_activations)

            # Recompute sync for output
            updated_post_history = self.memory.get_post_history()
            sync_repr = self.sync_computer.compute(updated_post_history)

        # Reshape back to [B, S, d_latent]
        new_state = new_state_flat.reshape(B, S, -1)

        return new_state, sync_repr, new_stream_state, new_hebbian_state, hebbian_lr_eff

# ═════════════════════════════════════════════════════════════════════════
# model.py
# ═════════════════════════════════════════════════════════════════════════

class FeatureEncoder(nn.Module):
    """
    Generic Feature Encoder backbone for multi-modal capability.
    Replaces the text token embedding with a projection of continuous features
    (e.g., from a Vision Transformer, ResNet, or audio frontend) into d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Placeholder for a real backbone. For now, it's just a linear projection
        # assuming the input is already a sequence of feature vectors of size d_model.
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, feature_dim]
        return self.proj(x)



class CTMTransformer(nn.Module):
    """
    Continuous Thought Machine Transformer (v2).

    Architecture flow:
    1. Embed input text → K, V for cross-attention (computed once)
    2. Initialize per-position latent states z_0 [B, S, d_latent]
    3. For T thought steps:
       a. Each position computes sync from its own post-activation history
       b. Sync → query, which cross-attends to text with CAUSAL MASK
       c. Synapse model mixes attention output with previous state
       d. NLM processes temporal pre-activation history
    4. Output head: concat z[i] + text_emb[i] → logits[i]
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config

        # ── Text Ingestion or Feature Encoder ───────────────────────────
        if config.use_feature_encoder:
            self.feature_encoder = FeatureEncoder(config.d_model)
            self.token_embedding = None
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.feature_encoder = None

        if config.use_positional_encoding:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_embedding = None

        self.embed_norm = nn.LayerNorm(config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Thought Layers ──────────────────────────────────────────────

        def _make_thought_layer(l_idx: int) -> ThoughtLayer:
            """Factory for thought layers with all v2 feature flags."""
            return ThoughtLayer(
                d_latent=config.d_latent,
                d_model=config.d_model,
                n_heads=config.n_heads,
                nlm_hidden_dim=config.nlm_hidden_dim,
                history_len=config.history_len,
                nlm_groups=config.nlm_groups,
                sync_method=config.sync_method,
                sync_rank=config.sync_rank,
                dropout=config.dropout,

                # v2 features
                use_matrix_streams=config.use_matrix_streams,
                n_streams=config.hpc_pc_n_streams if getattr(config, "use_hierarchical_pc", False) else config.n_streams,
                stream_gating=config.stream_gating,
                use_dssa=config.use_dssa,
                dssa_n_partitions=config.dssa_n_partitions,
                dssa_top_k=config.dssa_top_k,
                dssa_block_size=config.dssa_block_size,
                dssa_top_k_blocks=config.dssa_top_k_blocks,
                use_triton_attention=config.use_triton_attention,
                synapse_type=config.synapse_type,
                dendritic_n_branches=getattr(config, "dendritic_n_branches", 4),
                # Biological extensions (opt-in)
                use_hebbian=config.use_hebbian_synapse,
                hebbian_bottleneck_dim=config.hebbian_bottleneck_dim,
                hebbian_decay_init=config.hebbian_decay_init,
                hebbian_lr_init=config.hebbian_lr_init,
                hebbian_gate_init=config.hebbian_gate_init,
                hebbian_force_gate=getattr(config, "hebbian_force_gate", None),
                hebbian_update_rule=getattr(config, "hebbian_update_rule", "outer_product"),
                use_btsp=getattr(config, "use_btsp", False),
                btsp_lr_init=getattr(config, "btsp_lr_init", 0.05),
                btsp_kernel_decay_init=getattr(config, "btsp_kernel_decay_init", 0.9),
                btsp_salience_threshold=getattr(config, "btsp_salience_threshold", 0.0),
                btsp_weight_dependent=getattr(config, "btsp_weight_dependent", False),
                btsp_w_max=getattr(config, "btsp_w_max", 1.0),
                btsp_consolidate_prob=getattr(config, "btsp_consolidate_prob", 0.3),
                btsp_delay_shape=getattr(config, "btsp_delay_shape", 0.0),
                use_ip=getattr(config, "use_intrinsic_plasticity", False),
                ip_lr=getattr(config, "ip_lr", 0.01),
                ip_target=getattr(config, "ip_target", 0.1),
                ip_ema_decay=getattr(config, "ip_ema_decay", 0.99),
                use_thalamic_gating=getattr(config, "use_thalamic_gating", False),
                use_schema_routing=getattr(config, "use_schema_routing", False),
                schema_routing_ema_decay=getattr(config, "schema_routing_ema_decay", 0.99),
                schema_routing_temperature=getattr(config, "schema_routing_temperature", 0.1),
                schema_routing_novelty_threshold=getattr(config, "schema_routing_novelty_threshold", 0.1),
                schema_routing_btsp_scale=getattr(config, "schema_routing_btsp_scale", 2.0),
                hebbian_n_compartments=getattr(config, "hebbian_n_compartments", 1),
                use_stc=getattr(config, "use_stc", False),
                stc_tag_decay=getattr(config, "stc_tag_decay", 0.5),
                stc_threshold_tag=getattr(config, "stc_threshold_tag", 0.05),
                use_critical_init=getattr(config, "use_critical_init", False),
                critical_init_sym_frac=getattr(config, "critical_init_sym_frac", 0.6),
                critical_init_spectral_radius=getattr(config, "critical_init_spectral_radius", 0.999),
                use_multi_rate_streams=getattr(config, "use_multi_rate_streams", False),
                use_burstprop=getattr(config, "use_burstprop", False),
                burstprop_tau=getattr(config, "burstprop_tau", 100.0),
                burstprop_scale_init=getattr(config, "burstprop_scale_init", 1.0),
            )

        # ── Hyperloop or Standard Layer Construction ────────────────────
        if config.use_hyperloop:
            n_begin = config.hyperloop_n_begin
            n_middle = config.hyperloop_n_middle
            n_end = config.hyperloop_n_end
            # Total effective layers = n_begin + n_middle * middle_loops + n_end
            self.begin_layers = nn.ModuleList([
                _make_thought_layer(i) for i in range(n_begin)
            ])
            self.middle_layers = nn.ModuleList([
                _make_thought_layer(n_begin + i) for i in range(n_middle)
            ])
            self.end_layers = nn.ModuleList([
                _make_thought_layer(n_begin + n_middle + i)
                for i in range(n_end)
            ])
            self.layers = None  # Signal that we use hyperloop
            # Total effective layers for AttnRes and other per-layer bookkeeping
            self._effective_n_layers = (
                n_begin + n_middle * config.hyperloop_middle_loops + n_end
            )
        else:
            self.layers = nn.ModuleList([
                _make_thought_layer(l_idx)
                for l_idx in range(config.n_layers)
            ])
            self.begin_layers = None
            self.middle_layers = None
            self.end_layers = None
            self._effective_n_layers = config.n_layers



        # ── FEEC Integrator (v2) ────────────────────────────────────────
        if config.use_feec:
            self.feec = FEECIntegrator(
                d_latent=config.d_latent,
                n_layers=self._effective_n_layers,
                dt_init=config.feec_dt_init,
                damping_init=config.feec_damping_init,
                clamp_dt=config.feec_clamp_dt,
                state_clip=getattr(config, "feec_state_clip", 0.0),
            )
        else:
            self.feec = None

        # ── Hierarchical Predictive Coding (Mechanism 1) ────────────────
        if config.use_hierarchical_pc:
            gen_hidden = config.hpc_generative_hidden_dim or (2 * config.d_latent)
            self.pc_layers = nn.ModuleList([
                PCLayer(
                    d_latent=config.d_latent,
                    layer_idx=i,
                    n_layers=self._effective_n_layers,
                    generative_type=config.hpc_generative_type,
                    generative_hidden_dim=gen_hidden,
                    dropout=config.dropout,
                )
                for i in range(self._effective_n_layers)
            ])
            pc_n_streams = config.hpc_pc_n_streams
            self.pc_state_mgr = PCStateManager(
                n_streams=pc_n_streams,
                d_latent=config.d_latent,
            )
            # Learnable top-level prior μ_top for the highest layer.
            # In the PC hierarchy, the topmost generative model predicts
            # from this prior. When teacher_z is available it replaces
            # this prior at runtime.
            self.pc_top_prior = nn.Parameter(torch.zeros(config.d_latent))
        else:
            self.pc_layers = None
            self.pc_state_mgr = None
            self.pc_top_prior = None

        # ── Loop Position Embeddings (v2) ───────────────────────────────
        if config.use_loop_pos_emb:
            self.loop_pos_emb = nn.Embedding(
                config.max_thought_steps, config.d_latent
            )
        else:
            self.loop_pos_emb = None

        # ── Attention Residuals (Kimi AttnRes) ──────────────────────────
        n_layers_for_attnres = self._effective_n_layers
        self.attn_res_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(config.d_latent))
            for _ in range(n_layers_for_attnres)
        ])
        self.attn_res_norm = nn.RMSNorm(config.d_latent)

        # ── Output Head ─────────────────────────────────────────────────
        # Three configurations, mutually exclusive:
        #   1. per_tick_heads:    8 separate adapters, 1 shared head
        #                         (1.07B params at V=131072 — old default)
        #   2. use_shared_head_film: 1 shared adapter, 1 shared head, with
        #                         per-tick FiLM modulation between them
        #                         (135M params + 16K FiLM — recommended)
        #   3. neither (legacy):  1 shared output_proj used at every tick,
        #                         no tick conditioning — collapses ticks
        if config.per_tick_heads and config.use_shared_head_film:
            raise ValueError(
                "per_tick_heads and use_shared_head_film are mutually "
                "exclusive. Choose one (use_shared_head_film is the "
                "recommended successor)."
            )

        if config.use_shared_head_film:
            T = config.max_thought_steps
            # Shared adapter: latent + text → d_model with norm.
            self.shared_adapter = nn.Sequential(
                nn.Linear(config.d_latent + config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            # Per-tick FiLM parameters. Initialize gamma=1, beta=0 so the
            # initial behavior matches the un-modulated shared head; the
            # network learns to differentiate ticks during training.
            self.film_gamma = nn.Parameter(torch.ones(T, config.d_model))
            self.film_beta = nn.Parameter(torch.zeros(T, config.d_model))
            # Shared LM head — single [d_model, vocab_size] linear.
            # When tie_embeddings is on, the weight is shared with
            # token_embedding (set up further below).
            self.lm_head = nn.Linear(config.d_model, config.vocab_size,
                                     bias=False)
            self.tick_adapters = None
            self.output_proj = None
        elif config.per_tick_heads:
            T = config.max_thought_steps
            self.tick_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.d_latent + config.d_model, config.d_model),
                    nn.GELU(),
                    nn.LayerNorm(config.d_model),
                )
                for _ in range(T)
            ])
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)
            self.output_proj = None
            self.shared_adapter = None
            self.film_gamma = None
            self.film_beta = None
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(config.d_latent + config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.vocab_size),
            )
            self.tick_adapters = None
            self.lm_head = None
            self.shared_adapter = None
            self.film_gamma = None
            self.film_beta = None

        # ── Tied Embeddings ─────────────────────────────────────────────
        # Share weights between token_embedding and lm_head. Only meaningful
        # when use_shared_head_film=True (per_tick_heads has its own
        # lm_head we could tie, but the combination is not recommended;
        # with the legacy output_proj path there is no separable head to
        # tie to). The Embedding stores [V, d_model]; nn.Linear stores
        # [V, in_features=d_model] (same shape). Direct assignment of
        # the underlying parameter is the canonical way to tie.
        if config.tie_embeddings:
            if not config.use_shared_head_film:
                raise ValueError(
                    "tie_embeddings requires use_shared_head_film=True "
                    "(no canonical head to tie to otherwise)."
                )
            if self.token_embedding is None:
                raise ValueError(
                    "tie_embeddings requires a token_embedding (cannot "
                    "tie to a feature encoder)."
                )
            # Share the underlying Parameter object. Both modules now point
            # to the same memory; gradients accumulate correctly via
            # autograd's standard handling of shared parameters.
            self.lm_head.weight = self.token_embedding.weight

        # ── Initial State ───────────────────────────────────────────────
        self.z0 = nn.Parameter(torch.zeros(config.d_latent))

        # ── Distillation Alignment ──────────────────────────────────────
        # teacher_z_proj is created whenever feature distillation is enabled
        # and the dimensions differ (or always if we want a learned mapping).
        # The student-side LayerNorm + teacher-side LayerNorm are needed for
        # the "cosine" and "mse_normed" feature-distillation methods, so the
        # two architectures' hidden states are compared on a magnitude-
        # neutral footing.
        if config.use_distillation and config.distill_feature_weight > 0:
            if config.teacher_d_model != config.d_latent:
                self.teacher_z_proj = nn.Linear(config.teacher_d_model, config.d_latent)
            else:
                self.teacher_z_proj = None
            self.distill_student_norm = nn.LayerNorm(config.d_latent, elementwise_affine=False)
            self.distill_teacher_norm = nn.LayerNorm(config.d_latent, elementwise_affine=False)
        else:
            self.teacher_z_proj = None
            self.distill_student_norm = None
            self.distill_teacher_norm = None

        # ── Cerebellar Readout (Predictive Coding) ──────────────────────
        # Per-tick predictor of the chosen target. See
        # biological.CerebellarReadout for details. Only constructed if
        # the config flag is on; when off, no parameters are added.
        if config.use_predictive_coding:
            pc_hidden = config.pc_hidden_dim if config.pc_hidden_dim > 0 else None
            # target_dim depends on the target mode:
            #   "teacher"    → teacher_d_model (predict teacher hidden state)
            #   "next_tick"  → d_latent       (predict the student's own next-tick z)
            #   "final_tick" → d_latent       (predict the student's own final-tick z)
            pc_target = getattr(config, "pc_target", "teacher")
            if pc_target == "teacher":
                target_dim = config.teacher_d_model
            elif pc_target in ("next_tick", "final_tick"):
                target_dim = config.d_latent
            else:
                raise ValueError(
                    f"Unknown pc_target={pc_target!r}. Must be one of "
                    f"'teacher', 'next_tick', 'final_tick'."
                )
            self.cerebellar_readout = CerebellarReadout(
                target_dim=target_dim,
                d_latent=config.d_latent,
                hidden_dim=pc_hidden,
                dropout=config.pc_dropout,
            )
        else:
            self.cerebellar_readout = None

        # Velocity initial state (for FEEC)
        if config.use_feec:
            self.velocity_0 = nn.Parameter(torch.zeros(config.d_latent))
        else:
            self.velocity_0 = None

        # ── CUDA Graph wrapper (v2) ─────────────────────────────────────
        if config.use_cuda_graphs:
            self._cuda_graph = CUDAGraphThoughtLoop(enabled=True)
        else:
            self._cuda_graph = None

        # ── Training step counter (non-persistent) ──────────────────────
        self.register_buffer(
            "_train_step",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )

        # ── Sleep consolidation: cross-batch Hebbian carry-over ──────────
        # When use_sleep_consolidation=True, stores the mean fast-weight
        # matrix [m, n] per layer on CPU after each forward pass.
        # The next forward pass warm-starts Hebbian states from this
        # carry-over rather than zeros, making associative memory
        # persistent across batches. Reset to None at training start.
        # NOT a registered buffer — transient runtime state.
        self._hebbian_carry: list | None = None

        # ── R2 Burstprop: previous-tick PC errors (apical top-down signal) ──
        # Populated by _thought_step when use_burstprop is on; consumed one
        # tick later by the Hebbian fast-weight update. Transient runtime state.
        self._last_pc_errors: list | None = None
        if getattr(config, "use_burstprop", False):
            if not config.use_hebbian_synapse:
                raise ValueError(
                    "use_burstprop requires use_hebbian_synapse=True "
                    "(Burstprop modulates the Hebbian fast-weight update)."
                )
            if not getattr(config, "use_hierarchical_pc", False):
                raise ValueError(
                    "use_burstprop requires use_hierarchical_pc=True "
                    "(the apical burst signal is the previous tick's PC error)."
                )

        # ── BCM Sliding Threshold ─────────────────────────────────────────
        # EMA of mean squared latent magnitude, updated during the inference
        # phase of the EM loop. Acts as the Bienenstock-Cooper-Munro
        # metaplasticity threshold to prevent runaway latent excitation.
        if getattr(config, "use_bcm_threshold", False):
            self.register_buffer("_bcm_threshold", torch.tensor(1.0))
        else:
            self._bcm_threshold = None

        # ── Amortized Inference Network ─────────────────────────────────
        if (getattr(config, "use_amortized_inference", False)
                and getattr(config, "use_prospective_config", False)):
            amortized_hidden = getattr(config, "amortized_hidden_dim", 0) or (2 * config.d_model)
            self.amortized_net = AmortizedInferenceNet(
                d_model=config.d_model,
                d_latent=config.d_latent,
                hidden_dim=amortized_hidden,
            )
        else:
            self.amortized_net = None

        # ── R1: EFE / PonderNet thought-step controller ─────────────────
        if getattr(config, "use_efe_controller", False):
            efe_hidden = getattr(config, "efe_controller_hidden_dim", 0) or max(
                32, config.d_latent // 8
            )
            self.efe_controller = EFEThoughtController(
                hidden_dim=efe_hidden,
                halt_prior=getattr(config, "efe_halt_prior", 0.1),
            )
        else:
            self.efe_controller = None

        # ── R3: Meta-RL reward / action input channel ───────────────────
        # Reward = probability the model assigned to the previous realized
        # token (scalar in [0, 1]); action = the previous-position argmax token
        # (reuses the token embedding). Both are shifted by one position and
        # injected into the input embedding via a learnable gate. _meta_inner
        # guards the no_grad pre-pass against recursion / double side-effects.
        self._meta_inner = False
        if getattr(config, "use_meta_rl", False):
            if config.use_feature_encoder:
                raise ValueError(
                    "use_meta_rl requires a token embedding (the action channel "
                    "reuses it); it is incompatible with use_feature_encoder."
                )
            self.meta_reward_proj = nn.Linear(1, config.d_model)
            self.meta_inject_scale = nn.Parameter(
                torch.tensor(float(getattr(config, "meta_rl_reward_scale_init", 0.1)))
            )
        else:
            self.meta_reward_proj = None
            self.meta_inject_scale = None

        # ── R10: REM adversarial dreamer ────────────────────────────────
        if getattr(config, "use_rem_dreaming", False):
            if not getattr(config, "use_sleep_consolidation", False):
                raise ValueError(
                    "use_rem_dreaming requires use_sleep_consolidation=True "
                    "(the REM phase augments the sleep-consolidation loss)."
                )
            self.rem_dreamer = REMDreamer(
                d_latent=config.d_latent,
                noise_dim=getattr(config, "rem_noise_dim", 0),
                hidden_dim=getattr(config, "rem_hidden_dim", 0),
            )
            self._last_rem_disc_acc = 0.0
        else:
            self.rem_dreamer = None
        # Set by the sleep cycle (via the DDP-wrapped module) so the adversarial
        # REM loss is computed INSIDE forward — keeping its generator/
        # discriminator gradients inside the DDP-tracked graph (allreduced).
        self._rem_sleep_active = False

        # ── Episodic DND ──────────────────────────────────────────────────────
        # Non-parametric key-value bank for zero-shot retrieval. No model
        # parameters are added; the DND is purely runtime state (not saved in
        # state_dict). Keys = mean-pooled text_emb[:key_dim]; values =
        # mean-pooled z_curr from high-certainty passes.
        if getattr(config, "use_dnd", False):
            self.dnd: EpisodicDND | None = EpisodicDND(
                capacity=getattr(config, "dnd_capacity", 1000),
                key_dim=getattr(config, "dnd_key_dim", 64),
                d_latent=config.d_latent,
                confidence_threshold=getattr(config, "dnd_confidence_threshold", 0.98),
                hopfield_beta=getattr(config, "dnd_hopfield_beta", 4.0),
                min_novelty=getattr(config, "dnd_min_novelty", 0.05),
            )
        else:
            self.dnd = None

        self.apply(self._init_weights)

        # Re-apply Engram-specific inits
        for m in self.modules():
            reset_fn = getattr(m, "_reset_special_inits", None)
            if callable(reset_fn) and m is not self:
                reset_fn()




    def _init_weights(self, module):
        """Standard transformer weight initialization.

        Note: LayerNorm with elementwise_affine=False has weight/bias = None,
        so we must guard against that. Several modules in this codebase
        use non-affine LayerNorm intentionally (distill_*_norm,
        biological.HebbianSynapse.readout_norm, biological.CerebellarReadout's
        student/teacher norms) — without these guards, .apply(_init_weights)
        crashes with "NoneType has no attribute fill_".
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_embedding is not None:
            n_params -= self.pos_embedding.weight.numel()
        return n_params

    @torch.no_grad()
    def renormalize_spectral_radius(self, target: float = 1.0) -> list[tuple[int, float, float]]:
        """Synaptic-homeostasis (SHY) pass over all thought layers.

        Projects each layer's synapse z-recurrence matrix back to spectral
        radius ≤ target, downscaling supercritical recurrences toward
        criticality. De-duplicates shared layers (hyperloop reuses the middle
        block) so a weight isn't rescaled more than once. Deterministic given
        the weights, so it stays in sync across DDP ranks without communication.

        Returns [(layer_idx, rho_before, rho_after), ...].
        """
        seen: set[int] = set()
        out: list[tuple[int, float, float]] = []
        for i, layer in enumerate(self._get_layers_sequence()):
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            before, after = layer.renormalize_synapse_spectral_radius(target)
            out.append((i, before, after))
        return out

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens or continuous features. Returns [batch, seq_len, d_model]."""
        B, S = input_ids.shape[:2]
        if self.config.use_feature_encoder:
            # Assume input_ids is actually a continuous feature tensor
            tok_emb = self.feature_encoder(input_ids)
        else:
            tok_emb = self.token_embedding(input_ids)

        if self.pos_embedding is not None:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            tok_emb = tok_emb + self.pos_embedding(positions)

        return self.embed_dropout(self.embed_norm(tok_emb))

    def _output_logits(
        self,
        z: torch.Tensor,
        text_emb: torch.Tensor,
        tick: int = 0,
    ) -> torch.Tensor:
        """
        Project per-position latent states to logits.

        Args:
            z: [B, S, d_latent] — per-position latent states.
            text_emb: [B, S, d_model] — text embeddings.
            tick: which thought tick this projection is for.

        Returns:
            logits: [B, S, vocab_size]
        """
        combined = torch.cat([z, text_emb], dim=-1)

        if self.shared_adapter is not None:
            # Shared head + per-tick FiLM modulation.
            # adapter: [B, S, d_latent + d_model] → [B, S, d_model]
            x = self.shared_adapter(combined)
            # FiLM: y = gamma_t * x + beta_t. Per-tick scalars broadcast
            # over the [B, S] dimensions.
            # Clamp tick index: under prospective config, inference may
            # exceed the original max_thought_steps allocation. Ticks
            # beyond the table reuse the last (most refined) entry.
            film_tick = min(tick, self.film_gamma.shape[0] - 1)
            gamma = self.film_gamma[film_tick]   # [d_model]
            beta = self.film_beta[film_tick]     # [d_model]
            x = x * gamma + beta
            return self.lm_head(x)

        if self.tick_adapters is not None:
            adapter_tick = min(tick, len(self.tick_adapters) - 1)
            adapted = self.tick_adapters[adapter_tick](combined)
            return self.lm_head(adapted)

        return self.output_proj(combined)

    def _get_layers_sequence(self) -> list[ThoughtLayer]:
        """Get the sequence of layers to iterate over (respecting Hyperloop)."""
        if self.layers is not None:
            return list(self.layers)

        # Hyperloop: begin + middle*loops + end
        seq = list(self.begin_layers)
        for _ in range(self.config.hyperloop_middle_loops):
            seq.extend(list(self.middle_layers))
        seq.extend(list(self.end_layers))
        return seq

    def _thought_step(
        self,
        z: torch.Tensor,
        t: int,
        text_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,

        pre_states: list[torch.Tensor] | None = None,
        post_states: list[torch.Tensor] | None = None,
        velocity: torch.Tensor | None = None,
        stream_states: list[torch.Tensor] | None = None,
        targets: torch.Tensor | None = None,
        teacher_log_probs: torch.Tensor | None = None,
        teacher_top_indices: torch.Tensor | None = None,
        distill_temp: float = 1.0,
        hebbian_states: list[torch.Tensor] | None = None,
        teacher_z_for_pc: torch.Tensor | None = None,
        T_total: int = 1,
        prev_certainty: torch.Tensor | None = None,
        compute_logits: bool = True,
        burst_prev_errors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], torch.Tensor | None, list[torch.Tensor], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        """
        Execute all layers in the sequence for one thought tick.

        Returns:
            z_new: [B, S, d_latent]
            logits_t: [B, S, V] or None (if training)
            new_pre_states, new_post_states: v1 memory
            velocity_new: FEEC velocity
            new_stream_states: v2 streams
            ce_loss_t: Scalar or per-token loss
            kl_loss_t: Scalar KL loss
            certainty_t: [B, S] certainty score
            new_hebbian_states: list of Hebbian fast-weight matrices (one per layer)
            pc_loss_t: Predictive-coding auxiliary loss for this tick (scalar) or None
            hebbian_lr_eff_t: Mean effective Hebbian lr across positions (for logging), or None
        """
        B, S = z.shape[:2]
        layers = self._get_layers_sequence()
        layer_outputs = []
        new_stream_states = []
        new_hebbian_states = []
        hebbian_lr_effs = []  # per-layer effective lr, for diagnostics

        velocity_new = velocity
        if velocity is not None:
            velocity_new = velocity.clone()

        # ── Compute per-position Hebbian lr modulator from prev certainty ──
        # Biological motivation: locus coeruleus releases norepinephrine in
        # response to surprise, and NE gates plasticity for subsequent
        # events. Functionally: uncertain positions write more aggressively
        # to fast-weight memory than confident ones.
        #
        # Modulator shape: [B, S], value (1 + alpha * uncertainty).
        # alpha=0 disables (uniform lr). prev_certainty is None at tick 0
        # → also no modulation. Computed under no_grad: this is a routing
        # signal, not a learnable path, so we don't want grad flowing back
        # through the certainty computation.
        alpha = getattr(self.config, "hebbian_cert_lr_alpha", 0.0)
        hebbian_lr_modulator = None
        if (alpha > 0
                and prev_certainty is not None
                and self.config.use_hebbian_synapse):
            with torch.no_grad():
                # certainty is in [0, 1]; we use
                # uncertainty = clamp(1 - certainty, 0, 1) to be safe.
                uncertainty = (1.0 - prev_certainty).clamp(min=0.0, max=1.0)
                hebbian_lr_modulator = 1.0 + alpha * uncertainty
                hebbian_lr_modulator = hebbian_lr_modulator.to(z.dtype)

        for l_idx, layer in enumerate(layers):
            # Attention Residuals (Kimi AttnRes)
            # Stack all previous outputs (at least the input z)
            V = torch.stack(layer_outputs + [z], dim=0)
            
            V_norm = V.to(self.attn_res_norm.weight.dtype)
            K = self.attn_res_norm(V_norm) # [N, B, S, D]
            w_l = self.attn_res_queries[l_idx] if l_idx < len(self.attn_res_queries) else self.attn_res_queries[-1]
            scores = torch.einsum('d,nbsd->nbs', w_l, K) # [N, B, S]
            attn_weights = F.softmax(scores, dim=0)
            z_in = torch.einsum('nbs,nbsd->bsd', attn_weights, V) # [B, S, D]

            # Get stream state for this layer
            layer_stream = None
            if self.config.use_matrix_streams and stream_states is not None and l_idx < len(stream_states):
                layer_stream = stream_states[l_idx]

            # Get Hebbian state for this layer
            layer_heb = None
            if hebbian_states is not None and l_idx < len(hebbian_states):
                layer_heb = hebbian_states[l_idx]

            # Burstprop apical signal: previous tick's PC error for this layer.
            layer_top_down = None
            if burst_prev_errors is not None and l_idx < len(burst_prev_errors):
                layer_top_down = burst_prev_errors[l_idx]

            z_out, _sync_repr, new_layer_stream, new_layer_heb, layer_lr_eff = layer(
                text_emb,
                text_emb,
                z_in,
                key_padding_mask,

                stream_state=layer_stream,
                hebbian_state=layer_heb,
                hebbian_lr_modulator=hebbian_lr_modulator,
                hebbian_top_down=layer_top_down,
            )
            if layer_lr_eff is not None:
                hebbian_lr_effs.append(layer_lr_eff)

            if self.config.use_matrix_streams and new_layer_stream is not None:
                # Multi-rate cadence: slow streams hold their value between
                # refreshes (cross-frequency-coupling analogue). No-op unless
                # use_multi_rate_streams is enabled.
                if (getattr(self.config, "use_multi_rate_streams", False)
                        and layer.stream is not None
                        and layer_stream is not None):
                    new_layer_stream = layer.stream.apply_update_cadence(
                        t, layer_stream, new_layer_stream
                    )
                new_stream_states.append(new_layer_stream)
            # Keep new_hebbian_states parallel to `layers`: append None
            # for layers without a Hebbian synapse so indexing by l_idx
            # stays stable across calls. Today the flag is global so
            # either every layer has hebbian or none does, but the
            # robust pattern costs nothing and matches stream_states.
            new_hebbian_states.append(new_layer_heb)

            # ── FEEC Integration ────────────────────────────────────────
            if self.feec is not None and velocity_new is not None:
                force = z_out - z_in  # Force field = layer's contribution
                z_out, velocity_new = self.feec.step(
                    z_in, velocity_new, force, layer_idx=l_idx
                )

            layer_outputs.append(z_out)

        z_new = layer_outputs[-1]

        # ── Schema novelty: maybe wake a dormant stream ──────────────────
        # After all layers have run, check whether the schema router on any
        # layer is reporting consistently low max-similarity (novel input).
        # If so, activate one dormant stream slot for that layer.
        # Only fires during training; guarded with no_grad since activate_stream
        # is a buffer mutation.
        if self.training and getattr(self.config, "use_schema_routing", False):
            with torch.no_grad():
                seen_stream_ids: set[int] = set()
                for layer in layers:
                    if (layer.stream is None
                            or layer.stream.schema_router is None
                            or id(layer.stream) in seen_stream_ids):
                        continue
                    seen_stream_ids.add(id(layer.stream))
                    last_sim = getattr(layer.stream, "_last_max_similarity", None)
                    if last_sim is not None:
                        layer.stream.maybe_activate_novel_stream(last_sim)

        # ── Hierarchical PC Inference Step ───────────────────────────────
        # After the standard forward pass produces per-layer μ values
        # (layer_outputs), run one PC inference iteration:
        #   1. Top-down error sweep: ε_ℓ = μ_ℓ − f_ℓ(μ_{ℓ+1})
        #   2. Precision estimation: π_ℓ from local stats
        #   3. Bottom-up μ update: μ_ℓ -= η·(π_ℓ·ε_ℓ − Wᵀ·π_{ℓ-1}·ε_{ℓ-1})
        #   4. Local loss: Σ π_ℓ·‖ε_ℓ‖² for each generative function
        # All cross-layer deps are .detach()ed — gradients are fully local.
        hpc_local_loss_t = None
        hpc_free_energy_t = None
        hpc_mean_precision_t = None
        hpc_mean_error_norm_t = None
        if self.pc_layers is not None and len(layer_outputs) > 1:
            n_pc = min(len(self.pc_layers), len(layer_outputs))
            pc_errors = []
            pc_precisions = []
            pc_predictions = []
            pc_local_losses = []

            # Get top-level prior (teacher_z if available, else learnable)
            if teacher_z_for_pc is not None and teacher_z_for_pc.shape[-1] == self.config.d_latent:
                mu_top = teacher_z_for_pc.detach()
            else:
                mu_top = self.pc_top_prior.unsqueeze(0).unsqueeze(0).expand(
                    B, S, -1
                )

            # Top-down error sweep (from highest layer down)
            for l_idx in range(n_pc - 1, -1, -1):
                mu_current = layer_outputs[l_idx]
                if l_idx == n_pc - 1:
                    mu_above = mu_top
                else:
                    mu_above = layer_outputs[l_idx + 1]

                prediction, error, precision = self.pc_layers[l_idx].compute_error(
                    mu_current, mu_above
                )
                pc_errors.insert(0, error)
                pc_precisions.insert(0, precision)
                pc_predictions.insert(0, prediction)

                # Local loss for this layer's generative function
                local_l = self.pc_layers[l_idx].local_loss(
                    prediction, mu_current, precision
                )
                pc_local_losses.append(local_l)

            # Bottom-up μ update (from lowest layer up)
            for l_idx in range(n_pc):
                error_below = pc_errors[l_idx - 1] if l_idx > 0 else None
                prec_below = pc_precisions[l_idx - 1] if l_idx > 0 else None

                delta_mu = self.pc_layers[l_idx].compute_mu_update(
                    error=pc_errors[l_idx],
                    precision=pc_precisions[l_idx],
                    error_below=error_below,
                    precision_below=prec_below,
                    inference_lr=self.config.hpc_inference_lr,
                )

                # When FEEC is active, use δμ as the force vector for
                # the symplectic integrator. This routes the PC inference
                # dynamics through the Hamiltonian stepper, which:
                #   1. Maintains energy conservation during long inference
                #   2. Provides natural momentum for faster convergence
                #   3. Prevents oscillation via the built-in damping
                if self.feec is not None and velocity_new is not None:
                    mu_updated, velocity_new = self.feec.step(
                        layer_outputs[l_idx], velocity_new,
                        force=delta_mu, layer_idx=l_idx,
                    )
                    layer_outputs[l_idx] = mu_updated
                else:
                    # Fallback: direct additive update
                    layer_outputs[l_idx] = layer_outputs[l_idx] + delta_mu

            z_new = layer_outputs[-1]

            # Diagnostics
            if pc_local_losses:
                hpc_local_loss_t = torch.stack(pc_local_losses).sum()
            with torch.no_grad():
                err_norms = [e.norm(dim=-1).mean() for e in pc_errors]
                hpc_mean_error_norm_t = torch.stack(err_norms).mean()
                prec_vals = [p.mean() for p in pc_precisions]
                hpc_mean_precision_t = torch.stack(prec_vals).mean()
                hpc_free_energy_t = self.pc_state_mgr.free_energy(
                    pc_errors, pc_precisions
                )
                # Per-layer precision scalars for precision-weighted gradient scaling.
                # Stored on the model so forward() can publish them without
                # expanding the _thought_step return tuple.
                self._last_per_layer_precision = [
                    float(p.mean().item()) for p in pc_precisions
                ]
                # R2 Burstprop: stash this tick's per-layer PC errors (detached)
                # so the *next* tick's Hebbian update can use them as the apical
                # top-down credit signal. Read in the BPTT loop (outside any
                # gradient-checkpoint region), so it stays checkpoint-safe.
                if getattr(self.config, "use_burstprop", False):
                    self._last_pc_errors = [e.detach() for e in pc_errors]

        # ── Output + Loss + Certainty (In-Loop) ─────────────────────────
        # In pure_pc_mode, we detach z_new before producing logits. This allows the 
        # LM head to be trained by the Cross Entropy and Distillation losses without 
        # allowing those global gradients to backpropagate into the encoder weights, 
        # preserving the purely local PC learning for the transformer layers.
        logits_t = None
        ce_loss_t = None
        kl_loss_t = None
        certainty_t = None

        if compute_logits:
            z_out = z_new.detach() if getattr(self.config, "pure_pc_mode", False) else z_new
            logits_t = self._output_logits(z_out, text_emb, tick=t)
            
            # Certainty (entropy-based, always normalized to [0, 1])
            with torch.no_grad():
                probs_t = F.softmax(logits_t, dim=-1)
                entropy_t = -(probs_t * (probs_t + 1e-10).log()).sum(dim=-1)
                max_entropy = math.log(self.config.vocab_size)
                certainty_t = 1.0 - (entropy_t / max_entropy)
            
            if targets is not None:
                V_size = self.config.vocab_size
                logits_flat = logits_t.reshape(-1, V_size)
                targets_flat = targets.reshape(-1)
                
                # CE Loss (always mean reduction — per-tick aggregation
                # is handled in forward() for legacy BPTT; under prospective
                # config only the single consolidation pass matters)
                ce_loss_t = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
                    
                # KL Distillation
                if teacher_log_probs is not None:
                    student_logits_flat = logits_flat.float()
                    if teacher_top_indices is not None:
                        student_top_logits = student_logits_flat.gather(-1, teacher_top_indices)
                        student_log_probs = F.log_softmax(student_top_logits / distill_temp, dim=-1)
                    else:
                        student_log_probs = F.log_softmax(student_logits_flat / distill_temp, dim=-1)
                    
                    kl_t = F.kl_div(
                        student_log_probs,
                        teacher_log_probs,
                        reduction="batchmean",
                        log_target=True,
                    ) * (distill_temp * distill_temp)
                    kl_loss_t = kl_t
                
                # If we are training and computed losses, we can drop logits_t to save memory.
                if self.training:
                    logits_t = None

        # Read out v1 memory state
        new_pre_states = []
        new_post_states = []
        if not self.config.use_matrix_streams:
            for layer in layers:
                if layer.memory is not None:
                    new_pre_states.append(layer.memory.pre_history)
                    new_post_states.append(layer.memory.post_history)

        # ── Predictive-Coding / Prospective-Clamping ──────────────────────
        # The CerebellarReadout is now a target projector (single-arg) for
        # Prospective Configuration. Under EM mode, it projects the target
        # into d_latent for HPC clamping. Under legacy BPTT, the HPC
        # local losses provide the PC gradient signal. The old two-arg
        # cerebellar prediction pathway has been deprecated.
        pc_loss_t = None

        # Mean effective Hebbian lr for diagnostics (across all layers
        # with an active Hebbian module). None when Hebbian is off.
        hebbian_lr_eff_t = None
        if hebbian_lr_effs:
            hebbian_lr_eff_t = torch.stack(hebbian_lr_effs).mean()

        return z_new, logits_t, new_pre_states, new_post_states, velocity_new, new_stream_states, ce_loss_t, kl_loss_t, certainty_t, new_hebbian_states, pc_loss_t, hebbian_lr_eff_t, hpc_local_loss_t, hpc_free_energy_t, hpc_mean_precision_t, hpc_mean_error_norm_t

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        max_thought_steps: int | None = None,
        teacher_logits: torch.Tensor | None = None,
        teacher_z: torch.Tensor | None = None,
        cached_top_indices: torch.Tensor | None = None,
        cached_top_values: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass with per-position latent states and causal masking.

        Distillation paths (mutually exclusive — supply one OR the other,
        not both):
          - teacher_logits:  full [B, S, V] logits from a live teacher.
              The model computes top-K internally. Used by the legacy
              online-teacher path.
          - cached_top_indices, cached_top_values:
              precomputed top-K indices [B, S, K] (long) and the
              corresponding RAW logit values [B, S, K] (float). Used by
              the offline teacher cache path (scripts/cache_teacher_logits.py).
              When supplied, the model skips the topk computation and
              feeds these straight into the KL loss. cuda:1 can sit
              empty.
        """
        T = max_thought_steps or self.config.max_thought_steps
        B, S = input_ids.shape[:2]
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # ── Embed text ──────────────────────────────────────────────────
        text_emb = self._embed_text(input_ids)

        # ── R3: Meta-RL reward / action channel ──────────────────────────
        # Treat each sequence as an episode. A cheap no_grad pre-pass yields
        # the model's own per-position prediction; from it we form the causal
        # reward r_{s-1} = p(token_{s-1}) and action a_{s-1} = argmax_{s-1},
        # shift them one position forward, and add them (gated) to the input
        # embedding so the thought loop sees how well it predicted recent
        # tokens. _meta_inner guards the pre-pass against recursion and
        # suppresses its side-effects.
        meta_active = (
            getattr(self.config, "use_meta_rl", False)
            and self.meta_reward_proj is not None
            and not self._meta_inner
            and targets is not None
        )
        if meta_active:
            self._meta_inner = True
            # Run the probe in eval mode: _thought_step drops per-tick logits
            # while self.training, so we'd get no prediction back otherwise.
            # eval mode also makes the throwaway pass deterministic and
            # side-effect-free (dropout off, no IP/STC/schema EMA updates).
            _was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    inner = self.forward(
                        input_ids,
                        targets=targets,
                        key_padding_mask=key_padding_mask,
                        max_thought_steps=max_thought_steps,
                    )
            finally:
                self.train(_was_training)
                self._meta_inner = False
            inner_logits = inner.get("logits")
            if inner_logits is not None:
                with torch.no_grad():
                    logp = F.log_softmax(inner_logits.float(), dim=-1)
                    # reward = probability assigned to the realized token ∈ [0,1]
                    reward = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).exp()
                    action = inner_logits.argmax(dim=-1)
                    reward_prev = torch.zeros_like(reward)
                    reward_prev[:, 1:] = reward[:, :-1]
                    action_prev = torch.zeros_like(action)
                    action_prev[:, 1:] = action[:, :-1]
                reward_emb = self.meta_reward_proj(
                    reward_prev.unsqueeze(-1).to(text_emb.dtype)
                )
                action_emb = self.token_embedding(action_prev).to(text_emb.dtype)
                # Re-normalize after injecting: text_emb feeds cross-attention K/V
                # every thought tick and the meta channel (meta_reward_proj +
                # meta_inject_scale) is otherwise an *unbounded* additive
                # perturbation on the post-embed_norm signal. Without this the
                # injection magnitude grows with training and can push the
                # prospective inference relaxation supercritical → NaN.
                text_emb = self.embed_norm(
                    text_emb + self.meta_inject_scale * (reward_emb + action_emb)
                )

        # ── Episodic DND: instant retrieval for previously solved contexts ────
        # Key = mean-pooled text embedding (content fingerprint of the input).
        # If a near-exact key is in the DND, bypass the thought loop entirely:
        # expand the cached latent to [B, S, d_latent] and project to logits.
        # The output head (adapter + lm_head) is NOT frozen — CE loss still
        # flows gradients through it, so the head keeps improving even when
        # the thought loop is bypassed.
        if self.dnd is not None:
            with torch.no_grad():
                _key_dim = getattr(self.config, "dnd_key_dim", 64)
                _dnd_pool = text_emb.detach().float().mean(dim=1)  # [B, d_model]
                if _key_dim < _dnd_pool.shape[-1]:
                    _dnd_pool = _dnd_pool[..., :_key_dim]
                _retrieved = [self.dnd.query(_dnd_pool[b]) for b in range(B)]
                _all_hit = all(v is not None for v, _ in _retrieved)

            if _all_hit:
                _z_stacked = torch.stack([v for v, _ in _retrieved]).to(device=device, dtype=dtype)
                _z_exp = _z_stacked.unsqueeze(1).expand(B, S, -1)  # [B, S, d_latent]
                _logits = self._output_logits(_z_exp, text_emb, tick=T - 1)
                _dnd_result: dict = {
                    "logits": _logits,
                    "certainties": None,
                    "all_logits": [],
                    "loss": torch.tensor(0.0, device=device, dtype=dtype),
                    "dnd_hit": True,
                    "dnd_max_sim": float(max(sim for _, sim in _retrieved)),
                }
                if targets is not None:
                    _dnd_result["loss"] = F.cross_entropy(
                        _logits.reshape(-1, self.config.vocab_size),
                        targets.reshape(-1),
                        reduction="mean",
                    )
                return _dnd_result

        # ── Initialize latent states ────────────────────────────────────
        z = self.z0.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()

        # FEEC velocity initialization
        velocity = None
        if self.feec is not None and self.velocity_0 is not None:
            velocity = self.velocity_0.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()

        # ── Initialize memory / streams ─────────────────────────────────
        layers = self._get_layers_sequence()
        BS = B * S

        # v2: matrix stream states
        stream_states = None
        if self.config.use_matrix_streams:
            stream_states = []
            for layer in layers:
                if layer.stream is not None:
                    stream_states.append(
                        layer.stream.init_state(B, S, device, dtype)
                    )
                else:
                    stream_states.append(None)
        else:
            # v1: reset FIFO memory buffers. MUST be paired with
            # use_matrix_streams=False (else the NLM path's get_post_history()
            # returns None and ThoughtLayer.forward crashes). This is
            # orthogonal to use_hebbian_synapse — the Hebbian fast-weights
            # are an *additional* working-memory mechanism that augments
            # whichever residual-stream path is active.
            for layer in layers:
                if layer.memory is not None:
                    layer.reset_memory(BS, device, dtype)

        # Hebbian fast-weight states (one matrix per layer, per (B, S))
        # Initialised to zero by default; updated in-loop by each ThoughtLayer.
        # When use_sleep_consolidation=True, warm-started from the cross-batch
        # carry-over instead of zeros — making associative memory persistent.
        # When use_hebbian_synapse is off, this list is None and the layers
        # short-circuit the read/update path entirely.
        hebbian_states = None
        if self.config.use_hebbian_synapse:
            hebbian_states = []
            use_carry = (
                getattr(self.config, "use_sleep_consolidation", False)
                and self._hebbian_carry is not None
            )
            for i, layer in enumerate(layers):
                if layer.hebbian is not None:
                    if (use_carry
                            and i < len(self._hebbian_carry)
                            and self._hebbian_carry[i] is not None):
                        # Expand mean carry-over [...] → [B, S, ...] (works for flat [m,n] and compartmentalized [K,m_k,m_k])
                        carry = self._hebbian_carry[i].to(device=device, dtype=dtype)
                        hebbian_states.append(
                            carry.unsqueeze(0).unsqueeze(0).expand(B, S, *[-1] * carry.ndim).clone()
                        )
                    else:
                        hebbian_states.append(layer.hebbian.init_state(B, S, device, dtype))
                else:
                    hebbian_states.append(None)

        # ── Distillation Prep ───────────────────────────────────────────
        teacher_log_probs = None
        teacher_top_indices = None
        distill_temp = float(self.config.distill_temperature)
        if self.config.use_distillation and targets is not None:
            if teacher_logits is not None or cached_top_indices is not None:
                if cached_top_indices is not None:
                    B_, S_, K_ = cached_top_indices.shape
                    teacher_top_indices = cached_top_indices.reshape(B_ * S_, K_).long()
                    cached_values_flat = cached_top_values.reshape(B_ * S_, K_)
                    teacher_log_probs = F.log_softmax(cached_values_flat.float() / distill_temp, dim=-1)
                else:
                    B_, S_, V_ = teacher_logits.shape
                    teacher_logits_flat = teacher_logits.reshape(B_ * S_, V_)
                    top_k = int(getattr(self.config, "distill_top_k", 0) or 0)
                    if top_k > 0 and top_k < V_:
                        teacher_top_logits, teacher_top_indices = teacher_logits_flat.topk(top_k, dim=-1)
                        teacher_log_probs = F.log_softmax(teacher_top_logits / distill_temp, dim=-1)
                    else:
                        teacher_log_probs = F.log_softmax(teacher_logits_flat / distill_temp, dim=-1)

        # ── Expectation-Maximization (EM) Training Loop ─────────────────
        prospective_active = getattr(self.config, "use_prospective_config", False)

        result = {
            "logits": None,
            "certainties": None,
            "all_logits": [],
            "loss": torch.tensor(0.0, device=device, dtype=dtype),
        }

        use_per_step_ckpt = (
            self.training
            and self.config.gradient_checkpointing
            and T >= getattr(self.config, "gradient_checkpointing_min_T", 5)
        )

        # Base loop setup
        z_curr = z
        velocity_curr = velocity
        stream_states_curr = stream_states
        hebbian_states_curr = hebbian_states

        # v1 Memory initial states
        pre_states = []
        post_states = []
        if not self.config.use_matrix_streams:
            pre_states = [layer.memory.pre_history for layer in layers if layer.memory is not None]
            post_states = [layer.memory.post_history for layer in layers if layer.memory is not None]

        # Target clamping for Prospective Configuration
        clamped_target = None
        if prospective_active and self.cerebellar_readout is not None and targets is not None:
            # Targets are [B, S] tokens. Embed them to get [B, S, d_model]
            target_emb = self._embed_text(targets)
            clamped_target = self.cerebellar_readout(target_emb) # [B, S, d_latent]

        # ── Phase 1: Inference (Relaxation) ─────────────────────────────
        # If Prospective Configuration is active, we run the thought loop
        # purely as an inference process to find the equilibrium state \mu^*.
        # We wrap this in torch.no_grad() to save massive amounts of VRAM.
        #
        # Amortized inference: if use_amortized_inference is on, a lightweight
        # feedforward network predicts z* in one pass and warm-starts z_curr,
        # cutting required relaxation steps from O(1000) to O(5).
        if prospective_active:
            z_hat = None
            if self.amortized_net is not None:
                # Single-pass prediction of equilibrium state
                z_hat = self.amortized_net(text_emb)  # [B, S, d_latent]
                z_curr = z_hat.detach()               # warm-start inference loop
                if max_thought_steps is not None:
                    max_inference_steps = max_thought_steps
                else:
                    max_inference_steps = getattr(self.config, "amortized_inference_steps", 5)
            else:
                if max_thought_steps is not None:
                    max_inference_steps = max_thought_steps
                else:
                    max_inference_steps = getattr(self.config, "max_inference_steps", 1000)
            energy_tol = getattr(self.config, "inference_energy_tol", 1e-4)

            prev_certainty = None
            prev_energy = None

            bcm_active = self._bcm_threshold is not None
            bcm_decay = getattr(self.config, "bcm_ema_decay", 0.99)

            # ── R1 (prospective): EFE halting over the inference relaxation ──
            # PonderNet adapted to the EM loop: the controller learns *when the
            # relaxation has settled* and halts it (active inference — stop when
            # epistemic value is exhausted). The objective is the Expected Free
            # Energy Σ_t p_t·F_t over the recorded inference steps (F_t = the
            # variational/FEEC free energy the loop is already minimizing), so
            # the policy is trained to put halting mass on the low-free-energy
            # steps. All features are scalars → memory-safe (no per-step logits).
            efe_active = (
                getattr(self.config, "use_efe_controller", False)
                and self.efe_controller is not None
            )
            efe_feats: list = []     # per-step [info_gain, rel_delta, progress] (detached)
            efe_settle: list = []    # per-step free-energy / settle cost (detached scalar)
            z_prev_efe = z_curr.detach()
            efe_inv_dim = 1.0 / math.sqrt(self.config.d_latent)
            efe_cum_halt = 0.0
            efe_cum_continue = 1.0
            z_safe = z_curr   # last finite latent (divergence-guard fallback)

            with torch.no_grad():
                actual_inference_steps = 0
                for t in range(max_inference_steps):
                    actual_inference_steps = t + 1
                    step_res = self._thought_step(
                        z_curr, t, text_emb, key_padding_mask,
                        pre_states, post_states, velocity_curr, stream_states_curr,
                        targets, teacher_log_probs, teacher_top_indices, distill_temp,
                        None, clamped_target, max_inference_steps, prev_certainty,
                        compute_logits=False
                    )

                    (z_curr, logits_t, pre_states, post_states, velocity_curr, stream_states_curr,
                     ce_loss_t, kl_loss_t, certainty_t, _new_hebbian_states, pc_loss_t,
                     hebbian_lr_eff_t, hpc_local_loss_t, hpc_free_energy_t,
                     hpc_mean_precision_t, hpc_mean_error_norm_t) = step_res

                    if certainty_t is not None:
                        prev_certainty = certainty_t

                    # BCM: update sliding threshold EMA from current latent magnitude.
                    # Skip during the R3 meta inner-pass (a throwaway probe): it
                    # also runs this inference loop and would double-update θ_M,
                    # making the threshold track z² faster and weakening the very
                    # homeostat meant to prevent latent runaway.
                    if bcm_active and not self._meta_inner:
                        z_sq_mean = z_curr.detach().float().pow(2).mean()
                        # Finite-guard (sync-free): if the latent is non-finite
                        # (a transient diverged batch), feed the threshold its own
                        # value so the EMA is a no-op — otherwise θ_M latches to
                        # NaN and poisons every later forward, defeating the
                        # NaN-skip optimizer guard's recovery.
                        z_sq_mean = torch.where(
                            torch.isfinite(z_sq_mean), z_sq_mean, self._bcm_threshold
                        )
                        self._bcm_threshold.mul_(bcm_decay).add_(z_sq_mean * (1.0 - bcm_decay))

                    # Epistemic value (latent information gain this step).
                    if efe_active:
                        info_gain = (
                            (z_curr - z_prev_efe).norm(dim=-1).mean() * efe_inv_dim
                        ).float()
                        z_prev_efe = z_curr.detach()

                    # Dynamic termination based on energy stabilization
                    curr_energy = None
                    rel_delta_val = None
                    if self.feec is not None and velocity_curr is not None:
                        curr_energy = self.feec.energy(z_curr, velocity_curr)
                        if prev_energy is not None:
                            rel_delta_val = (
                                (curr_energy - prev_energy).abs() / (curr_energy.abs() + 1e-6)
                            )
                        prev_energy = curr_energy

                    # ── Divergence guard ─────────────────────────────────
                    # The relaxation is the unbounded part of the model (FEEC
                    # velocity / HPC precision feedback over up to
                    # max_inference_steps). If the energy goes non-finite, abort
                    # and fall back to the last finite latent so one bad batch
                    # can't hand NaNs to the learning pass / the R3 meta probe.
                    if curr_energy is not None and not torch.isfinite(curr_energy):
                        z_curr = z_safe
                        break
                    z_safe = z_curr.detach()

                    # ── R1 feature recording + eval early-halt ───────────
                    if efe_active:
                        if curr_energy is not None:
                            settle = curr_energy.detach().float().mean()
                        elif hpc_free_energy_t is not None:
                            settle = hpc_free_energy_t.detach().float().reshape(())
                        else:
                            settle = z_curr.detach().float().pow(2).mean()
                        rd = (
                            rel_delta_val.detach().float().reshape(())
                            if rel_delta_val is not None
                            else torch.ones((), device=device, dtype=torch.float32)
                        )
                        progress = torch.tensor(
                            t / max(max_inference_steps - 1, 1),
                            device=device, dtype=torch.float32,
                        )
                        efe_feats.append(torch.stack([info_gain, rd, progress]))
                        efe_settle.append(settle)
                        # Eval/generate: let the controller end the relaxation early.
                        if not self.training:
                            lam = torch.sigmoid(
                                self.efe_controller.halt_logit(info_gain, rd, progress)
                            ).item()
                            efe_cum_halt += efe_cum_continue * lam
                            efe_cum_continue *= (1.0 - lam)
                            if efe_cum_halt >= getattr(self.config, "efe_halt_threshold", 0.5):
                                break

                    if rel_delta_val is not None and rel_delta_val < energy_tol:
                        break

            result["inference_steps"] = actual_inference_steps

            # ── Diagnostic: stable rank of the equilibrium latent ──────────
            # srank(Z) = ‖Z‖_F² / σ_max²  over the [B·S, d] latent matrix.
            # If a dominant fixed bias (e.g. a saturated/anisotropic fast-weight
            # carry) forces every position along one direction, the per-position
            # latents collapse to ~1 effective dimension → flat, input-independent
            # output → certainty cliff. This metric makes that visible: healthy ≫1,
            # collapsed ≈1. Cheap (power-iteration σ_max, no full SVD).
            with torch.no_grad():
                _Z = z_curr.detach().reshape(-1, z_curr.shape[-1]).float()
                _fro2 = _Z.pow(2).sum()
                _v = torch.randn(_Z.shape[1], device=_Z.device)
                _v = _v / (_v.norm() + 1e-12)
                for _ in range(4):
                    _u = _Z @ _v
                    _u = _u / (_u.norm() + 1e-12)
                    _v = _Z.t() @ _u
                    _v = _v / (_v.norm() + 1e-12)
                _smax2 = (_Z @ _v).pow(2).sum()
                result["latent_srank"] = (_fro2 / _smax2.clamp_min(1e-12)).item()

            # ── Certainty at equilibrium (for NE modulation + logging) ──
            # The inference loop runs with compute_logits=False to save VRAM.
            # One extra no-grad step on z* with compute_logits=True gives the
            # model's output entropy at the settled state — the correct surprise
            # signal for neuromodulation, not a single-step estimate from z_init.
            with torch.no_grad():
                _eq_step = self._thought_step(
                    z_curr, actual_inference_steps, text_emb, key_padding_mask,
                    pre_states, post_states, velocity_curr, stream_states_curr,
                    targets, teacher_log_probs, teacher_top_indices, distill_temp,
                    None, clamped_target, actual_inference_steps + 1, prev_certainty,
                    compute_logits=True
                )
                _certainty_eq = _eq_step[8]   # certainty_t position in return tuple
                if _certainty_eq is not None:
                    result["certainties"] = _certainty_eq.unsqueeze(0)  # [1, B, S]

            # ── Phase 2: Learning (Consolidation) ───────────────────────
            # Execute a single forward pass with gradients enabled.
            # We set the "target" of the predictive coding layers to the
            # detached \mu^* discovered in Phase 1.
            
            # Re-initialize states for the single gradient-tracked pass
            z_learn = z.clone()
            velocity_learn = velocity.clone() if velocity is not None else None
            
            # v1 Memory initial states
            pre_states_learn = []
            post_states_learn = []
            if not self.config.use_matrix_streams:
                pre_states_learn = [layer.memory.pre_history for layer in layers if layer.memory is not None]
                post_states_learn = [layer.memory.post_history for layer in layers if layer.memory is not None]
            
            # Hebbian fast-weights are updated at equilibrium only.
            # Pass the original (freshly-initialised) hebbian_states so
            # the single outer-product captures z* ⊗ a*.
            #
            # R2 Burstprop: the apical credit signal for the consolidation
            # write is the equilibrium PC error discovered during Phase 1
            # (stashed in self._last_pc_errors). High top-down error at
            # equilibrium → more bursting → stronger consolidation at that
            # layer/position.
            burst_eq_errors = (
                getattr(self, "_last_pc_errors", None)
                if getattr(self.config, "use_burstprop", False) else None
            )

            step_res_learn = self._thought_step(
                z_learn, 0, text_emb, key_padding_mask,
                pre_states_learn, post_states_learn, velocity_learn, stream_states,
                targets, teacher_log_probs, teacher_top_indices, distill_temp,
                hebbian_states, clamped_target, 1, None,
                burst_prev_errors=burst_eq_errors,
            )

            (z_learn, logits_t, _, _, velocity_learn, _,
             ce_loss_t, kl_loss_t, certainty_t, new_hebbian_states, pc_loss_t,
             _, hpc_local_loss_t, hpc_free_energy_t,
             _, _) = step_res_learn

            if hpc_free_energy_t is not None:
                result["hpc_free_energy"] = hpc_free_energy_t.detach()

            loss = torch.tensor(0.0, device=device, dtype=dtype)

            # The local loss from the HPC layers represents the true prospective configuration gradient
            if hpc_local_loss_t is not None:
                loss = loss + getattr(self.config, "hpc_local_loss_weight", 1.0) * hpc_local_loss_t
                result["hpc_local_loss"] = hpc_local_loss_t.detach()

            # We can also add the cross-entropy of the single consolidated pass
            if ce_loss_t is not None:
                loss = loss + ce_loss_t.mean()

            # BCM homeostatic penalty: penalize z² exceeding the sliding threshold
            # Prevents runaway excitation during rapid one-shot learning.
            if bcm_active:
                bcm_weight = getattr(self.config, "bcm_loss_weight", 0.1)
                z_sq = z_learn.pow(2).mean()
                bcm_excess = torch.relu(z_sq - self._bcm_threshold.to(z_sq.dtype).detach())
                bcm_penalty = bcm_weight * bcm_excess.pow(2)
                loss = loss + bcm_penalty
                result["bcm_loss"] = bcm_penalty.detach()
                result["bcm_threshold"] = self._bcm_threshold.detach()

            # Amortized auxiliary loss: train the inference network to predict z*.
            # z_curr here is the equilibrium state discovered by Phase 1; we
            # supervise z_hat toward it so the warm-start improves over training.
            if z_hat is not None:
                amortized_weight = getattr(self.config, "amortized_aux_loss_weight", 0.1)
                amortized_loss = F.mse_loss(z_hat, z_curr.detach().to(z_hat.dtype))
                loss = loss + amortized_weight * amortized_loss
                result["amortized_loss"] = amortized_loss.detach()

            # ── R1 (prospective): Expected-Free-Energy halting objective ─
            # Re-run the controller (with grad) on the recorded inference-step
            # features and train its halting distribution p_t to minimise the
            # expected free energy Σ_t p_t·F_t (+ KL to a geometric prior). F_t
            # is normalised per-call to [0,1] so the term is scale-free. The
            # controller thereby learns to halt the relaxation at the
            # low-free-energy (settled) step → fewer inference steps at eval.
            if efe_active and len(efe_feats) > 0:
                feats = torch.stack(efe_feats).to(self.efe_controller.mlp[0].weight.dtype)
                halt_logits = self.efe_controller.mlp(feats).squeeze(-1).float()  # [n]
                lambdas = torch.sigmoid(halt_logits)
                n_steps = halt_logits.shape[0]
                if n_steps > 1:
                    lam = torch.cat([lambdas[:-1], lambdas.new_ones(1)])
                else:
                    lam = lambdas.new_ones(1)
                one_minus = (1.0 - lam).clamp(min=1e-6)
                cont = torch.cat([
                    one_minus.new_ones(1), torch.cumprod(one_minus, dim=0)[:-1]
                ])
                p = lam * cont
                p = p / p.sum().clamp(min=1e-6)
                settle = torch.stack(efe_settle).float()           # [n]
                s_min = settle.min()
                s_rng = (settle.max() - s_min).clamp(min=1e-6)
                settle_norm = (settle - s_min) / s_rng
                efe_recon = (p * settle_norm).sum()
                halt_prior = float(getattr(self.config, "efe_halt_prior", 0.1))
                t_idx = torch.arange(n_steps, device=device, dtype=torch.float32)
                g = halt_prior * (1.0 - halt_prior) ** t_idx
                g = g / g.sum().clamp(min=1e-6)
                efe_kl = (
                    p * (torch.log(p.clamp(min=1e-6)) - torch.log(g.clamp(min=1e-6)))
                ).sum()
                kl_w = float(getattr(self.config, "efe_ponder_kl_weight", 0.01))
                efe_w = float(getattr(self.config, "efe_weight", 0.1))
                loss = loss + (efe_w * (efe_recon + kl_w * efe_kl)).to(loss.dtype)
                result["efe_loss"] = efe_recon.detach()
                result["efe_ponder_kl"] = efe_kl.detach()
                with torch.no_grad():
                    result["efe_expected_halt"] = float((p * t_idx).sum().item()) + 1.0

            result["loss"] = loss
            if logits_t is not None:
                result["logits"] = logits_t

            # R10: expose the grad-carrying consolidation latent for REM dreaming;
            # add the adversarial REM loss in-graph during sleep replay.
            if getattr(self.config, "use_rem_dreaming", False) and not self._meta_inner:
                _rem_pool = z_learn.mean(dim=1)   # [B, d_latent]
                result["rem_latent_pool"] = _rem_pool
                if self._rem_sleep_active and self.rem_dreamer is not None:
                    result["loss"] = result["loss"] + self.rem_loss(_rem_pool)

        else:
            # ── Legacy BPTT Thought Loop (Fallback) ─────────────────────
            all_logits = []
            all_certainties = []
            per_tick_ce_losses = []
            all_hpc_fe: list = []

            prev_certainty = None
            # R2 Burstprop: previous tick's per-layer PC errors (apical signal).
            burst_prev_errors = None
            burstprop_on = getattr(self.config, "use_burstprop", False)

            # ── R1: EFE / PonderNet controller bookkeeping ──────────────
            efe_active = (
                getattr(self.config, "use_efe_controller", False)
                and self.efe_controller is not None
            )
            efe_halt_logits: list = []
            efe_ambiguities: list = []
            efe_epistemics: list = []
            z_prev_efe = z_curr.detach()
            efe_inv_dim = 1.0 / math.sqrt(self.config.d_latent)
            efe_cum_halt = 0.0       # cumulative halt prob (eval early-exit)
            efe_cum_continue = 1.0   # running Π(1 − λ)

            for t in range(T):
                if use_per_step_ckpt:
                    step_res = torch_checkpoint.checkpoint(
                        self._thought_step,
                        z_curr, t, text_emb, key_padding_mask,
                        pre_states, post_states, velocity_curr, stream_states_curr,
                        targets, teacher_log_probs, teacher_top_indices, distill_temp,
                        hebbian_states_curr, None, T, prev_certainty,
                        True, burst_prev_errors,
                        use_reentrant=False,
                    )
                else:
                    step_res = self._thought_step(
                        z_curr, t, text_emb, key_padding_mask,
                        pre_states, post_states, velocity_curr, stream_states_curr,
                        targets, teacher_log_probs, teacher_top_indices, distill_temp,
                        hebbian_states_curr, None, T, prev_certainty,
                        burst_prev_errors=burst_prev_errors,
                    )

                (z_curr, logits_t, pre_states, post_states, velocity_curr, stream_states_curr,
                 ce_loss_t, kl_loss_t, certainty_t, new_hebbian_states, pc_loss_t,
                 hebbian_lr_eff_t, hpc_local_loss_t, hpc_free_energy_t,
                 hpc_mean_precision_t, hpc_mean_error_norm_t) = step_res

                # R2: hand this tick's PC errors to the next tick's burst signal.
                # Read outside any checkpoint region → checkpoint-safe.
                if burstprop_on:
                    burst_prev_errors = getattr(self, "_last_pc_errors", None)

                if certainty_t is not None:
                    prev_certainty = certainty_t
                if ce_loss_t is not None:
                    per_tick_ce_losses.append(ce_loss_t)
                if logits_t is not None:
                    all_logits.append(logits_t)
                if certainty_t is not None:
                    all_certainties.append(certainty_t)

                if new_hebbian_states and any(h is not None for h in new_hebbian_states):
                    hebbian_states_curr = new_hebbian_states
                if hpc_free_energy_t is not None:
                    all_hpc_fe.append(hpc_free_energy_t.detach())

                # ── R1: EFE halting features (+ eval early-exit) ────────
                if efe_active:
                    with torch.no_grad():
                        if certainty_t is not None:
                            ambiguity = (1.0 - certainty_t).clamp(0.0, 1.0).mean().float()
                        else:
                            ambiguity = torch.zeros((), device=device, dtype=torch.float32)
                        info_gain = (
                            (z_curr - z_prev_efe).norm(dim=-1).mean() * efe_inv_dim
                        ).float()
                        progress = torch.tensor(
                            t / max(T - 1, 1), device=device, dtype=torch.float32
                        )
                    z_prev_efe = z_curr.detach()
                    halt_logit_t = self.efe_controller.halt_logit(
                        ambiguity, info_gain, progress
                    )
                    efe_halt_logits.append(halt_logit_t)
                    efe_ambiguities.append(ambiguity)
                    efe_epistemics.append(info_gain)

                    # Training always runs full T (the PonderNet distribution
                    # must cover every tick); only eval / generate early-exits.
                    if not self.training:
                        lam = torch.sigmoid(halt_logit_t).item()
                        efe_cum_halt += efe_cum_continue * lam
                        efe_cum_continue *= (1.0 - lam)
                        if efe_cum_halt >= getattr(self.config, "efe_halt_threshold", 0.5):
                            break

            # ── Loss assembly ────────────────────────────────────────────
            if efe_active and per_tick_ce_losses:
                n_ticks = len(per_tick_ce_losses)
                halt_logits = torch.stack(efe_halt_logits[:n_ticks])
                lambdas = torch.sigmoid(halt_logits)
                # Force the final tick to absorb the remaining mass (λ_last = 1).
                if n_ticks > 1:
                    lam = torch.cat([lambdas[:-1], lambdas.new_ones(1)])
                else:
                    lam = lambdas.new_ones(1)
                one_minus = (1.0 - lam).clamp(min=1e-6)
                cont = torch.cat([
                    one_minus.new_ones(1), torch.cumprod(one_minus, dim=0)[:-1]
                ])
                p = lam * cont
                p = p / p.sum().clamp(min=1e-6)

                ce_vec = torch.stack(per_tick_ce_losses)
                recon = (p * ce_vec).sum()

                # KL(p ‖ Geometric(halt_prior)) over the n ticks.
                halt_prior = float(getattr(self.config, "efe_halt_prior", 0.1))
                t_idx = torch.arange(n_ticks, device=device, dtype=torch.float32)
                g = halt_prior * (1.0 - halt_prior) ** t_idx
                g = g / g.sum().clamp(min=1e-6)
                kl = (
                    p * (torch.log(p.clamp(min=1e-6)) - torch.log(g.clamp(min=1e-6).to(p.dtype)))
                ).sum()

                # EFE shaping: pull halting mass toward low (ambiguity − epistemic).
                amb_vec = torch.stack(efe_ambiguities[:n_ticks]).to(p.dtype)
                epi_vec = torch.stack(efe_epistemics[:n_ticks]).to(p.dtype)
                efe_term = (p * (amb_vec - epi_vec)).sum()

                kl_w = float(getattr(self.config, "efe_ponder_kl_weight", 0.01))
                efe_w = float(getattr(self.config, "efe_weight", 0.1))
                result["loss"] = recon + kl_w * kl + efe_w * efe_term
                result["efe_ponder_kl"] = kl.detach()
                result["efe_loss"] = efe_term.detach()
                with torch.no_grad():
                    result["efe_expected_halt"] = float((p * t_idx).sum().item()) + 1.0
            elif per_tick_ce_losses:
                per_tick_loss_tensor = torch.stack(per_tick_ce_losses)
                result["loss"] = per_tick_loss_tensor.mean()
            if all_logits:
                result["logits"] = all_logits[-1]
                result["all_logits"] = all_logits
            if all_certainties:
                result["certainties"] = torch.stack(all_certainties, dim=0)
            if all_hpc_fe:
                result["hpc_free_energy"] = torch.stack(all_hpc_fe).mean()

            # R10: expose the grad-carrying final latent for REM dreaming;
            # add the adversarial REM loss in-graph during sleep replay.
            if getattr(self.config, "use_rem_dreaming", False) and not self._meta_inner:
                _rem_pool = z_curr.mean(dim=1)   # [B, d_latent]
                result["rem_latent_pool"] = _rem_pool
                if self._rem_sleep_active and self.rem_dreamer is not None:
                    result["loss"] = result["loss"] + self.rem_loss(_rem_pool)

        # ── Update cross-batch Hebbian carry-over (sleep consolidation) ──
        # Persist the mean fast-weight state across batches so the next
        # forward pass warm-starts from accumulated associations rather
        # than zeros. Mean over (B, S) gives a compact [m, n] summary
        # that broadcasts correctly to any future (B', S') shape.
        if (getattr(self.config, "use_sleep_consolidation", False)
                and self.config.use_hebbian_synapse and not self._meta_inner):
            # In the BPTT path, hebbian_states_curr is updated each tick and
            # ends up holding the final-tick Hebbian states.
            # In the prospective path, hebbian_states_curr is never updated
            # (still holds the initial zeros/carry-over), so we prefer the
            # new_hebbian_states from the single learning pass instead.
            # Both variables are always defined by this point (set via
            # step_res destructuring in both branches of the if/else).
            if prospective_active:
                final_heb = new_hebbian_states
            else:
                final_heb = hebbian_states_curr
            if final_heb is not None:
                # The carry is an integrator: forward() warm-starts M from it and
                # re-saves mean(M_final) ≈ carry + ΔM, so without a bound it grows
                # without limit across (especially frequent) sleeps — saturating
                # the output distribution (certainty collapse) and eventually
                # diverging to NaN. Clip each layer's carry to a max Frobenius
                # norm (0 = off). feec_state_clip bounds z; this bounds M.
                _carry_clip = getattr(self.config, "sleep_carry_max_norm", 0.0)
                carry = []
                for h in final_heb:
                    if h is not None:
                        # Mean over (batch, seq) → [m, n] on CPU
                        c = h.detach().float().mean(dim=(0, 1)).cpu()
                        if _carry_clip and _carry_clip > 0.0:
                            n = c.norm()
                            if n > _carry_clip:
                                c = c * (_carry_clip / n)
                        carry.append(c)
                    else:
                        carry.append(None)
                self._hebbian_carry = carry
                norms = [c.norm().item() for c in carry if c is not None]
                if norms:
                    result["hebbian_carry_norm"] = sum(norms) / len(norms)

        # Expose per-layer HPC precision for precision-weighted gradient scaling.
        # Updated by _thought_step whenever HPC layers are active; available in
        # both prospective-config and BPTT paths (reflects the final thought tick).
        if hasattr(self, "_last_per_layer_precision"):
            result["hpc_per_layer_precision"] = list(self._last_per_layer_precision)

        # SWIL semantic vector: mean-pooled final-tick latent for cosine-similarity
        # replay sampling.  z_curr holds the final equilibrium state (prospective
        # path) or the last-tick latent (BPTT path) — both are meaningful semantic
        # summaries of the processed sequence.  Optionally truncated to
        # swil_embed_dim for storage efficiency.
        if getattr(self.config, "use_swil", False) and not self._meta_inner:
            embed_dim = getattr(self.config, "swil_embed_dim", 64)
            _pool = z_curr.detach().float().mean(dim=1)   # [B, d_latent]
            if embed_dim > 0 and embed_dim < _pool.shape[-1]:
                _pool = _pool[..., :embed_dim]
            result["final_latent_pool"] = _pool.cpu()     # [B, d] on CPU

        # ── Episodic DND: write high-confidence latents ──────────────────────
        # After a normal forward pass, cache the final latent when the model is
        # sufficiently certain. Future calls with the same (or near-identical)
        # input will bypass the thought loop and retrieve this cached latent.
        if self.dnd is not None and not self._meta_inner:
            _certs = result.get("certainties")
            if _certs is not None:
                with torch.no_grad():
                    _mean_cert = float(_certs.float().mean().item())
                    if _mean_cert >= getattr(self.config, "dnd_write_confidence", 0.9):
                        _key_dim = getattr(self.config, "dnd_key_dim", 64)
                        _kpool = text_emb.detach().float().mean(dim=1)  # [B, d_model]
                        if _key_dim < _kpool.shape[-1]:
                            _kpool = _kpool[..., :_key_dim]
                        _vpool = z_curr.detach().float().mean(dim=1)    # [B, d_latent]
                        for b in range(B):
                            self.dnd.push(_kpool[b], _vpool[b])
                        result["dnd_written"] = True
            result["dnd_hit"] = False
            result["dnd_max_sim"] = self.dnd._last_max_sim

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len:]
            result = self(idx, max_thought_steps=self.config.max_thought_steps)
            logits = result["logits"][:, -1, :]

            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def sleep_consolidation_step(
        self,
        episodic_buffer: list,
        device,
        amp_dtype: torch.dtype,
        replay_batch_size: int = 4,
    ) -> dict:
        """Run one sleep consolidation pass over stored episodic sequences.

        Samples sequences from the episodic buffer and runs a full forward
        pass with the persistent Hebbian carry-over state warm-starting the
        fast weights. The resulting LM loss — scaled by `sleep_loss_weight`
        — drives upward distillation of episodic fast-weight knowledge into
        the slow (gradient-trained) backbone weights, mirroring the role of
        slow-wave sleep in biological memory consolidation.

        The carry-over state is updated as a side-effect of the forward pass,
        so replayed sequences also contribute to the running associative memory.

        Args:
            episodic_buffer: List of [S] int64 CPU tensors (input_ids from
                past wake-phase batches).
            device: Training device (same as the model).
            amp_dtype: dtype for torch.amp.autocast.
            replay_batch_size: How many sequences to sample per replay pass.

        Returns:
            result dict with "loss" already scaled by sleep_loss_weight.
            Caller should call result["loss"].backward() followed by
            optimizer.step() (with zeroed gradients beforehand).
        """
        import random

        if not episodic_buffer:
            return {"loss": torch.tensor(0.0, device=device)}

        n = min(replay_batch_size, len(episodic_buffer))
        sampled = random.sample(list(episodic_buffer), n)
        x = torch.stack(sampled, dim=0).to(device)  # [n, S]
        # Targets: standard next-token prediction (shift by 1, wrap last)
        y = torch.cat([x[:, 1:], x[:, :1]], dim=1)

        device_type = str(device).split(":")[0]
        with torch.amp.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=(amp_dtype != torch.float32),
        ):
            result = self(x, targets=y)

        sleep_weight = getattr(self.config, "sleep_loss_weight", 0.3)
        result["loss"] = result["loss"] * sleep_weight
        return result

    def rem_loss(self, real_latents: torch.Tensor) -> torch.Tensor:
        """R10 — adversarial REM dreaming loss (already scaled).

        Meant to be added to the sleep-consolidation loss before backward, so
        the GAN + backbone train in the same allreduced step.

        Args:
            real_latents: [n, d_latent] grad-carrying pooled latents from a
                replay forward (``result["rem_latent_pool"]``).
        Returns:
            Scalar loss scaled by rem_loss_weight (0 if the dreamer is absent).
        """
        if self.rem_dreamer is None or real_latents is None:
            ref = real_latents if real_latents is not None else self.z0
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
        w = getattr(self.config, "rem_loss_weight", 0.1)
        lam = getattr(self.config, "rem_grl_lambda", 1.0)
        loss, acc = self.rem_dreamer.loss(real_latents, grl_lambda=lam)
        self._last_rem_disc_acc = float(acc.item())
        return w * loss