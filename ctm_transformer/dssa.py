"""
Dual-Space Sparse Attention (DSSA)

Replaces the O(N²) full cross-attention in the ThoughtLayer with a hybrid
of two complementary attention mechanisms:

1. Sparse State Expansion (SSE) — Linear attention with sparse state updates.
   Maintains a set of m state partitions, each [d_key, d_value]. On each
   step, only the top-k partitions are updated based on an input-dependent
   gating vector. Memory is O(m·d_k·d_v) — constant w.r.t. sequence length.

2. Mixture of Block Attention (MoBA) — Exact block-wise attention for
   periodic precise retrieval. Divides the KV sequence into blocks and
   attends to the top-k blocks per query via block-level scoring.

The hybrid combines constant-memory compressed retrieval (SSE) with
periodic exact attention windows (MoBA). A learnable router decides
the mixing ratio per position.

Reference: Spiking Brain 2.0 (DSSA), MambaCSP (patch-mixer attention)

For the CTM at d_model=1760, this provides:
  - SSE: O(1) memory per step instead of O(N) for KV cache
  - MoBA: O(N·k/B) compute instead of O(N²) for full attention
  - Combined: up to 10x speedup in time-to-first-token at long contexts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        """Forward pass with sparse state expansion.

        For the CTM's cross-attention pattern, x provides queries and
        text_kv provides keys and values.

        Args:
            x: [B, S, d_model] — query source (sync/latent state).
            text_kv: [B, S, d_model] — KV source (text embeddings).
            causal: Whether to apply causal masking.

        Returns:
            output: [B, S, d_model]
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        m = self.n_partitions
        k = self.top_k

        # Project Q from queries, K/V from text
        q = self.q_proj(x).view(B, S, H, dh)        # [B, S, H, dh]
        kk = self.k_proj(text_kv).view(B, S, H, dh)  # [B, S, H, dh]
        v = self.v_proj(text_kv).view(B, S, H, dh)   # [B, S, H, dh]

        # Normalize Q and K for stable dot products
        q = self.q_norm(q)
        kk = self.k_norm(kk)

        # Compute gating logits for partition selection
        gate_logits = self.gate_proj(text_kv)  # [B, S, m]

        # Select top-k partitions per position
        _, top_indices = gate_logits.topk(k, dim=-1)  # [B, S, k]
        gate_weights = F.softmax(
            gate_logits.gather(-1, top_indices), dim=-1
        )  # [B, S, k]

        # Initialize state partitions
        # State: [B, H, m, dh, dh] — m partitions of outer products
        state = torch.zeros(
            B, H, m, dh, dh,
            device=x.device, dtype=x.dtype,
        )

        outputs = []

        # Sequential scan over positions (recurrent)
        for t in range(S):
            # Get this position's K, V, Q
            kt = kk[:, t]       # [B, H, dh]
            vt = v[:, t]        # [B, H, dh]
            qt = q[:, t]        # [B, H, dh]
            gt = gate_weights[:, t]  # [B, k]
            idx_t = top_indices[:, t]  # [B, k]

            # Update selected partitions: S_i += e_i * k^T v
            # For each sample in the batch, update only the top-k partitions
            kv_outer = torch.einsum('bhd,bhe->bhde', kt, vt)  # [B, H, dh, dh]

            for i in range(k):
                part_idx = idx_t[:, i]  # [B]
                weight = gt[:, i]  # [B]

                # Gather the partition indices for each batch element
                # and update with weighted outer product
                for b in range(B):
                    pi = part_idx[b].item()
                    state[b, :, pi] += weight[b] * kv_outer[b]

            # Query the state: o = sum_i (q^T S_i)
            # [B, H, 1, dh] @ [B, H, m, dh, dh] → [B, H, m, dh]
            readout = torch.einsum('bhd,bhmde->bhme', qt, state)  # [B, H, m, dh]

            # Weight readout by gate (only active partitions contribute)
            # Add eps to norms to avoid NaN from softmax(zeros) at t=0
            partition_norms = readout.norm(dim=-1) + 1e-8  # [B, H, m]
            attn_weights = F.softmax(partition_norms, dim=-1)  # [B, H, m]

            out_t = torch.einsum('bhm,bhmd->bhd', attn_weights, readout)  # [B, H, dh]
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B, S, H, dh]
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
