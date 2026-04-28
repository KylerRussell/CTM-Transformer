"""
Thought Layer — Single processing block iterated T times during the thought loop.

Each thought step performs:
1. Query Generation:  Project synchronization state → attention query
2. Cross-Attention:   Query attends to text embeddings (K, V from input tokens)
                      WITH CAUSAL MASK — position i can only attend to positions ≤ i
3. Synapse Model:     Mix attention output with previous neuron state → pre-activations
4. NLM Processing:    Per-neuron MLPs process pre-activation history → post-activations
5. Sync Update:       Recompute synchronization from updated post-activation history

CTM-v2 ADDITIONS:
  - Matrix-valued residual streams (replaces NLM + Sync when enabled)
  - DSSA attention (replaces O(N²) cross-attention when enabled)
  - Accelerated attention via Triton kernels (when available)

CRITICAL DESIGN NOTE:
  The latent state z has shape [B, S, d_latent] — one independent state per sequence
  position. The cross-attention uses a causal mask so that position i's query can only
  attend to text keys at positions 0..i. This prevents target leakage (future tokens
  informing past predictions) while preserving autoregressive correctness.

  For NLM and memory operations, we flatten B*S into the batch dimension since each
  position's NLM and memory are independent. For attention, we reshape back to
  [B, S, ...] to apply proper multi-head attention with causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ctm_transformer.nlm import NeuronLevelModels
from ctm_transformer.memory import TemporalMemory, SynchronizationComputer
from ctm_transformer.engram import EngramGate


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
        engram_enabled: If True, this layer fuses Engram memory into attn_out.
            The Engram K/V are projected by the parent model (not here) and
            passed in via `engram_kv` at forward time.
        engram_conv_kernel/dilation/use_conv: see EngramGate.
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
        engram_enabled: bool = False,
        engram_use_conv: bool = True,
        engram_conv_kernel: int = 4,
        engram_conv_dilation: int = 3,
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
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_matrix_streams = use_matrix_streams
        self.use_dssa = use_dssa
        self.use_triton_attention = use_triton_attention

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads

        # ── v2: Matrix-Valued Residual Stream (replaces Sync+NLM) ────────
        if use_matrix_streams:
            from ctm_transformer.matrix_stream import MatrixResidualStream
            self.stream = MatrixResidualStream(
                d_latent=d_latent,
                n_streams=n_streams,
                d_model=d_model,
                gating=stream_gating,
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
            )

            # ── Memory Buffers (v1 path) ────────────────────────────────
            self.memory = TemporalMemory(d_latent=d_latent, history_len=history_len)

        # ── v2: DSSA Cross-Attention (replaces manual Q/K/V) ────────────
        if use_dssa:
            from ctm_transformer.dssa import DualSpaceSparseAttention
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
        self.synapse = nn.Sequential(
            nn.Linear(d_model + d_latent, d_latent * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.synapse_gate = nn.Linear(d_model + d_latent, d_latent)

        # ── Post-NLM Layer Norm ─────────────────────────────────────────
        self.post_norm = nn.LayerNorm(d_latent)

        # ── Engram Gate (optional) ──────────────────────────────────────
        if engram_enabled:
            self.engram_gate = EngramGate(
                d_query=d_model,
                d_out=d_model,
                kernel_size=engram_conv_kernel,
                dilation=engram_conv_dilation,
                use_conv=engram_use_conv,
            )
        else:
            self.engram_gate = None

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
    ) -> torch.Tensor:
        """Standard O(N²) cross-attention with causal masking.

        Args:
            queries: [B, S, d_model]
            text_keys: [B, S, d_model]
            text_values: [B, S, d_model]
            key_padding_mask: [B, S] or None

        Returns:
            attn_out: [B, S, d_model]
        """
        B, S, D = queries.shape

        q = self.q_norm(queries)
        k = self.k_proj(text_keys)
        v = self.v_proj(text_values)

        # Check if we should use accelerated attention
        if self.use_triton_attention:
            from ctm_transformer.triton_kernels import accelerated_causal_attention
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
            return attn_out

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

        return attn_out

    def forward(
        self,
        text_keys: torch.Tensor,
        text_values: torch.Tensor,
        prev_state: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        engram_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        stream_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Execute one thought step with per-position states and causal masking.

        Args:
            text_keys:   [batch, seq_len, d_model] — text embeddings for K projection.
            text_values: [batch, seq_len, d_model] — text embeddings for V projection.
            prev_state:  [batch, seq_len, d_latent] — per-position neuron states z_{t-1}.
            key_padding_mask: [batch, seq_len] — True for padded positions.
            engram_kv: optional (k, v) tuple, each [batch, seq_len, d_model].
            stream_state: [batch, seq_len, n_streams, d_latent] — matrix stream state (v2).

        Returns:
            new_state:   [batch, seq_len, d_latent] — updated per-position states z_t.
            sync_repr:   [batch*seq_len, sync_dim] — sync repr (v1) or dummy (v2).
            new_stream_state: [batch, seq_len, n_streams, d_latent] or None.
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
            # v2 DSSA path
            attn_out = self.dssa(queries, text_keys, causal=True)
        else:
            # Standard attention (possibly Triton-accelerated)
            attn_out = self._standard_attention(
                queries, text_keys, text_values, key_padding_mask
            )

        # ── Step 2.5: Engram Fusion (optional) ──────────────────────────
        if self.engram_gate is not None and engram_kv is not None:
            engram_k, engram_v = engram_kv
            engram_out = self.engram_gate(attn_out, engram_k, engram_v)
            attn_out = attn_out + engram_out

        # Flatten attention output
        attn_flat = attn_out.reshape(BS, D)

        # ── Step 3: Synapse Model ───────────────────────────────────────
        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            collapsed = self.stream.collapse(stream_state)  # [B, S, d_latent]
            prev_flat = collapsed.reshape(BS, -1)
        else:
            prev_flat = prev_state.reshape(BS, -1)

        synapse_input = torch.cat([attn_flat, prev_flat], dim=1)
        synapse_input = self.synapse_norm(synapse_input)

        gate = torch.sigmoid(self.synapse_gate(synapse_input))
        candidate = self.synapse(synapse_input)
        pre_activations = gate * candidate + (1 - gate) * prev_flat

        # ── Step 4: Post-processing ─────────────────────────────────────
        new_stream_state = None

        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            # v2 path: update matrix streams (replaces NLM)
            post_activations = self.post_norm(pre_activations)
            new_state_flat = post_activations
            # Update stream state
            layer_output = post_activations.reshape(B, S, -1)
            new_stream_state = self.stream.mix_post(stream_state, layer_output)
        else:
            # v1 path: NLM processing
            self.memory.push_pre(pre_activations)
            pre_history = self.memory.get_pre_history()
            post_activations = self.nlm(pre_history)
            self.memory.push_post(post_activations)
            new_state_flat = self.post_norm(post_activations)

            # Recompute sync for output
            updated_post_history = self.memory.get_post_history()
            sync_repr = self.sync_computer.compute(updated_post_history)

        # Reshape back to [B, S, d_latent]
        new_state = new_state_flat.reshape(B, S, -1)

        return new_state, sync_repr, new_stream_state