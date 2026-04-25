"""
Thought Layer — Single processing block iterated T times during the thought loop.

Each thought step performs:
1. Query Generation:  Project synchronization state → attention query
2. Cross-Attention:   Query attends to text embeddings (K, V from input tokens)
                      WITH CAUSAL MASK — position i can only attend to positions ≤ i
3. Synapse Model:     Mix attention output with previous neuron state → pre-activations
4. NLM Processing:    Per-neuron MLPs process pre-activation history → post-activations
5. Sync Update:       Recompute synchronization from updated post-activation history

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
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads

        # ── Synchronization Computer ────────────────────────────────────
        self.sync_computer = SynchronizationComputer(
            d_latent=d_latent,
            history_len=history_len,
            method=sync_method,
            rank=sync_rank,
        )
        sync_dim = self.sync_computer.output_dim

        # ── Query Generation from Synchronization State ─────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(sync_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Cross-Attention ─────────────────────────────────────────────
        # Queries come from per-position sync states.
        # Keys and Values from text embeddings.
        # CAUSAL MASK: query at position i attends only to keys 0..i.
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

        # ── Neuron-Level Models ─────────────────────────────────────────
        self.nlm = NeuronLevelModels(
            d_latent=d_latent,
            history_len=history_len,
            nlm_hidden_dim=nlm_hidden_dim,
            nlm_groups=nlm_groups,
            dropout=dropout,
        )

        # ── Memory Buffers ──────────────────────────────────────────────
        self.memory = TemporalMemory(d_latent=d_latent, history_len=history_len)

        # ── Post-NLM Layer Norm ─────────────────────────────────────────
        self.post_norm = nn.LayerNorm(d_latent)

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize memory buffers for a new sequence/batch."""
        self.memory.reset(batch_size, device, dtype)

    def forward(
        self,
        text_keys: torch.Tensor,
        text_values: torch.Tensor,
        prev_state: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute one thought step with per-position states and causal masking.

        Args:
            text_keys:   [batch, seq_len, d_model] — text embeddings for K projection.
            text_values: [batch, seq_len, d_model] — text embeddings for V projection.
            prev_state:  [batch, seq_len, d_latent] — per-position neuron states z_{t-1}.
            key_padding_mask: [batch, seq_len] — True for padded positions.

        Returns:
            new_state:   [batch, seq_len, d_latent] — updated per-position states z_t.
            sync_repr:   [batch*seq_len, sync_dim] — sync repr (flattened for output head).
        """
        B, S, D = text_keys.shape
        BS = B * S

        # ── Flatten B*S for per-position NLM/memory operations ──────────
        # prev_state: [B, S, d_latent] → [B*S, d_latent]
        prev_flat = prev_state.reshape(BS, -1)

        # ── Step 1: Query Generation from Synchronization ───────────────
        # Memory operates on flat B*S dimension (each position independent)
        post_history = self.memory.get_post_history()  # [B*S, H, d_latent]
        sync_repr = self.sync_computer.compute(post_history)  # [B*S, sync_dim]

        # Project sync → per-position queries
        queries_flat = self.query_proj(sync_repr)  # [B*S, d_model]

        # ── Step 2: Causally-Masked Cross-Attention ─────────────────────
        # Reshape queries back to [B, S, d_model] for proper attention
        queries = queries_flat.reshape(B, S, D)  # [B, S, d_model]

        q = self.q_norm(queries)                                # [B, S, d_model]
        k = self.k_proj(text_keys)                              # [B, S, d_model]
        v = self.v_proj(text_values)                            # [B, S, d_model]

        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, S, head_dim]
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, S, head_dim]
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, S, head_dim]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, heads, S, S]

        # ── CAUSAL MASK ─────────────────────────────────────────────────
        # Position i can only attend to positions 0..i (upper triangle = -inf)
        causal_mask = torch.triu(
            torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
        )  # [S, S] — True above diagonal (future positions)
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )  # [B, heads, S, S]

        # Padding mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)                          # [B, heads, S, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)    # [B, S, d_model]
        attn_out = self.attn_out_proj(attn_out)                           # [B, S, d_model]

        # Flatten attention output back to [B*S, d_model]
        attn_flat = attn_out.reshape(BS, D)

        # ── Step 3: Synapse Model ───────────────────────────────────────
        # All subsequent ops are per-position (flattened B*S)
        synapse_input = torch.cat([attn_flat, prev_flat], dim=1)  # [B*S, d_model + d_latent]
        synapse_input = self.synapse_norm(synapse_input)

        gate = torch.sigmoid(self.synapse_gate(synapse_input))    # [B*S, d_latent]
        candidate = self.synapse(synapse_input)                   # [B*S, d_latent]
        pre_activations = gate * candidate + (1 - gate) * prev_flat  # [B*S, d_latent]

        # ── Step 4: NLM Processing ──────────────────────────────────────
        self.memory.push_pre(pre_activations)
        pre_history = self.memory.get_pre_history()  # [B*S, history_len, d_latent]
        post_activations = self.nlm(pre_history)     # [B*S, d_latent]

        # ── Step 5: Update Post-Activation History ──────────────────────
        self.memory.push_post(post_activations)

        new_state_flat = self.post_norm(post_activations)  # [B*S, d_latent]

        # Recompute sync for output (using updated history)
        updated_post_history = self.memory.get_post_history()
        output_sync = self.sync_computer.compute(updated_post_history)  # [B*S, sync_dim]

        # Reshape back to [B, S, d_latent]
        new_state = new_state_flat.reshape(B, S, -1)

        return new_state, output_sync
