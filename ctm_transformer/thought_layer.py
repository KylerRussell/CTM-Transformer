"""
Thought Layer — Single processing block iterated T times during the thought loop.

Each thought step performs:
1. Query Generation:  Project synchronization state → attention query
2. Cross-Attention:   Query attends to text embeddings (K, V from input tokens)
3. Synapse Model:     Mix attention output with previous neuron state → pre-activations
4. NLM Processing:    Per-neuron MLPs process pre-activation history → post-activations
5. Sync Update:       Recompute synchronization from updated post-activation history

Conceptual mapping from Living-Brain engine_torch.py:
- engine_torch settle loop:
    basal_drive = conductance(W_basal @ rho + input)    →  cross-attention + synapse model
    apical_drive = conductance(W_apical @ rho)          →  (folded into synapse residual)
    rho = get_soma(basal, apical, cahva, ...)           →  NLM processing
    repeat until convergence                            →  thought loop for T steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ctm_transformer.nlm import NeuronLevelModels
from ctm_transformer.memory import TemporalMemory, SynchronizationComputer


class ThoughtLayer(nn.Module):
    """
    A single thought processing block. Multiple ThoughtLayers can be stacked,
    and the entire stack is iterated T times in the thought loop.

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
        # Projects the flattened sync representation into attention queries.
        # S_action_t → q_t = W_in @ flatten(S_action_t)
        self.query_proj = nn.Sequential(
            nn.Linear(sync_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Cross-Attention ─────────────────────────────────────────────
        # Queries come from internal state (sync), Keys and Values from text.
        # This is the fundamental shift: the model decides what text to read
        # based on its current thought state, not its position.
        self.q_norm = nn.LayerNorm(d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # ── Synapse Model ───────────────────────────────────────────────
        # Mixes cross-attention output with previous neuron state to produce
        # pre-activations. Acts like the transformer's residual connections
        # and linear projections combined with recurrent state mixing.
        #
        # Conceptual parallel to engine_torch:
        #   dv_basal = G_L*(E_L - V) + g_E*(E_E - V) + g_I*(E_I - V) + bias + input
        # Here: pre_act = W_syn @ [attn_out; prev_state] + bias
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
        Execute one thought step.

        Args:
            text_keys:   [batch, seq_len, d_model] — text embeddings for K projection.
            text_values: [batch, seq_len, d_model] — text embeddings for V projection.
            prev_state:  [batch, d_latent] — previous neuron state (z_{t-1}).
            key_padding_mask: [batch, seq_len] — True for padded positions.

        Returns:
            new_state:     [batch, d_latent] — updated neuron state (z_t).
            sync_repr:     [batch, sync_dim] — synchronization representation for output.
        """
        B, S, D = text_keys.shape

        # ── Step 1: Query Generation from Synchronization ───────────────
        # Compute sync from post-activation history
        post_history = self.memory.get_post_history()  # [B, H, d_latent]
        sync_repr = self.sync_computer.compute(post_history)  # [B, sync_dim]

        # Project sync → attention queries
        queries = self.query_proj(sync_repr)  # [B, d_model]
        queries = queries.unsqueeze(1)        # [B, 1, d_model] (single query per thought step)

        # ── Step 2: Cross-Attention ─────────────────────────────────────
        # Q from internal state, K and V from text embeddings
        q = self.q_norm(queries)                                # [B, 1, d_model]
        k = self.k_proj(text_keys)                              # [B, S, d_model]
        v = self.v_proj(text_values)                            # [B, S, d_model]

        # Reshape for multi-head attention
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)     # [B, heads, 1, head_dim]
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)     # [B, heads, S, head_dim]
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)     # [B, heads, S, head_dim]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale       # [B, heads, 1, S]

        if key_padding_mask is not None:
            # Expand mask: [B, S] → [B, 1, 1, S]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)                          # [B, heads, 1, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, D)    # [B, 1, d_model]
        attn_out = self.attn_out_proj(attn_out).squeeze(1)                # [B, d_model]

        # ── Step 3: Synapse Model ───────────────────────────────────────
        # Concatenate attention output with previous state
        synapse_input = torch.cat([attn_out, prev_state], dim=1)  # [B, d_model + d_latent]
        synapse_input = self.synapse_norm(synapse_input)

        # Gated residual synapse (like GRU-style gating)
        gate = torch.sigmoid(self.synapse_gate(synapse_input))    # [B, d_latent]
        candidate = self.synapse(synapse_input)                   # [B, d_latent]
        pre_activations = gate * candidate + (1 - gate) * prev_state  # [B, d_latent]

        # ── Step 4: NLM Processing ──────────────────────────────────────
        # Push pre-activations into the FIFO buffer
        self.memory.push_pre(pre_activations)

        # Process temporal history through per-neuron MLPs
        pre_history = self.memory.get_pre_history()  # [B, history_len, d_latent]
        post_activations = self.nlm(pre_history)     # [B, d_latent]

        # ── Step 5: Update Post-Activation History ──────────────────────
        self.memory.push_post(post_activations)

        # Layer norm on the new state
        new_state = self.post_norm(post_activations)  # [B, d_latent]

        # Recompute sync for output (using updated history)
        updated_post_history = self.memory.get_post_history()
        output_sync = self.sync_computer.compute(updated_post_history)  # [B, sync_dim]

        return new_state, output_sync
