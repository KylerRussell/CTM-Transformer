"""
Matrix-Valued Residual Stream — Hyperloop-style manifold mixing

Replaces the per-neuron NLM FIFO buffers and O(D²) synchronization
computer with a set of n parallel residual streams that evolve through
the thought loop. Synchronization occurs via learned manifold mixing
(diagonal gating matrices) rather than explicit D×D summary computation.

Key advantages over the NLM+Sync design:
  1. Memory: O(n·D) instead of O(H·D) per layer — constant in history_len
  2. Compute: Diagonal gating is O(n·D) instead of O(D²) for sync
  3. Scalability: No FIFO buffer management, no roll operations
  4. Expressivity: Input-dependent gating learns co-activation patterns
     that the fixed NLM architecture cannot represent

Design from Hyperloop Transformers:
  - State is an [n × C] matrix (n streams, C = d_latent)
  - Three gating matrices per layer:
      H_pre:  [n, n] — mixes streams BEFORE the layer processes
      H_post: [n, n] — mixes streams AFTER the layer processes  
      H_res:  [n]    — diagonal gating on the residual (cheap but effective)
  - The Birkhoff polytope constraint (convex combination) is enforced via
    softmax normalization, keeping stream magnitudes bounded.

Integration:
  - Replaces TemporalMemory, SynchronizationComputer, and NeuronLevelModels
  - The 'to_query' method replaces sync_computer.compute() for query generation
  - Stream state replaces the (pre_states, post_states) lists in model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Broadcast learnable init to [B, S, n, D]
        state = self.stream_init.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1, -1
        ).clone().to(dtype=dtype)
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
    ) -> torch.Tensor:
        """Post-layer mixing with residual gating.

        After the thought layer processes the collapsed stream into a
        single-vector output, this method updates the multi-stream state
        by combining the layer output with the existing streams via
        learned gating.

        Args:
            stream: [B, S, n_streams, d_latent] — pre-layer stream state
            layer_output: [B, S, d_latent] — output from the thought layer

        Returns:
            updated: [B, S, n_streams, d_latent]
        """
        B, S, N, D = stream.shape

        # Expand layer output to all streams
        # [B, S, D] → [B, S, 1, D] → broadcast to [B, S, N, D]
        expanded = layer_output.unsqueeze(2).expand_as(stream)

        # Post-mixing weights (input-dependent)
        mean_state = stream.mean(dim=2)  # [B, S, D]
        mix_logits = self.post_mix_proj(mean_state)  # [B, S, N*N]
        mix_logits = mix_logits.view(B, S, N, N)
        mix_weights = F.softmax(mix_logits, dim=-1)

        # Mix the expanded output across streams
        new_content = torch.matmul(mix_weights, expanded)  # [B, S, N, D]

        # Residual gating: blend old stream with new content
        # gate ∈ (0, 1) via sigmoid; 0.5 at init (both contribute equally)
        gate = torch.sigmoid(self.res_gate)  # [N]
        gate = gate.view(1, 1, N, 1)  # broadcast shape

        updated = gate * new_content + (1 - gate) * stream

        return updated

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
