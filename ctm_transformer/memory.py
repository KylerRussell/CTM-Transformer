"""
Memory Buffers and Synchronization Computation

Maintains FIFO history buffers for pre-activations and post-activations,
and computes neural synchronization matrices from the post-activation history.

Conceptual mapping from Living-Brain engine_torch.py:
- engine_torch: rho_slow_states (myelinated temporal filtering / EMA of activations)
- CTM Memory:   explicit FIFO buffer of full activation snapshots across thought steps
- engine_torch: W @ rho (sparse connectivity matrix capturing inter-neuron coupling)
- CTM Sync:     S_t = Z_t @ Z_t^T (synchronization matrix from activation history)
"""

import torch
import torch.nn as nn


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

        # Buffers are registered as non-persistent state (not saved with model params)
        # They are initialized lazily on first push() to match batch size and device.
        self.register_buffer("pre_history", None, persistent=False)
        self.register_buffer("post_history", None, persistent=False)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Initialize or clear both history buffers to zeros.

        Args:
            batch_size: Current batch size.
            device: Target device.
            dtype: Data type for buffers.
        """
        self.pre_history = torch.zeros(
            batch_size, self.history_len, self.d_latent,
            device=device, dtype=dtype
        )
        self.post_history = torch.zeros(
            batch_size, self.history_len, self.d_latent,
            device=device, dtype=dtype
        )

    def push_pre(self, pre_activations: torch.Tensor):
        """
        Push new pre-activations into the FIFO buffer.

        Args:
            pre_activations: [batch, d_latent] — new pre-activation values.
        """
        # Shift left: drop oldest (index 0), append new at the end
        self.pre_history = torch.cat([
            self.pre_history[:, 1:, :],
            pre_activations.unsqueeze(1)
        ], dim=1)

    def push_post(self, post_activations: torch.Tensor):
        """
        Push new post-activations into the FIFO buffer.

        Args:
            post_activations: [batch, d_latent] — new post-activation values.
        """
        self.post_history = torch.cat([
            self.post_history[:, 1:, :],
            post_activations.unsqueeze(1)
        ], dim=1)

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

    Args:
        d_latent: Neuron count.
        history_len: FIFO buffer depth.
        method: Synchronization computation method.
        rank: Rank for low_rank method.
    """

    def __init__(
        self,
        d_latent: int,
        history_len: int,
        method: str = "diag_summary",
        rank: int = 32,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len
        self.method = method
        self.rank = rank

    def compute(self, post_history: torch.Tensor) -> torch.Tensor:
        """
        Compute synchronization representation from post-activation history.

        Args:
            post_history: [batch, history_len, d_latent] — the Z_t matrix.

        Returns:
            sync_repr: [batch, sync_dim] — flattened synchronization representation.
        """
        B, H, D = post_history.shape

        # S_t = Z_t^T @ Z_t → [batch, d_latent, d_latent]
        # Note: We compute Z^T @ Z (not Z @ Z^T) to get neuron-neuron coupling
        # Z_t is [batch, history_len, d_latent], so:
        # Z_t^T @ Z_t = [batch, d_latent, history_len] @ [batch, history_len, d_latent]
        #             = [batch, d_latent, d_latent]
        S = torch.bmm(post_history.transpose(1, 2), post_history)

        # Normalize by history length for stability
        S = S / max(H, 1)

        if self.method == "full":
            # Flatten entire matrix
            return S.reshape(B, D * D)

        elif self.method == "diag_summary":
            # Efficient summary: diagonal + row means + column means
            diag = torch.diagonal(S, dim1=1, dim2=2)        # [batch, d_latent]
            row_means = S.mean(dim=2)                         # [batch, d_latent]
            col_means = S.mean(dim=1)                         # [batch, d_latent]
            return torch.cat([diag, row_means, col_means], dim=1)  # [batch, 3*d_latent]

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
        else:
            raise ValueError(f"Unknown sync method: {self.method}")
