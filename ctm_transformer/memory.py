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