"""
Neuron-Level Models (NLMs)

Replaces static activation functions (ReLU, GELU) with privately-parameterized
per-neuron MLPs that process temporal history. Each neuron has its own learned
activation dynamics, making the model stateful and temporally aware.

Conceptual mapping from Living-Brain engine_torch.py:
- engine_torch: get_soma() computes somatic output from basal/apical/CaHVA compartments
- CTM NLM:      per-neuron MLP computes post-activation from pre-activation history

Implementation uses batched operations for GPU efficiency:
- Weights are stored as [d_latent, in_dim, out_dim] tensors
- Forward pass uses einsum or grouped convolutions for parallelism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    ):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len
        self.nlm_hidden_dim = nlm_hidden_dim
        self.nlm_groups = nlm_groups

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

        # Activation: GELU (smooth, avoids dead neurons unlike ReLU)
        h = F.gelu(h)
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
