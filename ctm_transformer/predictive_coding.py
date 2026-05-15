"""
Hierarchical Predictive Coding for CTM-Transformer (Mechanism 1).

Formalizes the CTM thought loop as iterative free-energy minimization
over a deep generative hierarchy. T thought ticks become T inference
iterations of a predictive coding network.

Key components:
  1. PredictionErrorComputer — per-layer ε = μ − f(μ_above) computation
     with configurable generative function (MLP or UNet).
  2. PrecisionComputer — per-token, per-layer learned precision π that
     gates which errors dominate learning (automatic curriculum).
  3. PCStateManager — splits MatrixResidualStream into value (μ) and
     error (ε) channels.

Gradient locality: all weight updates use fully local rules with
.detach() at layer boundaries. At the PC fixed point, these local
gradients equal standard backprop gradients (Millidge et al., Neural
Computation 2022; Whittington & Bogacz, Neural Computation 2017).

References:
  - Rao & Ballard, Nature Neuroscience 1999
  - Friston, Phil. Trans. B 2005
  - Millidge, Tschantz & Buckley, Neural Computation 2022 (arXiv 2006.04182)
  - Salvatori et al., NeurIPS 2022 / ICLR 2023
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ════════════════════════════════════════════════════════════════════════
# Generative Functions (top-down prediction)
# ════════════════════════════════════════════════════════════════════════

class GenerativeMLP(nn.Module):
    """Lightweight 2-layer MLP as the top-down generative function f_ℓ.

    Predicts the activity of the current layer from the layer above:
        f_ℓ(μ_{ℓ+1}) ≈ μ_ℓ

    Initialized near-identity so initial prediction errors ≈ 0 and the
    system starts in a quasi-equilibrium state.

    Args:
        d_latent: Width of the latent state.
        hidden_dim: Hidden layer width (default: 2 × d_latent).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_latent: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (2 * d_latent)
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent),
        )
        # Near-identity init: final layer starts small so f(x) ≈ 0
        # and prediction ≈ 0 → initial error ≈ μ (untrained prior).
        # The first tick's gradient is then dominated by matching μ to
        # the prior, which is the correct cold-start behavior.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu_above: torch.Tensor) -> torch.Tensor:
        """Predict current-layer activity from the layer above."""
        return self.net(mu_above)


class GenerativeUNet(nn.Module):
    """1D U-Net generative function for richer spatial mixing.

    Same architecture as UNetSynapse but used as the top-down generative
    model f_ℓ. More expressive than the MLP (~5× params) but captures
    multi-scale structure in the latent dimension.

    Args:
        d_latent: Width of the latent state.
        dropout: Dropout rate.
    """

    def __init__(self, d_latent: int, dropout: float = 0.1):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.GELU()
        )
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.GELU()
        )
        self.pool2 = nn.MaxPool1d(2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv1d(32 + 16, 16, kernel_size=3, padding=1), nn.GELU()
        )
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv1d(16 + 1, 1, kernel_size=3, padding=1), nn.GELU()
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent),
        )
        # Small init for output projection
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, mu_above: torch.Tensor) -> torch.Tensor:
        """Predict current-layer activity from the layer above.

        Args:
            mu_above: [*, d_latent] — can be any leading dims.
        Returns:
            prediction: [*, d_latent]
        """
        orig_shape = mu_above.shape
        x = mu_above.reshape(-1, orig_shape[-1])  # [N, d_latent]
        x_in = x.unsqueeze(1)  # [N, 1, d_latent]

        d1 = self.down1(x_in)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        u1 = self.up1(p2)
        if u1.size(2) != d1.size(2):
            u1 = F.interpolate(u1, size=d1.size(2))
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        if u2.size(2) != x_in.size(2):
            u2 = F.interpolate(u2, size=x_in.size(2))
        u2 = torch.cat([u2, x_in], dim=1)
        u2 = self.conv_up2(u2)

        out = u2.squeeze(1)  # [N, d_latent]
        out = self.proj(out)
        return out.view(orig_shape)


# ════════════════════════════════════════════════════════════════════════
# Prediction Error Computer
# ════════════════════════════════════════════════════════════════════════

class PredictionErrorComputer(nn.Module):
    """Computes layer-wise prediction errors for hierarchical PC.

    Each layer ℓ has a generative function f_ℓ that predicts the activity
    of the current layer from the layer above:

        ε_ℓ = μ_ℓ − f_ℓ(μ_{ℓ+1})

    The top layer uses a learnable prior (or teacher_z when available).

    Args:
        d_latent: Latent state width.
        generative_type: "mlp" or "unet" — type of generative function.
        hidden_dim: Hidden dim for MLP generative function.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_latent: int,
        generative_type: str = "mlp",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.generative_type = generative_type

        if generative_type == "unet":
            self.generative_fn = GenerativeUNet(d_latent, dropout=dropout)
        else:
            self.generative_fn = GenerativeMLP(
                d_latent, hidden_dim=hidden_dim, dropout=dropout
            )

    def forward(
        self,
        mu_current: torch.Tensor,
        mu_above: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prediction and prediction error.

        Both mu_current and mu_above are DETACHED from the upstream
        computation graph. The only trainable path is through
        self.generative_fn's parameters. This is the core of the
        fully-local PC gradient rule.

        Args:
            mu_current: [B, S, d_latent] — this layer's value (detached).
            mu_above:   [B, S, d_latent] — layer above's value (detached).

        Returns:
            prediction: [B, S, d_latent] — f_ℓ(μ_{ℓ+1}), has grad through f_ℓ.
            error:      [B, S, d_latent] — ε = μ_current - prediction.
        """
        prediction = self.generative_fn(mu_above)
        error = mu_current - prediction
        return prediction, error


# ════════════════════════════════════════════════════════════════════════
# Precision Computer
# ════════════════════════════════════════════════════════════════════════

class PrecisionComputer(nn.Module):
    """Per-token, per-layer learned precision weights.

    Precision-weighting is what neuromodulators do in the brain:
    high-precision errors dominate learning (attention/curriculum),
    low-precision errors are suppressed (noise gating).

    π is computed from local activity statistics (error magnitude,
    μ magnitude) via a small MLP, then passed through softplus to
    ensure positivity.

    Biologically-graded initialization: lower layers start with higher
    precision (sensory layers are more certain about their inputs),
    upper layers start with lower precision (abstract representations
    are more uncertain).

    Args:
        d_latent: Latent state width.
        layer_idx: Index of this layer in the hierarchy (0 = bottom).
        n_layers: Total number of layers (for graded init).
    """

    def __init__(
        self,
        d_latent: int,
        layer_idx: int = 0,
        n_layers: int = 1,
    ):
        super().__init__()
        self.d_latent = d_latent

        # Small MLP: [error_stats, mu_stats] → scalar precision
        # Input: 4 features (error_mean, error_var, mu_mean, mu_var)
        self.precision_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Learnable baseline log-precision (biologically graded).
        # Lower layers (layer_idx=0) get higher precision (more certain),
        # upper layers get lower precision (more uncertain).
        # Scale: precision ∈ [0.5, 2.0] at init across the hierarchy.
        if n_layers > 1:
            # Linear interpolation: layer 0 → log(2.0), layer L-1 → log(0.5)
            frac = layer_idx / (n_layers - 1)
            init_val = math.log(2.0) * (1 - frac) + math.log(0.5) * frac
        else:
            init_val = 0.0  # log(1.0)
        self.log_precision_baseline = nn.Parameter(
            torch.tensor(init_val)
        )

        # Init precision net to output ~0 so baseline dominates at start
        nn.init.zeros_(self.precision_net[-1].weight)
        nn.init.zeros_(self.precision_net[-1].bias)

    def forward(
        self,
        error: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Compute precision weight from local activity statistics.

        Args:
            error: [B, S, d_latent] — prediction error at this layer.
            mu:    [B, S, d_latent] — value at this layer.

        Returns:
            precision: [B, S, 1] — positive scalar per token per layer.
        """
        # Compute local statistics (detached — precision is a routing
        # signal, not a path for gradient to flow through the error
        # computation itself).
        with torch.no_grad():
            e = error.detach().float()
            m = mu.detach().float()
            stats = torch.stack([
                e.mean(dim=-1),           # error mean
                e.var(dim=-1, correction=0),  # error variance
                m.mean(dim=-1),           # mu mean
                m.var(dim=-1, correction=0),  # mu variance
            ], dim=-1)  # [B, S, 4]

        # Precision = softplus(baseline + net(stats))
        stats = stats.to(self.log_precision_baseline.dtype)
        adjustment = self.precision_net(stats)  # [B, S, 1]
        log_pi = self.log_precision_baseline + adjustment
        precision = F.softplus(log_pi)  # [B, S, 1], always positive

        return precision


# ════════════════════════════════════════════════════════════════════════
# PC State Manager
# ════════════════════════════════════════════════════════════════════════

class PCStateManager(nn.Module):
    """Manages value/error split in the MatrixResidualStream.

    With n_streams=8: streams 0–3 are μ (value) channels,
    streams 4–7 are ε (error) channels.

    Args:
        n_streams: Total streams (must be even).
        d_latent: Latent state width.
    """

    def __init__(self, n_streams: int, d_latent: int):
        super().__init__()
        assert n_streams % 2 == 0, (
            f"n_streams must be even for PC split, got {n_streams}"
        )
        self.n_streams = n_streams
        self.n_value = n_streams // 2
        self.n_error = n_streams - self.n_value
        self.d_latent = d_latent

    def split_streams(
        self, stream_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split stream state into value and error channels.

        Args:
            stream_state: [B, S, n_streams, d_latent]

        Returns:
            mu_streams:      [B, S, n_value, d_latent]
            epsilon_streams: [B, S, n_error, d_latent]
        """
        mu = stream_state[:, :, : self.n_value, :]
        eps = stream_state[:, :, self.n_value :, :]
        return mu, eps

    def merge_streams(
        self,
        mu_streams: torch.Tensor,
        epsilon_streams: torch.Tensor,
    ) -> torch.Tensor:
        """Merge value and error channels back into stream state.

        Args:
            mu_streams:      [B, S, n_value, d_latent]
            epsilon_streams: [B, S, n_error, d_latent]

        Returns:
            stream_state: [B, S, n_streams, d_latent]
        """
        return torch.cat([mu_streams, epsilon_streams], dim=2)

    def collapse_values(self, mu_streams: torch.Tensor) -> torch.Tensor:
        """Collapse value streams to a single vector (mean).

        Args:
            mu_streams: [B, S, n_value, d_latent]

        Returns:
            mu: [B, S, d_latent]
        """
        return mu_streams.mean(dim=2)

    def write_errors(
        self,
        epsilon_streams: torch.Tensor,
        error: torch.Tensor,
    ) -> torch.Tensor:
        """Write a prediction error into the error streams.

        Distributes the error across all error streams (broadcast).

        Args:
            epsilon_streams: [B, S, n_error, d_latent] — current errors.
            error:           [B, S, d_latent] — new error to write.

        Returns:
            updated_epsilon: [B, S, n_error, d_latent]
        """
        return error.unsqueeze(2).expand_as(epsilon_streams)

    def free_energy(
        self,
        errors: list[torch.Tensor],
        precisions: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute variational free energy F = Σ_ℓ (0.5 * π_ℓ · mean(ε_ℓ²) - 0.5 * log(π_ℓ)).

        Dimension-normalized to match the local loss scale and prevent massive
        metric values when d_latent is large.

        Args:
            errors: List of [B, S, d_latent] per-layer unnormalized errors.
            precisions: List of [B, S, 1] per-layer precisions.

        Returns:
            F: Scalar free energy, averaged over batch and sequence.
        """
        F_total = torch.tensor(0.0, device=errors[0].device, dtype=errors[0].dtype)
        for eps, pi in zip(errors, precisions):
            # pi: [B, S, 1], eps: [B, S, d_latent]
            weighted_sq = 0.5 * pi * (eps * eps).mean(dim=-1, keepdim=True)
            log_pi = 0.5 * torch.log(pi + 1e-8)
            layer_F = (weighted_sq - log_pi).mean()
            F_total = F_total + layer_F
        return F_total


# ════════════════════════════════════════════════════════════════════════
# Per-Layer PC Module (bundles error + precision for one layer)
# ════════════════════════════════════════════════════════════════════════

class PCLayer(nn.Module):
    """One layer of the predictive coding hierarchy.

    Bundles the generative function, prediction error computation, and
    precision estimation for a single layer. Also provides the local
    weight update computation.

    Args:
        d_latent: Latent state width.
        layer_idx: Position in the hierarchy (0 = bottom).
        n_layers: Total layers.
        generative_type: "mlp" or "unet".
        generative_hidden_dim: Hidden dim for MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_latent: int,
        layer_idx: int,
        n_layers: int,
        generative_type: str = "mlp",
        generative_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_latent = d_latent

        self.error_computer = PredictionErrorComputer(
            d_latent=d_latent,
            generative_type=generative_type,
            hidden_dim=generative_hidden_dim,
            dropout=dropout,
        )
        self.precision = PrecisionComputer(
            d_latent=d_latent,
            layer_idx=layer_idx,
            n_layers=n_layers,
        )

        # Feedback projection: maps error from the layer below back
        # into this layer's space. This is the Wᵀ in the PC update
        # rule: μ_ℓ ← μ_ℓ − η·(ε_ℓ − Wᵀ·ε_{ℓ-1}).
        self.feedback_proj = nn.Linear(d_latent, d_latent, bias=False)
        nn.init.orthogonal_(self.feedback_proj.weight)

    def compute_error(
        self,
        mu_current: torch.Tensor,
        mu_above: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prediction error and precision at this layer.

        All inputs are detached for fully-local gradient computation.
        The only gradient path is through self.error_computer.generative_fn.

        Args:
            mu_current: [B, S, d_latent] — this layer's value.
            mu_above:   [B, S, d_latent] — layer above's value.

        Returns:
            prediction: [B, S, d_latent] — f_ℓ(μ_{ℓ+1}), has local grad.
            error:      [B, S, d_latent] — ε = μ - f(μ_above).
            precision:  [B, S, 1] — π_ℓ.
        """
        # Detach inputs for local gradient computation
        mu_curr_det = mu_current.detach()
        mu_above_det = mu_above.detach()

        prediction, error = self.error_computer(mu_curr_det, mu_above_det)
        # We pass the unnormalized error to the precision computer.
        # LayerNorming the error forces it to variance 1.0, destroying magnitude info!
        pi = self.precision(error, mu_curr_det)

        return prediction, error, pi

    def compute_mu_update(
        self,
        error: torch.Tensor,
        precision: torch.Tensor,
        error_below: torch.Tensor | None,
        precision_below: torch.Tensor | None,
        inference_lr: float,
    ) -> torch.Tensor:
        """Compute the PC state update Δμ for this layer.

        μ_ℓ ← μ_ℓ − η · (π_ℓ · ε_ℓ − Wᵀ · π_{ℓ-1} · ε_{ℓ-1})

        Args:
            error: [B, S, d_latent] — this layer's prediction error.
            precision: [B, S, 1] — this layer's precision.
            error_below: [B, S, d_latent] — layer below's error, or None.
            precision_below: [B, S, 1] — layer below's precision, or None.
            inference_lr: Learning rate η for the inference step.

        Returns:
            delta_mu: [B, S, d_latent] — update to apply to μ_ℓ.
        """
        # Top-down error signal (this layer's own prediction error)
        top_down = precision * error

        # Bottom-up correction (from the layer below)
        if error_below is not None and precision_below is not None:
            weighted_below = precision_below * error_below
            bottom_up = self.feedback_proj(weighted_below.detach())
        else:
            bottom_up = torch.zeros_like(top_down)

        delta_mu = -inference_lr * (top_down.detach() - bottom_up)
        return delta_mu

    def local_loss(
        self,
        prediction: torch.Tensor,
        mu_target: torch.Tensor,
        precision: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the local free-energy loss for this layer's generative fn and precision.

        This is the loss that drives local weight updates for f_ℓ and π_ℓ:
            L_ℓ = 0.5 * π_ℓ · ‖μ_ℓ − f_ℓ(μ_{ℓ+1})‖² - 0.5 * d_latent * log(π_ℓ)

        The gradient of this loss w.r.t. f_ℓ and π_ℓ parameters is the fully-
        local PC learning rule. μ_ℓ and μ_{ℓ+1} are both detached (done
        in compute_error), so the gradient only flows through local parameters.

        Args:
            prediction: [B, S, d_latent] — f_ℓ(μ_{ℓ+1}), has grad.
            mu_target:  [B, S, d_latent] — μ_ℓ (detached target).
            precision:  [B, S, 1] — π_ℓ (has grad).

        Returns:
            Scalar local loss for this layer.
        """
        error_sq = (mu_target.detach() - prediction).pow(2).mean(dim=-1, keepdim=True)
        # Dimension-normalized Gaussian free energy: 0.5 * pi * mean_error^2 - 0.5 * log(pi)
        # The log(pi) term penalizes high precision, preventing trivial pi -> inf.
        loss = 0.5 * precision * error_sq - 0.5 * torch.log(precision + 1e-8)
        return loss.mean()
