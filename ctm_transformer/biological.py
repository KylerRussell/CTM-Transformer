"""
Biological learning extensions for CTM-Transformer.

This module adds two neuroscience-inspired learning mechanisms that exploit
the CTM's recurrent thought loop:

  1. HebbianSynapse — a per-batch, per-position fast-weight matrix updated
     by outer products of (prev_state, attn_out) during the thought loop.
     Encodes instant, few-shot associations in *activation state* rather
     than in slow gradient-trained weights. The fast weights persist across
     thought ticks within a single forward pass and reset between batches.
     Composes additively (and gated) with the existing synapse MLP so the
     model can fall back to slow weights when the Hebbian trace is unhelpful.

  2. CerebellarReadout — a temporal-hierarchy Predictive Coding head
     attached at every thought tick. At tick t, it predicts the *teacher
     hidden state that the student will match at tick t+1*, producing a
     local prediction error against the cached teacher features. The error
     is returned as an auxiliary loss that the training loop adds to the
     main objective with a configurable weight (`pc_loss_weight`).
     
     Why this is "predictive coding": instead of free-energy minimization
     across a spatial layer hierarchy (e.g. Rao & Ballard 1999), the
     hierarchy here is *temporal* — the thought-loop tick index plays the
     role that cortical layer index plays in canonical PC. The readout is
     "cerebellar" in the loose anatomical sense: a thin feedforward
     projection that pattern-completes a target without going through the
     full recurrent loop again.

Both mechanisms are opt-in via config flags, default off, and gated by
runtime so that turning them off costs zero — no parameter creation, no
extra forward work.

Integration points (see model.py for the call sites):

    CTMTransformer.__init__:
        if config.use_predictive_coding:
            self.cerebellar_readout = CerebellarReadout(...)

    ThoughtLayer.__init__:
        if config.use_hebbian_synapse:
            self.hebbian = HebbianSynapse(...)

    ThoughtLayer.forward:
        - Initialise/carry M (the fast-weight matrix) alongside z and
          stream_state, threaded through `hebbian_state` argument.
        - After the standard synapse output, add gated Hebbian readout.
        - Update M via outer product before returning.

    CTMTransformer._thought_step:
        - On tick t, after computing z_new, call cerebellar_readout to
          predict teacher_z (one step ahead) and accumulate prediction
          error into pc_loss_t.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
# Hebbian Fast-Weights
# ════════════════════════════════════════════════════════════════════════

class HebbianSynapse(nn.Module):
    """
    Fast-weight associative memory that augments the synapse output.

    Maintains a [d_latent × d_model] association matrix M per (batch,
    position) updated by an outer product rule:

        M_{t+1} = decay · M_t + lr · z_t ⊗ a_t

    where z_t is the previous latent state and a_t is the cross-attention
    output. The "readout" of M is a vector r_t = M_t · a_t (i.e. a content-
    addressed lookup keyed by the current attention output), projected
    back into d_latent space and added — gated — to the synapse output.

    The decay and learning-rate are learnable scalars (sigmoid-parameterized
    to stay in (0, 1)). The contribution to the synapse output passes
    through a learnable sigmoid gate, initialized at zero so the layer
    behaves exactly like the unaugmented synapse at the start of training.

    Memory cost: O(B · S · d_latent · d_model) per layer per forward.
    For B=1, S=512, d_latent=d_model=1024 that's ~512 MB in bf16; reduce
    via the `bottleneck_dim` argument, which projects (z, a) into a
    smaller subspace before the outer product, giving
    O(B · S · bottleneck_dim²). With bottleneck_dim=64 the same config
    drops to ~2 MB per layer — usually plenty of capacity for within-
    sequence working memory.

    Args:
        d_latent: Width of the latent state.
        d_model: Width of the attention output.
        bottleneck_dim: If > 0, project z and a into R^{bottleneck_dim}
            before the outer product. The fast-weight matrix becomes
            [bottleneck_dim, bottleneck_dim] per position. If 0 or None,
            use the full d_latent × d_model matrix.
        decay_init: Initial decay value (after sigmoid). 0.9 ≈ logit 2.2.
        lr_init: Initial learning rate. 0.1 ≈ logit -2.2.
        gate_init: Initial output-gate logit. Negative → starts near zero
            contribution.
        force_gate: Diagnostic override (see below).
        update_rule: One of:
            "outer_product" — classic Hebbian:
                M ← decay·M + lr·(z ⊗ a)
                Accumulates all (z, a) co-occurrences. Simple, but the
                matrix saturates and grows in magnitude indefinitely
                because there is no error-correction term. Established
                in the original Hopfield / Kohonen literature.
            "delta" — error-correcting (Widrow-Hoff / Schlag-Schmidhuber):
                M ← decay·M + lr·(z - M·a) ⊗ a
                The (z - M·a) term is a *local prediction error* — only
                the part of z that M doesn't already retrieve from a is
                written. Reaches a steady state where M·a ≈ z and stops
                accumulating, rather than growing unbounded. Closer to
                the "delta rule" / regression formulation argued for in
                Nested Learning (Behrouz et al. 2025) and used in
                "Linear Transformers are Fast Weight Programmers"
                (Schlag & Schmidhuber 2021).
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        bottleneck_dim: int = 64,
        decay_init: float = 0.9,
        lr_init: float = 0.1,
        gate_init: float = -3.0,
        force_gate: float | None = None,
        update_rule: str = "outer_product",
        use_btsp: bool = False,
        btsp_lr_init: float = 0.05,
        btsp_kernel_decay_init: float = 0.9,
        btsp_salience_threshold: float = 0.0,
        n_compartments: int = 1,
        use_stc: bool = False,
        stc_tag_decay: float = 0.5,
        stc_threshold_tag: float = 0.05,
        use_critical_init: bool = False,
        critical_init_sym_frac: float = 0.6,
        critical_init_spectral_radius: float = 0.999,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.use_bottleneck = bottleneck_dim is not None and bottleneck_dim > 0
        self.bottleneck_dim = bottleneck_dim if self.use_bottleneck else None
        if update_rule not in ("outer_product", "delta"):
            raise ValueError(
                f"update_rule must be 'outer_product' or 'delta', got {update_rule!r}"
            )
        self.update_rule = update_rule

        # Diagnostic override: when not None, the readout gate is held
        # at this fixed value regardless of gate_logit. Used to isolate
        # "is the readout content useful?" from "is the gate blocked?":
        #   force_gate=0.9 with no CE change → readout is genuinely useless
        #   force_gate=0.9 hurts CE → readout content is actively bad
        #   force_gate=0.9 helps CE → the gate gradient was the bottleneck
        # NOT meant for production training — purely a diagnostic lever.
        # When set, gate_logit is still a parameter (it might still
        # collect gradient from elsewhere) but is not consulted in the
        # readout multiplication.
        self.force_gate = force_gate

        # ── Dendritic compartmentalization ──────────────────────────────
        # When n_compartments > 1, the bottleneck space is partitioned
        # into K independent branches.  Each branch has its own fast-weight
        # matrix M_k, per-compartment decay/LR, and a learnable plateau-
        # potential gate that gates the Hebbian readout bilinearly.
        # Requires bottleneck_dim divisible by n_compartments.
        if n_compartments > 1:
            if not self.use_bottleneck:
                raise ValueError(
                    "n_compartments > 1 requires bottleneck_dim > 0"
                )
            if bottleneck_dim % n_compartments != 0:
                raise ValueError(
                    f"bottleneck_dim ({bottleneck_dim}) must be divisible "
                    f"by n_compartments ({n_compartments})"
                )
        self.n_compartments = n_compartments
        self.is_compartmentalized = n_compartments > 1

        if self.use_bottleneck:
            self.z_proj = nn.Linear(d_latent, bottleneck_dim, bias=False)
            self.a_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
            # Read out (K * m_k = bottleneck_dim) → d_latent.  Same shape
            # whether compartmentalized or not.
            self.readout_proj = nn.Linear(bottleneck_dim, d_latent, bias=False)
            if self.is_compartmentalized:
                m_k = bottleneck_dim // n_compartments
                self.m_k = m_k
                # M shape per (B, S): [K, m_k, m_k]
                self._matrix_dims = (n_compartments, m_k, m_k)
            else:
                # M shape per (B, S): [bottleneck_dim, bottleneck_dim]
                self._matrix_dims = (bottleneck_dim, bottleneck_dim)
        else:
            self.z_proj = None
            self.a_proj = None
            # Readout from M·a ∈ R^{d_latent} — identity projection.
            self.readout_proj = nn.Linear(d_latent, d_latent, bias=False)
            # M shape per (B, S): [d_latent, d_model]
            self._matrix_dims = (d_latent, d_model)

        # Learnable decay and learning-rate scalars/vectors (logit space).
        # Per-compartment when is_compartmentalized, scalar otherwise.
        if self.is_compartmentalized:
            self.decay_logit = nn.Parameter(
                torch.full((n_compartments,), float(_logit(decay_init)))
            )
            self.lr_logit = nn.Parameter(
                torch.full((n_compartments,), float(_logit(lr_init)))
            )
            # Per-compartment plateau-potential gate (dendritic spike
            # threshold).  Initialized same as gate_init so each branch
            # starts near-silent and grows organically.
            self.plateau_gate_logit = nn.Parameter(
                torch.full((n_compartments,), float(gate_init))
            )
        else:
            self.decay_logit = nn.Parameter(torch.tensor(_logit(decay_init)))
            self.lr_logit = nn.Parameter(torch.tensor(_logit(lr_init)))
            self.plateau_gate_logit = None

        # Global output gate: starts near zero so the unaugmented synapse
        # path dominates at init.  Scalar for both flat and compartmentalized.
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        # ── Behavioral Timescale Synaptic Plasticity (BTSP) ──────────────
        # Adds a sequence-spanning causal update on top of the per-position
        # Hebbian rule.  A salient event at position s (high surprise) can
        # retro-potentiate the fast-weight matrices of earlier positions τ<s
        # through an exponentially decaying eligibility trace:
        #
        #   M_{t+1}[s] += η_BTSP · Σ_{τ≤s} β^(s−τ) · Φ(salience_τ) · (z_τ ⊗ a_τ)
        #
        # Computed as a lower-triangular kernel matmul — fully vectorised,
        # no sequential Python loop.  When lr_modulator is None (no certainty
        # signal), salience degrades to uniform=1 (pure causal trace).
        self.use_btsp = use_btsp
        if use_btsp:
            self.btsp_lr_logit = nn.Parameter(torch.tensor(_logit(btsp_lr_init)))
            self.btsp_decay_logit = nn.Parameter(torch.tensor(_logit(btsp_kernel_decay_init)))
            self.btsp_salience_threshold = btsp_salience_threshold

        # ── Synaptic Tagging and Capture (STC) ──────────────────────────
        # Tracks a fast-decaying EMA of the mean absolute delta-rule error
        # |z - M·a|.  High error → tag is active → this layer's slow weights
        # are eligible for updates when the global PRP is also high.
        # Stored as a plain Python float (CPU, outside the gradient graph).
        self.use_stc = use_stc
        if use_stc:
            self.stc_tag_decay = stc_tag_decay
            self.stc_threshold_tag = stc_threshold_tag
            self._tag_norm: float = 0.0

        # Stabilise the readout norm so the Hebbian contribution doesn't
        # blow up the synapse output before training has shaped the gate.
        self.readout_norm = nn.LayerNorm(d_latent, elementwise_affine=False)

        # Small init for the readout projection — same logic as the NLM's
        # layer-2 init: start near-identity-to-zero so the layer doesn't
        # bias the early-training synapse output.
        nn.init.normal_(self.readout_proj.weight, std=0.01)
        if self.use_bottleneck:
            # Standard Xavier for the projections.
            nn.init.xavier_uniform_(self.z_proj.weight)
            nn.init.xavier_uniform_(self.a_proj.weight)

        # ── Critical-symmetric fast-weight prior (optional) ─────────────
        # Instead of starting M at zero (all eigenvalues 0, the opposite of
        # critical), pre-warm it with a critically-normalized matrix so the
        # fast-weight readout has long-timescale, high-dimensional dynamics
        # from tick one — a "scaffold" before any (z, a) co-occurrences are
        # written. Stored as a fixed buffer (no grad): the decay rule erodes
        # it over ticks as learned associations accumulate. Only registered
        # when enabled, so existing checkpoints load unchanged.
        self.use_critical_prior = bool(use_critical_init)
        if self.use_critical_prior:
            if self.is_compartmentalized:
                K, m_k, _ = self._matrix_dims
                M_init = torch.zeros(K, m_k, m_k)
                for k in range(K):
                    critical_symmetric_init_(
                        M_init[k], critical_init_sym_frac,
                        critical_init_spectral_radius,
                    )
            else:
                m, n = self._matrix_dims
                M_init = torch.zeros(m, n)
                critical_symmetric_init_(
                    M_init, critical_init_sym_frac,
                    critical_init_spectral_radius,
                )
            self.register_buffer("M_init", M_init)

    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a fresh fast-weight matrix for a new sequence.

        Zeros by default; a broadcast critical-symmetric prior when
        ``use_critical_init`` was set (see ``M_init`` buffer).

        Non-compartmentalized: [B, S, m, n]
        Compartmentalized:     [B, S, K, m_k, m_k]
        """
        if self.use_critical_prior:
            prior = self.M_init.to(device=device, dtype=dtype)
            return prior.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_len, *([-1] * prior.ndim)
            ).clone()
        if self.is_compartmentalized:
            K, m_k, _ = self._matrix_dims
            return torch.zeros(batch_size, seq_len, K, m_k, m_k,
                               device=device, dtype=dtype)
        m, n = self._matrix_dims
        return torch.zeros(batch_size, seq_len, m, n, device=device, dtype=dtype)

    def forward(
        self,
        prev_state: torch.Tensor,
        attn_out: torch.Tensor,
        hebbian_state: torch.Tensor,
        lr_modulator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Hebbian readout and update the fast-weight matrix.

        Args:
            prev_state: [B, S, d_latent] — z_{t-1} (or current latent
                state, depending on call site).
            attn_out:   [B, S, d_model] — cross-attention output a_t.
            hebbian_state: [B, S, m, n] — current fast-weight matrix M_t.
            lr_modulator: [B, S] or None — per-position multiplier
                applied to the outer-product update (NOT to the readout).
                Caller supplies (1 + alpha * uncertainty) for
                certainty-based modulation. When None, uniform lr.

        Returns:
            readout:        [B, S, d_latent] — gated, LayerNormed readout
                            to be added to the synapse output.
            new_state:      [B, S, m, n] — updated fast-weight matrix M_{t+1}.
            effective_lr:   scalar — mean effective lr across positions
                            after modulation. Used for logging only.
        """
        B, S, _ = prev_state.shape

        if self.is_compartmentalized:
            return self._forward_compartmentalized(
                prev_state, attn_out, hebbian_state, lr_modulator
            )

        # ── Flat (non-compartmentalized) path ────────────────────────

        # Project into Hebbian space
        if self.use_bottleneck:
            z_proj = self.z_proj(prev_state)        # [B, S, bn]
            a_proj = self.a_proj(attn_out)          # [B, S, bn]
        else:
            z_proj = prev_state                     # [B, S, d_latent]
            a_proj = attn_out                       # [B, S, d_model]

        # Read: r = M · a (per position)
        # hebbian_state: [B, S, m, n], a_proj: [B, S, n] → [B, S, m]
        readout_raw = torch.einsum(
            "bsmn,bsn->bsm",
            hebbian_state.to(a_proj.dtype),
            a_proj,
        )
        readout = self.readout_proj(readout_raw)    # [B, S, d_latent]
        readout = self.readout_norm(readout)

        # STC tag update: EMA of mean |prediction error| (outside grad graph)
        if self.use_stc and self.training:
            with torch.no_grad():
                _err = (z_proj - readout_raw).abs().mean().item()
            self._tag_norm = (
                self.stc_tag_decay * self._tag_norm
                + (1.0 - self.stc_tag_decay) * _err
            )

        # Update: M_{t+1} = decay · M_t + lr_eff · z ⊗ a
        decay = torch.sigmoid(self.decay_logit)
        lr_base = torch.sigmoid(self.lr_logit)
        if lr_modulator is not None:
            lr_mod = lr_modulator.clamp(min=0.0, max=10.0).to(lr_base.dtype)
            lr_eff = lr_base * lr_mod.unsqueeze(-1).unsqueeze(-1)
            with torch.no_grad():
                effective_lr_scalar = lr_eff.mean().detach()
        else:
            lr_eff = lr_base
            effective_lr_scalar = lr_base.detach()

        if self.update_rule == "delta":
            error = z_proj - readout_raw.detach()
            outer = error.unsqueeze(-1) * a_proj.unsqueeze(-2)
            a_sq = (a_proj * a_proj).sum(dim=-1, keepdim=True).unsqueeze(-1)
            outer = outer / (a_sq + 1e-6)
        else:
            outer = z_proj.unsqueeze(-1) * a_proj.unsqueeze(-2)
        new_hebbian = decay * hebbian_state.to(outer.dtype) + lr_eff * outer

        # BTSP: sequence-spanning causal trace update
        # Adds: η_BTSP · btsp_K · (salience ⊙ outer)
        if self.use_btsp:
            m, n = self._matrix_dims
            B_b, S_b = outer.shape[0], outer.shape[1]
            btsp_lr = torch.sigmoid(self.btsp_lr_logit).to(outer.dtype)
            beta = torch.sigmoid(self.btsp_decay_logit).to(outer.dtype)

            if lr_modulator is not None:
                salience = lr_modulator.clamp(0.0, 10.0).to(outer.dtype)
                salience = (salience - self.btsp_salience_threshold).clamp(min=0.0)
            else:
                salience = torch.ones(B_b, S_b, device=outer.device, dtype=outer.dtype)

            weighted = salience.unsqueeze(-1).unsqueeze(-1) * outer  # [B, S, m, n]

            idx = torch.arange(S_b, device=outer.device, dtype=outer.dtype)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp(min=0)
            btsp_K = torch.tril(beta ** dist)                          # [S, S]

            w_flat = weighted.reshape(B_b, S_b, m * n)
            delta_flat = torch.matmul(btsp_K, w_flat)
            delta_M = delta_flat.reshape(B_b, S_b, m, n)
            new_hebbian = new_hebbian + btsp_lr * delta_M

        # Gate the readout (global scalar gate)
        if self.force_gate is not None:
            gate = torch.tensor(self.force_gate, dtype=readout.dtype, device=readout.device)
        else:
            gate = torch.sigmoid(self.gate_logit)
        gated_readout = gate * readout

        return gated_readout, new_hebbian, effective_lr_scalar

    def _forward_compartmentalized(
        self,
        prev_state: torch.Tensor,
        attn_out: torch.Tensor,
        hebbian_state: torch.Tensor,
        lr_modulator: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compartmentalized forward: K independent branches with bilinear plateau gating.

        hebbian_state shape: [B, S, K, m_k, m_k]
        """
        B, S, _ = prev_state.shape
        K, m_k, _ = self._matrix_dims

        # Project into Hebbian space (shared projections, then chunk)
        z_proj = self.z_proj(prev_state)   # [B, S, K*m_k]
        a_proj = self.a_proj(attn_out)     # [B, S, K*m_k]

        z_k = z_proj.view(B, S, K, m_k)   # [B, S, K, m_k]
        a_k = a_proj.view(B, S, K, m_k)   # [B, S, K, m_k]

        # Per-compartment read: M_k · a_k
        # hebbian_state: [B, S, K, m_k, m_k]
        readout_raw = torch.einsum(
            "bskmn,bskn->bskm",
            hebbian_state.to(a_k.dtype), a_k,
        )  # [B, S, K, m_k]

        # Bilinear plateau gate: per-compartment sigmoid scales the readout.
        # Biological analogy: each dendritic branch has an independent
        # plateau-potential threshold; only branches that exceed their
        # threshold contribute to the somatic readout.
        plateau = torch.sigmoid(self.plateau_gate_logit)   # [K]
        gated_branches = plateau.view(1, 1, K, 1) * readout_raw  # [B, S, K, m_k]

        # Merge branches → project to d_latent (K * m_k = bottleneck_dim)
        branch_flat = gated_branches.reshape(B, S, K * m_k)       # [B, S, bn]
        readout = self.readout_proj(branch_flat)                   # [B, S, d_latent]
        readout = self.readout_norm(readout)

        # STC tag update: EMA of mean |prediction error| across compartments
        if self.use_stc and self.training:
            with torch.no_grad():
                _err = (z_k - readout_raw).abs().mean().item()
            self._tag_norm = (
                self.stc_tag_decay * self._tag_norm
                + (1.0 - self.stc_tag_decay) * _err
            )

        # Per-compartment write
        decay_k = torch.sigmoid(self.decay_logit).view(1, 1, K, 1, 1)   # [K]→bcst
        lr_k_base = torch.sigmoid(self.lr_logit)                          # [K]

        if lr_modulator is not None:
            lr_mod = lr_modulator.clamp(0.0, 10.0).to(lr_k_base.dtype)  # [B, S]
            lr_eff = lr_k_base.view(1, 1, K, 1, 1) * lr_mod.view(B, S, 1, 1, 1)
            with torch.no_grad():
                effective_lr_scalar = lr_eff.mean().detach()
        else:
            lr_eff = lr_k_base.view(1, 1, K, 1, 1)
            effective_lr_scalar = lr_k_base.mean().detach()

        if self.update_rule == "delta":
            error = z_k - readout_raw.detach()                             # [B,S,K,m_k]
            outer = error.unsqueeze(-1) * a_k.unsqueeze(-2)               # [B,S,K,m_k,m_k]
            a_sq = (a_k * a_k).sum(dim=-1, keepdim=True).unsqueeze(-1)   # [B,S,K,1,1]
            outer = outer / (a_sq + 1e-6)
        else:
            outer = z_k.unsqueeze(-1) * a_k.unsqueeze(-2)                 # [B,S,K,m_k,m_k]

        new_hebbian = decay_k * hebbian_state.to(outer.dtype) + lr_eff * outer

        # BTSP: naturally extends — reshape [B,S,K,m_k,m_k] → [B,S,K*m_k*m_k]
        if self.use_btsp:
            btsp_lr = torch.sigmoid(self.btsp_lr_logit).to(outer.dtype)
            beta = torch.sigmoid(self.btsp_decay_logit).to(outer.dtype)

            if lr_modulator is not None:
                salience = lr_modulator.clamp(0.0, 10.0).to(outer.dtype)
                salience = (salience - self.btsp_salience_threshold).clamp(min=0.0)
            else:
                salience = torch.ones(B, S, device=outer.device, dtype=outer.dtype)

            # salience: [B, S] → [B, S, 1, 1, 1] to broadcast over [B,S,K,m_k,m_k]
            weighted = salience.view(B, S, 1, 1, 1) * outer

            idx = torch.arange(S, device=outer.device, dtype=outer.dtype)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp(min=0)
            btsp_K = torch.tril(beta ** dist)                              # [S, S]

            flat_dim = K * m_k * m_k
            w_flat = weighted.reshape(B, S, flat_dim)
            delta_flat = torch.matmul(btsp_K, w_flat)
            delta_M = delta_flat.reshape(B, S, K, m_k, m_k)
            new_hebbian = new_hebbian + btsp_lr * delta_M

        # Global output gate on the merged readout
        if self.force_gate is not None:
            gate = torch.tensor(self.force_gate, dtype=readout.dtype, device=readout.device)
        else:
            gate = torch.sigmoid(self.gate_logit)
        gated_readout = gate * readout

        return gated_readout, new_hebbian, effective_lr_scalar


@torch.no_grad()
def critical_symmetric_init_(
    W: torch.Tensor,
    sym_frac: float = 1.0,
    spectral_radius: float = 0.999,
) -> torch.Tensor:
    """In-place critical init of a 2D weight as a random dynamics matrix A.

    Implements the recipe from the critically-normalized random-matrix prior:
    dense Gaussian → subtract mean (global inhibition) → enforce (partial)
    symmetry → rescale so the leading eigenvalue magnitude ≈ ``spectral_radius``.
    A critically-scaled symmetric matrix yields a ~2/3 power-law variance
    spectrum and long-timescale dynamics, the proposed "scaffold for learning".

    Args:
        W: 2D tensor [out, in], modified in place.
        sym_frac: 1.0 → fully symmetric (real spectrum, ~2/3 exponent);
            0.0 → fully asymmetric. Interpolated in between. **Ignored when
            W is non-square** — symmetry is undefined, so the matrix is left
            asymmetric and only the spectral scaling is applied.
        spectral_radius: target leading-|eigenvalue| (square) or leading
            singular value (non-square). 0.999 keeps it just sub-critical.

    Returns:
        W (for chaining).
    """
    assert W.dim() == 2, f"critical_symmetric_init_ expects 2D, got {tuple(W.shape)}"
    out_d, in_d = W.shape
    A = torch.randn(out_d, in_d, device=W.device, dtype=torch.float32)
    A = A - A.mean()

    if out_d == in_d:
        # Square: build a (partially) symmetric matrix and scale by spectral radius.
        A.fill_diagonal_(0.0)
        A_sym = 0.5 * (A + A.t())
        A_eff = sym_frac * A_sym + (1.0 - sym_frac) * A
        rho = torch.linalg.eigvals(A_eff).abs().max().real
    else:
        # Non-square: no symmetry possible. Scale by the largest singular
        # value so the operator norm matches the requested radius.
        A_eff = A
        rho = torch.linalg.svdvals(A_eff).max()

    A_eff = A_eff * (spectral_radius / rho.clamp_min(1e-12))
    W.copy_(A_eff.to(W.dtype))
    return W


def _logit(p: float) -> float:
    """Stable logit for use in parameter init."""
    p = float(max(min(p, 1.0 - 1e-6), 1e-6))
    return math.log(p / (1.0 - p))


# ════════════════════════════════════════════════════════════════════════
# Predictive Coding via Temporal Hierarchy
# ════════════════════════════════════════════════════════════════════════

class CerebellarReadout(nn.Module):
    """
    Per-tick "cerebellar" projector of the target into latent space.

    Under Prospective Configuration, this module is repurposed. Instead of
    predicting the teacher from the student, it projects the target token
    (or teacher state) into the latent space. This projected representation
    acts as the top-down prior (mu_L) that clamps the highest predictive
    coding layer during the inference phase, driving the network toward a
    prospective configuration.

    Args:
        target_dim: Dimension of the target.
        d_latent: Student latent width (output of the projector).
        hidden_dim: Predictor hidden width (default 2× target_dim).
        dropout: Predictor dropout.
    """

    def __init__(
        self,
        target_dim: int,
        d_latent: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.target_dim = target_dim
        self.teacher_d_model = target_dim

        if hidden_dim is None:
            hidden_dim = max(2 * target_dim, d_latent)

        # Predictor: maps target into the latent space.
        self.predictor = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent),
        )

        nn.init.normal_(self.predictor[-1].weight, std=0.01)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project target into the latent space.

        Args:
            target: [B, S, target_dim]

        Returns:
            Projected state: [B, S, d_latent]
        """
        return self.predictor(target)


# ════════════════════════════════════════════════════════════════════════
# Neuromodulated Optimizer
# ════════════════════════════════════════════════════════════════════════

class NeuroPlasticOptimizer:
    """
    Surprise-modulated learning rate wrapper for biologically-inspired
    global neuromodulation.

    Biological motivation
    ---------------------
    The locus coeruleus releases norepinephrine (NE) in proportion to
    prediction error (surprise). NE gates synaptic plasticity across the
    entire cortex: high NE → higher effective learning rate (novel stimuli
    are encoded aggressively); low NE → suppressed plasticity (familiar
    inputs leave consolidated weights undisturbed).

    The CTM already implements this locally via per-position Hebbian LR
    modulation (HebbianSynapse + hebbian_cert_lr_alpha). This class
    extends the same principle globally to the slow-weight AdamW optimizer.

    Mechanism
    ---------
    At each optimizer step the base LR is temporarily scaled by:

        lr_eff = lr_base × clip(1 + alpha × surprise_ema, min_scale, max_scale)

    where:
        certainty ∈ [0, 1]  —  1 = fully confident, 0 = maximum surprise
        surprise  = 1 - certainty
        surprise_ema  is an EMA-smoothed surprise signal (stability)
        clip prevents destabilising extremes

    The base LR is restored immediately after the step so that external
    schedulers, logging, and gradient-norm scaling always see the unmodified
    LR. Checkpoints are fully compatible with vanilla AdamW (state_dict
    is a pure passthrough).

    Args:
        optimizer:  Inner PyTorch optimizer (AdamW, AdamW8bit, …).
        alpha:      Modulation strength. 0 → no modulation (uniform LR).
                    1 → at max surprise, lr scales by (1 + 1) = 2×.
        min_scale:  Floor for the LR multiplier. Prevents plasticity collapse
                    on highly predictable data.
        max_scale:  Ceiling for the LR multiplier. Prevents instability when
                    the model is very uncertain.
        ema_decay:  Smoothing factor for the surprise EMA. Higher values
                    produce slower, more stable modulation.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        alpha: float = 1.0,
        min_scale: float = 0.1,
        max_scale: float = 3.0,
        ema_decay: float = 0.95,
    ):
        self.optimizer = optimizer
        self.alpha = alpha
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ema_decay = ema_decay
        self._surprise_ema: float | None = None
        self._current_scale: float = 1.0

    def modulate(self, certainty: float) -> float:
        """Update the modulation scale from a new certainty observation.

        Should be called once per optimizer step, before :meth:`step`.

        Args:
            certainty: Mean model certainty ∈ [0, 1] for the current batch
                       (1 – entropy/max_entropy, averaged over ticks × positions).

        Returns:
            The LR scale factor applied on the next :meth:`step` call.
        """
        surprise = 1.0 - float(certainty)
        if self._surprise_ema is None:
            self._surprise_ema = surprise
        else:
            self._surprise_ema = (
                self.ema_decay * self._surprise_ema
                + (1.0 - self.ema_decay) * surprise
            )
        raw = 1.0 + self.alpha * self._surprise_ema
        self._current_scale = max(self.min_scale, min(self.max_scale, raw))
        return self._current_scale

    def step(self):
        """Step with the current modulation scale applied transiently."""
        if self.alpha == 0.0 or self._current_scale == 1.0:
            self.optimizer.step()
            return
        # Temporarily scale each param group's lr
        orig_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["lr"] * self._current_scale
        self.optimizer.step()
        # Restore: schedulers / logging always see the unmodified base LR
        for pg, lr in zip(self.optimizer.param_groups, orig_lrs):
            pg["lr"] = lr

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # ── Checkpoint compatibility ─────────────────────────────────────────
    def state_dict(self):
        """Pass through inner optimizer state dict (checkpoint-compatible)."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load inner optimizer state. Surprise EMA resets cleanly on resume."""
        self.optimizer.load_state_dict(state_dict)

    # ── Property delegation ─────────────────────────────────────────────
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def current_scale(self) -> float:
        """Current LR scale factor (for logging)."""
        return self._current_scale


# ════════════════════════════════════════════════════════════════════════
# Free-Energy Prioritized Replay Buffer
# ════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """Episodic replay buffer with free-energy priority and optional SWIL sampling.

    Biological motivation
    ---------------------
    The hippocampus does not replay experiences uniformly during sleep.
    It preferentially reactivates sequences that generated the highest
    prediction errors (surprise) during waking — boundary events, novel
    stimuli, and hard-to-learn patterns. Frequent, easy sequences are
    largely ignored during consolidation.

    CLS / SWIL extension
    --------------------
    Complementary Learning Systems theory predicts that replay should
    interleave memories that are STRUCTURALLY SIMILAR to the current
    context, not just high-surprise ones. When a query semantic vector is
    supplied to sample(), the sampling weight becomes:

        w_i = softmax( (F_i + λ · cos_sim(query, v_i)) / τ )

    where v_i is the stored semantic vector (mean-pooled final-tick latent).
    This focuses consolidation on episodes whose representations overlap
    with the ones currently being modified, preventing gradient interference
    while requiring fewer total replay steps.

    Mechanism
    ---------
    Each stored episode carries a scalar priority = variational free
    energy (or CE-loss proxy when HPC is inactive):

        F_ℓ = 0.5 · π_ℓ · ‖ε_ℓ‖² − 0.5 · log(π_ℓ)

    When the buffer is full, the incoming entry replaces the LOWEST-
    priority incumbent (most-consolidated), keeping hard examples alive.

    Base sampling (no query vector):
        w_i = softmax(F_i / τ)

    SWIL sampling (query vector provided):
        w_i = softmax( (F_i + λ · cos_sim(query, v_i)) / τ )

    τ=1   → direct softmax over raw FE (biological default)
    τ→∞   → uniform sampling (degrades to original behavior)
    τ→0   → greedy (always replay highest-FE / most-similar episode)

    Args:
        maxsize:     Maximum number of episodes to retain.
        temperature: Softmax temperature τ for sampling weights (default 1.0).
        sim_weight:  λ — scale of cosine-similarity term in SWIL mode. 0 = pure-FE.
    """

    def __init__(self, maxsize: int, temperature: float = 1.0, sim_weight: float = 1.0):
        # Each entry: (priority, seq, semantic_vec_or_None)
        self._buf: list[tuple[float, torch.Tensor, torch.Tensor | None]] = []
        self.maxsize = maxsize
        self.temperature = max(float(temperature), 1e-6)
        self.sim_weight = float(sim_weight)

    def push(
        self,
        priority: float,
        seq: torch.Tensor,
        semantic_vec: torch.Tensor | None = None,
    ) -> None:
        """Store an episode with optional semantic vector.

        When full, evicts the lowest-priority incumbent if the new priority
        is higher; otherwise discards the new entry.

        Args:
            priority:     Scalar free-energy (or CE-loss) priority.
            seq:          Input sequence tensor (stored on CPU).
            semantic_vec: Optional [d] float32 CPU tensor — mean-pooled
                          final-tick latent state for SWIL cosine similarity.
        """
        priority = float(priority)
        entry = (priority, seq, semantic_vec)
        if len(self._buf) < self.maxsize:
            self._buf.append(entry)
        else:
            min_idx = min(range(len(self._buf)), key=lambda i: self._buf[i][0])
            if priority > self._buf[min_idx][0]:
                self._buf[min_idx] = entry

    def sample(
        self,
        n: int,
        query_vec: torch.Tensor | None = None,
        sim_weight: float | None = None,
    ) -> list[torch.Tensor]:
        """Sample n episodes by priority, optionally biased by similarity.

        Args:
            n:          Number of episodes to sample.
            query_vec:  [d] float32 CPU tensor — current context's semantic
                        vector. When provided and stored vectors exist, the
                        sampling weight is:
                          softmax((F_i + λ·cos_sim(query, v_i)) / τ)
                        Falls back to pure-FE softmax if stored vectors are
                        absent or query_vec is None.
            sim_weight: Override for λ (uses self.sim_weight if None).

        Returns:
            List of sampled sequence tensors.
        """
        if not self._buf:
            return []
        n = min(n, len(self._buf))
        lam = sim_weight if sim_weight is not None else self.sim_weight

        scores = torch.tensor(
            [entry[0] for entry in self._buf], dtype=torch.float32
        )

        # SWIL: add cosine-similarity term when query and stored vecs available
        if query_vec is not None and lam != 0.0:
            stored_vecs = [entry[2] for entry in self._buf]
            if any(v is not None for v in stored_vecs):
                # Pad missing vecs with zeros so cosine sim = 0 (no bias)
                d = query_vec.shape[0]
                stack = torch.stack([
                    v if v is not None else torch.zeros(d)
                    for v in stored_vecs
                ])  # [N, d]
                q = query_vec.float().unsqueeze(0)   # [1, d]
                q_norm = F.normalize(q, dim=-1)
                s_norm = F.normalize(stack.float(), dim=-1)
                cos_sim = (q_norm * s_norm).sum(dim=-1)  # [N]
                scores = scores + lam * cos_sim

        weights = torch.softmax(scores / self.temperature, dim=0)
        replace = n > len(self._buf)
        indices = torch.multinomial(weights, num_samples=n, replacement=replace)
        return [self._buf[i][1] for i in indices.tolist()]

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def mean_priority(self) -> float:
        if not self._buf:
            return 0.0
        return sum(e[0] for e in self._buf) / len(self._buf)

    @property
    def max_priority(self) -> float:
        if not self._buf:
            return 0.0
        return max(e[0] for e in self._buf)

    @property
    def has_semantic_vecs(self) -> bool:
        """True if at least one stored episode has a semantic vector."""
        return any(e[2] is not None for e in self._buf)


# ════════════════════════════════════════════════════════════════════════
# Precision-Weighted Gradient Scaler
# ════════════════════════════════════════════════════════════════════════

class PrecisionWeightedGradientScaler:
    """Per-layer gradient scaling based on local HPC precision weights.

    Biological motivation
    ---------------------
    The LC-NE system gates plasticity locally: norepinephrine modulates
    individual synapses in proportion to their precision trace (calcium
    influx). Well-calibrated synapses (high precision, low error) are
    protected from overwriting. Uncertain synapses (low precision, high
    error) receive amplified updates.

    In artificial terms: parameters of a thought layer with high HPC
    precision π_ℓ > π_ref are scaled DOWN (stable schema — protect), while
    parameters of low-precision layers (novel/uncertain) are scaled UP.

    Mechanism
    ---------
    At each optimizer step, per-layer precision readings are EMA-smoothed
    to produce stable π̄_ℓ.  The per-layer gradient multiplier is then:

        scale_ℓ = clip(π_ref / π̄_ℓ, min_scale, max_scale)

    where π_ref = mean(π̄_ℓ) across all active layers.

    PC layer parameters are automatically excluded because their
    local_loss already incorporates π_ℓ (via 0.5 * π_ℓ * ε_ℓ²), and
    applying the inverse multiplier again would cancel it out.

    Requires use_hierarchical_pc=True so per-layer precision is available.
    """

    def __init__(
        self,
        n_layers: int,
        min_scale: float = 0.2,
        max_scale: float = 5.0,
        ema_decay: float = 0.95,
    ):
        self.n_layers = n_layers
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ema_decay = ema_decay
        self._precision_ema: list[float | None] = [None] * n_layers
        self._last_scales: list[float] = [1.0] * n_layers

    def update_and_scale(
        self,
        layers: list,
        per_layer_precision: list[float],
        excluded_param_ids: set | None = None,
    ) -> list[float]:
        """EMA-update precision estimates then apply per-layer gradient scaling.

        Args:
            layers:               ordered list of nn.Module thought layers
                                  (from _get_layers_sequence())
            per_layer_precision:  per-layer π scalars from HPC (same order)
            excluded_param_ids:   set of id(param) to skip — typically the
                                  PC generative-layer parameters whose local
                                  loss already embeds precision weighting

        Returns:
            Applied per-layer scales (for logging).
        """
        n = min(len(per_layer_precision), len(layers), self.n_layers)

        for i in range(n):
            pi = float(per_layer_precision[i])
            if self._precision_ema[i] is None:
                self._precision_ema[i] = pi
            else:
                self._precision_ema[i] = (
                    self.ema_decay * self._precision_ema[i]
                    + (1.0 - self.ema_decay) * pi
                )

        active = [e for e in self._precision_ema[:n] if e is not None and e > 1e-8]
        if not active:
            return [1.0] * n

        pi_ref = sum(active) / len(active)
        excluded = excluded_param_ids or set()
        seen = set()
        scales: list[float] = []

        for i in range(n):
            pi_l = self._precision_ema[i]
            if pi_l is None or pi_l < 1e-8:
                scale = float(self.max_scale)
            else:
                scale = float(pi_ref) / float(pi_l)
            scale = max(float(self.min_scale), min(float(self.max_scale), scale))
            scales.append(scale)
            self._last_scales[i] = scale

            layer = layers[i]
            lid = id(layer)
            if lid in seen:
                continue  # hyperloop: same module instance repeated — skip
            seen.add(lid)

            if abs(scale - 1.0) < 1e-6:
                continue

            for p in layer.parameters():
                if p.grad is not None and id(p) not in excluded:
                    p.grad.mul_(scale)

        return scales

    @property
    def mean_scale(self) -> float:
        """Mean absolute deviation of per-layer scales from 1.0 (for logging)."""
        valid = [s for s in self._last_scales if s != 1.0]
        return sum(valid) / len(valid) if valid else 1.0

    @property
    def precision_ema(self) -> list[float]:
        return [e if e is not None else 0.0 for e in self._precision_ema]


# ════════════════════════════════════════════════════════════════════════
# Structural Plasticity Controller
# ════════════════════════════════════════════════════════════════════════

class StructuralPlasticityController:
    """
    Online structural plasticity for MatrixResidualStream layers.

    Biological motivation
    ---------------------
    Biological networks continuously reorganise their topology via
    synaptogenesis (growth) and synaptic pruning, driven by activity
    homeostasis rather than external rewards. Under-active synapses are
    eliminated; under-capacity networks grow new connections.

    Mechanism
    ---------
    Uses the learned residual gate values of MatrixResidualStream as a
    utilization proxy:
        utilization_s = sigmoid(res_gate[s]) ∈ (0, 1)

    A gate near 0 means the stream is barely contributing new information
    (candidate for pruning). A gate near 1 means the stream is fully
    committed (capacity saturated — candidate for growing a new stream).

    Decision rules (evaluated every ``update_interval`` steps):
        Prune:  lowest-utilization active stream's EMA < prune_threshold
                AND n_active > min_active
        Grow:   mean active gate EMA > grow_threshold
                AND dormant streams exist

    Masked over-allocation
    ----------------------
    Tensors are never reallocated. MatrixResidualStream is pre-allocated
    with the full n_streams capacity. An ``active_mask`` boolean buffer
    logically activates/deactivates stream slots (just a multiply by 0/1).
    CUDA-graph safe — no control-flow change, only values change.

    Args:
        n_layers:         Number of ThoughtLayers with matrix streams.
        prune_threshold:  Gate EMA below this → stream is dormant; prune.
        grow_threshold:   Mean active gate EMA above this → at capacity; grow.
        ema_decay:        Smoothing for per-stream gate EMA. Slow decay
                          (0.99) prevents reacting to transient fluctuations.
        update_interval:  Training steps between grow/prune evaluations.
        min_active:       Minimum streams to keep active per layer.
    """

    def __init__(
        self,
        n_layers: int,
        prune_threshold: float = 0.05,
        grow_threshold: float = 0.85,
        ema_decay: float = 0.99,
        update_interval: int = 500,
        min_active: int = 1,
    ):
        self.prune_threshold = prune_threshold
        self.grow_threshold = grow_threshold
        self.ema_decay = ema_decay
        self.update_interval = update_interval
        self.min_active = min_active
        # Per-layer per-stream gate EMA (CPU float list, or None before first update)
        self._gate_ema: list[list[float] | None] = [None] * n_layers

    def update(self, stream_layers: list) -> None:
        """Update per-stream gate EMA from live res_gate values.

        Call every training step after the forward pass.

        Args:
            stream_layers: List of ThoughtLayer modules whose ``.stream``
                           is a MatrixResidualStream with ``active_mask``.
        """
        d = self.ema_decay
        for i, layer in enumerate(stream_layers):
            if i >= len(self._gate_ema):
                break
            stream = getattr(layer, "stream", None)
            if stream is None or not hasattr(stream, "active_mask"):
                continue

            gates = torch.sigmoid(stream.res_gate).detach().cpu().float().tolist()
            if self._gate_ema[i] is None:
                self._gate_ema[i] = gates[:]
            else:
                ema = self._gate_ema[i]
                while len(ema) < len(gates):
                    ema.append(0.5)
                for s, g in enumerate(gates):
                    ema[s] = d * ema[s] + (1.0 - d) * g

    def maybe_adapt(self, stream_layers: list) -> list[str]:
        """Evaluate grow/prune decisions for each stream layer.

        Returns a list of event strings for logging (empty if no change).
        Call every ``update_interval`` steps.
        """
        events: list[str] = []
        for layer_idx, layer in enumerate(stream_layers):
            if layer_idx >= len(self._gate_ema):
                break
            stream = getattr(layer, "stream", None)
            if stream is None or not hasattr(stream, "active_mask"):
                continue

            gate_ema = self._gate_ema[layer_idx]
            if gate_ema is None:
                continue

            mask = stream.active_mask
            n = mask.shape[0]
            active = [s for s in range(n) if mask[s].item()]
            dormant = [s for s in range(n) if not mask[s].item()]

            # ── Prune: deactivate least-used stream ──────────────────
            if len(active) > self.min_active:
                worst_s = min(active, key=lambda s: gate_ema[s])
                if gate_ema[worst_s] < self.prune_threshold:
                    stream.deactivate_stream(worst_s)
                    gate_ema[worst_s] = 0.0
                    events.append(f"L{layer_idx}:prune(s{worst_s},"
                                  f"gate={gate_ema[worst_s]:.3f})")

            # ── Grow: activate dormant stream if all active are saturated ──
            if dormant and active:
                mean_active_gate = sum(gate_ema[s] for s in active) / len(active)
                if mean_active_gate > self.grow_threshold:
                    new_s = dormant[0]
                    stream.activate_stream(new_s)
                    gate_ema[new_s] = 0.5
                    events.append(f"L{layer_idx}:grow(s{new_s},"
                                  f"mean_gate={mean_active_gate:.3f})")

        return events

    def active_counts(self, stream_layers: list) -> list[int]:
        """Return the number of active streams per layer (for logging)."""
        counts = []
        for layer in stream_layers:
            stream = getattr(layer, "stream", None)
            if stream is not None and hasattr(stream, "active_mask"):
                counts.append(int(stream.active_mask.sum().item()))
        return counts


# ════════════════════════════════════════════════════════════════════════
# Synaptic Tagging and Capture (STC) Gradient Gate
# ════════════════════════════════════════════════════════════════════════

class STCGradientGate:
    """
    Synaptic Tagging and Capture gradient gate for slow-weight consolidation.

    Biological motivation
    ---------------------
    Late-phase LTP (long-term structural change) requires two concurrent
    signals at the synapse:

      Tag (local):  A synapse experiencing high prediction error (early-LTP)
                    sets a transient biochemical marker.  This fades rapidly
                    when activity drops.
      PRP (global): Strong, surprising stimuli trigger soma-wide synthesis of
                    Plasticity-Related Proteins that diffuse and enable late-LTP
                    at tagged synapses.

    Only when BOTH signals exceed their thresholds does the synapse undergo
    permanent structural consolidation.  Routine, predictable inputs that
    generate neither local error nor global surprise cannot overwrite
    consolidated schemas.

    Computational analog
    --------------------
    Tag  = HebbianSynapse._tag_norm: per-layer fast-decaying EMA of
           |z - M·a|, the local delta-rule prediction error.
    PRP  = NeuroPlasticOptimizer._surprise_ema: global surprise EMA
           (0 if NeuroPlasticOptimizer is not enabled).

    After backward() but before grad_clip and optimizer.step():
    - For each ThoughtLayer with a tagged HebbianSynapse:
        if tag < threshold_tag  OR  prp < threshold_prp:
            scale all layer.parameters() gradients by min_gate
    - Fully gradient-path safe: modifies .grad in-place, no hooks needed.

    Args:
        threshold_tag: Minimum tag norm for the layer gate to open.
                       Increase to require stronger recent error signal.
        threshold_prp: Minimum global PRP for any gate to open.
                       Set to 0.0 if not using NeuroPlasticOptimizer.
        min_gate:      Gradient scale when the gate is closed.
                       0.01 = 1%% pass-through (near-zero but not zero,
                       so Adam state keeps updating slowly).
    """

    def __init__(
        self,
        threshold_tag: float = 0.05,
        threshold_prp: float = 0.3,
        min_gate: float = 0.01,
    ):
        self.threshold_tag = threshold_tag
        self.threshold_prp = threshold_prp
        self.min_gate = min_gate
        self._last_n_gated: int = 0
        self._last_n_total: int = 0

    def apply(self, layers: list, prp: float) -> None:
        """Scale slow-weight gradients by the STC gate.

        Args:
            layers: ThoughtLayer modules (from _stream_layers or similar).
            prp:    Global PRP = NeuroPlasticOptimizer._surprise_ema.
                    Pass 0.0 when the neuromodulated optimizer is absent.
        """
        prp_active = prp >= self.threshold_prp
        n_gated = 0
        n_total = 0

        for layer in layers:
            hebbian = getattr(layer, "hebbian", None)
            if hebbian is None or not getattr(hebbian, "use_stc", False):
                continue

            n_total += 1
            tag_active = hebbian._tag_norm >= self.threshold_tag

            if not (prp_active and tag_active):
                n_gated += 1
                for p in layer.parameters():
                    if p.grad is not None:
                        p.grad.mul_(self.min_gate)

        self._last_n_gated = n_gated
        self._last_n_total = n_total

    @property
    def gate_fraction(self) -> float:
        """Fraction of STC-enabled layers currently gated (for logging)."""
        if self._last_n_total == 0:
            return 0.0
        return self._last_n_gated / self._last_n_total

    def mean_tag_norm(self, layers: list) -> float:
        """Mean tag norm across all STC-enabled layers (for logging)."""
        vals = [
            layer.hebbian._tag_norm
            for layer in layers
            if getattr(layer, "hebbian", None) is not None
            and getattr(layer.hebbian, "use_stc", False)
        ]
        return sum(vals) / len(vals) if vals else 0.0


class EpisodicDND:
    """
    Episodic Differentiable Neural Dictionary (DND) for zero-shot retrieval.

    Biological motivation
    ---------------------
    Episodic Control in biological agents bypasses slow incremental learning
    for previously-encountered situations by directly retrieving successful
    responses from hippocampal episodic memory via pattern-completion.

    The DND is a non-parametric key-value bank:
      Keys  = truncated mean-pooled input embeddings (context fingerprint).
      Values = mean-pooled final-tick latent z from high-confidence passes.

    During forward(): if max cosine_sim(current_key, stored_keys) >= threshold,
    the thought loop is bypassed entirely — the retrieved value is expanded and
    passed through the output head to produce logits in O(1) instead of O(T).

    Memory cost: capacity × (key_dim + d_latent) × 4 bytes.
    At capacity=1000, key_dim=64, d_latent=1024: ≈4.3 MB — negligible.

    Retrieval uses Hopfield-style soft attention rather than hard argmax: the
    returned value is a softmax-weighted blend of the top-k stored values,
    making retrieval robust to small distributional shifts.

    Args:
        capacity:             Max number of stored key-value pairs.
        key_dim:              Dimensionality of stored keys.
        d_latent:             Dimensionality of stored values (final latent).
        confidence_threshold: Cosine similarity required to trigger bypass.
                              0.98 is deliberately stringent: only near-exact
                              matches short-circuit the thought loop.
        hopfield_beta:        Softmax inverse temperature for retrieval.
                              Higher → sharper (closer to argmax). 4.0 works well.
        min_novelty:          Min (1 − max_sim) required to accept a write.
                              Prevents near-duplicates from filling capacity.
                              0.05 = skip if sim > 0.95 entry already exists.
    """

    def __init__(
        self,
        capacity: int = 1000,
        key_dim: int = 64,
        d_latent: int = 64,
        confidence_threshold: float = 0.98,
        hopfield_beta: float = 4.0,
        min_novelty: float = 0.05,
    ):
        self.capacity = capacity
        self.key_dim = key_dim
        self.d_latent = d_latent
        self.confidence_threshold = confidence_threshold
        self.hopfield_beta = hopfield_beta
        self.min_novelty = min_novelty

        # Lazily initialized on first push/query (device not known at construction)
        self._keys: torch.Tensor | None = None    # [capacity, key_dim]
        self._values: torch.Tensor | None = None  # [capacity, d_latent]
        self._n_stored: int = 0
        self._write_ptr: int = 0  # FIFO write pointer

        # Diagnostics (updated by push/query)
        self._last_max_sim: float = 0.0
        self._last_hit: bool = False

    def _init_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._keys is None:
            self._keys = torch.zeros(
                self.capacity, self.key_dim, device=device, dtype=dtype
            )
            self._values = torch.zeros(
                self.capacity, self.d_latent, device=device, dtype=dtype
            )

    def push(self, key: torch.Tensor, value: torch.Tensor) -> bool:
        """Write key/value to the buffer (FIFO with novelty guard).

        Args:
            key:   [key_dim] — context fingerprint.
            value: [d_latent] — mean-pooled final-tick latent to cache.

        Returns:
            True if written, False if skipped (too similar to existing entry).
        """
        key = key.float()
        value = value.float()
        self._init_buffers(key.device, key.dtype)

        if self._n_stored > 0 and self.min_novelty > 0.0:
            k_norm = F.normalize(key.unsqueeze(0), dim=-1)             # [1, key_dim]
            K_norm = F.normalize(self._keys[:self._n_stored], dim=-1)  # [n, key_dim]
            max_sim = float((k_norm @ K_norm.T).squeeze(0).max().item())
            if max_sim >= (1.0 - self.min_novelty):
                return False  # Too similar — skip to avoid near-duplicate entries

        idx = self._write_ptr % self.capacity
        self._keys[idx] = key
        self._values[idx] = value
        self._write_ptr += 1
        self._n_stored = min(self._n_stored + 1, self.capacity)
        return True

    def query(self, key: torch.Tensor) -> tuple:
        """Hopfield-style retrieval.

        If max cosine similarity >= confidence_threshold, returns a
        softmax-weighted blend of the top matching stored values.
        Otherwise returns (None, max_sim).

        Args:
            key: [key_dim] — context fingerprint.

        Returns:
            (value: [d_latent] | None, max_sim: float)
        """
        if self._n_stored == 0 or self._keys is None:
            self._last_hit = False
            self._last_max_sim = 0.0
            return None, 0.0

        key = key.float()
        self._init_buffers(key.device, key.dtype)

        k_norm = F.normalize(key.unsqueeze(0), dim=-1)                   # [1, key_dim]
        K_norm = F.normalize(self._keys[:self._n_stored], dim=-1)        # [n, key_dim]
        sims = (k_norm @ K_norm.T).squeeze(0)                            # [n]
        max_sim = float(sims.max().item())
        self._last_max_sim = max_sim

        if max_sim < self.confidence_threshold:
            self._last_hit = False
            return None, max_sim

        attn = torch.softmax(self.hopfield_beta * sims, dim=0)           # [n]
        v_out = attn @ self._values[:self._n_stored]                     # [d_latent]
        self._last_hit = True
        return v_out, max_sim

    @property
    def n_stored(self) -> int:
        return self._n_stored