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

        if self.use_bottleneck:
            self.z_proj = nn.Linear(d_latent, bottleneck_dim, bias=False)
            self.a_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
            # Read out from M·a back into d_latent.
            self.readout_proj = nn.Linear(bottleneck_dim, d_latent, bias=False)
            # M shape per (B, S): [bottleneck_dim, bottleneck_dim]
            self._matrix_dims = (bottleneck_dim, bottleneck_dim)
        else:
            self.z_proj = None
            self.a_proj = None
            # Readout from M·a ∈ R^{d_latent} — identity projection.
            self.readout_proj = nn.Linear(d_latent, d_latent, bias=False)
            # M shape per (B, S): [d_latent, d_model]
            self._matrix_dims = (d_latent, d_model)

        # Learnable decay and learning-rate scalars (in logit space so
        # the sigmoid keeps them in (0, 1)).
        self.decay_logit = nn.Parameter(torch.tensor(_logit(decay_init)))
        self.lr_logit = nn.Parameter(torch.tensor(_logit(lr_init)))

        # Output gate: starts near zero so the unaugmented synapse path
        # dominates at init. Learnable scalar.
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

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

    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a fresh zero fast-weight matrix for a new sequence."""
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

        # ── Project into Hebbian space ────────────────────────────────
        if self.use_bottleneck:
            z_proj = self.z_proj(prev_state)        # [B, S, bn]
            a_proj = self.a_proj(attn_out)          # [B, S, bn]
        else:
            z_proj = prev_state                     # [B, S, d_latent]
            a_proj = attn_out                       # [B, S, d_model]

        # ── Read: r = M · a (per position) ────────────────────────────
        # hebbian_state: [B, S, m, n], a_proj: [B, S, n]
        # result: [B, S, m]
        readout_raw = torch.einsum(
            "bsmn,bsn->bsm",
            hebbian_state.to(a_proj.dtype),
            a_proj,
        )
        # Project back to d_latent (no-op shape-wise if not using
        # bottleneck, since m == d_latent there).
        readout = self.readout_proj(readout_raw)    # [B, S, d_latent]
        readout = self.readout_norm(readout)

        # ── Update: M_{t+1} = decay · M_t + lr_eff · z ⊗ a ────────────
        # We compute the update in the current activation dtype, then
        # store back into the buffer's dtype on return. autograd does
        # flow through the update — the gate, decay, and lr all receive
        # gradient signal from later ticks' readouts.
        #
        # lr_eff has shape [B, S, 1, 1] when modulated so it broadcasts
        # over the outer-product (m, n) dimensions. When unmodulated, lr
        # is a plain scalar. We keep this branch explicit because the
        # broadcasting cost is small but nonzero, and a scalar mul on
        # the unmodulated path is faster.
        decay = torch.sigmoid(self.decay_logit)
        lr_base = torch.sigmoid(self.lr_logit)
        if lr_modulator is not None:
            # lr_modulator is [B, S]; reshape to [B, S, 1, 1] for broadcast
            # over outer product. Clamp to a sane range to prevent runaway
            # updates from anomalous uncertainty values (e.g. NaNs from
            # an exploded softmax during early training).
            lr_mod = lr_modulator.clamp(min=0.0, max=10.0).to(lr_base.dtype)
            lr_eff = lr_base * lr_mod.unsqueeze(-1).unsqueeze(-1)
            with torch.no_grad():
                effective_lr_scalar = lr_eff.mean().detach()
        else:
            lr_eff = lr_base
            effective_lr_scalar = lr_base.detach()

        # outer product per position: [B, S, m, 1] * [B, S, 1, n] -> [B, S, m, n]
        if self.update_rule == "delta":
            # Delta rule / Widrow-Hoff / error-correcting Hebbian:
            #   M_{t+1} = decay·M_t + lr·(z − M·a) ⊗ a / (||a||² + ε)
            #
            # The ||a||² normalization is essential for numerical stability.
            # Without it, the recursion on r = M·a is
            #   r_{t+1} = (decay − lr·||a||²)·r_t + lr·||a||²·z
            # which only contracts if |decay − lr·||a||²| < 1. For any
            # bottleneck dim or unnormalized activations, ||a||² grows
            # with the dimension and the rule diverges.
            #
            # With the normalization, the recursion becomes
            #   r_{t+1} = (decay − lr)·r_t + lr·z
            # which is contractive for any lr ∈ (0, 1+decay) and converges
            # to r∞ = lr/(lr + 1 − decay) · z. With decay ≈ 0.9, lr ≈ 0.1
            # this converges in ~50 steps to roughly z/2 — useful working
            # memory without explosion. (Some literature calls this the
            # "normalized LMS" or "NLMS" rule.)
            #
            # We already computed M·a as `readout_raw` above. We detach
            # it inside the error term so the BPTT graph doesn't chain
            # through every prior tick's M state — see the long-loop
            # memory comment in the docstring.
            error = z_proj - readout_raw.detach()
            outer = error.unsqueeze(-1) * a_proj.unsqueeze(-2)
            # Per-position normalization by ||a||² + eps. a_proj is
            # [B, S, n]; the squared norm is [B, S, 1, 1] so it
            # broadcasts over (m, n).
            a_sq = (a_proj * a_proj).sum(dim=-1, keepdim=True).unsqueeze(-1)
            outer = outer / (a_sq + 1e-6)
        else:
            # Classic Hebbian outer-product accumulation.
            outer = z_proj.unsqueeze(-1) * a_proj.unsqueeze(-2)
        new_hebbian = decay * hebbian_state.to(outer.dtype) + lr_eff * outer

        # ── Gate the readout ──────────────────────────────────────────
        # Sigmoid-gate so the synapse output is not blown up by an
        # uninitialised fast-weight matrix on tick 0. The gate is shared
        # across all positions; per-position gating would over-parameterise.
        # Diagnostic mode: force_gate overrides the learned gate. Used
        # to isolate gate-gradient issues from readout-content issues.
        if self.force_gate is not None:
            gate = torch.tensor(self.force_gate, dtype=readout.dtype, device=readout.device)
        else:
            gate = torch.sigmoid(self.gate_logit)
        gated_readout = gate * readout

        return gated_readout, new_hebbian, effective_lr_scalar


def _logit(p: float) -> float:
    """Stable logit for use in parameter init."""
    p = float(max(min(p, 1.0 - 1e-6), 1e-6))
    return math.log(p / (1.0 - p))


# ════════════════════════════════════════════════════════════════════════
# Predictive Coding via Temporal Hierarchy
# ════════════════════════════════════════════════════════════════════════

class CerebellarReadout(nn.Module):
    """
    Per-tick "cerebellar" predictor of the teacher's hidden state.

    At thought tick t, this module takes the student's current latent z_t
    and produces a prediction zhat_{t+1} of what z_{t+1} *should* look
    like in the teacher's representation space. The prediction error
    between zhat and the actual teacher hidden state (supplied via
    teacher_z) becomes a local auxiliary loss. Crucially, this gradient
    short-circuits BPTT: it flows into the layer parameters that produced
    z_t directly, rather than only via the final-tick LM loss.

    This is "Predictive Coding" in the Rao-Ballard / Friston sense, but
    along the *temporal* axis of the CTM's thought loop rather than the
    *spatial* axis of cortical layers. The CTM's recurrence makes this
    swap natural: each tick is an opportunity to refine the prediction.

    Predictor architecture: a small MLP, kept intentionally cheap (this
    is anatomically "cerebellar" — a thin pattern-completing feedforward
    path, not a deep recurrent module).

    Args:
        d_latent: Student latent width (input to the predictor).
        target_dim: Dimension of the target. Equal to `teacher_d_model`
            when predicting teacher hidden states; equal to `d_latent`
            when predicting the student's own future latents (intrinsic
            PC, no teacher needed).
        hidden_dim: Predictor hidden width (default 2× d_latent).
        normalize: If "layernorm", LayerNorm both sides before computing
            error (recommended for cross-architecture distillation).
            If "none", use raw MSE.
        loss_type: "mse" or "cosine".
        dropout: Predictor dropout.
    """

    def __init__(
        self,
        d_latent: int,
        target_dim: int,
        hidden_dim: Optional[int] = None,
        normalize: str = "layernorm",
        loss_type: str = "cosine",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.target_dim = target_dim
        # Back-compat alias: legacy code/tests check for `teacher_d_model`.
        # The semantic role of target_dim is "whatever the predictor
        # outputs"; for the teacher-target mode it's literally
        # teacher_d_model. Exposing both names costs nothing.
        self.teacher_d_model = target_dim
        self.loss_type = loss_type
        self.normalize = normalize

        if hidden_dim is None:
            hidden_dim = max(2 * d_latent, target_dim)

        # Predictor: maps current student z_t into the target space.
        # Two layers with GELU + dropout — cheap, single hidden.
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

        if normalize == "layernorm":
            self.student_norm = nn.LayerNorm(target_dim, elementwise_affine=False)
            self.teacher_norm = nn.LayerNorm(target_dim, elementwise_affine=False)
        else:
            self.student_norm = None
            self.teacher_norm = None

        # Init the final layer to small weights so the predictor starts
        # near zero — the auxiliary loss is then dominated by the
        # teacher_norm output until the predictor learns. Prevents the
        # PC term from drowning the LM loss at init.
        nn.init.normal_(self.predictor[-1].weight, std=0.01)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the prediction error for one tick.

        Args:
            z_t: [B, S, d_latent] — student latent at current tick.
            target: [B, S, target_dim] — the thing to be predicted.
                Caller decides the semantics: teacher hidden state, the
                student's own next-tick latent (detached), etc.

        Returns:
            Scalar tensor: per-tick prediction error (mean over B, S).
        """
        zhat = self.predictor(z_t)  # [B, S, target_dim]

        if self.student_norm is not None:
            zhat_n = self.student_norm(zhat.float())
            tgt_n = self.teacher_norm(target.float())
        else:
            zhat_n = zhat.float()
            tgt_n = target.float()

        if self.loss_type == "cosine":
            # 1 - cos_sim. Magnitude-invariant; the right default when
            # student and teacher come from different architectures, same
            # logic as distill_feature_method="cosine".
            cos = F.cosine_similarity(zhat_n, tgt_n, dim=-1)
            err = (1.0 - cos).mean()
        else:
            err = F.mse_loss(zhat_n, tgt_n)

        return err