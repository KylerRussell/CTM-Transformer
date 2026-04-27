"""
Engram — Conditional Memory via Hashed N-gram Lookup

Reference:
    Cheng et al. "Conditional Memory via Scalable Lookup: A New Axis of
    Sparsity for Large Language Models." arXiv:2601.07372 (DeepSeek-AI, 2026).

This module gives the CTM-Transformer a constant-time O(1) static memory
primitive, structurally separated from neural computation. The intuition,
per the paper: a substantial fraction of language is local, static, and
stereotyped (named entities, formulaic phrases). Forcing a Transformer
backbone to *reconstruct* this content through depth wastes computational
capacity that could be allocated to higher-level reasoning. Engram offloads
this static reconstruction to a hashed embedding lookup.

──────────────────────────────────────────────────────────────────────────
ARCHITECTURE (split across three modules for clean caching)
──────────────────────────────────────────────────────────────────────────

  EngramTable
    - Hashed N-gram lookup. Token IDs → e_t [B, S, d_mem].
    - One nn.Embedding holds *all* (order, head, slot) weights as a single
      flat table; per-(order, head) offsets are pre-baked as buffers.
    - The hash is a deterministic multiplicative-XOR. No learned hash params.
    - Called ONCE per forward pass (e_t is constant across thought steps).

  EngramProjection
    - Per-injection-point W_K, W_V projections of e_t.
    - Computed ONCE per forward pass alongside EngramTable.
    - Per the paper, W_V is zero-initialized so the engram contribution
      starts at exactly zero — preserves the original CTM behavior at init.
    - One per fusion layer (each layer has its own K/V, paper-style branching).

  EngramGate
    - Per-thought-step gating + (optional) depthwise causal conv.
    - Lives inside ThoughtLayer; called once per thought iteration.
    - Uses the cross-attention output (CTM's evolving "thinking state") as
      the gating query. Because attn_out varies across thought steps but k, v
      are static, the gating decision varies dynamically — each thought
      iteration can re-evaluate whether to trust the memory.
    - Conv is zero-initialized so the SiLU(Conv(·)) + ṽ residual reduces
      to ṽ at init.

──────────────────────────────────────────────────────────────────────────
INTEGRATION
──────────────────────────────────────────────────────────────────────────

  CTMTransformer:
      e_t = engram_table(input_ids)                     # once
      kv  = {l: engram_projections[l](e_t)              # once per fusion layer
             for l in engram_layers}
      for thought_step in 1..T:
          for layer in layers:
              attn_out = cross_attention(...)           # dynamic query
              if layer in kv:
                  attn_out = attn_out + engram_gate(attn_out, *kv[l])
              synapse(attn_out, prev_state)             # NLM, etc.

──────────────────────────────────────────────────────────────────────────
WHY THIS DESIGN, NOT THE PAPER'S RESIDUAL-BEFORE-ATTENTION
──────────────────────────────────────────────────────────────────────────

The paper integrates Engram into a residual stream BEFORE attention:
H ← H + Y, then Attention(H), then MoE(H). In their setting, H *is*
the layer's hidden state, so the gating query (h_t) is naturally the
pre-attention hidden state.

In CTM, the analog of H — the latent state z — is per-position and gets
fed to a synapse model. The "what just happened with text" signal is
attn_out (post-cross-attention), so it serves as the most natural,
context-grounded gating query. We add Engram's contribution to attn_out
and let synapse mix it with prev_state. Residual add (not concat) keeps
the synapse input dim unchanged — important for backward-compatibility
with existing CTM checkpoints.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Multiplicative-XOR hash
# ──────────────────────────────────────────────────────────────────────────
#
# Per-head multipliers: large odd 32-bit constants drawn from well-known
# public hash constants (Knuth's Fibonacci-derived multiplier, MurmurHash3,
# PCG primes). Different multipliers per head decorrelate the K independent
# hash maps, so a token N-gram that collides under one head is unlikely to
# collide under any of the others.
_DEFAULT_HEAD_MULTIPLIERS = [
    2654435761,   # Knuth's Fibonacci-derived multiplier (golden ratio · 2^32)
    40503,        # Knuth's smaller variant
    2246822519,   # MurmurHash3 32-bit C1
    3266489917,   # MurmurHash3 32-bit C2
    668265263,    # PCG / Numerical Recipes
    374761393,    # MurmurHash2 alternate
    3432918353,   # rotation-friendly prime
    461845907,    # MurmurHash3 32-bit final mixer
    2106027353,   # additional spread (random odd 32-bit)
    1789363923,
    3988292384,
    2654435789,
    4099279,
    899809343,
    3884862473,
    998254319,
]
_HASH_MASK = (1 << 32) - 1   # restrict state to uint32 range


@torch.no_grad()
def compute_ngram_indices(
    token_ids: torch.Tensor,
    n_order: int,
    n_heads: int,
    table_size: int,
    multipliers: torch.Tensor,
    bos_id: int = 0,
) -> torch.Tensor:
    """Multiplicative-XOR hash of suffix N-grams to per-head table indices.

    Suffix convention (paper Eq. 1): the N-gram ending at position t covers
    tokens (x_{t-n+1}, ..., x_t). Positions t < n-1 are left-padded with
    `bos_id` so every position gets a defined N-gram window.

    Args:
        token_ids: [B, S] int64 tokens
        n_order: N-gram order (e.g. 2 or 3)
        n_heads: K independent hash heads per order
        table_size: M_{n,k} — slots per head's table; should be prime
        multipliers: [n_heads] int64, one large odd constant per head
        bos_id: padding id for short suffixes at the start of the sequence

    Returns:
        [B, S, n_heads] int64 indices in [0, table_size)
    """
    B, S = token_ids.shape

    # Left-pad so position 0's suffix N-gram is well-defined (filled with BOS)
    padded = F.pad(token_ids, (n_order - 1, 0), value=bos_id)  # [B, S + n_order - 1]

    # Build the suffix-N-gram windows: windows[b, t, i] = padded[b, t+i]
    #   = token at position (t - n_order + 1 + i) in the original sequence.
    windows = torch.stack(
        [padded[:, i:i + S] for i in range(n_order)], dim=-1
    )  # [B, S, n_order]

    # Per-head MX hash:  state ← (state ^ x) * mul, mod 2^32 each round
    #
    # Using int64 arithmetic with explicit masking. For very long sequences
    # / large vocabs this is cheap because token_ids fit comfortably under
    # 2^17 (vocab_size at most 128k) and the multiply stays within int64
    # before the mask trims it back to uint32 width.
    mul = multipliers.view(1, 1, -1)                                # [1, 1, K]
    state = torch.zeros(B, S, n_heads, dtype=torch.long, device=token_ids.device)
    for i in range(n_order):
        x = windows[..., i].unsqueeze(-1)                           # [B, S, 1]
        state = ((state ^ x) * mul) & _HASH_MASK                    # [B, S, K]

    return state % table_size


# ──────────────────────────────────────────────────────────────────────────
# EngramTable — multi-head, multi-order N-gram embedding lookup
# ──────────────────────────────────────────────────────────────────────────

class EngramTable(nn.Module):
    """Static memory: token-ID-derived hash → embedding lookup.

    Maintains len(orders) × n_heads independent embedding tables, packed
    into a single flat nn.Embedding so PyTorch's standard sparse-aware
    backward path works without indirection. Per-(order, head) base offsets
    are pre-baked into a buffer so the per-step lookup is just an index add
    plus an embedding gather.

    Args:
        ngram_orders: list of N-gram orders to track, e.g. [2, 3]
        n_heads:      K independent hash heads per order
        slots_per_table: M, the per-(order, head) table size. Choose a prime
                      to minimize hash-collision regularity. Default 65521
                      (largest prime ≤ 2^16) gives a ~256MB total table at
                      orders=[2,3], heads=8, d_head=64, fp32.
        d_head:       embedding dim per head. Final concatenated dim is
                      d_mem = len(ngram_orders) × n_heads × d_head.
        bos_id:       padding ID for short suffixes at sequence start.

    Output (forward):
        e_t [B, S, d_mem] — concatenated retrieved embeddings.
    """

    def __init__(
        self,
        ngram_orders: list[int],
        n_heads: int,
        slots_per_table: int,
        d_head: int,
        bos_id: int = 0,
    ):
        super().__init__()
        if not ngram_orders or any(n < 1 for n in ngram_orders):
            raise ValueError(f"ngram_orders must be a non-empty list of positives; got {ngram_orders}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {n_heads}")
        if slots_per_table < 2:
            raise ValueError(f"slots_per_table must be ≥ 2; got {slots_per_table}")

        self.ngram_orders = list(ngram_orders)
        self.n_heads = n_heads
        self.slots_per_table = slots_per_table
        self.d_head = d_head
        self.bos_id = bos_id

        self.n_tables = len(ngram_orders) * n_heads
        self.d_mem = self.n_tables * d_head

        # Single flat embedding holds *all* (order, head, slot) rows.
        # Indexing scheme: row at (order_idx, head_k, slot_s) =
        #     order_idx · (n_heads · slots) + head_k · slots + s
        total_slots = self.n_tables * slots_per_table
        self.tables = nn.Embedding(total_slots, d_head)

        # Per-head multipliers (extend deterministically if user requests
        # more heads than the curated default list provides).
        if n_heads > len(_DEFAULT_HEAD_MULTIPLIERS):
            mults = list(_DEFAULT_HEAD_MULTIPLIERS)
            seed = 12345
            while len(mults) < n_heads:
                # Linear-congruential generator (Numerical Recipes parameters)
                # — produces well-spread odd 32-bit constants
                seed = (seed * 1103515245 + 12345) & _HASH_MASK
                candidate = seed | 1   # force odd
                if candidate not in mults:
                    mults.append(candidate)
            multipliers = mults[:n_heads]
        else:
            multipliers = _DEFAULT_HEAD_MULTIPLIERS[:n_heads]

        self.register_buffer(
            "multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        # Flat-index offsets so the gather indexes the single big table.
        order_offsets = torch.tensor(
            [oi * n_heads * slots_per_table for oi in range(len(ngram_orders))],
            dtype=torch.long,
        )
        head_offsets = torch.arange(n_heads, dtype=torch.long) * slots_per_table

        self.register_buffer("order_offsets", order_offsets, persistent=True)
        self.register_buffer("head_offsets", head_offsets, persistent=True)

        self._reset_special_inits()

    def _reset_special_inits(self):
        """Small init so initial lookups don't dominate the residual stream.

        Called from __init__ and ALSO re-applied after the parent module's
        generic apply(_init_weights) walks the tree (which would otherwise
        clobber this with std=0.02).
        """
        nn.init.normal_(self.tables.weight, std=0.01)

    @property
    def num_lookup_params(self) -> int:
        """Total parameters in the lookup table (the bulk of Engram's params)."""
        return self.tables.num_embeddings * self.tables.embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, S] long → e_t [B, S, d_mem]."""
        B, S = token_ids.shape
        per_order_embeds = []

        for oi, n_order in enumerate(self.ngram_orders):
            # Per-head hash indices for this order:  [B, S, n_heads]
            indices = compute_ngram_indices(
                token_ids,
                n_order=n_order,
                n_heads=self.n_heads,
                table_size=self.slots_per_table,
                multipliers=self.multipliers,
                bos_id=self.bos_id,
            )
            # Promote to flat-table indices (add per-order and per-head bases)
            flat = indices + self.head_offsets.view(1, 1, -1) + self.order_offsets[oi]
            # Gather: [B, S, n_heads, d_head] → flatten the head dim
            embeds = self.tables(flat)                                # [B, S, K, d_head]
            per_order_embeds.append(embeds.flatten(2))                # [B, S, K·d_head]

        return torch.cat(per_order_embeds, dim=-1)                    # [B, S, d_mem]


# ──────────────────────────────────────────────────────────────────────────
# EngramProjection — layer-specific K, V projections (computed once)
# ──────────────────────────────────────────────────────────────────────────

class EngramProjection(nn.Module):
    """Layer-specific W_K, W_V projections of the static Engram memory.

    Crucial efficiency: e_t is constant across the thought loop, so we
    project it ONCE per forward pass instead of T·n_layers times. This
    reduces Engram's incremental compute from ~T·n_layers·O(B·S·d_mem·d_out)
    to T·n_layers·O(B·S·d_out)  (the per-step gate is just a dot product).

    Per the paper, W_V is zero-initialized — combined with the zero-init
    on the conv inside EngramGate, this guarantees the Engram contribution
    is exactly 0 at the start of training. The model bootstraps from its
    pre-Engram behavior and only gradually learns to use the memory.

    Args:
        d_mem:    Engram lookup output width
        d_query:  width of the K projection (matches the gating query)
        d_out:    width of the V projection (final Engram contribution width)
        zero_init_v:  if True, init W_V to 0 for identity-at-init
    """

    def __init__(
        self,
        d_mem: int,
        d_query: int,
        d_out: int,
        zero_init_v: bool = True,
    ):
        super().__init__()
        self.W_K = nn.Linear(d_mem, d_query, bias=False)
        self.W_V = nn.Linear(d_mem, d_out, bias=False)
        self.zero_init_v = zero_init_v
        self._reset_special_inits()

    def _reset_special_inits(self):
        # Standard small init for K — we want a meaningful gating signal
        # from the start so the model can learn to *trust or distrust* memory.
        nn.init.normal_(self.W_K.weight, std=0.02)
        # Zero init for V: makes the Engram contribution exactly 0 at step 0,
        # so the model starts from its pre-Engram solution and grows into
        # using memory rather than being thrown by an uninformed lookup.
        if self.zero_init_v:
            nn.init.zeros_(self.W_V.weight)
        else:
            nn.init.normal_(self.W_V.weight, std=0.02)

    def forward(self, e_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """e_t [B, S, d_mem] → (k [B, S, d_query], v [B, S, d_out])."""
        return self.W_K(e_t), self.W_V(e_t)


# ──────────────────────────────────────────────────────────────────────────
# EngramGate — per-thought-step gating + (optional) depthwise causal conv
# ──────────────────────────────────────────────────────────────────────────

class EngramGate(nn.Module):
    """Context-aware gating + depthwise causal conv (paper §2.3).

    Runs once per thought step inside each ThoughtLayer that uses Engram.
    Cheap: the only matmul-shaped op is a single dot product per position.

    Pipeline (paper Eq. 4–5):
        α_t = σ( <RMSNorm(q_t), RMSNorm(k_t)> / √d_query )       # [B, S, 1]
        ṽ_t = α_t · v_t                                          # [B, S, d_out]
        Y_t = SiLU(Conv1D(RMSNorm(ṽ_t))) + ṽ_t                   # [B, S, d_out]

    The conv is depthwise causal with kernel=4 and dilation=N_max (matches
    the paper's recommended config). It's zero-initialized so Y = ṽ at the
    start of training; combined with W_V zero-init in EngramProjection,
    this means Y_0 = 0 too — the entire Engram path is exactly 0 at init.

    Args:
        d_query:  K-projection / query width (must match q's last dim)
        d_out:    V-projection / output width (must match v's last dim)
        kernel_size: conv kernel (paper: 4)
        dilation:    conv dilation (paper: max N-gram order)
        use_conv:    set False to skip the conv refinement (cheaper, slight
                     loss in expressivity per the paper's ablation)
    """

    def __init__(
        self,
        d_query: int,
        d_out: int,
        kernel_size: int = 4,
        dilation: int = 3,
        use_conv: bool = True,
    ):
        super().__init__()
        self.d_query = d_query
        self.d_out = d_out
        self.use_conv = use_conv

        # RMSNorms on q and k stabilize the dot product magnitude — without
        # them, the gate saturates as the projections grow during training.
        self.q_norm = nn.RMSNorm(d_query)
        self.k_norm = nn.RMSNorm(d_query)
        self.scale = 1.0 / math.sqrt(d_query)

        if use_conv:
            self.conv_norm = nn.RMSNorm(d_out)
            # Depthwise: each channel evolves independently. groups=d_out.
            self.conv = nn.Conv1d(
                d_out, d_out,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=d_out,
                padding=0,            # we'll left-pad manually for causality
                bias=True,
            )
            # Effective receptive field: (kernel - 1) · dilation + 1
            # The left-padding amount is the receptive field minus 1.
            self.causal_pad = (kernel_size - 1) * dilation
            self._reset_special_inits()

    def _reset_special_inits(self):
        if self.use_conv:
            # Zero-init keeps the SiLU(Conv(·)) branch at 0 → Y = ṽ at init.
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        query: torch.Tensor,    # [B, S, d_query]
        k: torch.Tensor,        # [B, S, d_query]
        v: torch.Tensor,        # [B, S, d_out]
    ) -> torch.Tensor:
        """Returns Y [B, S, d_out] — the Engram contribution to add to attn_out."""
        # Per-position scalar gate (paper Eq. 4)
        q_n = self.q_norm(query)
        k_n = self.k_norm(k)
        # Dot product on the d_query axis → [B, S, 1] after keepdim sum
        gate = torch.sigmoid(
            (q_n * k_n).sum(dim=-1, keepdim=True) * self.scale
        )

        v_gated = gate * v                                            # [B, S, d_out]

        if not self.use_conv:
            return v_gated

        # Depthwise causal conv: paper Eq. 5
        v_norm = self.conv_norm(v_gated)
        x = v_norm.transpose(1, 2)                                    # [B, d_out, S]
        x = F.pad(x, (self.causal_pad, 0))                            # left-pad only (causal)
        x = self.conv(x)                                              # [B, d_out, S]
        x = x.transpose(1, 2)                                         # [B, S, d_out]
        return F.silu(x) + v_gated                                    # SiLU branch + residual
