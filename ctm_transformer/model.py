"""
CTMTransformer — Full Continuous Thought Machine Transformer Model

Assembles all components into the complete architecture:
  Input tokens → Embedding → Text KV
  Initialize per-position latent states z_0 = 0  [B, S, d_latent]
  For t in 1..T thought steps:
    Sync → Query → Causal Cross-Attention → Synapse → NLM → Output
  Return logits across all thought steps for temporal loss.

CAUSAL CORRECTNESS:
  Each sequence position maintains its own independent latent state.
  Cross-attention in the ThoughtLayer uses a causal mask so that position i's
  query can only attend to text keys at positions 0..i. This prevents any
  information leakage from future tokens to past predictions.

  The output head concatenates each position's latent state z[i] with its
  text embedding and projects to vocab logits. Since z[i] only contains
  information from positions 0..i (via causal masking), autoregressive
  correctness is maintained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import nullcontext

from ctm_transformer.config import CTMConfig
from ctm_transformer.thought_layer import ThoughtLayer
from ctm_transformer.memory import SynchronizationComputer


class CTMTransformer(nn.Module):
    """
    Continuous Thought Machine Transformer.

    Architecture flow:
    1. Embed input text → K, V for cross-attention (computed once)
    2. Initialize per-position latent states z_0 [B, S, d_latent]
    3. For T thought steps:
       a. Each position computes sync from its own post-activation history
       b. Sync → query, which cross-attends to text with CAUSAL MASK
       c. Synapse model mixes attention output with previous state
       d. NLM processes temporal pre-activation history
    4. Output head: concat z[i] + text_emb[i] → logits[i]
    """

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config

        # ── Text Ingestion ──────────────────────────────────────────────
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        if config.use_positional_encoding:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_embedding = None

        self.embed_norm = nn.LayerNorm(config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Thought Layers (stacked, iterated T times) ──────────────────
        self.layers = nn.ModuleList([
            ThoughtLayer(
                d_latent=config.d_latent,
                d_model=config.d_model,
                n_heads=config.n_heads,
                nlm_hidden_dim=config.nlm_hidden_dim,
                history_len=config.history_len,
                nlm_groups=config.nlm_groups,
                sync_method=config.sync_method,
                sync_rank=config.sync_rank,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # ── Attention Residuals (Kimi AttnRes) ──────────────────────────
        # Replaces fixed uniform residual accumulation (∑ h_i) with a
        # learned, softmax-weighted attention over all preceding layer
        # outputs. Bounds latent magnitude via convex combination — critical
        # for CTM since z is looped T thought steps (uncontrolled growth
        # would compound across both depth and time).
        #
        # Strict zero-init on queries → initial step is equal-weight
        # averaging, preventing training volatility (per Kimi paper).
        self.attn_res_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(config.d_latent))
            for _ in range(config.n_layers)
        ])

        # RMSNorm applied to keys (previous layer outputs) before scoring.
        # Shared across layers — stabilizes dot-product magnitudes without
        # adding per-layer parameters.
        self.attn_res_norm = nn.RMSNorm(config.d_latent)
        # ────────────────────────────────────────────────────────────────

        # ── Output Head ─────────────────────────────────────────────────
        # Per-position logits: concat z[i] (from thought loop) with text_emb[i],
        # then project to vocab. Gradients flow: output → z → NLM → layers.
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_latent + config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

        # ── Initial State ───────────────────────────────────────────────
        # Learned initial latent state, broadcast to all positions
        self.z0 = nn.Parameter(torch.zeros(config.d_latent))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard transformer weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_embedding is not None:
            n_params -= self.pos_embedding.weight.numel()
        return n_params

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens. Returns [batch, seq_len, d_model]."""
        B, S = input_ids.shape
        tok_emb = self.token_embedding(input_ids)

        if self.pos_embedding is not None:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            tok_emb = tok_emb + self.pos_embedding(positions)

        return self.embed_dropout(self.embed_norm(tok_emb))

    def _output_logits(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Project per-position latent states to logits.

        Each position i's logit is computed from [z[i]; text_emb[i]].
        Since z[i] was built using only causal attention (positions 0..i),
        no target leakage occurs.

        Args:
            z: [B, S, d_latent] — per-position latent states.
            text_emb: [B, S, d_model] — text embeddings.

        Returns:
            logits: [B, S, vocab_size]
        """
        # z is already [B, S, d_latent], just concat with text_emb
        combined = torch.cat([z, text_emb], dim=-1)  # [B, S, d_latent + d_model]
        return self.output_proj(combined)  # [B, S, vocab_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        max_thought_steps: int | None = None,
    ) -> dict:
        """
        Full forward pass with per-position latent states and causal masking.
        Memory-optimized: discards intermediate logits during training if checkpointing.
        """
        T = max_thought_steps or self.config.max_thought_steps
        B, S = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # ── Embed text (computed once) ──────────────────────────────────
        text_emb = self._embed_text(input_ids)  # [B, S, d_model]

        # ── Initialize per-position latent states ───────────────────────
        z = self.z0.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()  # [B, S, d_latent]

        # Reset memory buffers
        BS = B * S
        for layer in self.layers:
            layer.reset_memory(BS, device, dtype)

        # ── Thought Loop ────────────────────────────────────────────────
        all_logits = []
        all_certainties = []

        for t in range(T):
            is_last_step = (t == T - 1)
            is_checkpointed = (
                not is_last_step
                and self.training
                and self.config.gradient_checkpointing
            )

            # Dynamic grad-mode context. Background: detaching `z` at step
            # boundaries severs the graph through z, but the FIFO memory
            # buffers (pre_history, post_history) push via cat/roll which
            # holds the OLD buffer as a graph parent — chaining the graph
            # back through every prior thought step. End result without
            # this fix: ~T·n_layers (~192 at T=8, n_layers=24) layer-
            # forwards of saved activations alive at backward, plus all
            # the AttnRes K=norm(V) tensors (~225 MB/step → ~1.8 GB across
            # 8 steps just for K). That's the OOM at line 252.
            #
            # nullcontext (NOT torch.enable_grad) for active steps so eval-
            # time outer no_grad isn't overridden.
            step_context = torch.no_grad() if is_checkpointed else nullcontext()

            with step_context:
                # ── Attention Residuals: layer stack ─────────────────────
                # At each depth l, feed a softmax-weighted sum over ALL
                # previous layer outputs [z_0, z_1, ..., z_{l-1}] using
                # the layer's learned pseudo-query w_l.
                layer_outputs = [z]
                sync_repr = None

                for l_idx, layer in enumerate(self.layers):
                    V = torch.stack(layer_outputs, dim=0)              # [n, B, S, D]

                    # Match V to the norm weight's dtype so RMSNorm
                    # dispatches to its fused kernel instead of the slow
                    # unfused fallback. Cast ONCE so the downstream einsum
                    # also gets the matched dtype — otherwise attn_weights
                    # (bf16) × V (fp32) would type-promote z_in to fp32
                    # and push the mismatch one op downstream into the
                    # layer's bf16 weights. If V is already bf16, .to()
                    # is a no-op.
                    V = V.to(self.attn_res_norm.weight.dtype)

                    K = self.attn_res_norm(V)
                    w_l = self.attn_res_queries[l_idx]                  # [D]
                    scores = torch.einsum('d,nbsd->nbs', w_l, K)        # [n, B, S]
                    attn_weights = F.softmax(scores, dim=0)
                    z_in = torch.einsum('nbs,nbsd->bsd', attn_weights, V)

                    z_out, sync_repr = layer(
                        text_emb,
                        text_emb,
                        z_in,
                        key_padding_mask,
                    )
                    layer_outputs.append(z_out)

                # Final state = output of last layer (raw, not the
                # AttnRes-aggregated input — aggregation only feeds
                # INPUTS to layers).
                z = layer_outputs[-1]

                # ── Per-position logits ─────────────────────────────────
                logits_t = self._output_logits(z, text_emb)  # [B, S, V]

            # ── Certainty (always non-differentiable) ───────────────────
            # Explicit no_grad even on the last step. The original design
            # used certainty only as a non-differentiable weighting signal
            # for max_cert_component; letting gradients flow through it
            # creates a degenerate path where the model can lower loss by
            # becoming uniformly confident regardless of correctness.
            with torch.no_grad():
                probs_t = F.softmax(logits_t, dim=-1)
                entropy_t = -(probs_t * (probs_t + 1e-10).log()).sum(dim=-1)
                certainty_t = -entropy_t
            all_certainties.append(certainty_t)

            # ── Step-boundary cleanup ───────────────────────────────────
            if is_checkpointed:
                # Detach z so next step's forward starts from a fresh leaf.
                z = z.detach().requires_grad_(True)

                # Detach memory buffers — this is the actual leak fix.
                # NOT setting requires_grad_(True): doing so would cause
                # backward to accumulate a useless `.grad` of the buffer's
                # full shape on the leaf, since buffers aren't optimizer
                # parameters. Plain detach is enough — gradients through
                # the *new* activations pushed on the last step still flow
                # correctly via the cat/roll graph.
                for layer in self.layers:
                    layer.memory.pre_history = layer.memory.pre_history.detach()
                    layer.memory.post_history = layer.memory.post_history.detach()

                all_logits.append(None)
            else:
                all_logits.append(logits_t)

        all_certainties_tensor = torch.stack(all_certainties, dim=0)

        result = {
            "logits": all_logits[-1],
            "certainties": all_certainties_tensor,
        }

        if not self.training and None not in all_logits:
            result["all_logits"] = torch.stack(all_logits, dim=0)

        if targets is not None:
            result["loss"], result["per_tick_loss"] = self._compute_temporal_loss(
                all_logits, all_certainties_tensor, targets
            )

        return result

    def _compute_temporal_loss(
        self,
        all_logits: list,
        all_certainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal loss. When gradient_checkpointing is on, only the
        final step has gradients. Intermediate logits may be None.
        """
        T = len(all_logits)
        V = self.config.vocab_size
        targets_flat = targets.reshape(-1)

        per_tick_losses = []
        for t in range(T):
            if all_logits[t] is None:
                # Provide a zero-loss placeholder for dropped intermediate steps
                per_tick_losses.append(torch.tensor(0.0, device=targets.device))
                continue

            logits_t = all_logits[t].reshape(-1, V)
            loss_t = F.cross_entropy(logits_t, targets_flat, reduction="mean")
            per_tick_losses.append(loss_t)

        per_tick_loss_tensor = torch.stack(per_tick_losses)

        if self.training and self.config.gradient_checkpointing:
            loss = per_tick_losses[-1]
        else:
            temperature = 0.1
            min_loss_weights = F.softmax(-per_tick_loss_tensor / temperature, dim=0)
            min_loss_component = (min_loss_weights * per_tick_loss_tensor).sum()

            mean_certainty = all_certainties.mean(dim=(1, 2))
            max_cert_weights = F.softmax(mean_certainty / temperature, dim=0)
            max_cert_component = (max_cert_weights * per_tick_loss_tensor).sum()

            loss = (
                self.config.min_loss_weight * min_loss_component +
                self.config.max_cert_weight * max_cert_component +
                self.config.aux_loss_weight * per_tick_loss_tensor.mean()
            )

        return loss, per_tick_loss_tensor.detach()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len:]
            result = self(idx, max_thought_steps=self.config.max_thought_steps)
            logits = result["logits"][:, -1, :]

            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids