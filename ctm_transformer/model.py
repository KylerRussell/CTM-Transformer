"""
CTMTransformer — Full Continuous Thought Machine Transformer Model

Assembles all components into the complete architecture:
  Input tokens → Embedding → Text KV
  Initialize latent state z_0 = 0
  For t in 1..T thought steps:
    Sync → Query → Cross-Attention → Synapse → NLM → Output
  Return logits across all thought steps for temporal loss.

Conceptual mapping from Living-Brain engine_torch.py:
- engine_torch.settle():    Iterates IMEX dynamics until convergence → our thought loop
- engine_torch.state:       Single somatic state vector → our z_t latent state
- cerebellar_forward():     Purkinje readout from L5/6 → our output head from sync matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ctm_transformer.config import CTMConfig
from ctm_transformer.thought_layer import ThoughtLayer
from ctm_transformer.memory import SynchronizationComputer


class CTMTransformer(nn.Module):
    """
    Continuous Thought Machine Transformer.

    A dynamic, stateful transformer that processes text through iterative
    internal 'thought steps' driven by neural synchronization.

    Instead of a single feed-forward pass per token, the model iterates
    through decoupled thought steps where:
    - Queries derive from internal synchronization state, not text positions
    - Activation functions are replaced by learned per-neuron MLPs (NLMs)
    - Output is projected from synchronization matrices, not hidden states
    - Loss is aggregated across the temporal thought dimension

    Args:
        config: CTMConfig with all hyperparameters.
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

        # ── Output Head ─────────────────────────────────────────────────
        # Projects synchronization representation → logits
        # y_t = W_out @ flatten(S_out_t)
        sync_dim = self.layers[0].sync_computer.output_dim
        self.output_head = nn.Sequential(
            nn.Linear(sync_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

        # ── Initial State Projection ────────────────────────────────────
        # Learns an initial latent state z_0 (alternatively could be zeros)
        self.z0 = nn.Parameter(torch.zeros(config.d_latent))

        # Initialize weights
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
        """
        Embed input tokens into the text representation used for K and V.

        Args:
            input_ids: [batch, seq_len] — integer token IDs.

        Returns:
            text_embeddings: [batch, seq_len, d_model]
        """
        B, S = input_ids.shape
        tok_emb = self.token_embedding(input_ids)  # [B, S, d_model]

        if self.pos_embedding is not None:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)  # [1, S, d_model]
            tok_emb = tok_emb + pos_emb

        return self.embed_dropout(self.embed_norm(tok_emb))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        max_thought_steps: int | None = None,
    ) -> dict:
        """
        Full forward pass through the CTM-Transformer.

        For each position in the sequence, the model runs T thought steps
        before producing output logits. All per-tick logits are returned
        for temporal loss computation.

        Args:
            input_ids: [batch, seq_len] — input token IDs.
            targets: [batch, seq_len] — target token IDs (optional, for loss).
            key_padding_mask: [batch, seq_len] — True for padded positions.
            max_thought_steps: Override config.max_thought_steps if provided.

        Returns:
            dict with:
                "logits":        [batch, seq_len, vocab_size] — final thought step logits
                "all_logits":    [T, batch, seq_len, vocab_size] — logits at each thought step
                "loss":          scalar — temporal aggregated loss (if targets provided)
                "per_tick_loss": [T] — loss at each thought step (if targets provided)
                "certainties":   [T, batch, seq_len] — certainty at each tick
        """
        T = max_thought_steps or self.config.max_thought_steps
        B, S = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # ── Step 1: Ingestion ───────────────────────────────────────────
        # Embed input text → Keys and Values (fixed across thought steps)
        text_emb = self._embed_text(input_ids)  # [B, S, d_model]

        # We process each sequence position through the thought loop.
        # For efficiency, we process all positions in parallel by treating
        # each position as having its own latent state.

        # Initialize latent state: z_0 for all batch items and positions
        # [B, S, d_latent] — each position gets its own latent state
        z = self.z0.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()  # [B, S, d_latent]

        # Reset memory buffers in all layers
        # We reshape B*S into a flat batch dimension for memory
        BS = B * S
        for layer in self.layers:
            layer.reset_memory(BS, device, dtype)

        # Prepare text K, V tiled per-position (each position sees all text)
        # Actually, for cross-attention, each position attends to the FULL
        # input sequence. We repeat text embeddings for each position.
        # Shape: text_emb is [B, S, d_model]
        # For cross-attention: K, V are the full text, Q comes from each position's state.
        # We process all S positions together by flattening B*S into batch dim.
        text_k = text_emb  # [B, S, d_model] — all positions see the same keys
        text_v = text_emb  # [B, S, d_model] — all positions see the same values

        # For the thought loop, we need each position to attend to all text.
        # We'll repeat text for each position: [B*S, S, d_model]
        text_k_repeated = text_k.unsqueeze(1).expand(B, S, S, -1).reshape(BS, S, -1)
        text_v_repeated = text_v.unsqueeze(1).expand(B, S, S, -1).reshape(BS, S, -1)

        if key_padding_mask is not None:
            mask_repeated = key_padding_mask.unsqueeze(1).expand(B, S, S).reshape(BS, S)
        else:
            mask_repeated = None

        # ── Step 2-6: Thought Loop ──────────────────────────────────────
        all_logits = []
        all_certainties = []

        for t in range(T):
            # Flatten z: [B, S, d_latent] → [B*S, d_latent]
            z_flat = z.reshape(BS, -1)

            # Memory optimization: Detach the state between thought steps
            # to prevent the backward graph from growing as O(T * layers).
            # Only the LAST thought step retains full gradients for backprop.
            # This is the same strategy OpenMythos uses for recurrent blocks.
            is_last_step = (t == T - 1)
            if not is_last_step and self.training and self.config.gradient_checkpointing:
                z_flat = z_flat.detach().requires_grad_(True)

            # Pass through each thought layer sequentially
            sync_repr = None
            for layer in self.layers:
                z_flat, sync_repr = layer(
                    text_k_repeated,
                    text_v_repeated,
                    z_flat,
                    mask_repeated,
                )

            # Unflatten: [B*S, d_latent] → [B, S, d_latent]
            z = z_flat.reshape(B, S, -1)

            # ── Output projection from synchronization ──────────────────
            # sync_repr: [B*S, sync_dim]
            if is_last_step or not self.training:
                # Full computation with gradients for the last step
                logits_t = self.output_head(sync_repr)     # [B*S, vocab_size]
                logits_t = logits_t.reshape(B, S, -1)      # [B, S, vocab_size]
            else:
                # Intermediate steps: compute logits for monitoring but
                # detach to avoid building huge backward graph
                with torch.no_grad():
                    logits_t = self.output_head(sync_repr)
                    logits_t = logits_t.reshape(B, S, -1)
            all_logits.append(logits_t)

            # ── Certainty: negative entropy of softmax ──────────────────
            with torch.no_grad():
                probs_t = F.softmax(logits_t, dim=-1)
                entropy_t = -(probs_t * (probs_t + 1e-10).log()).sum(dim=-1)  # [B, S]
                certainty_t = -entropy_t  # Higher = more certain
            all_certainties.append(certainty_t)

        # Stack all thought step outputs
        all_logits_tensor = torch.stack(all_logits, dim=0)         # [T, B, S, V]
        all_certainties_tensor = torch.stack(all_certainties, dim=0)  # [T, B, S]

        result = {
            "logits": all_logits[-1],          # Final thought step
            "all_logits": all_logits_tensor,
            "certainties": all_certainties_tensor,
        }

        # ── Temporal Loss Aggregation ───────────────────────────────────
        if targets is not None:
            result["loss"], result["per_tick_loss"] = self._compute_temporal_loss(
                all_logits_tensor, all_certainties_tensor, targets
            )

        return result

    def _compute_temporal_loss(
        self,
        all_logits: torch.Tensor,
        all_certainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the temporal loss.

        When gradient_checkpointing is enabled, only the final thought step
        has gradients (intermediate steps are detached for memory savings).
        The loss is computed on the final step. Per-tick losses for all steps
        are computed for logging/monitoring but do not carry gradients.

        When gradient_checkpointing is disabled (e.g., during eval or small
        models), the full temporal aggregation across all ticks is used.

        Args:
            all_logits:      [T, B, S, V] — logits at each thought step.
            all_certainties: [T, B, S] — certainty at each tick.
            targets:         [B, S] — target token IDs.

        Returns:
            loss: scalar — loss for backpropagation.
            per_tick_loss: [T] — CE loss at each thought step (for logging).
        """
        T, B, S, V = all_logits.shape
        cfg = self.config
        targets_flat = targets.reshape(-1)

        # Compute per-tick cross-entropy loss (some may be detached)
        per_tick_losses = []
        for t in range(T):
            logits_t = all_logits[t].reshape(-1, V)
            loss_t = F.cross_entropy(logits_t, targets_flat, reduction="mean")
            per_tick_losses.append(loss_t)

        per_tick_loss_tensor = torch.stack(per_tick_losses)  # [T]

        if self.training and cfg.gradient_checkpointing:
            # Only the final step has gradients — use it directly
            loss = per_tick_losses[-1]
        else:
            # Full temporal aggregation (all steps have gradients)
            temperature = 0.1

            # Min-loss tick: softmin over per-tick losses
            min_loss_weights = F.softmax(-per_tick_loss_tensor / temperature, dim=0)
            min_loss_component = (min_loss_weights * per_tick_loss_tensor).sum()

            # Max-certainty tick
            mean_certainty = all_certainties.mean(dim=(1, 2))
            max_cert_weights = F.softmax(mean_certainty / temperature, dim=0)
            max_cert_component = (max_cert_weights * per_tick_loss_tensor).sum()

            loss = (
                cfg.min_loss_weight * min_loss_component +
                cfg.max_cert_weight * max_cert_component +
                cfg.aux_loss_weight * per_tick_loss_tensor.mean()
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
        """
        Autoregressive generation using the final thought step.

        Args:
            input_ids: [1, seq_len] — prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.

        Returns:
            generated: [1, seq_len + max_new_tokens] — full sequence.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            result = self(idx, max_thought_steps=self.config.max_thought_steps)
            logits = result["logits"][:, -1, :]  # [1, V]

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
