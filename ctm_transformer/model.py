"""
CTMTransformer — Full Continuous Thought Machine Transformer Model (v2)

Assembles all components into the complete architecture:
  Input tokens → Embedding → Text KV
  Initialize per-position latent states z_0 = 0  [B, S, d_latent]
  For t in 1..T thought steps:
    Sync → Query → Causal Cross-Attention → Synapse → NLM → Output
  Return logits across all thought steps for temporal loss.

CTM-v2 ADDITIONS (all backward-compatible, controlled by config flags):
  - FEEC Integrator: structure-preserving dynamics for the thought loop
  - Matrix-Valued Residual Streams: replace NLM FIFO + O(D²) sync
  - Dual-Space Sparse Attention: O(N) cross-attention via SSE + MoBA
  - Hyperloop Looped Middle Cycle: weight-sharing with depth preservation
  - Loop Position Embeddings: iteration-aware context injection
  - CUDA Graphs: kernel launch elimination for the thought loop
  - Triton Tiled Attention: accelerated Q·K computation

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
import torch.utils.checkpoint as torch_checkpoint
import math
from contextlib import nullcontext

from ctm_transformer.config import CTMConfig
from ctm_transformer.thought_layer import ThoughtLayer
from ctm_transformer.memory import SynchronizationComputer
from ctm_transformer.engram import EngramTable, EngramProjection


class FeatureEncoder(nn.Module):
    """
    Generic Feature Encoder backbone for multi-modal capability.
    Replaces the text token embedding with a projection of continuous features
    (e.g., from a Vision Transformer, ResNet, or audio frontend) into d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Placeholder for a real backbone. For now, it's just a linear projection
        # assuming the input is already a sequence of feature vectors of size d_model.
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, feature_dim]
        return self.proj(x)



class CTMTransformer(nn.Module):
    """
    Continuous Thought Machine Transformer (v2).

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

        # ── Text Ingestion or Feature Encoder ───────────────────────────
        if config.use_feature_encoder:
            self.feature_encoder = FeatureEncoder(config.d_model)
            self.token_embedding = None
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.feature_encoder = None

        if config.use_positional_encoding:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_embedding = None

        self.embed_norm = nn.LayerNorm(config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Thought Layers ──────────────────────────────────────────────
        # Resolve which layers fuse Engram.
        self.engram_layer_indices: list[int] = (
            config.resolve_engram_layers() if config.use_engram else []
        )
        engram_layer_set = set(self.engram_layer_indices)

        def _make_thought_layer(l_idx: int) -> ThoughtLayer:
            """Factory for thought layers with all v2 feature flags."""
            return ThoughtLayer(
                d_latent=config.d_latent,
                d_model=config.d_model,
                n_heads=config.n_heads,
                nlm_hidden_dim=config.nlm_hidden_dim,
                history_len=config.history_len,
                nlm_groups=config.nlm_groups,
                sync_method=config.sync_method,
                sync_rank=config.sync_rank,
                dropout=config.dropout,
                engram_enabled=(l_idx in engram_layer_set),
                engram_use_conv=config.engram_use_conv,
                engram_conv_kernel=config.engram_conv_kernel,
                engram_conv_dilation=config.engram_conv_dilation,
                # v2 features
                use_matrix_streams=config.use_matrix_streams,
                n_streams=config.n_streams,
                stream_gating=config.stream_gating,
                use_dssa=config.use_dssa,
                dssa_n_partitions=config.dssa_n_partitions,
                dssa_top_k=config.dssa_top_k,
                dssa_block_size=config.dssa_block_size,
                dssa_top_k_blocks=config.dssa_top_k_blocks,
                use_triton_attention=config.use_triton_attention,
                synapse_type=config.synapse_type,
            )

        # ── Hyperloop or Standard Layer Construction ────────────────────
        if config.use_hyperloop:
            n_begin = config.hyperloop_n_begin
            n_middle = config.hyperloop_n_middle
            n_end = config.hyperloop_n_end
            # Total effective layers = n_begin + n_middle * middle_loops + n_end
            self.begin_layers = nn.ModuleList([
                _make_thought_layer(i) for i in range(n_begin)
            ])
            self.middle_layers = nn.ModuleList([
                _make_thought_layer(n_begin + i) for i in range(n_middle)
            ])
            self.end_layers = nn.ModuleList([
                _make_thought_layer(n_begin + n_middle + i)
                for i in range(n_end)
            ])
            self.layers = None  # Signal that we use hyperloop
            # Total effective layers for AttnRes and other per-layer bookkeeping
            self._effective_n_layers = (
                n_begin + n_middle * config.hyperloop_middle_loops + n_end
            )
        else:
            self.layers = nn.ModuleList([
                _make_thought_layer(l_idx)
                for l_idx in range(config.n_layers)
            ])
            self.begin_layers = None
            self.middle_layers = None
            self.end_layers = None
            self._effective_n_layers = config.n_layers

        # ── Engram Memory (optional) ────────────────────────────────────
        if config.use_engram:
            self.engram_table = EngramTable(
                ngram_orders=config.engram_ngram_orders,
                n_heads=config.engram_n_heads,
                slots_per_table=config.engram_slots_per_table,
                d_head=config.engram_d_head,
                bos_id=config.engram_bos_id,
            )
            self.engram_projections = nn.ModuleDict({
                str(l_idx): EngramProjection(
                    d_mem=self.engram_table.d_mem,
                    d_query=config.d_model,
                    d_out=config.d_model,
                    zero_init_v=True,
                )
                for l_idx in self.engram_layer_indices
            })
        else:
            self.engram_table = None
            self.engram_projections = None

        # ── FEEC Integrator (v2) ────────────────────────────────────────
        if config.use_feec:
            from ctm_transformer.feec_integrator import FEECIntegrator
            self.feec = FEECIntegrator(
                d_latent=config.d_latent,
                n_layers=self._effective_n_layers,
                dt_init=config.feec_dt_init,
                damping_init=config.feec_damping_init,
                clamp_dt=config.feec_clamp_dt,
            )
        else:
            self.feec = None

        # ── Loop Position Embeddings (v2) ───────────────────────────────
        if config.use_loop_pos_emb:
            self.loop_pos_emb = nn.Embedding(
                config.max_thought_steps, config.d_latent
            )
        else:
            self.loop_pos_emb = None

        # ── Attention Residuals (Kimi AttnRes) ──────────────────────────
        n_layers_for_attnres = self._effective_n_layers
        self.attn_res_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(config.d_latent))
            for _ in range(n_layers_for_attnres)
        ])
        self.attn_res_norm = nn.RMSNorm(config.d_latent)

        # ── Output Head ─────────────────────────────────────────────────
        if config.per_tick_heads:
            T = config.max_thought_steps
            self.tick_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.d_latent + config.d_model, config.d_model),
                    nn.GELU(),
                    nn.LayerNorm(config.d_model),
                )
                for _ in range(T)
            ])
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)
            self.output_proj = None
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(config.d_latent + config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.vocab_size),
            )
            self.tick_adapters = None
            self.lm_head = None

        # ── Initial State ───────────────────────────────────────────────
        self.z0 = nn.Parameter(torch.zeros(config.d_latent))

        # ── Distillation Alignment ──────────────────────────────────────
        # teacher_z_proj is created whenever feature distillation is enabled
        # and the dimensions differ (or always if we want a learned mapping).
        # The student-side LayerNorm + teacher-side LayerNorm are needed for
        # the "cosine" and "mse_normed" feature-distillation methods, so the
        # two architectures' hidden states are compared on a magnitude-
        # neutral footing.
        if config.use_distillation and config.distill_feature_weight > 0:
            if config.teacher_d_model != config.d_latent:
                self.teacher_z_proj = nn.Linear(config.teacher_d_model, config.d_latent)
            else:
                self.teacher_z_proj = None
            self.distill_student_norm = nn.LayerNorm(config.d_latent, elementwise_affine=False)
            self.distill_teacher_norm = nn.LayerNorm(config.d_latent, elementwise_affine=False)
        else:
            self.teacher_z_proj = None
            self.distill_student_norm = None
            self.distill_teacher_norm = None

        # Velocity initial state (for FEEC)
        if config.use_feec:
            self.velocity_0 = nn.Parameter(torch.zeros(config.d_latent))
        else:
            self.velocity_0 = None

        # ── CUDA Graph wrapper (v2) ─────────────────────────────────────
        if config.use_cuda_graphs:
            from ctm_transformer.triton_kernels import CUDAGraphThoughtLoop
            self._cuda_graph = CUDAGraphThoughtLoop(enabled=True)
        else:
            self._cuda_graph = None

        # ── Training step counter (non-persistent) ──────────────────────
        self.register_buffer(
            "_train_step",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )

        self.apply(self._init_weights)

        # Re-apply Engram-specific inits
        for m in self.modules():
            reset_fn = getattr(m, "_reset_special_inits", None)
            if callable(reset_fn) and m is not self:
                reset_fn()

        # ── Ternary weight swap (optional) ──────────────────────────────
        if config.use_ternary:
            from ctm_transformer.ternary import replace_linears_with_ternary
            only = tuple(config.ternary_only_modules) if config.ternary_only_modules else None
            n_swapped = replace_linears_with_ternary(
                self,
                only_module_names=only,
                verbose=False,
            )
            self._n_ternary_layers = n_swapped
        else:
            self._n_ternary_layers = 0

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
        """Embed input tokens or continuous features. Returns [batch, seq_len, d_model]."""
        B, S = input_ids.shape[:2]
        if self.config.use_feature_encoder:
            # Assume input_ids is actually a continuous feature tensor
            tok_emb = self.feature_encoder(input_ids)
        else:
            tok_emb = self.token_embedding(input_ids)

        if self.pos_embedding is not None:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            tok_emb = tok_emb + self.pos_embedding(positions)

        return self.embed_dropout(self.embed_norm(tok_emb))

    def _output_logits(
        self,
        z: torch.Tensor,
        text_emb: torch.Tensor,
        tick: int = 0,
    ) -> torch.Tensor:
        """
        Project per-position latent states to logits.

        Args:
            z: [B, S, d_latent] — per-position latent states.
            text_emb: [B, S, d_model] — text embeddings.
            tick: which thought tick this projection is for.

        Returns:
            logits: [B, S, vocab_size]
        """
        combined = torch.cat([z, text_emb], dim=-1)

        if self.tick_adapters is not None:
            adapted = self.tick_adapters[tick](combined)
            return self.lm_head(adapted)

        return self.output_proj(combined)

    def _get_layers_sequence(self) -> list[ThoughtLayer]:
        """Get the sequence of layers to iterate over (respecting Hyperloop)."""
        if self.layers is not None:
            return list(self.layers)

        # Hyperloop: begin + middle*loops + end
        seq = list(self.begin_layers)
        for _ in range(self.config.hyperloop_middle_loops):
            seq.extend(list(self.middle_layers))
        seq.extend(list(self.end_layers))
        return seq

    def _thought_step(
        self,
        z: torch.Tensor,
        t: int,
        text_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        engram_kv: dict,
        pre_states: list[torch.Tensor],
        post_states: list[torch.Tensor],
        velocity: torch.Tensor | None = None,
        stream_states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor],
               torch.Tensor | None, list[torch.Tensor] | None]:
        """
        Run one thought-step iteration with explicit per-layer memory state.

        Args:
            z: [B, S, d_latent] — pre-step latent state.
            t: thought-tick index.
            text_emb: [B, S, d_model] — token embeddings.
            key_padding_mask: [B, S] or None.
            engram_kv: dict mapping layer index → (k, v) tuples.
            pre_states: list of pre_history buffers (v1 path).
            post_states: list of post_history buffers (v1 path).
            velocity: [B, S, d_latent] or None — FEEC velocity state.
            stream_states: list of stream states per layer (v2 path) or None.

        Returns:
            z_new, logits_t, new_pre_states, new_post_states, velocity_new, new_stream_states
        """
        layers = self._get_layers_sequence()

        # Restore v1 memory state into layers
        if not self.config.use_matrix_streams:
            for l_idx, layer in enumerate(layers):
                if layer.memory is not None and l_idx < len(pre_states):
                    layer.memory.pre_history = pre_states[l_idx]
                    layer.memory.post_history = post_states[l_idx]

        # ── Loop Position Embedding ─────────────────────────────────────
        if self.loop_pos_emb is not None:
            tick_idx = torch.tensor(t, device=z.device, dtype=torch.long)
            z = z + self.loop_pos_emb(tick_idx).unsqueeze(0).unsqueeze(0)

        # ── Attention Residuals: layer stack ─────────────────────────────
        layer_outputs = [z]
        velocity_new = velocity
        new_stream_states = stream_states if stream_states is not None else []

        for l_idx, layer in enumerate(layers):
            V = torch.stack(layer_outputs, dim=0)
            V = V.to(self.attn_res_norm.weight.dtype)

            K = self.attn_res_norm(V)
            w_l = self.attn_res_queries[l_idx] if l_idx < len(self.attn_res_queries) else self.attn_res_queries[-1]
            scores = torch.einsum('d,nbsd->nbs', w_l, K)
            attn_weights = F.softmax(scores, dim=0)
            z_in = torch.einsum('nbs,nbsd->bsd', attn_weights, V)

            # Get stream state for this layer (if matrix streams)
            layer_stream = None
            if self.config.use_matrix_streams and stream_states is not None and l_idx < len(stream_states):
                layer_stream = stream_states[l_idx]

            z_out, _sync_repr, new_layer_stream = layer(
                text_emb,
                text_emb,
                z_in,
                key_padding_mask,
                engram_kv=engram_kv.get(l_idx),
                stream_state=layer_stream,
            )

            # Update stream state
            if self.config.use_matrix_streams and new_layer_stream is not None:
                if l_idx < len(new_stream_states):
                    new_stream_states[l_idx] = new_layer_stream
                else:
                    new_stream_states.append(new_layer_stream)

            # ── FEEC Integration ────────────────────────────────────────
            if self.feec is not None and velocity_new is not None:
                force = z_out - z_in  # Force field = layer's contribution
                z_out, velocity_new = self.feec.step(
                    z_in, velocity_new, force, layer_idx=l_idx
                )

            layer_outputs.append(z_out)

        z_new = layer_outputs[-1]
        logits_t = self._output_logits(z_new, text_emb, tick=t)

        # Read out v1 memory state
        new_pre_states = []
        new_post_states = []
        if not self.config.use_matrix_streams:
            for layer in layers:
                if layer.memory is not None:
                    new_pre_states.append(layer.memory.pre_history)
                    new_post_states.append(layer.memory.post_history)

        return z_new, logits_t, new_pre_states, new_post_states, velocity_new, new_stream_states

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        max_thought_steps: int | None = None,
        teacher_logits: torch.Tensor | None = None,
        teacher_z: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass with per-position latent states and causal masking.
        """
        T = max_thought_steps or self.config.max_thought_steps
        B, S = input_ids.shape[:2]
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # ── Embed text ──────────────────────────────────────────────────
        text_emb = self._embed_text(input_ids)

        # ── Engram Lookup ───────────────────────────────────────────────
        engram_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.engram_table is not None:
            e_t = self.engram_table(input_ids)
            e_t = e_t.to(text_emb.dtype)
            for l_idx_str, proj in self.engram_projections.items():
                l_idx = int(l_idx_str)
                k, v = proj(e_t)
                engram_kv[l_idx] = (k, v)

        # ── Initialize latent states ────────────────────────────────────
        z = self.z0.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()

        # FEEC velocity initialization
        velocity = None
        if self.feec is not None and self.velocity_0 is not None:
            velocity = self.velocity_0.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()

        # ── Initialize memory / streams ─────────────────────────────────
        layers = self._get_layers_sequence()
        BS = B * S

        # v2: matrix stream states
        stream_states = None
        if self.config.use_matrix_streams:
            stream_states = []
            for layer in layers:
                if layer.stream is not None:
                    stream_states.append(
                        layer.stream.init_state(B, S, device, dtype)
                    )
                else:
                    stream_states.append(None)
        else:
            # v1: reset FIFO memory buffers
            for layer in layers:
                if layer.memory is not None:
                    layer.reset_memory(BS, device, dtype)

        # ── Thought Loop ────────────────────────────────────────────────
        all_logits = []
        all_certainties = []
        energy_penalties = []

        use_per_step_ckpt = (
            self.training
            and self.config.gradient_checkpointing
            and T >= self.config.gradient_checkpointing_min_T
        )

        # Initialize v1 memory state lists
        pre_states = []
        post_states = []
        if not self.config.use_matrix_streams:
            pre_states = [layer.memory.pre_history for layer in layers if layer.memory is not None]
            post_states = [layer.memory.post_history for layer in layers if layer.memory is not None]

        z_prev = z
        velocity_prev = velocity

        for t in range(T):
            if use_per_step_ckpt:
                result = torch_checkpoint.checkpoint(
                    self._thought_step,
                    z, t, text_emb, key_padding_mask, engram_kv,
                    pre_states, post_states, velocity, stream_states,
                    use_reentrant=False,
                )
            else:
                result = self._thought_step(
                    z, t, text_emb, key_padding_mask, engram_kv,
                    pre_states, post_states, velocity, stream_states,
                )

            z, logits_t, pre_states, post_states, velocity, stream_states = result

            # ── FEEC energy penalty ─────────────────────────────────────
            if self.feec is not None and velocity is not None:
                ep = self.feec.energy_penalty(z, velocity, z_prev, velocity_prev)
                energy_penalties.append(ep)
                z_prev = z
                velocity_prev = velocity

            # ── Certainty ───────────────────────────────────────────────
            with torch.no_grad():
                probs_t = F.softmax(logits_t, dim=-1)
                entropy_t = -(probs_t * (probs_t + 1e-10).log()).sum(dim=-1)
                if self.config.temporal_loss_type == "dynamic_aggregate":
                    max_entropy = math.log(self.config.vocab_size)
                    certainty_t = 1.0 - (entropy_t / max_entropy)
                else:
                    certainty_t = -entropy_t
            all_certainties.append(certainty_t)
            all_logits.append(logits_t)

        # Restore v1 memory state
        if not self.config.use_matrix_streams:
            for l_idx, layer in enumerate(layers):
                if layer.memory is not None and l_idx < len(pre_states):
                    layer.memory.pre_history = pre_states[l_idx]
                    layer.memory.post_history = post_states[l_idx]

        all_certainties_tensor = torch.stack(all_certainties, dim=0)

        result = {
            "logits": all_logits[-1],
            "certainties": all_certainties_tensor,
        }

        # FEEC energy diagnostics
        if self.feec is not None and velocity is not None:
            with torch.no_grad():
                result["feec_energy"] = self.feec.energy(z, velocity).item()

        # Always populate all_logits for diagnostics/testing
        # We store the list instead of stacking to save a massive allocation (vocab_size is large!)
        result["all_logits"] = all_logits

        if targets is not None:
            loss, per_tick_loss = self._compute_temporal_loss(
                all_logits, all_certainties_tensor, targets
            )

            # Add FEEC energy penalty to loss
            if energy_penalties and self.config.feec_energy_penalty_weight > 0:
                energy_pen = torch.stack(energy_penalties).mean()
                loss = loss + self.config.feec_energy_penalty_weight * energy_pen
                result["feec_energy_penalty"] = energy_pen.item()

            # ── Distillation Loss ────────────────────────────────────────
            distill_loss = torch.tensor(0.0, device=device)
            if self.config.use_distillation:
                # ── 1. Soft-Target Distillation ──────────────────────────
                # Several fixes vs. the original:
                #   (a) reduction is now per-token: we flatten [B, S, V] →
                #       [B*S, V] before kl_div so 'batchmean' divides by
                #       B*S (token count), matching the per-token-mean
                #       scale of the LM cross-entropy. The original divided
                #       by B only, making KL ~seq_len× larger than LM —
                #       which dominated training and produced the V-shaped
                #       per-tick loss curve.
                #   (b) temperature softening (Hinton 2015): both
                #       distributions are softened by T; the resulting KL
                #       is multiplied by T² so the gradient magnitude is
                #       T-invariant. T=1 disables softening.
                #   (c) softmax computed in fp32 for numerical stability
                #       with large (131k+) vocabularies under bf16 amp.
                #   (d) log_target=True for the KL itself — this avoids
                #       recomputing exp(teacher_log_probs) inside kl_div
                #       and is the numerically stable formulation.
                #   (e) per-tick KD: KL is computed at every thought tick
                #       and aggregated according to distill_tick_aggregation.
                #       Distilling only the last tick (the original
                #       behavior) creates a conflict with dynamic_aggregate
                #       temporal loss and corrupts the per-tick LM heads
                #       when --per_tick_heads is enabled.
                if teacher_logits is not None:
                    T_temp = float(self.config.distill_temperature)
                    T_temp_sq = T_temp * T_temp

                    B_, S_, V_ = teacher_logits.shape
                    teacher_logits_flat = teacher_logits.reshape(B_ * S_, V_)
                    # fp32 log_softmax for stability; reused across ticks.
                    teacher_log_probs = F.log_softmax(
                        teacher_logits_flat / T_temp,
                        dim=-1,
                        dtype=torch.float32,
                    )

                    n_ticks = len(all_logits)
                    per_tick_kl = []
                    for tk in range(n_ticks):
                        student_logits_flat = all_logits[tk].reshape(B_ * S_, V_)
                        student_log_probs = F.log_softmax(
                            student_logits_flat / T_temp,
                            dim=-1,
                            dtype=torch.float32,
                        )
                        kl_t = F.kl_div(
                            student_log_probs,
                            teacher_log_probs,
                            reduction="batchmean",
                            log_target=True,
                        ) * T_temp_sq
                        per_tick_kl.append(kl_t)
                    per_tick_kl_tensor = torch.stack(per_tick_kl)  # [T]

                    # Aggregate per-tick KLs.
                    agg = self.config.distill_tick_aggregation
                    if agg == "all":
                        # Uniform mean — recommended default. Every per-tick
                        # head gets the same KD pressure; no tick is
                        # uniquely degraded.
                        kl_loss = per_tick_kl_tensor.mean()
                    elif agg == "last":
                        # Legacy behavior. Strongly discouraged when
                        # combined with per_tick_heads or dynamic_aggregate.
                        kl_loss = per_tick_kl_tensor[-1]
                    elif agg == "lm_aligned":
                        if self.config.temporal_loss_type == "dynamic_aggregate":
                            # Per-token tick-selection would require
                            # per-token KL (~Gb of fp32 at 131k vocab),
                            # so approximate dynamic aggregation with a
                            # uniform mean. Same effect as "all".
                            kl_loss = per_tick_kl_tensor.mean()
                        else:
                            # Mirror the ramp_mono LM weighting so KD
                            # and LM agree on which ticks matter most.
                            ramp = torch.linspace(
                                self.config.tick_ramp_start,
                                self.config.tick_ramp_end,
                                n_ticks,
                                device=per_tick_kl_tensor.device,
                                dtype=per_tick_kl_tensor.dtype,
                            )
                            ramp = ramp * (n_ticks / ramp.sum())
                            kl_loss = (ramp * per_tick_kl_tensor).mean()
                    else:
                        raise ValueError(
                            f"Unknown distill_tick_aggregation: {agg!r}"
                        )

                    kl_loss = kl_loss.to(loss.dtype)
                    distill_loss = distill_loss + self.config.distill_logit_weight * kl_loss
                    result["distill_kl_loss"] = kl_loss.item()
                    result["distill_kl_per_tick"] = per_tick_kl_tensor.detach()

                # ── 2. Z-Alignment Feature Distillation ──────────────────
                # Off by default (distill_feature_weight=0). For cross-
                # architecture distillation (NemotronH hybrid Mamba →
                # FEEC-integrated CTM), the absolute magnitudes of the two
                # hidden states are not comparable. The "cosine" method
                # is magnitude-invariant and is the right default when
                # this term is enabled. Raw "mse" is retained for parity
                # with prior code but is NOT recommended.
                if (
                    teacher_z is not None
                    and self.config.distill_feature_weight > 0
                ):
                    teacher_z_aligned = teacher_z
                    if self.teacher_z_proj is not None:
                        teacher_z_aligned = self.teacher_z_proj(teacher_z)

                    method = self.config.distill_feature_method
                    if method == "cosine":
                        cos = F.cosine_similarity(
                            z.float(), teacher_z_aligned.float(), dim=-1
                        )  # [B, S]
                        feature_loss = (1.0 - cos).mean()
                    elif method == "mse_normed":
                        z_n = self.distill_student_norm(z.float())
                        t_n = self.distill_teacher_norm(teacher_z_aligned.float())
                        feature_loss = F.mse_loss(z_n, t_n)
                    elif method == "mse":
                        feature_loss = F.mse_loss(z, teacher_z_aligned)
                    else:
                        raise ValueError(
                            f"Unknown distill_feature_method: {method!r}"
                        )

                    feature_loss = feature_loss.to(loss.dtype)
                    distill_loss = (
                        distill_loss
                        + self.config.distill_feature_weight * feature_loss
                    )
                    result[f"distill_feature_{method}"] = feature_loss.item()

                loss = loss + distill_loss
                result["distill_loss"] = (
                    distill_loss.item()
                    if isinstance(distill_loss, torch.Tensor)
                    else float(distill_loss)
                )

            result["loss"] = loss
            result["per_tick_loss"] = per_tick_loss

        return result

    def _compute_temporal_loss(
        self,
        all_logits: list,
        all_certainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal loss aggregated over all thought steps.
        """
        T = len(all_logits)
        V = self.config.vocab_size
        targets_flat = targets.reshape(-1)

        if self.config.temporal_loss_type == "dynamic_aggregate":
            # Dynamic Loss Aggregation (Listing 4)
            per_tick_losses_unreduced = []
            per_tick_losses_reduced = []
            for t in range(T):
                logits_t = all_logits[t].reshape(-1, V)
                loss_t_unreduced = F.cross_entropy(logits_t, targets_flat, reduction="none")
                per_tick_losses_unreduced.append(loss_t_unreduced)
                per_tick_losses_reduced.append(loss_t_unreduced.mean())
            
            losses_unreduced = torch.stack(per_tick_losses_unreduced, dim=1) # [BS, T]
            per_tick_loss_tensor = torch.stack(per_tick_losses_reduced) # [T]
            
            # all_certainties is [T, B, S]. Flatten to [BS, T]
            cert = all_certainties.view(T, -1).transpose(0, 1) # [BS, T]
            
            lowest_idx = losses_unreduced.argmin(dim=-1) # [BS]
            certain_idx = cert.argmax(dim=-1) # [BS]
            
            # Gather
            loss_t1 = losses_unreduced.gather(1, lowest_idx.unsqueeze(1)).squeeze(1)
            loss_t2 = losses_unreduced.gather(1, certain_idx.unsqueeze(1)).squeeze(1)
            
            base_loss = ((loss_t1 + loss_t2) / 2.0).mean()
        else:
            # Standard path: Linear ramp weighting (default ramp_mono)
            per_tick_losses = []
            for t in range(T):
                logits_t = all_logits[t].reshape(-1, V)
                per_tick_losses.append(F.cross_entropy(logits_t, targets_flat, reduction="mean"))
            per_tick_loss_tensor = torch.stack(per_tick_losses)

            ramp = torch.linspace(
                self.config.tick_ramp_start, self.config.tick_ramp_end, T,
                device=per_tick_loss_tensor.device, dtype=per_tick_loss_tensor.dtype,
            )
            ramp = ramp * (T / ramp.sum())
            base_loss = (ramp * per_tick_loss_tensor).mean()

        # ── Monotonicity Penalty (Applied to all paths) ─────────────────
        if T > 1:
            diffs = per_tick_loss_tensor[1:] - per_tick_loss_tensor[:-1]
            mono_penalty = F.relu(diffs).mean()
        else:
            mono_penalty = torch.tensor(0.0, device=per_tick_loss_tensor.device)

        # Mono-penalty decay
        decay_until_frac = self.config.mono_penalty_decay_until_frac
        if decay_until_frac > 0 and self.config.max_steps > 0:
            decay_until_step = decay_until_frac * self.config.max_steps
            current_step = float(self._train_step.item())
            progress = min(current_step / max(decay_until_step, 1.0), 1.0)
            min_frac = self.config.mono_penalty_min_frac
            decay_factor = 1.0 - progress * (1.0 - min_frac)
        else:
            decay_factor = 1.0
        
        effective_mono_weight = self.config.mono_penalty_weight * decay_factor
        loss = base_loss + effective_mono_weight * mono_penalty

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