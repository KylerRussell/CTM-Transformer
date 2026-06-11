"""
CTM-Transformer Configuration

All hyperparameters for model architecture, training, and temporal loss.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CTMConfig:
    """Configuration for the Continuous Thought Machine Transformer."""

    # ── Model Architecture ──────────────────────────────────────────────
    use_feature_encoder: bool = False  # If True, replaces text token embedding with a generic FeatureEncoder
    vocab_size: int = 50257            # GPT-2 BPE tokenizer (tiktoken)
    d_model: int = 512                 # Text embedding / cross-attention dimension
    d_latent: int = 512                # Latent neuron count (internal state width)
    n_heads: int = 8                   # Cross-attention heads
    n_layers: int = 4                  # Stacked thought layers
    nlm_hidden_dim: int = 32           # Per-neuron MLP hidden size
    history_len: int = 8               # FIFO buffer depth (temporal window for NLMs)
    max_thought_steps: int = 8         # Max internal iterations T per token

    # ── Thought-step curriculum (optional) ──────────────────────────────
    # Curriculum on T: gradually deepen the thought loop over training.
    # When `t_curriculum=True`, training uses progressively larger T values
    # at progressively later phases. Empirically:
    #   - Starting at T=2 concentrates the gradient signal: with only 2
    #     steps, the model can't slip into the "produce identical output
    #     at every step" attractor as easily — there's no room to ignore
    #     refinement.
    #   - Once differentiation is established at T=2, increasing to T=4
    #     gives the model room to factor that refinement over more steps.
    #   - T=8 then allows genuinely multi-step strategies on top of an
    #     already-refining base.
    # The flat T=8 alternative tends to collapse to "all ticks identical"
    # because the symmetry argument is stronger at higher T from a cold
    # start.
    #
    # Wall-clock benefit: cheaper steps at lower T early in training.
    # T=2 runs ~3-4× faster per step than T=8 (with checkpointing on),
    # so the early phases do more work per wall-clock hour even though
    # the late phase is slower.
    t_curriculum: bool = False
    # Stage definitions: each stage is (T_value, end_fraction_of_max_steps).
    # Defaults: T=2 for steps 0..30% of max_steps, T=4 for 30..60%,
    # T=8 for 60..100%. Override by passing a different list, or via
    # the CLI flags --t_curriculum_t1, --t_curriculum_t1_until, etc.
    # Final stage's T should equal `max_thought_steps` so the model is
    # trained at its target depth by the end.
    t_curriculum_stages: list[tuple[int, float]] = field(
        default_factory=lambda: [(2, 0.30), (4, 0.60), (8, 1.00)]
    )
    max_seq_len: int = 1024            # Maximum input sequence length
    use_positional_encoding: bool = False  # Disabled by default — thought loop provides temporal structure
    dropout: float = 0.1              # Dropout rate for attention and projections

    # ── NLM Configuration ───────────────────────────────────────────────
    nlm_groups: int = 1                # Number of neuron groups (1 = true per-neuron MLPs)
                                       # Set to e.g. 32 for grouped NLMs (16 neurons per group)

    # ── CTM-v2: FEEC Integrator ─────────────────────────────────────────
    # Structure-preserving dynamics for the thought loop. Replaces heuristic
    # gating with a symplectic-like integration scheme.
    use_feec: bool = False               # Master switch for FEEC integrator
    feec_dt_init: float = 0.1            # Initial learnable step size per layer
    feec_damping_init: float = 0.1       # Initial damping coefficient γ
    feec_clamp_dt: float = 1.0           # Upper bound on dt for stability
    # Trust-region bound on the relaxation state. The prospective-config loop
    # iterates the FEEC integrator up to max_inference_steps; a non-normal
    # recurrence can transiently amplify the state even at ρ<1 (ρ=1 with a
    # Jordan block still grows ∝T), so the loop is the unbounded part of the
    # model. This caps the per-position L2 norm of z and velocity at
    # feec_state_clip·√d_latent each step — a hard ceiling that only engages
    # during pathological growth (healthy ‖z‖≈0.3·√d), so the relaxation can't
    # diverge to NaN regardless of ρ/σ_max. 0 = disabled.
    feec_state_clip: float = 0.0
    feec_energy_penalty_weight: float = 0.01  # Weight of energy growth penalty in loss

    # ── CTM-v2: Matrix-Valued Residual Streams ──────────────────────────
    # Replaces NLM FIFO buffers + O(D²) sync with Hyperloop-style parallel
    # residual streams and diagonal manifold mixing.
    use_matrix_streams: bool = False     # Master switch
    n_streams: int = 4                   # Number of parallel residual streams
    stream_gating: str = "diagonal"      # "diagonal" or "sigmoid"

    # ── CTM-v2: Dual-Space Sparse Attention (DSSA) ──────────────────────
    # Replaces O(N²) cross-attention with SSE + MoBA hybrid.
    use_dssa: bool = False               # Master switch
    dssa_n_partitions: int = 32          # SSE state partitions
    dssa_top_k: int = 8                  # Active SSE partitions per step
    dssa_block_size: int = 64            # MoBA block size
    dssa_top_k_blocks: int = 4           # MoBA blocks per query

    # ── CTM-v2: Hyperloop Looped Middle Cycle ───────────────────────────
    # Weight-shares the middle layers via looping. Preserves depth while
    # cutting unique layer parameters. Model is widened to maintain total
    # parameter count.
    use_hyperloop: bool = False          # Master switch
    hyperloop_n_begin: int = 2           # Non-shared begin layers
    hyperloop_n_middle: int = 4          # Shared middle layers
    hyperloop_n_end: int = 2             # Non-shared end layers
    hyperloop_middle_loops: int = 2      # Times to loop the middle block

    # ── CTM-v2: Loop Position Embeddings ────────────────────────────────
    # Parameter-efficient way to distinguish thought iterations.
    # Each thought step gets a learned embedding added to the latent state.
    use_loop_pos_emb: bool = False       # Master switch

    # ── Theta-clocked phase coding on thought-step index (R12) ──────────
    # A theta phase φ_t = 2π·(t mod cycle)/cycle is cycled across thought
    # ticks; each latent unit d has a learnable preferred phase φ_d, and the
    # tick's synaptic input is multiplicatively modulated by
    # (1 + amplitude·cos(φ_t − φ_d)). Phase precession encodes tick order on a
    # continuous variable (Lisman & Jensen 2013; O'Keefe & Recce 1993). Applied
    # to the *transient* per-layer input (not the persisted latent) so it does
    # not compound across ticks.
    use_theta_phase: bool = False
    theta_cycle_len: int = 8             # ticks per theta cycle
    theta_amplitude: float = 0.5         # modulation depth (gate ∈ [1−amp, 1+amp])

    # ── Active-sensing re-read gate (R15) ───────────────────────────────
    # Per thought tick, soft-attend over the input embeddings to select a span
    # and inject a gated re-read summary back into the thought state — the
    # embodied 'active sensing' ingredient (rereading to clarify a referent).
    # Standalone (the EFE halting controller has no action set); gate starts
    # near zero (no-op) and the re-read is added to the transient per-layer
    # input so it does not compound across ticks.
    use_reread: bool = False

    # ── Canonical-microcircuit laminar coupling (R13) ───────────────────
    # Repurpose two MatrixResidualStream slots as an 'error' stream (0) and a
    # 'prediction' stream (1), coupled each tick by a Rao-Ballard residual
    # (error ← error − g·f(pred); pred ← pred + g·error). Lightweight: reuses
    # existing slots (no memory doubling), trained end-to-end (no extra loss).
    # Requires use_matrix_streams with n_streams ≥ 2.
    use_laminar_coupling: bool = False

    # ── Morphogenetic pre-pattern hypernetwork init (R14) ───────────────
    # A small hypernetwork (learned 'morphogen code' → low-rank generators)
    # produces a structural pre-pattern for the inter-stream mixing matrices:
    # it seeds their init AND acts as a soft anchor (‖W − generate‖² regularizer)
    # so weights relax toward the (adapting) low-rank pre-pattern. Requires
    # use_matrix_streams. Speculative (Levin/Pezzulo bioelectric prior).
    use_morphogen_init: bool = False
    morphogen_dim: int = 32              # morphogen code dimensionality
    morphogen_rank: int = 4              # low-rank of the generated pre-pattern
    morphogen_reg_weight: float = 0.001  # anchor strength (‖W − pre-pattern‖²)

    # ── CTM-v2: Triton Acceleration ─────────────────────────────────────
    use_triton_attention: bool = False   # Use Triton tiled attention kernel
    use_cuda_graphs: bool = False        # Wrap thought loop in CUDA Graph
    tiled_schedule: bool = False         # Use N·log(N) tiled schedule

    # ── Per-tick output heads (Option 3 fix for thought-loop collapse) ──
    # When True, each thought tick gets its own output adapter (Linear→GELU→
    # LayerNorm) feeding into a SHARED LM head. Removes the gradient-
    # interference problem where a single output_proj forces all T per-tick
    # gradients into the same parameters, which empirically collapses the
    # thought loop to "produce identical output at every tick" once training
    # gets going. Cost: T × ~(d_latent+d_model)*d_model extra params,
    # typically ~1% of the backbone.
    per_tick_heads: bool = False

    # ── Shared Head + Per-Tick FiLM (post-restart architecture) ─────────
    # When True, replaces the 8 per-tick LM heads (1.07B params for V=131072)
    # with a single shared head + per-tick FiLM modulation (16K params total).
    # This matches the published CTM architecture (one shared output
    # projection across ticks) and reduces head FLOPs ~85% in forward and
    # backward. FiLM = element-wise affine: y = gamma_t * x + beta_t,
    # where (gamma_t, beta_t) are learned per-tick R^d_model vectors.
    # Mutually exclusive with per_tick_heads.
    use_shared_head_film: bool = False

    # ── Tied Embeddings ─────────────────────────────────────────────────
    # When True, the LM head's vocab projection shares weights with the
    # token embedding (lm_head.weight = token_embedding.weight). Saves
    # vocab_size*d_model parameters (134M at V=131072, d_model=1024) and
    # the corresponding optimizer state (~1.6 GB for AdamW). Standard
    # since Press & Wolf 2017; safe at student scale below ~1B params.
    # Only meaningful when use_shared_head_film=True (with per_tick_heads,
    # there's no canonical "the" head to tie to).
    tie_embeddings: bool = False

    # ── Synchronization ─────────────────────────────────────────────────
    sync_method: str = "diag_summary"  # "full", "diag_summary", "low_rank", or "sparse_decay"
    sync_rank: int = 32                # Rank for low_rank sync method
    sync_sparse_pairs: int = 256       # Number of (i, j) pairs for sparse_decay sync method

    # ── Synapse ─────────────────────────────────────────────────────────
    synapse_type: str = "mlp"          # "mlp", "unet", or "dendritic"
    dendritic_n_branches: int = 4      # Dendritic compartments (synapse_type="dendritic")

    # ── Critical-Symmetric Dynamics Prior ───────────────────────────────
    # Initialize recurrent dynamics matrices (the synapse z-recurrence slice
    # and, if enabled, the Hebbian fast-weight M_0) as critically-normalized
    # symmetric random matrices instead of the default normal_(std=0.02) /
    # zeros. Produces a ~2/3 power-law variance spectrum and long-timescale
    # dynamics as a "scaffold for learning". sym_frac=1.0 → fully symmetric
    # (real spectrum); 0.0 → asymmetric; ~0.6 sits near the biological regime.
    use_critical_init: bool = False
    critical_init_sym_frac: float = 0.6
    critical_init_spectral_radius: float = 0.999

    # ── Synaptic Homeostasis (SHY) — Spectral Renormalization ───────────
    # Periodically project the synapse z-recurrence matrix back to spectral
    # radius ≤ target, downscaling supercritical recurrences toward criticality
    # (the homeostatic counterpart to use_critical_init: init near-critical,
    # then keep it there as training deforms the spectrum). Slow-wave-sleep
    # synaptic downscaling analogue.
    use_spectral_renorm: bool = False
    # 10, not 200: a sleep micro-cycle shifts the dynamics into a regime where ρ
    # drifts back toward/over 1.0 over the *following* normal steps (not just at
    # sleep). Exact-ρ renorm is ~2.6s for 26 layers, so per-step is too slow;
    # every 10 steps caps the post-sleep drift at ~0.99→0.994 (well under 1)
    # while keeping overhead ~25%. Paired with a renorm after EACH sleep step
    # (see worker_multi_gpu / train) so the sleep cycle itself can't leave ρ>1.
    # Drop to 5 if the SHY log still shows max ρ creeping > ~0.998.
    spectral_renorm_interval: int = 10     # training steps between renorm passes
    # Subcritical (<1.0) on purpose: weights are bf16, and the in-place rescale
    # injects ~±0.4% (2^-8) per-element spectral-radius noise, so a target of
    # exactly 1.0 lands in ~[0.996, 1.004] — supercritical half the time. A
    # target of 0.99 keeps the recurrence safely contracting even after a sleep
    # micro-cycle kicks ρ above 1.0 (target × (1+0.4%) ≈ 0.994 < 1.0).
    spectral_renorm_target: float = 0.99   # max allowed spectral radius

    # ── Multi-Rate Thought Loop (cross-frequency coupling) ──────────────
    # Matrix streams update at different timescales: stream i refreshes only on
    # ticks where t % period_i == 0 (power-of-2 schedule [1,1,2,4,8,...]).
    # Requires use_matrix_streams=True.
    use_multi_rate_streams: bool = False

    # ── Neuromodulated Optimizer ─────────────────────────────────────────
    use_neuromod_optimizer: bool = False  # Wrap AdamW with surprise-modulated LR
    neuromod_alpha: float = 1.0           # Modulation strength (0 = disabled)
    neuromod_min_scale: float = 0.1      # Minimum LR multiplier
    neuromod_max_scale: float = 3.0      # Maximum LR multiplier
    neuromod_ema_decay: float = 0.95     # EMA smoothing of surprise signal

    # ── Adaptive Intrinsic Plasticity ────────────────────────────────────
    # Unsupervised, cell-autonomous adaptation of each NLM neuron's activation
    # threshold to maintain a target mean firing rate.  Mirrors the biological
    # mechanism whereby neurons shift their intrinsic excitability to maximise
    # information transmission.
    #
    # During training each NLM hidden unit tracks an EMA of its post-GELU
    # activation.  A non-gradient ip_bias (analogous to a membrane threshold)
    # is updated in-place every forward pass using:
    #
    #   β ← β + η_IP · (target − ĥ)
    #
    # where ĥ is the running activation EMA.  When the unit is too active
    # (ĥ > target) the bias decreases, suppressing it; when too sparse it
    # increases, driving it into range.  This happens entirely outside the
    # gradient graph (no backprop needed).
    #
    # Requires use_matrix_streams=False (NLM v1 path).
    use_intrinsic_plasticity: bool = False
    ip_lr: float = 0.01            # η_IP — intrinsic plasticity learning rate
    ip_target: float = 0.1         # Target mean activation (≈10% sparsity)
    ip_ema_decay: float = 0.99     # EMA decay for activation tracking

    # ── Online Structural Plasticity ──────────────────────────────────────
    # Grow/prune MatrixResidualStream slots based on gate EMA utilization.
    # Requires use_matrix_streams=True. When OFF, no overhead.
    use_structural_plasticity: bool = False
    plasticity_prune_threshold: float = 0.05   # Gate EMA below this → prune
    plasticity_grow_threshold: float = 0.85    # Mean active gate EMA above this → grow
    plasticity_ema_decay: float = 0.99         # Smoothing for per-stream gate EMA
    plasticity_update_interval: int = 500      # Steps between grow/prune evaluations
    plasticity_min_active: int = 1             # Min stream slots to keep per layer

    # ── Dynamic Schema Routing ────────────────────────────────────────────
    # Routes latents to the most-similar MatrixResidualStream slot via cosine
    # similarity matching against per-stream EMA prototypes.  Schema-matched
    # streams receive a larger share of the layer update (fast assimilation);
    # novel inputs can wake a dormant stream rather than overwriting an active
    # schema.  Augments the Hebbian/BTSP lr_modulator for matched schemas.
    # Requires use_matrix_streams=True.
    use_schema_routing: bool = False
    schema_routing_ema_decay: float = 0.99      # EMA decay for prototype update
    schema_routing_temperature: float = 0.1     # Softmax temperature (lower = sharper)
    schema_routing_novelty_threshold: float = 0.1  # Max-sim below this → try wake dormant
    schema_routing_btsp_scale: float = 2.0      # Hebbian/BTSP LR boost at full match

    # Successor-Feature Schema Router (R6): replace the raw-feature EMA
    # prototypes with successor features ψ(s) ≈ φ(s) + γ·ψ(s'), so streams
    # cluster on *where states lead* (discounted future occupancy along the
    # sequence) rather than where they currently are.  ψ is a gradient-free
    # buffer matrix trained by a local TD delta rule; routing prototypes are
    # the EMA of ψ.  Requires use_schema_routing=True.
    use_successor_features: bool = False
    successor_gamma: float = 0.95               # SR discount over sequence positions
    successor_lr: float = 0.05                  # local TD learning rate for ψ

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100_000
    warmup_steps: int = 1000
    gradient_checkpointing: bool = True
    # Minimum T at which gradient_checkpointing actually activates. When
    # the curriculum-resolved (or fixed) T is below this threshold,
    # per-step checkpointing is skipped — recovering the throughput cost
    # since smaller activation graphs fit in VRAM without recomputation.
    # Default 5 means: T=2 and T=4 phases run without checkpointing
    # (saving ~20-30% step time), while T=5+ activates checkpointing
    # to keep memory bounded. Adjust based on your hardware: lower the
    # threshold if you OOM at T=4, raise it if you have headroom at T=5+.
    gradient_checkpointing_min_T: int = 5
    gradient_accumulation_steps: int = 1   # 1 = no accumulation. With N>1, runs
                                           # N micro-batches per optimizer step;
                                           # effective batch = batch_size · N.
    dtype: str = "bfloat16"            # "float32", "float16", or "bfloat16"
    grad_clip: float = 1.0            # Gradient clipping max norm
    eval_interval: int = 500          # Steps between evaluations
    log_interval: int = 50            # Steps between logging

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer: str = "adamw"
    # AdamW betas.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    # 8-bit AdamW via bitsandbytes. Cuts AdamW state memory ~8× (fp32 m+v
    # → quantized 8-bit blocks) with no measurable accuracy loss on
    # transformer-shaped models. On a 200M model that's ~2.8 GB recovered.
    use_8bit_adam: bool = False

    # Legacy fields kept for backward compatibility with old checkpoints
    # that may reference them. Not used by the new temporal loss.
    temporal_loss_type: str = "ramp_mono" # [DEPRECATED]
    tick_ramp_start: float = 0.5       # [DEPRECATED]
    tick_ramp_end: float = 1.5         # [DEPRECATED]
    mono_penalty_weight: float = 0.5   # [DEPRECATED]
    mono_penalty_min_frac: float = 0.1 # [DEPRECATED]
    mono_penalty_decay_until_frac: float = 0.3 # [DEPRECATED]
    aux_loss_weight: float = 0.1       # [DEPRECATED]
    min_loss_weight: float = 0.5       # [DEPRECATED]
    max_cert_weight: float = 0.5       # [DEPRECATED]

    # ── Data ────────────────────────────────────────────────────────────
    data_path: Optional[str] = None    # Path to local training text file
    eval_data_path: Optional[str] = None
    seq_len: int = 512                 # Training sequence length (context window)
    tokenizer: str = "gpt2"            # Tiktoken encoding name ("gpt2", "r50k_base", "cl100k_base")
    dataset: str = ""                  # HF dataset name ("fineweb-edu" or "" for local file)
    dataset_subset: str = "sample-10BT"  # FineWeb-Edu subset (sample-10BT, sample-100BT, etc.)
    checkpoint_dir: str = "checkpoints"  # Directory for saving checkpoints

    # ── Data Curriculum (Two-Phase) ─────────────────────────────────────
    use_two_phase_curriculum: bool = False
    phase1_tokens: int = 500_000_000   # Tokens to train in Phase 1 before switching to Phase 2
    
    # Phase 1: Logic Priming (Math-MIND and SFT data)
    phase1_datasets: list[str] = field(default_factory=lambda: [
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-CC-Math-v1"
        # , "nvidia/Nemotron-Pretraining-SFT-v1"
    ])
    phase1_dataset_subsets: list[str] = field(default_factory=lambda: [
        "Nemotron-Pretraining-RQA",
        "Nemotron-Pretraining-InfiniByte-Reasoning",
        "Nemotron-Pretraining-Wiki-Rewrite",
        "Nemotron-Pretraining-Scientific-Coding",
        "Nemotron-Pretraining-Math-Textbooks",
        "Nemotron-Pretraining-STEM-SFT",
        "Nemotron-Pretraining-Code-Concepts",
        "Nemotron-Pretraining-Unconditional-Algorithmic",
        "Nemotron-Pretraining-Formal-Logic",
        "Nemotron-Pretraining-Economics",
        "Nemotron-Pretraining-Multiple-Choice",
        "4plus_MIND"
        # , "default"
    ])
    phase1_dataset_weights: list[float] = field(default_factory=lambda: [
        0.10, 0.10, 0.05, 0.05, 0.10, 0.10,
        0.10, 0.05, 0.10, 0.05, 0.10, 
        0.10
        # , 0.20
    ])

    # Phase 2: Full Competence (High-Signal Hybrid Mix)
    phase2_datasets: list[str] = field(default_factory=lambda: [
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-Pretraining-Specialized-v1.1",
        "nvidia/Nemotron-CC-Math-v1"
        # , "nvidia/Nemotron-Pretraining-SFT-v1"
        # , "nvidia/Nemotron-Pretraining-Code-v2"
        # , "nvidia/Nemotron-CC-v2"
        # , "nvidia/Nemotron-CC-v2.1"
    ])
    phase2_dataset_subsets: list[str] = field(default_factory=lambda: [
        "Nemotron-Pretraining-RQA",
        "Nemotron-Pretraining-InfiniByte-Reasoning",
        "Nemotron-Pretraining-Wiki-Rewrite",
        "Nemotron-Pretraining-Scientific-Coding",
        "Nemotron-Pretraining-Math-Textbooks",
        "Nemotron-Pretraining-STEM-SFT",
        "Nemotron-Pretraining-Code-Concepts",
        "Nemotron-Pretraining-Unconditional-Algorithmic",
        "Nemotron-Pretraining-Formal-Logic",
        "Nemotron-Pretraining-Economics",
        "Nemotron-Pretraining-Multiple-Choice",
        "4plus_MIND"
        # , "default"
        # , "default"
        # , "default"
        # , "default"
    ])
    phase2_dataset_weights: list[float] = field(default_factory=lambda: [
        0.05, 0.05, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.05, 0.05, 0.10, 
        0.10
        # , 0.10
        # , 0.10
        # , 0.05
        # , 0.05
    ])

    # ── Distillation Strategy ───────────────────────────────────────────
    use_distillation: bool = False
    teacher_model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    teacher_d_model: int = 3136        # Teacher hidden dimension for projection
    distill_logit_weight: float = 1.0  # Weight for KL divergence loss
    distill_feature_weight: float = 0.0 # Weight for Z-alignment loss (default OFF — see distill_feature_method)

    # Device placement for the teacher model. By default the teacher is
    # colocated with the student ("auto" → same device as the rest of
    # training). On multi-GPU systems with limited per-GPU VRAM, setting
    # this to a separate device (e.g. "cuda:1") lets the student use the
    # full memory of cuda:0 while the (frozen) teacher runs in parallel
    # on cuda:1. The teacher inputs are shuttled across devices using
    # non-blocking copies so the cross-GPU transfer overlaps with student
    # compute.
    #
    # "auto" — same device as the student (the existing behavior).
    # "cuda:N" — explicit device for the teacher (e.g. "cuda:1").
    teacher_device: str = "auto"

    # Temperature for soft-target distillation. Standard KD uses T in [2, 5].
    # The KL loss is multiplied by T^2 so the gradient magnitude is invariant
    # to T (per Hinton 2015). Set to 1.0 to disable temperature softening.
    distill_temperature: float = 4.0

    # Which ticks to apply distillation to:
    #   "all"           — uniform mean over every tick. Strong KD signal at
    #                     every step. CAVEAT: when the teacher signal is
    #                     larger in magnitude than the base CE loss, this
    #                     pulls every intermediate tick toward the teacher's
    #                     final-answer-shaped distribution and flattens the
    #                     thought loop — every tick predicts the same thing
    #                     because every tick is being supervised against the
    #                     same target. Use with mono_penalty + ramp_mono if
    #                     you want the loop to develop genuine refinement.
    #   "first"         — distill ONLY the first tick. The teacher
    #                     constrains the model's initial guess; subsequent
    #                     ticks refine via base CE + mono_penalty. Use when
    #                     "all" is flattening the thought loop and you
    #                     can't reduce distill_logit_weight enough to
    #                     compensate. Cleanest interpretation: teacher
    #                     handles fast pattern-matching, thought loop
    #                     handles iterative refinement.
    #   "last"          — distill ONLY the final tick. WARNING: empirically
    #                     unstable in this codebase. The model's CE-best
    #                     trajectory across ticks is often "rolling toward
    #                     the right answer from a different direction than
    #                     the teacher's prediction"; the final-tick
    #                     distillation jerks tick T toward the teacher,
    #                     producing per-tick CE shaped like
    #                     [10, 9, 8, 7, 6, 9] (decreases through the loop,
    #                     jumps up at the end), which mono_penalty then
    #                     punishes severely.
    #   "decay_ramp"    — linearly-decaying weights from tick 0 (full
    #                     weight) to tick T-1 (10% weight). Front-loaded
    #                     teacher signal: early ticks teacher-constrained,
    #                     later ticks freer. Hybrid of "all" and "first".
    #                     Renormalized so mean weight is 1.0, keeping
    #                     distill_logit_weight semantics consistent.
    #   "lm_aligned"    — mirror the temporal_loss_type aggregation:
    #                     dynamic_aggregate → KD on lowest-CE/highest-cert ticks
    #                     ramp_mono         → ramp-weighted KD across ticks
    distill_tick_aggregation: str = "all"

    # Feature-distillation method (when distill_feature_weight > 0):
    #   "mse"     — raw MSE between student z and projected teacher z.
    #               Strongly NOT recommended for cross-architecture distillation
    #               (e.g. NemotronH→CTM) — magnitude scales differ.
    #   "cosine"  — 1 - cos_sim(z, teacher_z_aligned). Magnitude-invariant;
    #               the right default when student and teacher come from
    #               different architectures.
    #   "mse_normed" — LayerNorm both sides before MSE. Compromise between
    #                  the two; preserves dimensional structure while
    #                  removing magnitude mismatch.
    distill_feature_method: str = "cosine"

    # Top-K soft-target distillation. When > 0, KL is computed only over
    # the teacher's top-K most-likely tokens at each position rather than
    # the full vocabulary. The student is gathered at those same indices
    # and softmaxed over only those K tokens, so both distributions live
    # on the same support and the standard KL formulation applies.
    #
    # Why this is essentially free in accuracy: a calibrated 4B-class
    # teacher puts >99% of its probability mass in its top-256 tokens for
    # next-token prediction (the long tail is essentially noise from the
    # softmax over an oversized vocabulary). The student matching the
    # teacher's top-K distribution captures all the signal that wasn't
    # already going to be drowned out anyway.
    #
    # Why it speeds up: the fp32 log_softmax over [B*S, V] at V=131072
    # allocates ~268 MB per call and is a major slice of step time when
    # distillation is on. With K=256 that becomes [B*S, 256] = ~512 KB —
    # a ~512× reduction in softmax memory traffic, and the surrounding
    # KL is correspondingly smaller.
    #
    # CALIBRATION NOTE: top-K KL has a larger absolute magnitude than
    # full-vocab KL because the loss is no longer averaged across the
    # near-zero tail. In synthetic tests the magnitude rises ~2-4× while
    # the gradient cosine vs. full-vocab KL stays above 0.98. If you see
    # KL dominating the LM loss after enabling this, drop
    # distill_logit_weight by a similar factor (e.g. 1.0 → 0.3).
    #
    # 0 = full-vocab KD (legacy / safe fallback).
    # 128-512 are reasonable values; 256 is a good default.
    distill_top_k: int = 256

    # ── Offline Teacher Cache ───────────────────────────────────────────
    # When True, the training loop reads pre-computed teacher top-K
    # logits from disk (produced by scripts/cache_teacher_logits.py) and
    # does NOT load a teacher model. cuda:1 is freed for student-side
    # data parallelism, and the per-step `teacher_issue` cost vanishes.
    #
    # When this is enabled, `use_distillation` is implicitly true (the
    # cache only exists if you intend to distill), and `teacher_model_name`
    # / `teacher_device` are ignored.
    use_cached_teacher: bool = False
    teacher_cache_dir: str = ""

    # ── Device ──────────────────────────────────────────────────────────
    device: str = "auto"               # "auto", "cuda", "cpu"

    # ── Biological Learning Extensions ──────────────────────────────────
    # Two opt-in mechanisms that exploit the CTM's recurrent thought loop.
    # Both default OFF. When OFF, no parameters are created and the
    # training loop runs the original code path with zero overhead.

    # 1) Hebbian fast-weights inside the Synapse.
    #    Each ThoughtLayer maintains a per-(batch, position) fast-weight
    #    matrix M updated by outer product of (prev_state, attn_out) at
    #    every tick. The readout M·a is added (gated) to the synapse
    #    output, giving the model an instant, per-sequence associative
    #    memory that lives entirely in activation state — no gradient
    #    update needed. M resets between batches, persists across ticks.
    #
    #    Memory cost scales with bottleneck_dim²:
    #      bottleneck_dim=64  →  ~2 MB per layer per fwd pass at B=1, S=512
    #      bottleneck_dim=0   →  full d_latent × d_model (very expensive)
    use_hebbian_synapse: bool = False
    hebbian_bottleneck_dim: int = 64        # 0 → no bottleneck (full matrix)
    hebbian_n_compartments: int = 1         # >1 → dendritic compartmentalization
    hebbian_decay_init: float = 0.9         # Persistence of M across ticks
    hebbian_lr_init: float = 0.1            # Outer-product update magnitude
    hebbian_gate_init: float = -3.0         # Logit; -3 ≈ 5% initial readout
    # ── Diagnostic: force the Hebbian readout gate to a fixed value ──
    # When set (e.g. 0.9), the gate logit is bypassed and the readout
    # contribution is multiplied by this constant. Used to isolate
    # "is the readout content useful?" from "is the gate blocked from
    # opening?". Three diagnostic outcomes:
    #   No CE change vs baseline → readout content adds nothing
    #   CE worse than baseline   → readout is actively noisy
    #   CE better than baseline  → gate gradient was the bottleneck
    # Default None = use learned gate (normal training).
    hebbian_force_gate: float | None = None
    # Hebbian update rule selection:
    #   "outer_product" (default) — classic M ← decay·M + lr·(z⊗a).
    #                Simple, but saturates: |M| grows unbounded since
    #                every co-occurrence is accumulated regardless of
    #                whether M already encodes it.
    #   "delta"      — error-correcting rule from Nested Learning /
    #                Schlag-Schmidhuber FW Programmers:
    #                  M ← decay·M + lr·(z - M·a)⊗a
    #                The bracketed (z - M·a) term is a local prediction
    #                error. Only writes corrections to whatever the
    #                current M fails to retrieve. Reaches steady-state
    #                M·a ≈ z and stops growing rather than saturating.
    #                Better choice if observed |M| growth correlates
    #                with degraded readout quality at scale.
    hebbian_update_rule: str = "outer_product"

    # ── Behavioral Timescale Synaptic Plasticity (BTSP) ─────────────────
    # Augments the per-position Hebbian fast-weight update with a causal
    # sequence-spanning eligibility trace.  At each thought tick, each
    # position s receives extra potentiation from all prior positions τ≤s,
    # weighted by an exponential decay kernel β^(s−τ) and the per-position
    # salience (surprise).  This lets a salient event at position s
    # instantly retro-potentiate associations formed earlier in the sequence
    # — enabling true one-shot learning over temporal delays without
    # backpropagation.
    #
    # Salience source: hebbian_cert_lr_alpha > 0 provides per-position
    # surprise (1 + α·uncertainty).  When cert_lr_alpha=0, BTSP defaults to
    # uniform salience (pure causal exponential MA of outer products).
    #
    # Requires use_hebbian_synapse=True.
    use_btsp: bool = False
    btsp_lr_init: float = 0.05          # Initial BTSP learning rate η_BTSP
    btsp_kernel_decay_init: float = 0.9  # Initial kernel decay β (causal span)
    btsp_salience_threshold: float = 0.0 # Φ threshold; 0 = continuous (no gate)

    # ── Synaptic Tagging and Capture (STC) ───────────────────────────────
    # Biological late-phase LTP requires two concurrent signals:
    #   Tag (local):  A synapse experiencing high prediction error sets a
    #                 transient biochemical marker (early-LTP trace).
    #   PRP (global): Strong, surprising stimuli trigger the soma to synthesize
    #                 Plasticity-Related Proteins that diffuse globally.
    # Only when BOTH signals exceed threshold does the synapse undergo
    # permanent structural consolidation (late-LTP / slow-weight update).
    #
    # Computationally:
    #   Tag = fast-decaying EMA of |z - M·a| in HebbianSynapse (per layer).
    #   PRP = NeuroPlasticOptimizer._surprise_ema (global scalar).
    #   Gate: if tag < threshold_tag OR prp < threshold_prp,
    #         slow-weight gradients for that layer are multiplied by min_gate.
    #
    # Effect: routine, low-salience inputs cannot erode consolidated schemas.
    # Reduces reliance on brute-force sleep replay to protect old memories.
    #
    # Requires use_hebbian_synapse=True. Works best with use_neuromod_optimizer=True
    # to provide a meaningful PRP signal; if absent, PRP is 0 and the gate
    # acts purely on the local tag (threshold_prp should be set to 0.0).
    use_stc: bool = False
    stc_tag_decay: float = 0.5        # Fast-decaying EMA coefficient for local tag
    stc_threshold_tag: float = 0.05   # Tag norm threshold; below → layer is "stale"
    stc_threshold_prp: float = 0.3    # PRP (surprise EMA) threshold; below → routine input
    stc_min_gate: float = 0.01        # Gradient multiplier when gate is closed
    # Quick-win #6: gate STC by the Hebbian BURST probability (apical credit)
    # instead of the NPO surprise EMA. False → legacy surprise-EMA gating.
    stc_prp_from_burst: bool = False

    # ── Similarity-Weighted Interleaved Learning (SWIL / CLS Priority #3) ──
    # Augments the PrioritizedReplayBuffer with semantic indexing so sleep
    # replay is biased toward structurally related past episodes rather than
    # high-surprise episodes alone.
    #
    # Each stored episode carries a semantic vector — the mean-pooled final-tick
    # latent z_curr, optionally truncated to swil_embed_dim for storage efficiency.
    # When sampling for sleep, the cosine similarity between the current batch's
    # semantic vector (query) and each stored episode's vector is computed and
    # added to the free-energy priority:
    #
    #   w_i = softmax( (F_i + λ · cos_sim(query, v_i)) / τ )
    #
    # Episodes that are both high-surprise AND structurally similar to the current
    # context are sampled preferentially, preventing gradient interference with
    # representations currently being modified. Pure-surprise episodes that are
    # unrelated to the current context are down-weighted.
    #
    # Requires use_sleep_consolidation=True and use_prioritized_replay=True.
    use_swil: bool = False
    swil_sim_weight: float = 1.0   # λ — cosine similarity scale relative to FE
    swil_embed_dim: int = 64       # Stored vector dim; 0 = use full d_latent

    # ── Episodic Differentiable Neural Dictionaries (DNDs) ────────────────────
    # Non-parametric key-value bank for zero-shot retrieval of previously-solved
    # contexts. Keys = truncated mean-pooled input embeddings; values =
    # mean-pooled final-tick latent z from high-confidence forward passes.
    # When cosine_sim(current_key, stored_key) >= dnd_confidence_threshold,
    # the thought loop is bypassed entirely — the cached latent is expanded and
    # projected to logits via the output head in O(1). Implements Episodic
    # Control: deterministic algorithmic tasks are solved without re-running T
    # thought steps once the model has mastered them.
    #
    # Enable with --use_dnd. Works with any other flag combination.
    use_dnd: bool = False
    dnd_capacity: int = 1000           # Max key-value pairs in the episodic bank
    dnd_key_dim: int = 64             # Key dim (first N dims of mean-pooled text embedding)
    dnd_confidence_threshold: float = 0.98   # Cosine sim threshold to bypass thought loop
    dnd_write_confidence: float = 0.9  # Min mean token certainty to write a new entry
    dnd_hopfield_beta: float = 4.0    # Softmax inverse temperature for Hopfield retrieval
    dnd_min_novelty: float = 0.05     # Min (1 - max_sim) required to write (skip near-dupes)

    # ── Neuron–Astrocyte Dense Associative Memory (R7) ────────────────────
    # Long-term schema memory alongside the (short-term) DND.  Retrieval uses a
    # higher-order separation function softmax(β·relu(sim)^order) — the
    # Krotov–Hopfield dense-associative-memory rule whose pattern capacity
    # scales K_max ∝ N^(order-1) (order=4 ≈ N³, the supralinear neuron–astrocyte
    # regime of Kozachkov, Slotine & Krotov, PNAS 2025).  Eviction is driven by
    # astrocyte importance traces (a slow EMA of retrieval co-activation), so
    # frequently-recalled schemas persist rather than being evicted FIFO.  At
    # the bypass point the model routes to whichever of {DND, AstroDND} hits
    # with the lower normalized retrieval entropy.
    #
    # NOTE: the literal [d,d,d_v] coupling tensor from the source write-up is
    # infeasible (d³ floats); the N³ figure is a *capacity* result and is
    # realized here via the separation function over an M×d pattern bank.
    use_astro_memory: bool = False
    astro_capacity: int = 4000         # Larger than DND — long-term schema store
    astro_key_dim: int = 64            # Key dim (first N dims of mean-pooled text embedding)
    astro_order: int = 4               # Separation order; capacity ~ N^(order-1)
    astro_beta: float = 8.0            # Softmax inverse temperature (sharper than DND)
    astro_confidence_threshold: float = 0.9   # Cosine sim to bypass (schema-level, looser than DND)
    astro_write_confidence: float = 0.85      # Min mean certainty to write a new entry
    astro_trace_decay: float = 0.99    # Astrocyte importance-trace EMA decay
    astro_min_novelty: float = 0.02    # Min (1 - max_sim) required to write (skip near-dupes)

    # ── Dendritic Gated Network parallel head (R5, Sezener et al. 2021) ───
    # A parallel LM head where each token routes to 1-of-K dendritic branches
    # via a frozen random gate; the active branch's weights are updated by a
    # local delta (log-loss) rule (no backprop) for forgetting resistance. Its
    # logits are added to the main head scaled by a learned blend gate, and a
    # small CE aux loss on the blended logits trains the blend. The dendritic
    # input is a frozen random projection to dgn_proj_dim to keep w feasible
    # (w[K, proj_dim, vocab]; ~134 MB at K=8, proj_dim=32, vocab=131072).
    use_dgn_head: bool = False
    dgn_n_branches: int = 8
    dgn_proj_dim: int = 32
    dgn_eta: float = 0.05              # local delta-rule learning rate
    dgn_aux_weight: float = 0.1        # weight of the DGN CE aux loss (trains blend)

    # ── GFlowNet thought-trajectory sampler (R8, E. Bengio et al. 2021) ───
    # A stochastic policy samples one of M learned "thought operators" per
    # thought tick (a small learned perturbation of the latent), trained by a
    # Trajectory-Balance loss with reward R(τ)=exp(−β·seq_loss) so diverse
    # high-reward latent reasoning paths are sampled ∝ reward. Modes are sampled
    # (no_grad) during the inference relaxation and the policy is re-run with
    # grad afterward for the TB loss (mirrors the EFE controller). Default off.
    use_gflownet: bool = False
    gflownet_n_modes: int = 4          # M discrete thought operators
    gflownet_op_scale: float = 0.1     # operator perturbation magnitude
    gflownet_hidden_dim: int = 32      # policy MLP hidden width
    gflownet_reward_beta: float = 1.0  # logR = −beta · sequence_loss
    gflownet_tb_weight: float = 0.1    # weight of the Trajectory-Balance loss
    gflownet_tb_huber_delta: float = 10.0  # robust-TB threshold: |residual|>delta → linear (bounded grad)

    # ── Thalamic Multiplicative Gating ─────────────────────────────────────
    # Intercepts cross-attention output and scales by an entropy-derived gain.
    # Low entropy (focused attention) → gain ≈ 1 (signal passes).
    # High entropy (diffuse/noisy attention) → gain ≈ 0 (state protected).
    use_thalamic_gating: bool = False

    # ── Prospective Configuration ───────────────────────────────────────
    # Biologically motivated Expectation-Maximization (EM) training.
    # 
    # Replaces Backpropagation Through Time (BPTT). When enabled:
    # 1. Inference Phase: The neural activities relax into an equilibrium
    #    state that accounts for the target outcome (clamped top layer).
    #    Weights are frozen during this phase. Loop terminates when energy
    #    stabilizes (rel_delta < inference_energy_tol).
    # 2. Learning Phase: Synaptic weights are updated to consolidate the
    #    new activity pattern using purely local gradients evaluated at
    #    the steady-state prospective configuration.
    #
    # Requires use_feec=True.
    use_prospective_config: bool = False
    
    # Maximum steps allowed in the inference loop. Should be set very high
    # (e.g. 1000) as inference relies on energy stabilization to terminate,
    # but provides a safety bound against floating point divergence.
    max_inference_steps: int = 1000

    # Threshold for energy stabilization (tau). The inference loop terminates
    # when the relative change in FEEC energy drops below this value.
    inference_energy_tol: float = 1e-4

    # ── Amortized Inference ─────────────────────────────────────────────────
    # Single-pass prediction of the prospective configuration equilibrium state.
    # A lightweight feedforward network q(z*|x) maps text_emb → z_hat, which
    # warm-starts the inference loop near the true equilibrium. This reduces
    # the required relaxation steps from O(1000) → amortized_inference_steps.
    # An auxiliary MSE loss trains the network to match the discovered z*.
    # Only meaningful when use_prospective_config=True.
    use_amortized_inference: bool = False
    amortized_inference_steps: int = 5      # relaxation steps after warm-start
    amortized_hidden_dim: int = 0           # 0 → auto (2 × d_model)
    amortized_aux_loss_weight: float = 0.1  # weight on MSE(z_hat, z*) auxiliary loss
    # Quick-win #13: L1 firing-rate penalty on the thought-loop latent (sparsity).
    # 0.0 = OFF (no-op). Small values (~1e-4–1e-3) encourage sparse activations.
    l1_rate_weight: float = 0.0
    # Quick-win #14: critical-period freeze. After this fraction of max_steps,
    # freeze the EARLY thought layers (begin_layers) so their learned features
    # lock in while later layers stay plastic. 0.0 = OFF (never freezes).
    critical_period_freeze_pct: float = 0.0
    # Anti-collapse regularization (Halvagal & Zenke 2023 / VICReg) on the
    # thought-loop latent — counters the BYOL/SimSiam-class representational
    # collapse (srank→1, |z|→0) that pure-local PC against a detached target
    # admits. A1 = variance-maximization (−log var), A2 = off-diagonal covariance
    # decorrelation. False = OFF (no-op). Added to the loss (Phase-2 weight update).
    use_pc_var_reg: bool = False
    pc_var_weight: float = 0.1      # λ_var (A1): variance-maximization weight
    pc_cov_weight: float = 1e-3     # λ_cov (A2): off-diagonal covariance weight (≈1/d)

    # ── Hypersphere latent constraint (nGPT-style anti-collapse experiment) ──
    # When on, project the latent thought-state z onto a fixed-radius hypersphere
    # at the end of every thought tick (radius = hypersphere_z_scale·√d_latent,
    # matching the PC clamp target's radius). Removes the magnitude degree of
    # freedom so the (possibly over-damped) relaxation keeps z at the scale the
    # generative dynamics expect, giving z* a chance to reach the diverse
    # token-distinct target directions instead of decaying into a low-norm
    # rank-1 attractor. Magnitude-only constraint — does NOT save memory.
    use_hypersphere_z: bool = False
    hypersphere_z_scale: float = 0.3  # radius = this · √d_latent (matches clamped_target)

    # ── BCM Sliding Threshold (metaplasticity for the EM loop) ─────────────
    # Tracks EMA of mean squared latent magnitude during the inference phase.
    # Penalizes the learning pass when z² exceeds the historical threshold,
    # preventing runaway excitation and catastrophic forgetting.
    # Only active when use_prospective_config=True.
    use_bcm_threshold: bool = False
    bcm_ema_decay: float = 0.99    # EMA smoothing for the sliding threshold
    bcm_loss_weight: float = 0.1   # Weight on the homeostatic BCM penalty

    # ── Precision-Weighted Gradient Scaling (Priority #2) ─────────────────
    # Applies per-layer gradient multipliers derived from HPC precision π_ℓ.
    # High-precision layers (well-learned schemas) are scaled DOWN to protect
    # consolidated knowledge. Low-precision layers are scaled UP for amplified
    # learning on novel/uncertain inputs.
    #
    #   scale_ℓ = clip(π_ref / π̄_ℓ, min_scale, max_scale)
    #
    # PC layer parameters are excluded since their local_loss already
    # incorporates π_ℓ (0.5 * π_ℓ * ε_ℓ²).
    # Requires use_hierarchical_pc=True.
    use_precision_neuromod: bool = False
    precision_neuromod_min_scale: float = 0.2   # floor for per-layer gradient multiplier
    precision_neuromod_max_scale: float = 5.0   # ceiling for per-layer gradient multiplier
    precision_neuromod_ema_decay: float = 0.95  # EMA smoothing of per-layer π readings

    # ── ACh / NE Dual-Uncertainty Channels (R9, Yu & Dayan 2005) ──────────
    # ACh (expected uncertainty) = EMA of the per-step error → down-weights HPC
    # precision (up-weights bottom-up evidence). NE (unexpected uncertainty) =
    # change-detector on ACh → drives the NeuroPlasticOptimizer LR and, on a
    # spike (context switch), triggers a reset broadcast: flush the Hebbian
    # carry, warm-start from the amortized net, and transiently raise the
    # schema-router temperature. Each coupling is optional:
    #   ACh→precision needs use_hierarchical_pc; NE→LR needs
    #   use_neuromod_optimizer; the temperature bump needs use_schema_routing.
    use_achne: bool = False
    achne_ach_decay: float = 0.8           # EMA decay for ACh (expected uncertainty)
    achne_ne_decay: float = 0.95           # EMA decay for the NE deviation signal
    achne_ne_threshold: float = 3.0        # reset fires when NE > threshold × baseline
    achne_ne_baseline_decay: float = 0.999  # slow baseline the NE spike is measured against
    achne_ach_baseline_decay: float = 0.99   # ACh baseline → scale-free relative ACh (precision recovers ~70 steps after FE stabilizes)
    achne_warmup: int = 50                 # min steps before a reset can fire
    achne_refractory: int = 20             # steps after a reset blocking re-fire
    achne_ach_precision_scale: float = 1.0  # precision_mod = 1/(1 + scale·ACh)
    achne_reset_temp_mult: float = 3.0     # schema-router temperature × this on NE reset
    # Quick-win #12: on an NE reset, transiently raise the Hebbian force_gate by
    # this factor (clamped ≤1.0), relaxing back over the next forwards — mirrors
    # the schema-temperature bump. 1.0 = OFF (no-op). Requires hebbian_force_gate set.
    force_gate_ne_mult: float = 1.0

    prospective_beta: float = 2.0           # [DEPRECATED]
    # Certainty-modulated learning rate ("neurochemical modulation"):
    # When > 0, the Hebbian outer-product update is scaled per-position
    # by (1 + alpha * uncertainty), where uncertainty = 1 - certainty at
    # the PREVIOUS tick. Biologically: locus coeruleus releases NE in
    # response to surprise, which then potentiates plasticity for
    # subsequent events. Functionally: it writes to fast-weight memory
    # preferentially in surprising contexts rather than every position
    # equally, which we hope gives the Hebbian path informative content
    # rather than noise. Tick 0 has no predecessor → no modulation.
    # alpha=0 disables (same as base behavior). alpha=2 means an
    # uncertainty of 1.0 triples the lr at that position.
    hebbian_cert_lr_alpha: float = 0.0

    # ── Sleep-Based Memory Consolidation ────────────────────────────────
    # Implements a dual-phase wake/sleep training paradigm to prevent
    # destruction of fast-weight memory between batches.
    #
    # Wake phase: At the end of each forward pass, the mean Hebbian
    #   fast-weight matrix (averaged over batch and sequence dimensions)
    #   is stored as a cross-batch carry-over. The NEXT forward pass
    #   warm-starts from this carry-over instead of zeros, making the
    #   Hebbian associative memory persistent across batches (analogous
    #   to the hippocampus retaining episodic traces between waking hours).
    #
    # Sleep phase: Every `sleep_interval` optimizer steps, stored
    #   sequences from the episodic cache are replayed through the model
    #   with the warm-started carry-over active. The standard LM loss on
    #   these replays drives upward distillation of fast-weight episodic
    #   knowledge into the slow (gradient-trained) weights — mirroring
    #   neocortical consolidation during slow-wave sleep.
    #
    # Requires use_hebbian_synapse=True.
    use_sleep_consolidation: bool = False
    sleep_interval: int = 100        # Optimizer steps between sleep micro-cycles
    sleep_buffer_size: int = 32      # Max sequences retained in episodic cache
    sleep_replay_steps: int = 4      # Consolidation gradient steps per sleep cycle
    sleep_loss_weight: float = 0.3   # Scale factor applied to consolidation loss
    # Max Frobenius norm of the per-layer Hebbian carry-over (0 = unbounded).
    # The carry is an integrator (forward warm-starts M from it and re-saves
    # mean(M_final) ≈ carry+ΔM), so it grows without bound across sleeps,
    # saturating the output (certainty collapse) and eventually NaN-ing. This
    # caps it; healthy carry ≈ 0.3, so ~1.0 is a generous safety bound.
    sleep_carry_max_norm: float = 0.0
    # Isolate the optimizer state across the sleep micro-cycle: snapshot the
    # optimizer's momentum/variance buffers before the replay steps and restore
    # them after. Sleep still updates the *weights* (consolidation) but its
    # gradients no longer pollute the main AdamW EMA — which otherwise carries
    # the sleep direction into subsequent normal steps and drives the recurrence
    # supercritical (the post-sleep ρ drift). Standard "auxiliary-phase optimizer
    # state isolation". Snapshot is taken to CPU to avoid a VRAM spike.
    isolate_sleep_optimizer: bool = False

    # ── Free-Energy Prioritized Episodic Replay (Priority #3) ─────────────
    # Upgrades the uniform-random sleep replay to hippocampal-style priority
    # replay: high-free-energy (hard/surprising) episodes are replayed more
    # often; low-FE (fully consolidated) episodes are evicted when the buffer
    # fills.  Priority = hpc_free_energy when HPC is active; CE loss otherwise.
    #
    # Requires use_sleep_consolidation=True.
    use_prioritized_replay: bool = False
    replay_fe_temperature: float = 1.0  # softmax temperature τ for sampling weights

    # 2) Predictive Coding via temporal hierarchy.
    #    A per-tick "cerebellar" readout predicts the teacher's hidden
    #    state at tick t (with the *same* teacher_z target supplied each
    #    tick — the student should converge toward it). Yields an
    #    auxiliary loss that gives local gradient at every tick rather
    #    than only at the final-tick LM head. Requires the cached
    #    teacher to provide hidden-state features (teacher_z); if
    #    teacher_z is missing the PC loss is silently zero.
    use_predictive_coding: bool = False
    pc_loss_weight: float = 0.05            # Weight of PC loss in total
    pc_hidden_dim: int = 0                  # 0 → auto: max(2*d_latent, teacher_d_model)
    pc_normalize: str = "layernorm"         # "layernorm" or "none"
    pc_loss_type: str = "cosine"            # "cosine" or "mse"
    pc_dropout: float = 0.0
    # What the cerebellar readout tries to predict at each tick:
    #   "teacher"    — the teacher's hidden state at the same position.
    #                  Most faithful to canonical PC-as-distillation, but
    #                  REQUIRES teacher_z to be available (either via a
    #                  live teacher, or by extending the cache with
    #                  scripts/extend_teacher_cache_with_z.py). If
    #                  teacher_z is absent, PC silently degrades to zero.
    #   "next_tick"  — at tick t, predict z_{t+1} (the student's own
    #                  next-tick latent, detached). Zero extra compute,
    #                  no teacher_z needed. This is the most direct
    #                  realisation of canonical Predictive Coding
    #                  (Friston: brain predicts its own next state).
    #                  Recommended when teacher_z is unavailable.
    #   "final_tick" — at tick t, predict z_{T-1} (the student's own
    #                  final-tick latent, detached). Stronger
    #                  "converge toward the answer" pressure than
    #                  next_tick; cheap (one stored tensor) but less
    #                  truly local.
    pc_target: str = "teacher"
    # Which ticks to apply PC loss to:
    #   "all"   — every tick (default; richest signal)
    #   "early" — first floor(T/2) ticks (focus the predictor on early refinement)
    #   "last"  — final tick only (equivalent to feature-distillation, mostly here for ablation)
    pc_tick_aggregation: str = "all"

    # ── Hierarchical Predictive Coding (Mechanism 1) ─────────────────────
    # Formalizes the thought loop as iterative free-energy minimization
    # over a deep generative hierarchy. T thought ticks become T inference
    # iterations of a predictive coding network where each layer predicts
    # the layer below, and only prediction errors propagate upward.
    #
    # Requires use_matrix_streams=True. When enabled, n_streams is
    # automatically doubled (half value μ, half error ε channels).
    #
    # Gradient locality: all weight updates use fully local rules with
    # .detach() at layer boundaries. At the PC fixed point, these local
    # gradients equal backprop (Millidge et al., Neural Computation 2022).
    #
    # The existing CerebellarReadout (use_predictive_coding) is subsumed:
    # the top-layer PC error against teacher_z IS the predictive-coding
    # signal. Both can coexist but are redundant.
    pure_pc_mode: bool = False                  # Detach z between thought ticks to disable BPTT
    use_hierarchical_pc: bool = False           # Master switch
    hpc_inference_lr: float = 0.1               # η in μ ← μ − η·(ε − Wᵀε)
    hpc_generative_type: str = "mlp"            # "mlp" or "unet"
    hpc_generative_hidden_dim: int = 0          # 0 → auto (2 × d_latent)
    hpc_error_as_loss_weight: bool = True        # Weight per-tick CE by mean precision
    hpc_local_loss_weight: float = 0.1           # Weight of local PC losses in total
    hpc_pc_n_streams: int = 8                    # Streams when PC enabled (4μ + 4ε)

    # ── R1: Expected-Free-Energy Thought-Step Controller (Active Inference) ──
    # A PonderNet-style halting policy over the thought loop, shaped by an
    # expected-free-energy (EFE) objective. At each thought tick t the
    # controller emits a per-position halting probability λ_t from EFE
    # features (ambiguity = 1 − certainty; epistemic value = latent info
    # gain ‖z_t − z_{t-1}‖). The per-position halting distribution
    #   p_t = λ_t · Π_{t'<t}(1 − λ_{t'})
    # reweights the per-position cross-entropy (Σ_t p_t·CE_t replaces the
    # uniform per-tick mean) so the model is rewarded for emitting early
    # when confident. Three loss terms are added:
    #   1. PonderNet reconstruction:  Σ_t p_t · CE_t
    #   2. KL(p ‖ Geometric(efe_halt_prior)) — keeps the loop from collapsing
    #      to "halt at tick 0".
    #   3. EFE shaping: Σ_t p_t · (ambiguity_t − epistemic_t) — pushes halting
    #      mass toward low-EFE ticks (confident + info-gain exhausted).
    # At eval / generate, the loop early-exits once the mean cumulative halt
    # probability crosses efe_halt_threshold — the wall-clock payoff.
    #
    # Only active on the legacy BPTT thought loop (not under
    # use_prospective_config, which has no per-tick loop). Default OFF.
    use_efe_controller: bool = False
    efe_halt_prior: float = 0.1          # Geometric prior λ for the PonderNet KL
    efe_ponder_kl_weight: float = 0.01   # Weight on KL(p ‖ Geometric)
    efe_weight: float = 0.1              # Weight on the EFE shaping term
    efe_halt_threshold: float = 0.5      # Mean cumulative halt prob to early-exit (eval/gen)
    efe_controller_hidden_dim: int = 0   # 0 → auto (max(32, d_latent // 8))

    # ── R2: Burst-multiplexed dendritic credit assignment (Burstprop) ────────
    # Payeur, Guerguiev, Zenke, Richards & Naud (Nat Neurosci 2021): postsynaptic
    # activity factorizes into events E (any spike) and bursts B (high-frequency
    # spikes gated by an apical top-down signal). The fast-weight update replaces
    # the unsigned event factor with the burst-vs-event residual:
    #   ΔM ∝ (B − P̄·E) ⊗ a ,  B = burst_p · E ,  P̄ = running burst probability
    # The apical burst probability burst_p is derived from the *previous* thought
    # tick's per-layer hierarchical-PC prediction error — top-down credit
    # multiplexed onto the same fast-weight write. High top-down error → more
    # bursting → stronger credit assignment at that layer/position.
    #
    # Requires use_hebbian_synapse=True AND use_hierarchical_pc=True (the burst
    # signal source). Default OFF.
    use_burstprop: bool = False
    burstprop_tau: float = 100.0         # EMA window for the running burst probability P̄
    burstprop_scale_init: float = 1.0    # Initial slope mapping top-down error → burst prob

    # ── R3: Prefrontal meta-RL reward/action channel (self-supervised) ───────
    # Wang et al. (Nat Neurosci 2018) / RL² (Wang et al. 2016): a recurrent
    # network meta-trained with (obs_t, a_{t-1}, r_{t-1}) in its input learns an
    # in-context RL algorithm whose state lives in activations, not weights.
    # For an autoregressive LM, each sequence is an episode; the causal reward
    # r_{s-1} = p(token_{s-1}) (the probability the model assigned to the realized
    # previous token) and action a_{s-1} = argmax prediction at s-1. Under
    # teacher forcing these are recovered with a cheap no_grad pre-pass, then
    # shifted by one position and injected into the input embedding alongside the
    # token. The thought loop thereby sees how well it predicted recent tokens
    # and can adapt in-context.
    #
    # Requires a token embedding (not use_feature_encoder). Cost: one extra
    # no_grad forward per step. Default OFF.
    use_meta_rl: bool = False
    meta_rl_reward_scale_init: float = 0.1  # Initial gate on the reward/action injection

    # ── R11: BTSP-v2 — weight-dependent / stochastic / delayed plasticity ────
    # Upgrades the additive BTSP eligibility trace with three empirically-
    # grounded properties (opt-in; when off, the original additive BTSP rule
    # is used unchanged):
    #   (a) Milstein et al. (eLife 2021) bidirectional inverse-weight-dependence:
    #       plateaus potentiate weak synapses and depress strong ones — the
    #       update drives M toward sign(eligibility)·w_max gated by the current
    #       weight, making the fast-weight self-stabilising rather than runaway.
    #   (b) Jain et al. (Nature 2024) stochastic CaMKII consolidation: each
    #       plateau consolidates only with probability btsp_consolidate_prob
    #       (per-position Bernoulli during training; expectation at eval).
    #   (c) Delayed kernel: a gamma-shaped causal kernel (rises then decays,
    #       peak set by btsp_delay_shape) approximates the 10–100 s post-plateau
    #       CaMKII delay. btsp_delay_shape=0 → the original immediate kernel.
    # Requires use_btsp=True (and therefore use_hebbian_synapse=True).
    btsp_weight_dependent: bool = False
    btsp_w_max: float = 1.0              # Saturating bound for the fast-weight magnitude
    btsp_consolidate_prob: float = 0.3   # Per-plateau stochastic consolidation probability
    btsp_delay_shape: float = 0.0        # Gamma-kernel delay shape (0 = immediate)

    # ── R4: Sharp-Wave-Ripple content-tagged replay ─────────────────────────
    # Yang et al. (Science 2024): awake SPW-Rs tag a *subset* of experiences and
    # the same content is preferentially replayed in NREM. Augments the
    # PrioritizedReplayBuffer with a heuristic tagger — an episode is tagged when
    # its free-energy/surprise priority exceeds a running EMA threshold AND a
    # stochastic gate fires at the target tag rate. Sleep replay then samples
    # preferentially from tagged episodes (falls back to priority sampling when
    # nothing is tagged). Requires use_sleep_consolidation + use_prioritized_replay.
    use_swr_tagging: bool = False
    swr_tag_rate: float = 0.3                # Target fraction of episodes tagged
    swr_tag_threshold_decay: float = 0.99    # EMA decay for the running surprise threshold

    # ── R10: REM-style generative replay (adversarial dreaming) ─────────────
    # Deperrois et al. (Comm Biol 2022): a REM sub-phase generates novel latent
    # samples (the generative pathway) which the recognition pathway is trained
    # to handle adversarially. Implemented as a latent-space GAN with a
    # gradient-reversal layer so generator, discriminator, and the backbone's
    # real-latent encoder all train in a SINGLE allreduced backward (keeps DDP /
    # flat-allreduce replicas in sync — a per-rank internal GAN optimizer would
    # diverge). The REM loss augments the sleep-consolidation loss.
    # Requires use_sleep_consolidation=True.
    use_rem_dreaming: bool = False
    rem_noise_dim: int = 0           # Generator noise dim (0 → d_latent)
    rem_hidden_dim: int = 0          # Generator/discriminator hidden width (0 → max(64, d_latent//4))
    rem_loss_weight: float = 0.1     # Weight of the adversarial REM loss in the sleep loss
    rem_grl_lambda: float = 1.0      # Gradient-reversal strength for the generator

    @property
    def sync_dim(self) -> int:
        """Dimension of the flattened synchronization representation."""
        if self.sync_method == "full":
            return self.d_latent * self.d_latent
        elif self.sync_method == "diag_summary":
            # diag(S) + row_means(S) + col_means(S)
            return 3 * self.d_latent
        elif self.sync_method == "low_rank":
            return self.sync_rank * self.d_latent
        elif self.sync_method == "sparse_decay":
            return self.sync_sparse_pairs
        else:
            raise ValueError(f"Unknown sync_method: {self.sync_method}")

    @property
    def neurons_per_group(self) -> int:
        """Number of neurons sharing each NLM group's parameters."""
        assert self.d_latent % self.nlm_groups == 0, \
            f"d_latent ({self.d_latent}) must be divisible by nlm_groups ({self.nlm_groups})"
        return self.d_latent // self.nlm_groups

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device string."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"



    def resolve_thought_steps(self, step: int) -> int:
        """Return the T value for the current training step under curriculum.

        When `t_curriculum=False`, always returns `max_thought_steps`.
        When `t_curriculum=True`, walks `t_curriculum_stages` and picks the
        first stage whose `end_fraction_of_max_steps` exceeds the current
        progress fraction. Steps past the final stage's end fraction stay
        at the final stage's T.

        The T values must all be ≤ max_thought_steps because the per-tick
        adapter list is sized at construction time. Using a curriculum T
        smaller than max_thought_steps simply uses fewer adapters.
        """
        if not self.t_curriculum:
            return self.max_thought_steps

        if not self.t_curriculum_stages:
            return self.max_thought_steps

        progress = step / max(self.max_steps, 1)
        for T_value, end_frac in self.t_curriculum_stages:
            if T_value > self.max_thought_steps:
                raise ValueError(
                    f"Curriculum stage T={T_value} exceeds max_thought_steps="
                    f"{self.max_thought_steps}. The per-tick adapters are sized "
                    f"at construction; lower stage T values or raise max_thought_steps."
                )
            if progress < end_frac:
                return T_value

        # Past the final stage's end fraction → use its T
        return self.t_curriculum_stages[-1][0]