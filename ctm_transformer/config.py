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