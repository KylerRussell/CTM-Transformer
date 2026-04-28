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

    # ── Synchronization ─────────────────────────────────────────────────
    sync_method: str = "diag_summary"  # "full", "diag_summary", or "low_rank"
    sync_rank: int = 32                # Rank for low_rank sync method

    # ── Ternary Weight Quantization (optional, paper-faithful TWN) ──────
    # Master switch. When False, all backbone Linears are standard nn.Linear.
    use_ternary: bool = False
    # Which subtrees to quantize. Only Linears under these module names are
    # eligible (after the never-quantize skip list of token_embedding /
    # output_proj / nlm). Empty = no whitelist; quantize everything eligible.
    # Recommended default: leave empty so all backbone projections are
    # quantized but the small / boundary-precision-critical layers are skipped.
    ternary_only_modules: list[str] = field(default_factory=list)


    # Master switch. When False, all Engram fields below are ignored and
    # the model is identical to the original CTM-Transformer.
    use_engram: bool = False
    # N-gram orders to track. Paper recommends [2, 3]; ablation in Fig 5
    # shows 4-grams hurt at fixed memory budget (capacity dilution).
    engram_ngram_orders: list[int] = field(default_factory=lambda: [2, 3])
    # K independent hash heads per order. Mitigates collision bias.
    engram_n_heads: int = 8
    # M, slots per (order, head) table. Prime preferred. Defaults give a
    # ~64K-slot table (16-bit indexing space), which keeps the total table
    # small enough to fit on a consumer GPU even with d_head=64. Scale up
    # via --engram_slots_per_table for larger memory budgets.
    engram_slots_per_table: int = 65521         # largest prime ≤ 2^16
    # Per-head embedding dim. Concatenated d_mem = len(orders)·n_heads·d_head.
    # Default: matches d_model/n_heads to keep the projection matrix square-ish.
    engram_d_head: int = 64
    # ID used to left-pad short suffixes at sequence start.
    engram_bos_id: int = 0
    # Layer indices where Engram fuses into the thought loop. Empty = auto:
    # for n_layers ≥ 4, picks {1, n_layers // 2} (paper-style early + mid),
    # else just {0} (single early injection).
    engram_layers: list[int] = field(default_factory=list)
    # Depthwise causal conv: kernel size and dilation (paper Eq. 5).
    # Set use_conv=False to skip it (slight loss per Fig 5 ablation, ~30% fewer engram params).
    engram_use_conv: bool = True
    engram_conv_kernel: int = 4
    engram_conv_dilation: int = 3               # = max N-gram order
    # Engram embedding LR multiplier. Paper uses 5× the backbone LR with no
    # weight decay — the lookup tables are sparsely-updated and benefit from
    # a more aggressive step on the rows that actually receive gradient.
    engram_lr_mult: float = 5.0
    engram_weight_decay: float = 0.0

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
    optimizer: str = "adamw"           # "adamw" or "adamuon"
    # AdamW betas (used as the AdamW group when optimizer="adamuon" too).
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    # 8-bit AdamW via bitsandbytes. Cuts AdamW state memory ~8× (fp32 m+v
    # → quantized 8-bit blocks) with no measurable accuracy loss on
    # transformer-shaped models. On a 200M model that's ~2.8 GB recovered.
    # Affects the AdamW path(s) in build_optimizers; AdaMuon stays in fp32.
    use_8bit_adam: bool = False
    # AdaMuon hyperparameters (paper defaults).
    adamuon_beta: float = 0.95         # shared β for first/second momentum
    adamuon_eps: float = 1e-8          # variance denominator floor
    adamuon_ns_steps: int = 5          # Newton-Schulz iterations
    adamuon_rms_target: float = 0.2    # target update RMS (matches Adam)
    # Per-paper, AdaMuon and its AdamW companion both use wd=0.1. Keeping
    # this as a separate knob since the user's existing AdamW default is
    # 0.01 and we don't want to silently change AdamW-only behavior.
    adamuon_weight_decay: float = 0.1

    # ── Temporal Loss ───────────────────────────────────────────────────
    # Linear ramp weights across thought ticks. ramp_start < ramp_end means
    # later ticks contribute more to the gradient — pressures the model to
    # prioritize getting later ticks right, breaking the symmetry that
    # otherwise lets "produce identical output at every tick" win.
    # Defaults give a 3x ramp (last tick weighted 3x more than first).
    tick_ramp_start: float = 0.5
    tick_ramp_end: float = 1.5
    # Monotonicity penalty: ReLU(loss[t+1] - loss[t]) — punishes ticks that
    # are *worse* than their predecessor. One-sided, so improvements are
    # free. 0.5 is a moderate setting; raise to 1.0+ to enforce strict
    # monotonic refinement, lower to 0.1 to merely discourage regression.
    mono_penalty_weight: float = 0.5
    # Mono-penalty decay schedule. The penalty is helpful early in training
    # (when the model could otherwise collapse to "all ticks identical")
    # but becomes counterproductive at steady state — it forces the model
    # to maintain strict no-regression on every batch, including noise-
    # induced single-tick wobbles, making the model conservative ("better
    # to pin all later ticks to step 1's output than risk a tiny
    # regression"). Decaying the penalty lets the model explore more
    # aggressive refinement once basic differentiation is established.
    #
    # Schedule: linear decay from `mono_penalty_weight` to
    # `mono_penalty_weight * mono_penalty_min_frac` over the first
    # `mono_penalty_decay_until_frac` of max_steps, then constant at the
    # floor. Set decay_until_frac=0 to disable decay (penalty stays at
    # the full weight forever).
    mono_penalty_min_frac: float = 0.1     # floor as fraction of base weight
    mono_penalty_decay_until_frac: float = 0.3   # decay over first 30% of max_steps

    # Legacy fields kept for backward compatibility with old checkpoints
    # that may reference them. Not used by the new temporal loss.
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

    # ── Device ──────────────────────────────────────────────────────────
    device: str = "auto"               # "auto", "cuda", "cpu"

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

    @property
    def engram_d_mem(self) -> int:
        """Total Engram lookup output dim: len(orders) × n_heads × d_head."""
        return len(self.engram_ngram_orders) * self.engram_n_heads * self.engram_d_head

    def resolve_engram_layers(self) -> list[int]:
        """Resolve `engram_layers`: explicit list or auto-pick {1, n_layers//2}.

        Auto rule: paper places Engram at layers [2, 15] of a 30-layer model
        — early + mid. We map this to {1, n_layers // 2} for any n_layers ≥ 4
        (the smallest depth where mid is meaningfully different from early).
        For shallower models we collapse to a single early layer.
        """
        if self.engram_layers:
            for l in self.engram_layers:
                if l < 0 or l >= self.n_layers:
                    raise ValueError(
                        f"engram_layers contains {l}, outside [0, n_layers={self.n_layers})"
                    )
            return sorted(set(self.engram_layers))
        if self.n_layers >= 4:
            return [1, self.n_layers // 2]
        return [0]

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