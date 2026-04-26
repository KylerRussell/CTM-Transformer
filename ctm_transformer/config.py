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
    max_seq_len: int = 1024            # Maximum input sequence length
    use_positional_encoding: bool = False  # Disabled by default — thought loop provides temporal structure
    dropout: float = 0.1              # Dropout rate for attention and projections

    # ── NLM Configuration ───────────────────────────────────────────────
    nlm_groups: int = 1                # Number of neuron groups (1 = true per-neuron MLPs)
                                       # Set to e.g. 32 for grouped NLMs (16 neurons per group)

    # ── Synchronization ─────────────────────────────────────────────────
    sync_method: str = "diag_summary"  # "full", "diag_summary", or "low_rank"
    sync_rank: int = 32                # Rank for low_rank sync method

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100_000
    warmup_steps: int = 1000
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"            # "float32", "float16", or "bfloat16"
    grad_clip: float = 1.0            # Gradient clipping max norm
    eval_interval: int = 500          # Steps between evaluations
    log_interval: int = 50            # Steps between logging

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer: str = "adamw"           # "adamw" or "adamuon"
    # AdamW betas (used as the AdamW group when optimizer="adamuon" too).
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
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
    aux_loss_weight: float = 0.1       # Weight for mean-across-ticks auxiliary loss
    min_loss_weight: float = 0.5       # Weight for min-loss tick
    max_cert_weight: float = 0.5       # Weight for max-certainty tick

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