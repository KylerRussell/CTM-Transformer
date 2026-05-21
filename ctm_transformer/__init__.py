"""
CTM-Transformer: Continuous Thought Machine Transformer (v2)

A dynamic, stateful transformer architecture that processes text through
iterative internal "thought steps" driven by neural synchronization rather
than static feed-forward computation.

This package was consolidated from a per-component file layout into four
top-level modules:

    config.py   — CTMConfig (every hyperparameter and architectural switch)
    model.py    — every nn.Module that makes up the model architecture
    train.py    — training script (single-GPU, torchrun-DDP, and dual-GPU
                  mp.spawn modes); also bundles PhaseTimer and
                  CachedTeacherDataset since they're training-side tools
    extras.py   — opt-in helpers: CTMCPUOffloadEngine

Public re-exports below mirror the old per-file imports so existing
external code that did `from ctm_transformer import CTMTransformer` keeps
working unchanged. New code can also import from the consolidated
modules directly:

    from ctm_transformer.model import CTMTransformer, ThoughtLayer
    from ctm_transformer.config import CTMConfig
"""

from ctm_transformer.config import CTMConfig

from ctm_transformer.model import (
    # Top-level model
    CTMTransformer,
    FeatureEncoder,
    # Layers
    ThoughtLayer,
    UNetSynapse,
    # Memory + sync
    TemporalMemory,
    SynchronizationComputer,
    # Per-neuron MLPs
    NeuronLevelModels,
    # Matrix-valued residual streams
    MatrixResidualStream,
    # FEEC integrator
    FEECIntegrator,
    # Dual-Space Sparse Attention
    SparseStateExpansion,
    MixtureOfBlockAttention,
    DualSpaceSparseAttention,
    # Triton + CUDA-graph helpers
    accelerated_causal_attention,
    pytorch_causal_attention,
    triton_causal_attention,
    CUDAGraphThoughtLoop,
    compute_tiled_schedule,
)

from ctm_transformer.extras import (
    CTMCPUOffloadEngine,
)

from ctm_transformer.biological import (
    HebbianSynapse,
    CerebellarReadout,
)

from ctm_transformer.predictive_coding import (
    PCLayer,
    PCStateManager,
)

from ctm_transformer.validation import (
    compute_validation_metrics,
    compute_validation_metrics_t_sweep,
    compute_spectrum_metrics,
    format_validation_report,
    format_t_sweep_report,
    format_spectrum_report,
)


__all__ = [
    "CTMConfig",
    # Model
    "CTMTransformer",
    "FeatureEncoder",
    "ThoughtLayer",
    "UNetSynapse",
    "TemporalMemory",
    "SynchronizationComputer",
    "NeuronLevelModels",
    "MatrixResidualStream",
    "FEECIntegrator",
    "SparseStateExpansion",
    "MixtureOfBlockAttention",
    "DualSpaceSparseAttention",
    "accelerated_causal_attention",
    "pytorch_causal_attention",
    "triton_causal_attention",
    "CUDAGraphThoughtLoop",
    "compute_tiled_schedule",
    # Extras
    "CTMCPUOffloadEngine",
    # Biological
    "HebbianSynapse",
    "CerebellarReadout",
    # Predictive Coding
    "PCLayer",
    "PCStateManager",
    # Validation
    "compute_validation_metrics",
    "compute_validation_metrics_t_sweep",
    "compute_spectrum_metrics",
    "format_validation_report",
    "format_t_sweep_report",
    "format_spectrum_report",
]
__version__ = "2.0.0"