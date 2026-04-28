"""
CTM-Transformer: Continuous Thought Machine Transformer (v2)

A dynamic, stateful transformer architecture that processes text through
iterative internal "thought steps" driven by neural synchronization rather
than static feed-forward computation.

Integrates concepts from bio-plausible predictive coding:
- Neuron-Level Models (NLMs) replacing static activation functions
- FIFO memory buffers with temporal dynamics
- Synchronization matrices for inter-neuron coupling
- Cross-attention driven by internal state, not token position
- Optional Engram conditional memory (hashed N-gram lookup) — offloads
  static factual recall to constant-time embedding retrieval, freeing the
  thought loop to focus on reasoning.

v2 additions:
- FEEC Integrator: structure-preserving dynamics for bounded gradients
- Matrix-Valued Residual Streams: replace NLM O(D²) sync with O(n·D) streams
- Dual-Space Sparse Attention: O(N) cross-attention via SSE + MoBA
- Hyperloop Looped Middle Cycle: parameter-efficient weight sharing
- Loop Position Embeddings: iteration-aware context injection
- Triton Tiled Attention: accelerated Q·K computation
- CUDA Graphs: kernel launch elimination for the thought loop
"""

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.engram import (
    EngramTable,
    EngramProjection,
    EngramGate,
)
from ctm_transformer.ternary import (
    TernaryLinear,
    ternarize,
    pack_ternary,
    unpack_ternary,
    replace_linears_with_ternary,
)
from ctm_transformer.feec_integrator import FEECIntegrator
from ctm_transformer.matrix_stream import MatrixResidualStream
from ctm_transformer.dssa import (
    SparseStateExpansion,
    MixtureOfBlockAttention,
    DualSpaceSparseAttention,
)

__all__ = [
    "CTMConfig",
    "CTMTransformer",
    "EngramTable",
    "EngramProjection",
    "EngramGate",
    "TernaryLinear",
    "ternarize",
    "pack_ternary",
    "unpack_ternary",
    "replace_linears_with_ternary",
    "FEECIntegrator",
    "MatrixResidualStream",
    "SparseStateExpansion",
    "MixtureOfBlockAttention",
    "DualSpaceSparseAttention",
]
__version__ = "2.0.0"