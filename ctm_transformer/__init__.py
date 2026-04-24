"""
CTM-Transformer: Continuous Thought Machine Transformer

A dynamic, stateful transformer architecture that processes text through
iterative internal "thought steps" driven by neural synchronization rather
than static feed-forward computation.

Integrates concepts from bio-plausible predictive coding:
- Neuron-Level Models (NLMs) replacing static activation functions
- FIFO memory buffers with temporal dynamics
- Synchronization matrices for inter-neuron coupling
- Cross-attention driven by internal state, not token position
"""

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer

__all__ = ["CTMConfig", "CTMTransformer"]
__version__ = "0.1.0"
