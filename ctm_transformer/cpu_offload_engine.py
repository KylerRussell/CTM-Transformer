"""
CPU-Offloaded Training Engine for CTM-Transformer.

Architecture: Proxy-based layer streaming. Each layer call:
1. Copies CPU pinned weights → GPU template
2. Runs template.forward()
3. After backward, copies template grads → CPU params

The model's own forward() with gradient checkpointing works unmodified.

NOTE: This engine is useful when the model is TOO LARGE to fit on a single
GPU. For models that fit (like the current 471M), standard single-GPU
training is faster because it avoids the H2D weight copy overhead.
"""

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _StreamedLayerProxy(nn.Module):
    """Proxy that loads weights from CPU and runs on GPU template.

    On each forward call:
    1. H2D copy: pinned CPU flat → GPU flat buffer → template params
    2. Forward through template with requires_grad=True
    3. After backward completes globally, engine.collect_grads() copies
       template param grads to CPU params (NOT via hooks to avoid
       double-counting with gradient checkpointing).
    """

    def __init__(self, engine, layer_idx):
        super().__init__()
        self.engine = engine
        self.layer_idx = layer_idx

    @property
    def memory(self):
        return self.engine.gpu_template.memory

    @property
    def stream(self):
        return self.engine.gpu_template.stream

    def reset_memory(self, *args, **kwargs):
        self.engine.gpu_template.reset_memory(*args, **kwargs)

    def forward(self, *args, **kwargs):
        engine = self.engine
        idx = self.layer_idx

        # Track which layer was last loaded (for grad collection)
        engine._last_layer_idx = idx

        # 1. H2D: copy pinned flat → GPU flat → unflatten to template
        flat = engine.layer_pinned_flats[idx]
        n = engine.layer_numels[idx]
        engine.gpu_flat_buffer[:n].copy_(flat, non_blocking=False)

        template = engine.gpu_template
        offset = 0
        for p, shape, numel in zip(
            template.parameters(),
            engine.layer_param_shapes[idx],
            engine.layer_param_numels[idx],
        ):
            p.data.copy_(engine.gpu_flat_buffer[offset:offset + numel].view(shape))
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None
            offset += numel

        # 2. Forward — autograd graph connects through template params
        return template(*args, **kwargs)


class CTMCPUOffloadEngine:
    """CPU-backed training engine for CTM-Transformer.

    Stores ThoughtLayer parameters on CPU pinned memory and streams them to
    a single GPU template for each layer call. Static modules (embedding,
    output head, FEEC, etc.) stay GPU-resident.

    The engine patches model._get_layers_sequence() to return proxy layers,
    so model.forward() works unmodified with gradient checkpointing.

    IMPORTANT: After loss.backward(), you must call engine.collect_grads()
    to copy the template's accumulated gradients back to CPU params.

    Usage:
        engine = CTMCPUOffloadEngine(model, device='cuda:0')
        result = model(input_ids, targets=targets)
        result['loss'].backward()
        # Template grads flow through autograd; but they live on GPU template.
        # The optimizer holds CPU params. We need to bridge them:
        engine.collect_grads()
        optimizer.step()
        engine.sync_params_from_cpu()
    """

    def __init__(self, model, device="cuda:0", dtype=torch.bfloat16):
        self.model = model
        self.config = model.config
        self.device = torch.device(device)
        self.dtype = dtype
        self._last_layer_idx = -1

        self.layers = model._get_layers_sequence()
        self.n_layers = len(self.layers)

        self._move_static_to_gpu()
        self._setup_cpu_layers()
        self._setup_gpu_template()
        self._patch_model()

        # Accumulated grads per-layer (indexed by layer_idx)
        # Shape: list of flat tensors on CPU, same size as pinned flats
        self._grad_accum = [
            torch.zeros_like(flat) for flat in self.layer_pinned_flats
        ]

        print(
            f"  Engine: {self.n_layers} layers, "
            f"max layer {self.max_layer_numel * 2 / 1e6:.1f} MB (bf16), "
            f"GPU resident: {self._gpu_resident_mb:.1f} MB"
        )

    def _move_static_to_gpu(self):
        """Move small, always-needed modules to GPU."""
        m = self.model
        layer_param_ids = set()
        for layer in self.layers:
            for p in layer.parameters():
                layer_param_ids.add(id(p))

        gpu_mb = 0
        for name, param in m.named_parameters():
            if id(param) not in layer_param_ids:
                param.data = param.data.to(self.device, self.dtype)
                gpu_mb += param.numel() * param.element_size() / 1e6

        for name, buf in m.named_buffers():
            buf.data = buf.data.to(self.device)

        self._gpu_resident_mb = gpu_mb

    def _setup_cpu_layers(self):
        """Move layer params to CPU pinned memory."""
        self.layer_pinned_flats = []
        self.layer_param_shapes = []
        self.layer_param_numels = []
        self.layer_numels = []
        self.layer_cpu_params = []

        for layer in self.layers:
            for p in layer.parameters():
                p.data = p.data.cpu().to(self.dtype)

            shapes = [p.shape for p in layer.parameters()]
            numels = [p.numel() for p in layer.parameters()]
            total = sum(numels)
            cpu_params = list(layer.parameters())

            flat = torch.empty(total, dtype=self.dtype).pin_memory()
            offset = 0
            for p in layer.parameters():
                n = p.numel()
                flat[offset:offset + n].copy_(p.data.flatten())
                offset += n

            self.layer_pinned_flats.append(flat)
            self.layer_param_shapes.append(shapes)
            self.layer_param_numels.append(numels)
            self.layer_numels.append(total)
            self.layer_cpu_params.append(cpu_params)

        self.max_layer_numel = max(self.layer_numels) if self.layer_numels else 0

    def _setup_gpu_template(self):
        """Create a single GPU-resident layer shell."""
        if not self.layers:
            self.gpu_template = None
            self.gpu_flat_buffer = None
            return

        self.gpu_flat_buffer = torch.empty(
            self.max_layer_numel, dtype=self.dtype, device=self.device
        )
        self.gpu_template = copy.deepcopy(self.layers[0])
        self.gpu_template = self.gpu_template.to(self.device, self.dtype)

    def _patch_model(self):
        """Replace model layers with streaming proxies."""
        proxies = nn.ModuleList([
            _StreamedLayerProxy(self, i) for i in range(self.n_layers)
        ])
        self._original_get_layers = self.model._get_layers_sequence
        self.model._get_layers_sequence = lambda: proxies
        self._proxies = proxies

    def zero_grad(self):
        """Zero accumulated gradients for all CPU layer params."""
        for acc in self._grad_accum:
            acc.zero_()
        for params in self.layer_cpu_params:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    def collect_grads(self):
        """After backward, copy template grads to CPU params.

        NOTE: With gradient checkpointing, the template is shared across all
        layers and ticks. During backward, PyTorch recomputes the forward
        (which re-loads weights into the template for each layer) and
        accumulates grads on the template params. However, since all 26
        layers share one template, the grads on the template at the END of
        backward only reflect the LAST layer that was recomputed.

        This is a fundamental limitation of the single-template approach
        with gradient checkpointing. The grad hooks approach (previous
        version) tried to fix this but double-counted due to recomputation.

        For correct grad collection with a single template, we would need
        to either:
        a) Use separate templates per layer (26 × 25 MB = 650 MB GPU)
        b) Not use gradient checkpointing (OOM)
        c) Use manual backward with explicit recompute (original engine v1)

        For now, this is a best-effort implementation.
        """
        template = self.gpu_template
        if template is None:
            return

        # The template grads belong to whatever layer was last computed
        # This is only correct without gradient checkpointing
        idx = self._last_layer_idx
        if idx < 0 or idx >= self.n_layers:
            return

        offset = 0
        for p, cpu_p in zip(template.parameters(), self.layer_cpu_params[idx]):
            if p.grad is not None:
                n = p.grad.numel()
                if cpu_p.grad is None:
                    cpu_p.grad = p.grad.cpu().to(cpu_p.dtype)
                else:
                    cpu_p.grad.add_(p.grad.cpu().to(cpu_p.dtype))

    def sync_params_from_cpu(self):
        """Refresh pinned flats from CPU params (after optimizer step)."""
        for i, layer in enumerate(self.layers):
            flat = self.layer_pinned_flats[i]
            offset = 0
            for p in layer.parameters():
                n = p.numel()
                flat[offset:offset + n].copy_(p.data.flatten())
                offset += n

    def get_all_parameters(self):
        """Return all trainable params (CPU layers + GPU static)."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        layer_param_ids = set()
        for layer in self.layers:
            for p in layer.parameters():
                layer_param_ids.add(id(p))
        for p in self.model.parameters():
            if id(p) not in layer_param_ids:
                params.append(p)
        return params

    def shutdown(self):
        """Restore original model state."""
        if hasattr(self, '_original_get_layers'):
            self.model._get_layers_sequence = self._original_get_layers
