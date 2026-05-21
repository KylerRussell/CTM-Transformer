"""
CTM-Transformer Optional / Reusable Components

Standalone modules that aren't part of the model architecture proper but
are wired into training when their corresponding flag is set:

  1. cpu_offload_engine.py — CTMCPUOffloadEngine: stores ThoughtLayer
                             params on CPU pinned memory and streams
                             them to a single GPU template per layer call.
                             For models too large to fit on a single GPU.

These could just as well live next to model.py, but keeping them in their
own file makes the dependency story clearer: the model file is pure
architecture, the train file is pure orchestration, and this file is
where 'opt-in helpers' live.
"""

from __future__ import annotations

import torch
import copy
import logging
import torch.nn as nn



# ═════════════════════════════════════════════════════════════════════════
# cpu_offload_engine.py
# ═════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


# Fixed positional order of ThoughtLayer.forward's tensor-or-None inputs.
# The proxy normalizes its call into this order before handing off to the
# custom autograd Function so backward can reconstruct the call exactly.
#   (text_keys, text_values, prev_state, key_padding_mask,
#    stream_state, hebbian_state, hebbian_lr_modulator)
_N_LAYER_INPUTS = 7


class _StreamedLayerFn(torch.autograd.Function):
    """Per-layer autograd boundary for the CPU-offload engine.

    The whole reason the previous engine produced corrupt gradients is that
    all layers shared a single GPU `template` whose leaf parameters were
    overwritten in-place per call. Standard autograd then accumulated one
    tangled gradient onto that single shared parameter set.

    This Function makes each layer its own recompute-based autograd unit so
    grads are correct *and* parameter memory stays offloaded:

      forward:  stream layer L's weights CPU→GPU, run the template under
                no_grad (so no per-layer activation graph is retained), and
                save only the inputs + RNG state.
      backward: restore RNG, re-stream layer L's weights, recompute the
                forward *with* grad enabled, then `torch.autograd.grad` to
                get grads w.r.t. the (detached, re-leafed) inputs and the
                template parameters. Parameter grads are accumulated into a
                per-layer GPU buffer; input grads flow back into the model's
                normal autograd graph (attn-residual mix, FEEC, embed, head).

    Because each layer reloads its own weights inside its own backward, the
    shared template is never relied upon to hold L layers' gradients at once.

    Recompute side-effects (RNG, in-place buffer EMA, v1 TemporalMemory FIFO
    pushes) follow the same semantics as torch.utils.checkpoint, which this
    codebase already uses. RNG is saved/restored so dropout masks match
    between forward and recompute; the matrix-streams (v2) path is the
    intended configuration for offloaded training.
    """

    @staticmethod
    def forward(ctx, engine, layer_idx, *layer_inputs):
        ctx.engine = engine
        ctx.layer_idx = layer_idx

        # Record which positions carry a tensor (vs None) so backward can
        # rebuild the exact positional call.
        ctx.tensor_positions = [
            i for i, a in enumerate(layer_inputs) if torch.is_tensor(a)
        ]
        ctx.save_for_backward(*[layer_inputs[i] for i in ctx.tensor_positions])

        # Snapshot RNG for a faithful recompute (dropout, etc.).
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.had_cuda_rng = torch.cuda.is_available()
        if ctx.had_cuda_rng:
            ctx.cuda_device = engine.device
            ctx.cuda_rng_state = torch.cuda.get_rng_state(engine.device)

        # Stream weights and run the layer with grad disabled — no per-layer
        # activation graph is kept; it is reconstructed in backward.
        engine._stream_weights(layer_idx, requires_grad=False)
        with torch.no_grad():
            outputs = engine.gpu_template(*layer_inputs)

        # autograd.Function.forward must not return an input tensor as-is
        # (e.g. ThoughtLayer's passthrough `new_hebbian_state = hebbian_state`).
        # Clone any output that aliases an input.
        input_ids = {id(a) for a in layer_inputs if torch.is_tensor(a)}
        safe_outputs = tuple(
            o.clone() if (torch.is_tensor(o) and id(o) in input_ids) else o
            for o in outputs
        )
        return safe_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        engine = ctx.engine
        idx = ctx.layer_idx

        # ── Rebuild the positional input tuple from saved tensors ────────
        saved = list(ctx.saved_tensors)
        by_pos: dict[int, torch.Tensor] = {}
        for pos, t in zip(ctx.tensor_positions, saved):
            by_pos[pos] = t

        # `ctx.needs_input_grad` aligns to (engine, layer_idx, *inputs);
        # the i-th layer input is at index i + 2.
        diff_positions: list[int] = []
        recompute_args: list = []
        diff_inputs: list[torch.Tensor] = []
        for pos in range(_N_LAYER_INPUTS):
            t = by_pos.get(pos, None)
            if t is None:
                recompute_args.append(None)
                continue
            wants_grad = ctx.needs_input_grad[pos + 2] and t.is_floating_point()
            if wants_grad:
                leaf = t.detach().clone().requires_grad_(True)
                diff_positions.append(pos)
                diff_inputs.append(leaf)
                recompute_args.append(leaf)
            else:
                recompute_args.append(t.detach())

        # ── Restore RNG and re-stream this layer's weights (grad on) ─────
        torch.set_rng_state(ctx.cpu_rng_state)
        if ctx.had_cuda_rng:
            torch.cuda.set_rng_state(ctx.cuda_rng_state, ctx.cuda_device)
        engine._stream_weights(idx, requires_grad=True)
        template_params = list(engine.gpu_template.parameters())

        # ── Recompute forward with grad, then differentiate ──────────────
        with torch.enable_grad():
            outputs = engine.gpu_template(*recompute_args)

        sel_outputs: list[torch.Tensor] = []
        sel_grads: list[torch.Tensor] = []
        for o, g in zip(outputs, grad_outputs):
            if torch.is_tensor(o) and o.requires_grad and g is not None:
                sel_outputs.append(o)
                sel_grads.append(g)

        n_diff = len(diff_inputs)
        if not sel_outputs:
            # Nothing differentiable downstream — no grads to compute.
            return (None, None) + (None,) * _N_LAYER_INPUTS

        grads = torch.autograd.grad(
            sel_outputs,
            diff_inputs + template_params,
            grad_outputs=sel_grads,
            allow_unused=True,
            retain_graph=False,
        )
        input_grads = grads[:n_diff]
        param_grads = grads[n_diff:]

        # ── Accumulate parameter grads into this layer's GPU buffer ──────
        buf = engine._grad_gpu[idx]
        offset = 0
        for pg, numel in zip(param_grads, engine.layer_param_numels[idx]):
            if pg is not None:
                buf[offset:offset + numel].add_(pg.reshape(-1).float())
            offset += numel

        # ── Assemble grads aligned to forward's (engine, layer_idx, *inputs)
        out_grads: list = [None, None]
        pos_to_grad = {pos: g for pos, g in zip(diff_positions, input_grads)}
        for pos in range(_N_LAYER_INPUTS):
            out_grads.append(pos_to_grad.get(pos, None))
        return tuple(out_grads)


class _StreamedLayerProxy(nn.Module):
    """Proxy that runs one offloaded ThoughtLayer via `_StreamedLayerFn`.

    The proxy stands in for a real ThoughtLayer in the model's layer
    sequence. It normalizes the model's call (positional + keyword) into the
    fixed positional order the autograd Function expects, then delegates to
    the Function which handles streaming + correct gradients.
    """

    def __init__(self, engine, layer_idx):
        super().__init__()
        self.engine = engine
        self.layer_idx = layer_idx

    # Out-of-forward reads (initial-state construction, schema-novelty
    # checks) must hit this layer's *real* module, not the shared template.
    # The template's params are only correct for whichever layer was last
    # streamed in; per-layer init params (e.g. stream_init) live on the real
    # CPU layer, and their gradients flow through normal autograd from there.
    # The template's own forward still uses its own `self.stream`/`self.memory`.
    @property
    def _real_layer(self):
        return self.engine.layers[self.layer_idx]

    @property
    def memory(self):
        return self._real_layer.memory

    @property
    def stream(self):
        return self._real_layer.stream

    @property
    def hebbian(self):
        return self._real_layer.hebbian

    def reset_memory(self, *args, **kwargs):
        self._real_layer.reset_memory(*args, **kwargs)

    def forward(self, *args, **kwargs):
        engine = self.engine
        engine._last_layer_idx = self.layer_idx

        # Normalize to the fixed positional order of ThoughtLayer.forward.
        text_keys = args[0] if len(args) > 0 else kwargs.get("text_keys")
        text_values = args[1] if len(args) > 1 else kwargs.get("text_values")
        prev_state = args[2] if len(args) > 2 else kwargs.get("prev_state")
        key_padding_mask = args[3] if len(args) > 3 else kwargs.get("key_padding_mask")
        stream_state = kwargs.get("stream_state", args[4] if len(args) > 4 else None)
        hebbian_state = kwargs.get("hebbian_state", args[5] if len(args) > 5 else None)
        hebbian_lr_modulator = kwargs.get(
            "hebbian_lr_modulator", args[6] if len(args) > 6 else None
        )

        return _StreamedLayerFn.apply(
            engine,
            self.layer_idx,
            text_keys,
            text_values,
            prev_state,
            key_padding_mask,
            stream_state,
            hebbian_state,
            hebbian_lr_modulator,
        )


class CTMCPUOffloadEngine:
    """CPU-backed training engine for CTM-Transformer.

    Stores ThoughtLayer parameters on CPU pinned memory and streams them to
    a single GPU template for each layer call. Static modules (embedding,
    output head, FEEC, attn-residual, etc.) stay GPU-resident.

    The engine patches model._get_layers_sequence() to return proxy layers.
    Each proxy runs its layer through `_StreamedLayerFn`, a per-layer
    recompute-based autograd boundary that produces correct per-layer
    gradients while keeping only one layer's parameters GPU-resident.

    Because the engine owns recompute, it disables the model's per-thought-
    step gradient checkpointing while active (restored on shutdown) to avoid
    nested/duplicate recompute. Layer-internal activations are still freed
    (forward runs under no_grad), so the dominant per-layer activation cost
    is checkpointed; the inter-layer residual-stream activations scale with
    the thought-step count T, so prefer modest T for offloaded training.

    Usage:
        engine = CTMCPUOffloadEngine(model, device='cuda:0')
        engine.zero_grad()
        result = model(input_ids, targets=targets)
        result['loss'].backward()   # fills per-layer GPU grad buffers
        engine.collect_grads()      # GPU grad buffers → CPU param .grad
        optimizer.step()            # optimizer holds get_all_parameters()
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

        # Per-layer GPU gradient accumulators (fp32 for accumulation
        # accuracy, regardless of the bf16 weight stream). Indexed by
        # layer_idx, flat, same numel as the layer's pinned flat.
        self._grad_gpu = [
            torch.zeros(n, dtype=torch.float32, device=self.device)
            for n in self.layer_numels
        ]

        # The engine provides its own recompute; disable the model's
        # per-thought-step checkpointing to avoid nested recompute.
        self._orig_grad_ckpt = getattr(self.config, "gradient_checkpointing", False)
        self.config.gradient_checkpointing = False

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

    def _stream_weights(self, layer_idx, requires_grad):
        """Copy layer_idx's pinned CPU weights into the GPU template.

        One H2D copy of the flat buffer, then cheap D2D views into the
        template parameters. `requires_grad` is set so backward can
        differentiate w.r.t. the template params via autograd.grad.
        """
        flat = self.layer_pinned_flats[layer_idx]
        n = self.layer_numels[layer_idx]
        self.gpu_flat_buffer[:n].copy_(flat, non_blocking=False)

        template = self.gpu_template
        offset = 0
        for p, shape, numel in zip(
            template.parameters(),
            self.layer_param_shapes[layer_idx],
            self.layer_param_numels[layer_idx],
        ):
            p.data.copy_(self.gpu_flat_buffer[offset:offset + numel].view(shape))
            p.requires_grad_(requires_grad)
            if p.grad is not None:
                p.grad = None
            offset += numel

    def zero_grad(self):
        """Zero per-layer GPU grad buffers and all param .grad fields."""
        for buf in self._grad_gpu:
            buf.zero_()
        for p in self.get_all_parameters():
            p.grad = None

    def collect_grads(self):
        """Copy per-layer GPU grad buffers into CPU param .grad fields.

        Each layer's grads were accumulated independently in backward, so no
        cross-layer overwrite occurs. For Hyperloop (a shared middle layer
        repeated in the sequence), the same CPU param object appears under
        multiple layer indices; we accumulate so its total grad is the sum
        over occurrences.
        """
        if self.gpu_template is None:
            return

        for idx in range(self.n_layers):
            buf = self._grad_gpu[idx]
            offset = 0
            for cpu_p, numel, shape in zip(
                self.layer_cpu_params[idx],
                self.layer_param_numels[idx],
                self.layer_param_shapes[idx],
            ):
                g = buf[offset:offset + numel].view(shape).to(cpu_p.dtype).cpu()
                if cpu_p.grad is None:
                    cpu_p.grad = g.clone()
                else:
                    cpu_p.grad = cpu_p.grad + g
                offset += numel

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
        """Return all trainable params (CPU layers + GPU static), de-duped.

        De-duping by identity matters for Hyperloop, where a shared middle
        layer is repeated in the sequence and would otherwise be handed to
        the optimizer multiple times.
        """
        params = []
        seen = set()
        for layer in self.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        for p in self.model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def shutdown(self):
        """Restore original model state."""
        if hasattr(self, '_original_get_layers'):
            self.model._get_layers_sequence = self._original_get_layers
        if hasattr(self, '_orig_grad_ckpt'):
            self.config.gradient_checkpointing = self._orig_grad_ckpt
