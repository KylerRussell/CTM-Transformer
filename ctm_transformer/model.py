"""
CTM-Transformer — All Architectural Components

This file holds the entire model architecture for the CTM-Transformer (v2),
consolidated from what used to be one-class-per-file. The original module
structure is preserved as section banners below.

Section order respects dependencies — leaves first, composite modules later:
  1. ternary.py            — TernaryLinear (TWN drop-in for nn.Linear)
  2. triton_kernels.py     — Triton/CUDA-graph attention helpers
  3. engram.py             — Hashed N-gram conditional memory
  4. dssa.py               — Dual-Space Sparse Attention
  5. memory.py             — FIFO history + synchronization
  6. nlm.py                — Per-neuron MLPs
  7. matrix_stream.py      — Hyperloop-style residual streams (NLM/sync replacement)
  8. feec_integrator.py    — Symplectic-like integrator for the thought loop
  9. thought_layer.py      — One iteration block of the thought loop
 10. model.py              — CTMTransformer assembly + forward + generate
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional
import torch.utils.checkpoint as torch_checkpoint
from contextlib import nullcontext

from ctm_transformer.biological import HebbianSynapse, CerebellarReadout


# ═════════════════════════════════════════════════════════════════════════
# ternary.py
# ═════════════════════════════════════════════════════════════════════════

class _TernaryWeightSTE(torch.autograd.Function):
    """
    Forward:  emit (α · W̃) where W̃ ∈ {-1, 0, +1} per Eq. 3 of TWN paper.
    Backward: pass dL/d(αW̃) straight through to dL/dW (identity STE).

    The straight-through estimator is the standard trick for training through
    non-differentiable quantizations: pretend the quantizer is the identity
    during backward. The gradient that arrives at W is the gradient that
    *would have* gone to the ternary output, which is a reasonable approximation
    because for entries far from ±Δ the local sensitivity of W̃ to W is zero,
    while for entries near ±Δ a tiny perturbation of W flips W̃ — both regimes
    are handled adequately by passing the upstream gradient unchanged.

    Note: we deliberately do NOT clip the gradient (unlike BitNet's Clip(·)
    + STE combo). TWN doesn't require it — the +1/0/-1 range plus per-tensor
    scaling α is well-bounded enough that the latent weight's natural
    distribution stays in a reasonable range.
    """

    @staticmethod
    def forward(ctx, W: torch.Tensor) -> torch.Tensor:
        # Δ = 0.75 · E[|W|]   (TWN paper, Sec 2.3, derived assuming W ~ N(0, σ²))
        # We use mean(|W|) over the entire tensor — per-tensor quantization,
        # not per-row, because per-row would require K extra reductions per
        # forward and barely changes accuracy at the model sizes we care about.
        abs_W = W.abs()
        delta = 0.75 * abs_W.mean()

        # Ternarize: where |W| > Δ → sign(W); else 0
        # Done as (sign · mask) rather than torch.where for numerical efficiency
        # — sign is fused into the masking on most backends.
        mask = abs_W > delta                      # bool, [out, in]
        ternary = torch.sign(W) * mask            # ∈ {-1, 0, +1}, same dtype as W

        # α = mean of |W[i,j]| over the active set { |W[i,j]| > Δ }
        # If the entire weight is below Δ (degenerate — only happens with
        # an all-zero weight at init, or after weight decay annihilates it),
        # fall back to α=0 so the layer cleanly outputs zero.
        n_active = mask.sum()
        # Avoid creating a 0-dim tensor branch — always do the masked sum but
        # guard the divide. Keeps the autograd graph clean.
        active_sum = (abs_W * mask).sum()
        alpha = torch.where(
            n_active > 0,
            active_sum / n_active.clamp(min=1).to(W.dtype),
            torch.zeros_like(active_sum),
        )

        # Cache for backward — but we don't actually use it (STE is identity)
        # so we don't save anything. Keeps ctx light.
        return alpha * ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Identity STE: dL/dW = dL/d(αW̃)
        # We don't even need a clamp here — TWN's α scaling keeps the
        # effective output magnitude bounded, so the upstream gradient
        # is already well-scaled.
        return grad_output


def ternarize(W: torch.Tensor) -> torch.Tensor:
    """Ternarize W (with STE backward). Returns α · W̃, same shape as W."""
    return _TernaryWeightSTE.apply(W)


# ──────────────────────────────────────────────────────────────────────────
# 2-bit packing for inference-time weight storage
# ──────────────────────────────────────────────────────────────────────────
#
# Each ternary value uses 2 bits:
#   00 → 0
#   01 → +1
#   11 → -1   (sign-extended; or use 10 — the choice doesn't matter as long
#              as encode/decode agree)
# We pack 4 trits per uint8 byte. For a 768×3072 weight that's:
#   fp16:  768 × 3072 × 2 bytes  =  4.72 MiB
#   2-bit: 768 × 3072 × 0.25 + α  =   0.59 MiB + 4 bytes
# An ~8× reduction. The α scalar is stored alongside in fp32 (4 bytes).

@torch.no_grad()
def pack_ternary(ternary_signed: torch.Tensor) -> torch.Tensor:
    """Pack a ±1/0 tensor into uint8 with 4 trits per byte.

    Args:
        ternary_signed: any shape, values in {-1, 0, +1}, any dtype.

    Returns:
        uint8 tensor of shape ternary_signed.shape with the LAST dim padded
        and divided by 4. We embed the original last-dim size in the first
        4 elements of the buffer so unpack can recover the shape exactly.
    """
    flat = ternary_signed.flatten().to(torch.int8)
    # Encode {-1, 0, +1} → {2, 0, 1}  (uses 2 bits cleanly, decode with sign extension)
    codes = torch.zeros_like(flat, dtype=torch.uint8)
    codes[flat == 1] = 1
    codes[flat == -1] = 2

    # Pad to multiple of 4
    pad = (-codes.numel()) % 4
    if pad:
        codes = F.pad(codes, (0, pad), value=0)

    # Reshape to [N/4, 4] and combine 4 trits per byte
    codes = codes.view(-1, 4)
    packed = (
        codes[:, 0]
        | (codes[:, 1] << 2)
        | (codes[:, 2] << 4)
        | (codes[:, 3] << 6)
    )
    return packed.contiguous()


@torch.no_grad()
def unpack_ternary(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Inverse of pack_ternary — returns a fp32 ±1/0 tensor of the given shape."""
    n = int(torch.tensor(shape).prod().item())
    # Extract 4 trits per byte
    c0 = (packed       ) & 0b11
    c1 = (packed >> 2) & 0b11
    c2 = (packed >> 4) & 0b11
    c3 = (packed >> 6) & 0b11
    codes = torch.stack([c0, c1, c2, c3], dim=-1).flatten()[:n]
    # Decode: 0→0, 1→+1, 2→-1
    out = torch.zeros(n, dtype=torch.float32, device=packed.device)
    out[codes == 1] = 1.0
    out[codes == 2] = -1.0
    return out.view(shape)


# ──────────────────────────────────────────────────────────────────────────
# TernaryLinear: drop-in nn.Linear with TWN-style training
# ──────────────────────────────────────────────────────────────────────────

class TernaryLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weights during forward.

    Two operating modes, controlled by `freeze_to_ternary()`:
      * Training (default): forward path applies the on-the-fly ternarization
        + STE on every call. The latent weight is updated normally by the
        optimizer. Bit-exactly faithful to nn.Linear in the limit of identity
        STE on continuous weights.
      * Frozen / inference: ternarization done once, packed as a 2-bit
        buffer, latent weight optionally dropped. Forward unpacks on the fly.
        Use `freeze_to_ternary(pack=True)` before saving for deployment.

    Args mirror nn.Linear except that bias defaults to True for parity, and
    the weight init follows Kaiming uniform (same default as nn.Linear).

    Note on dtype: the latent weight stays in float32 regardless of the
    surrounding model's autocast / dtype. This is necessary for stable
    training — see module docstring. The ternarized output is cast to the
    input's dtype before the matmul so autograd / autocast play well.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Latent fp32 weight — *always* fp32, regardless of `dtype`
        # argument. The `dtype` arg controls bias precision and forward
        # cast target only.
        self._fwd_dtype = dtype  # may be None → uses input dtype
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        if bias:
            bias_kwargs = {"device": device, "dtype": dtype if dtype is not None else torch.float32}
            self.bias = nn.Parameter(torch.empty(out_features, **bias_kwargs))
        else:
            self.register_parameter("bias", None)

        # Frozen-mode buffers (populated by freeze_to_ternary())
        # Two storage formats: unpacked ±1/0 fp32 ('frozen_weight') or
        # 2-bit packed uint8 ('frozen_packed'). Only one is populated.
        self.register_buffer("frozen_weight", None, persistent=False)
        self.register_buffer("frozen_packed", None, persistent=False)
        self.register_buffer("frozen_alpha", None, persistent=False)
        self._frozen = False
        self._frozen_shape: torch.Size | None = None

        self.reset_parameters()

    def reset_parameters(self):
        # Same init as nn.Linear, applied to the latent fp32 weight.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._frozen:
            return self._frozen_forward(x)

        # Training / live forward: ternarize the fp32 latent weight on the fly.
        # The cast to x.dtype happens AFTER ternarization so that the matmul
        # itself runs in the model's compute dtype (bf16 typically).
        W_tern = ternarize(self.weight)              # fp32, same shape as self.weight
        W_tern = W_tern.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W_tern, bias)

    def _frozen_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference-mode forward: weight is already ternarized. We just need
        # to scale by α and matmul. If packed, unpack first.
        if self.frozen_packed is not None:
            ternary = unpack_ternary(self.frozen_packed, self._frozen_shape)
            ternary = ternary.to(device=x.device, dtype=x.dtype)
        else:
            ternary = self.frozen_weight.to(x.dtype)

        alpha = self.frozen_alpha.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        # Equivalent to F.linear(x, alpha * ternary, bias) but does the scalar
        # multiply post-matmul to save one full materialization of αW̃.
        out = F.linear(x, ternary, None) * alpha
        if bias is not None:
            out = out + bias
        return out

    # ── Freeze / unfreeze for inference ─────────────────────────────────

    @torch.no_grad()
    def freeze_to_ternary(self, pack: bool = True, drop_latent: bool = False):
        """Bake the current latent weight into ternary form for inference.

        After this call, forward uses the cached ternary buffer instead of
        re-running the ternarization op every time. Subsequent `loss.backward()`
        on this layer will fail — the layer is no longer trainable.

        Args:
            pack: if True, store the ternary trits as 2-bit packed uint8
                (~8× smaller than fp16 unpacked). If False, store as fp32
                ±1/0 — useful for debugging or if the inference kernel
                wants direct access to the unpacked form.
            drop_latent: if True, also free the fp32 latent weight. Saves
                memory but makes the layer un-fine-tunable. Default False
                so you can `unfreeze()` and resume training if needed.
        """
        # Recompute the ternarization (without STE — pure forward math).
        abs_W = self.weight.abs()
        delta = 0.75 * abs_W.mean()
        mask = abs_W > delta
        ternary = (torch.sign(self.weight) * mask).to(torch.float32)
        n_active = mask.sum().clamp(min=1)
        alpha = (abs_W * mask).sum() / n_active.to(self.weight.dtype)

        self._frozen_shape = ternary.shape
        self.frozen_alpha = alpha.detach().clone().to(torch.float32)

        if pack:
            self.frozen_packed = pack_ternary(ternary)
            self.frozen_weight = None
        else:
            self.frozen_weight = ternary
            self.frozen_packed = None

        if drop_latent:
            # Replace the parameter with a 0-element placeholder so state_dict
            # is still consistent but memory is reclaimed.
            self.weight = nn.Parameter(
                torch.empty(0, dtype=torch.float32, device=self.weight.device),
                requires_grad=False,
            )

        self._frozen = True

    @torch.no_grad()
    def unfreeze(self):
        """Resume training mode. Requires the latent weight to still be present
        (i.e. you didn't pass drop_latent=True). The frozen buffers are cleared."""
        if self.weight.numel() == 0:
            raise RuntimeError(
                "Cannot unfreeze: latent weight was dropped via drop_latent=True. "
                "Reload from a non-dropped checkpoint to resume training."
            )
        self.frozen_weight = None
        self.frozen_packed = None
        self.frozen_alpha = None
        self._frozen_shape = None
        self._frozen = False

    # ── Diagnostics ─────────────────────────────────────────────────────

    @torch.no_grad()
    def ternary_stats(self) -> dict:
        """Compute density / sparsity statistics of the current ternarization.

        Useful for monitoring whether the network is healthily using its
        ternary capacity. Pathological extremes:
          * density → 0: latent weights collapsing toward zero (over-decay)
          * density → 1: Δ threshold too small relative to weight scale
                         (loss of TWN's compression benefit)
        Healthy range is roughly density ∈ [0.4, 0.7] per the TWN paper.
        """
        abs_W = self.weight.abs()
        delta = 0.75 * abs_W.mean()
        mask = abs_W > delta
        n_active = mask.sum().item()
        n_total = self.weight.numel()
        alpha = (abs_W * mask).sum() / max(n_active, 1)
        return {
            "delta": delta.item(),
            "alpha": alpha.item(),
            "density": n_active / n_total,
            "n_pos": (self.weight > delta).sum().item(),
            "n_neg": (self.weight < -delta).sum().item(),
            "n_zero": n_total - n_active,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, frozen={self._frozen}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Helper: walk a module tree and swap eligible nn.Linears with TernaryLinear
# ──────────────────────────────────────────────────────────────────────────

# Modules whose Linears we NEVER quantize, regardless of placement.
# These either: (a) are too small for it to matter, (b) would lose critical
# expressiveness from per-output quantization (NLM stacks have per-neuron
# dynamics), or (c) are at the model boundaries where precision matters most
# (token embedding, LM head).
_NEVER_QUANTIZE_MODULE_NAMES = (
    "token_embedding",
    "output_proj",     # LM head (legacy single-head mode)
    "lm_head",         # LM head (per-tick mode — shared across ticks)
    "tick_adapters",   # Per-tick output adapters (boundary-precision-critical)
    "nlm",             # NeuronLevelModels — per-neuron MLPs
)


def replace_linears_with_ternary(
    module: nn.Module,
    skip_module_names: tuple[str, ...] = _NEVER_QUANTIZE_MODULE_NAMES,
    only_module_names: tuple[str, ...] | None = None,
    verbose: bool = False,
) -> int:
    """Recursively swap nn.Linear children for TernaryLinear, in-place.

    Routing:
      * `skip_module_names`: any path component matching these is left alone.
        E.g. "token_embedding" prevents quantizing anything inside the token
        embedding subtree (defensive — there usually aren't Linears there
        anyway, but the LM head DOES live under "output_proj").
      * `only_module_names`: if given, ONLY paths containing one of these
        components are quantized. None means "quantize everything not skipped".

    Args:
        module: root model. Modified in place.
        skip_module_names: subtrees to skip.
        only_module_names: optional whitelist of subtree names.
        verbose: print each swap if True.

    Returns:
        Number of layers swapped.
    """
    n_swapped = 0

    def _walk(parent: nn.Module, path: str):
        nonlocal n_swapped
        for child_name, child in list(parent.named_children()):
            full_path = f"{path}.{child_name}" if path else child_name
            path_parts = full_path.split(".")

            # Skip-list takes precedence
            if any(part in skip_module_names for part in path_parts):
                continue

            # Whitelist filter (if provided)
            if only_module_names is not None:
                if not any(part in only_module_names for part in path_parts):
                    # Recurse but don't quantize at this node
                    _walk(child, full_path)
                    continue

            if isinstance(child, nn.Linear):
                # Build the replacement and copy weights over so we keep
                # whatever the model's __init__ already initialized.
                new = TernaryLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                with torch.no_grad():
                    new.weight.copy_(child.weight.to(torch.float32))
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(parent, child_name, new)
                n_swapped += 1
                if verbose:
                    print(f"  [ternary] {full_path}  ({child.in_features}→{child.out_features})")
            else:
                _walk(child, full_path)

    _walk(module, "")
    return n_swapped

# ═════════════════════════════════════════════════════════════════════════
# triton_kernels.py
# ═════════════════════════════════════════════════════════════════════════

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────
# Triton Kernel: Fused Flash-style Causal Attention
# ──────────────────────────────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def _fused_attention_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Fused Q·K^T with causal masking, online softmax, and multiplication by V.
        
        Grid: (batch * n_heads, ceil(S / BLOCK_S))
        """
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)

        # Query rows this tile handles
        q_start = pid_s * BLOCK_S
        q_offsets = q_start + tl.arange(0, BLOCK_S)
        q_mask = q_offsets < S
        safe_q = tl.where(q_mask, q_offsets, 0)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        safe_d = tl.where(d_mask, d_offsets, 0)

        # Accumulators in float32 for precision
        out_acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)
        m_i = tl.full([BLOCK_S], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_S], dtype=tl.float32)

        # Load Q tile: [BLOCK_S, BLOCK_D]
        q_ptrs = Q_ptr + pid_bh * stride_qh + safe_q[:, None] * stride_qs + safe_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs, mask=(q_mask[:, None] & d_mask[None, :]), other=0.0)

        # Loop over key/value tiles
        for k_start in range(0, S, BLOCK_S):
            k_offsets = k_start + tl.arange(0, BLOCK_S)
            k_mask = k_offsets < S
            safe_k = tl.where(k_mask, k_offsets, 0)

            # Load K tile: [BLOCK_S, BLOCK_D]
            k_ptrs = K_ptr + pid_bh * stride_kh + safe_k[:, None] * stride_ks + safe_d[None, :] * stride_kd
            k_tile = tl.load(k_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)

            # Q·K^T: [BLOCK_S, BLOCK_S]
            qk = tl.dot(q_tile, tl.trans(k_tile)) * scale

            # Causal mask: position q can only attend to position k <= q
            causal = q_offsets[:, None] >= k_offsets[None, :]
            qk = tl.where(causal & q_mask[:, None] & k_mask[None, :],
                          qk, float('-inf'))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            
            # alpha = exp(m_i - m_new), zero if m_i is -inf
            alpha = tl.where(m_i == float('-inf'), 0.0, tl.exp(m_i - m_new))
            
            # p = exp(qk - m_new), zero if m_new is -inf
            p = tl.where(m_new[:, None] == float('-inf'), 0.0, tl.exp(qk - m_new[:, None]))
            
            # Load V tile: [BLOCK_S, BLOCK_D]
            v_ptrs = V_ptr + pid_bh * stride_vh + safe_k[:, None] * stride_vs + safe_d[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0)

            # Update output accumulator: Out = Out * alpha + Softmax(QK) * V
            out_acc = out_acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        # Final normalization
        out_acc = out_acc / l_i[:, None]

        # Store output: [BLOCK_S, BLOCK_D]
        out_ptrs = Out_ptr + pid_bh * stride_oh + safe_q[:, None] * stride_os + safe_d[None, :] * stride_od
        tl.store(out_ptrs, out_acc.to(Out_ptr.dtype.element_ty), mask=(q_mask[:, None] & d_mask[None, :]))


def triton_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Triton-accelerated fused causal attention.

    Args:
        q: [B*H, S, D] queries
        k: [B*H, S, D] keys
        v: [B*H, S, D] values
        scale: 1/sqrt(d_head)

    Returns:
        output: [B*H, S, D]
    """
    BH, S, D = q.shape

    # Power of 2 larger than or equal to D
    BLOCK_D = 1 << (D - 1).bit_length()
    
    # Adjust BLOCK_S to fit in shared memory (standard 96KB-100KB limit)
    # Fused kernel needs 4 tiles (Q, K, V, Acc). 
    # With D=220 (BLOCK_D=256), each float32 tile is 64KB if BLOCK_S=64.
    # 4 * 16 * 256 * 4 bytes = 16KB.
    # 4 * 32 * 256 * 4 bytes = 32KB.
    # On most GPUs, 32 would work, but 16 is safer and avoids fragmentation.
    if BLOCK_D >= 256:
        BLOCK_S = 16
    elif BLOCK_D >= 128:
        BLOCK_S = 32
    else:
        BLOCK_S = 64

    # Output buffer
    output = torch.empty_like(q)

    grid = (BH, (S + BLOCK_S - 1) // BLOCK_S)

    _fused_attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(0),
        output.stride(1), output.stride(2),
        S=S, D=D,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        scale=scale,
    )

    return output


# ──────────────────────────────────────────────────────────────────────────
# CUDA Graph Wrapper for the Thought Loop
# ──────────────────────────────────────────────────────────────────────────

class CUDAGraphThoughtLoop:
    """Wraps the thought loop in a CUDA Graph for kernel launch elimination.

    The thought loop consists of T sequential steps, each launching
    multiple small CUDA kernels (attention, synapse, NLM/stream, norms).
    At T=8 with 8 layers, that's ~200+ kernel launches per forward pass.
    CUDA Graphs capture the entire sequence of launches into a single
    graph and replay it in one GPU operation.

    Lifecycle:
      1. warmup() — runs one forward with recording (initializes graph)
      2. replay() — efficient replay on subsequent calls (same shapes)
      3. invalidate() — called when shapes change (new B, S, T)

    Important constraints:
      - All tensor shapes must be identical between capture and replay.
      - No CPU-side branching during captured execution.
      - No dynamic memory allocation during replay.

    Args:
        enabled: Whether to use CUDA Graphs (disabled on CPU or AMD).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: dict = {}
        self._static_outputs: dict = {}
        self._captured_shapes: Optional[tuple] = None

    def _should_recapture(self, B: int, S: int, T: int) -> bool:
        """Check if we need to re-capture the graph (shape change)."""
        shapes = (B, S, T)
        if self._captured_shapes != shapes:
            return True
        return False

    @contextmanager
    def capture_context(self, B: int, S: int, T: int):
        """Context manager for CUDA graph capture.

        Usage:
            with cuda_graph.capture_context(B, S, T):
                # Run the thought loop — this is being captured
                result = model._thought_loop(...)

        Yields control to the caller, who should run exactly one
        forward pass inside the context.
        """
        if not self.enabled:
            yield
            return

        # Warmup (non-captured run to allocate all tensors)
        torch.cuda.synchronize()

        if self._graph is not None:
            self._graph = None

        self._graph = torch.cuda.CUDAGraph()
        self._captured_shapes = (B, S, T)

        # Capture
        with torch.cuda.graph(self._graph):
            yield

        torch.cuda.synchronize()

    def replay(self):
        """Replay the captured graph."""
        if self._graph is not None:
            self._graph.replay()
        else:
            raise RuntimeError(
                "No CUDA Graph captured. Call capture_context() first."
            )

    def invalidate(self):
        """Invalidate the captured graph (e.g., on shape change)."""
        self._graph = None
        self._captured_shapes = None
        self._static_inputs.clear()
        self._static_outputs.clear()

    @property
    def is_captured(self) -> bool:
        return self._graph is not None


# ──────────────────────────────────────────────────────────────────────────
# N·log(N) Tiled Schedule for Thought Step Reuse
# ──────────────────────────────────────────────────────────────────────────

def compute_tiled_schedule(T: int, n_layers: int, tile_size: int = 4) -> list[list[tuple[int, int]]]:
    """Compute a tiled execution schedule for the thought loop.

    Instead of processing thought steps 1, 2, ..., T sequentially where
    each step independently reads its history from HBM, this schedule
    groups steps into tiles so that intermediate results stay in SRAM
    across multiple iterations.

    The schedule produces groups of (step, layer) pairs that can share
    temporary data. Within each group, steps are processed sequentially
    but their intermediate results are kept in fast memory.

    Reference: Recurrent Transformer IO-aware tiling algorithm.

    Algorithm:
      - Divide T steps into tiles of size `tile_size`
      - Within each tile, all layer outputs are kept in SRAM
      - Between tiles, only the boundary states need HBM round-trips
      - Total HBM traffic: O(T/tile_size * n_layers * D) = O(T·log(T)·D)
        when tile_size = O(log(T))

    Args:
        T: Number of thought steps.
        n_layers: Number of layers per step.
        tile_size: Steps per tile (auto-tuned to ~log(T) if 0).

    Returns:
        List of tiles, where each tile is a list of (step, layer) pairs.
    """
    if tile_size <= 0:
        # Auto-tune: use log2(T) rounded up, minimum 2
        tile_size = max(2, int(math.ceil(math.log2(max(T, 2)))))

    schedule = []
    for tile_start in range(0, T, tile_size):
        tile_end = min(tile_start + tile_size, T)
        tile = []
        for t in range(tile_start, tile_end):
            for l in range(n_layers):
                tile.append((t, l))
        schedule.append(tile)

    return schedule


# ──────────────────────────────────────────────────────────────────────────
# Pure PyTorch Fallback: Flash-style Causal Attention
# ──────────────────────────────────────────────────────────────────────────

def pytorch_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """PyTorch fallback for causal attention (used when Triton unavailable).

    Uses torch.nn.functional.scaled_dot_product_attention when available
    (PyTorch 2.0+), otherwise manual implementation.

    Args:
        q: [B, H, S, D] queries
        k: [B, H, S, D] keys
        v: [B, H, S, D] values
        scale: scaling factor (1/sqrt(d_head))
        dropout_p: dropout probability
        training: whether in training mode

    Returns:
        output: [B, H, S, D]
    """
    if hasattr(F, 'scaled_dot_product_attention'):
        # PyTorch 2.0+ — uses FlashAttention or memory-efficient backend
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout_p if training else 0.0,
            scale=scale,
        )

    # Manual fallback
    S = q.size(-2)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    causal_mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
    )
    attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

    attn_weights = F.softmax(attn_weights, dim=-1)
    if training and dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    return torch.matmul(attn_weights, v)


# ──────────────────────────────────────────────────────────────────────────
# Accelerated Attention Dispatcher
# ──────────────────────────────────────────────────────────────────────────

def accelerated_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    dropout_p: float = 0.0,
    training: bool = False,
    use_triton: bool = True,
) -> torch.Tensor:
    """Dispatch to the fastest available causal attention implementation.

    Priority: Triton tiled kernel > PyTorch SDPA > Manual.

    Args:
        q: [B, S, n_heads, d_head] queries (will be transposed internally)
        k: [B, S, n_heads, d_head] keys
        v: [B, S, n_heads, d_head] values
        n_heads: number of attention heads
        dropout_p: dropout probability
        training: whether in training mode
        use_triton: try Triton first

    Returns:
        output: [B, S, n_heads * d_head]
    """
    B, S, H, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Transpose to [B, H, S, D] for attention
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if use_triton and _HAS_TRITON and q.is_cuda:
        # Triton path: reshape to [B*H, S, D]
        q_flat = q.reshape(B * H, S, D).contiguous()
        k_flat = k.reshape(B * H, S, D).contiguous()
        v_flat = v.reshape(B * H, S, D).contiguous()
        out = triton_causal_attention(q_flat, k_flat, v_flat, scale)
        out = out.view(B, H, S, D).transpose(1, 2).reshape(B, S, H * D)
        return out

    # PyTorch SDPA / manual fallback
    out = pytorch_causal_attention(q, k, v, scale, dropout_p, training)
    return out.transpose(1, 2).reshape(B, S, H * D)

# ═════════════════════════════════════════════════════════════════════════
# engram.py
# ═════════════════════════════════════════════════════════════════════════

_DEFAULT_HEAD_MULTIPLIERS = [
    2654435761,   # Knuth's Fibonacci-derived multiplier (golden ratio · 2^32)
    40503,        # Knuth's smaller variant
    2246822519,   # MurmurHash3 32-bit C1
    3266489917,   # MurmurHash3 32-bit C2
    668265263,    # PCG / Numerical Recipes
    374761393,    # MurmurHash2 alternate
    3432918353,   # rotation-friendly prime
    461845907,    # MurmurHash3 32-bit final mixer
    2106027353,   # additional spread (random odd 32-bit)
    1789363923,
    3988292384,
    2654435789,
    4099279,
    899809343,
    3884862473,
    998254319,
]
_HASH_MASK = (1 << 32) - 1   # restrict state to uint32 range


@torch.no_grad()
def compute_ngram_indices(
    token_ids: torch.Tensor,
    n_order: int,
    n_heads: int,
    table_size: int,
    multipliers: torch.Tensor,
    bos_id: int = 0,
) -> torch.Tensor:
    """Multiplicative-XOR hash of suffix N-grams to per-head table indices.

    Suffix convention (paper Eq. 1): the N-gram ending at position t covers
    tokens (x_{t-n+1}, ..., x_t). Positions t < n-1 are left-padded with
    `bos_id` so every position gets a defined N-gram window.

    Args:
        token_ids: [B, S] int64 tokens
        n_order: N-gram order (e.g. 2 or 3)
        n_heads: K independent hash heads per order
        table_size: M_{n,k} — slots per head's table; should be prime
        multipliers: [n_heads] int64, one large odd constant per head
        bos_id: padding id for short suffixes at the start of the sequence

    Returns:
        [B, S, n_heads] int64 indices in [0, table_size)
    """
    B, S = token_ids.shape

    # Left-pad so position 0's suffix N-gram is well-defined (filled with BOS)
    padded = F.pad(token_ids, (n_order - 1, 0), value=bos_id)  # [B, S + n_order - 1]

    # Build the suffix-N-gram windows: windows[b, t, i] = padded[b, t+i]
    #   = token at position (t - n_order + 1 + i) in the original sequence.
    windows = torch.stack(
        [padded[:, i:i + S] for i in range(n_order)], dim=-1
    )  # [B, S, n_order]

    # Per-head MX hash:  state ← (state ^ x) * mul, mod 2^32 each round
    #
    # Using int64 arithmetic with explicit masking. For very long sequences
    # / large vocabs this is cheap because token_ids fit comfortably under
    # 2^17 (vocab_size at most 128k) and the multiply stays within int64
    # before the mask trims it back to uint32 width.
    mul = multipliers.view(1, 1, -1)                                # [1, 1, K]
    state = torch.zeros(B, S, n_heads, dtype=torch.long, device=token_ids.device)
    for i in range(n_order):
        x = windows[..., i].unsqueeze(-1)                           # [B, S, 1]
        state = ((state ^ x) * mul) & _HASH_MASK                    # [B, S, K]

    return state % table_size


# ──────────────────────────────────────────────────────────────────────────
# EngramTable — multi-head, multi-order N-gram embedding lookup
# ──────────────────────────────────────────────────────────────────────────

class EngramTable(nn.Module):
    """Static memory: token-ID-derived hash → embedding lookup.

    Maintains len(orders) × n_heads independent embedding tables, packed
    into a single flat nn.Embedding so PyTorch's standard sparse-aware
    backward path works without indirection. Per-(order, head) base offsets
    are pre-baked into a buffer so the per-step lookup is just an index add
    plus an embedding gather.

    Args:
        ngram_orders: list of N-gram orders to track, e.g. [2, 3]
        n_heads:      K independent hash heads per order
        slots_per_table: M, the per-(order, head) table size. Choose a prime
                      to minimize hash-collision regularity. Default 65521
                      (largest prime ≤ 2^16) gives a ~256MB total table at
                      orders=[2,3], heads=8, d_head=64, fp32.
        d_head:       embedding dim per head. Final concatenated dim is
                      d_mem = len(ngram_orders) × n_heads × d_head.
        bos_id:       padding ID for short suffixes at sequence start.

    Output (forward):
        e_t [B, S, d_mem] — concatenated retrieved embeddings.
    """

    def __init__(
        self,
        ngram_orders: list[int],
        n_heads: int,
        slots_per_table: int,
        d_head: int,
        bos_id: int = 0,
    ):
        super().__init__()
        if not ngram_orders or any(n < 1 for n in ngram_orders):
            raise ValueError(f"ngram_orders must be a non-empty list of positives; got {ngram_orders}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {n_heads}")
        if slots_per_table < 2:
            raise ValueError(f"slots_per_table must be ≥ 2; got {slots_per_table}")

        self.ngram_orders = list(ngram_orders)
        self.n_heads = n_heads
        self.slots_per_table = slots_per_table
        self.d_head = d_head
        self.bos_id = bos_id

        self.n_tables = len(ngram_orders) * n_heads
        self.d_mem = self.n_tables * d_head

        # Single flat embedding holds *all* (order, head, slot) rows.
        # Indexing scheme: row at (order_idx, head_k, slot_s) =
        #     order_idx · (n_heads · slots) + head_k · slots + s
        total_slots = self.n_tables * slots_per_table
        self.tables = nn.Embedding(total_slots, d_head)

        # Per-head multipliers (extend deterministically if user requests
        # more heads than the curated default list provides).
        if n_heads > len(_DEFAULT_HEAD_MULTIPLIERS):
            mults = list(_DEFAULT_HEAD_MULTIPLIERS)
            seed = 12345
            while len(mults) < n_heads:
                # Linear-congruential generator (Numerical Recipes parameters)
                # — produces well-spread odd 32-bit constants
                seed = (seed * 1103515245 + 12345) & _HASH_MASK
                candidate = seed | 1   # force odd
                if candidate not in mults:
                    mults.append(candidate)
            multipliers = mults[:n_heads]
        else:
            multipliers = _DEFAULT_HEAD_MULTIPLIERS[:n_heads]

        self.register_buffer(
            "multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        # Flat-index offsets so the gather indexes the single big table.
        order_offsets = torch.tensor(
            [oi * n_heads * slots_per_table for oi in range(len(ngram_orders))],
            dtype=torch.long,
        )
        head_offsets = torch.arange(n_heads, dtype=torch.long) * slots_per_table

        self.register_buffer("order_offsets", order_offsets, persistent=True)
        self.register_buffer("head_offsets", head_offsets, persistent=True)

        self._reset_special_inits()

    def _reset_special_inits(self):
        """Small init so initial lookups don't dominate the residual stream.

        Called from __init__ and ALSO re-applied after the parent module's
        generic apply(_init_weights) walks the tree (which would otherwise
        clobber this with std=0.02).
        """
        nn.init.normal_(self.tables.weight, std=0.01)

    @property
    def num_lookup_params(self) -> int:
        """Total parameters in the lookup table (the bulk of Engram's params)."""
        return self.tables.num_embeddings * self.tables.embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, S] long → e_t [B, S, d_mem]."""
        B, S = token_ids.shape
        per_order_embeds = []

        for oi, n_order in enumerate(self.ngram_orders):
            # Per-head hash indices for this order:  [B, S, n_heads]
            indices = compute_ngram_indices(
                token_ids,
                n_order=n_order,
                n_heads=self.n_heads,
                table_size=self.slots_per_table,
                multipliers=self.multipliers,
                bos_id=self.bos_id,
            )
            # Promote to flat-table indices (add per-order and per-head bases)
            flat = indices + self.head_offsets.view(1, 1, -1) + self.order_offsets[oi]
            # Gather: [B, S, n_heads, d_head] → flatten the head dim
            embeds = self.tables(flat)                                # [B, S, K, d_head]
            per_order_embeds.append(embeds.flatten(2))                # [B, S, K·d_head]

        return torch.cat(per_order_embeds, dim=-1)                    # [B, S, d_mem]


# ──────────────────────────────────────────────────────────────────────────
# EngramProjection — layer-specific K, V projections (computed once)
# ──────────────────────────────────────────────────────────────────────────

class EngramProjection(nn.Module):
    """Layer-specific W_K, W_V projections of the static Engram memory.

    Crucial efficiency: e_t is constant across the thought loop, so we
    project it ONCE per forward pass instead of T·n_layers times. This
    reduces Engram's incremental compute from ~T·n_layers·O(B·S·d_mem·d_out)
    to T·n_layers·O(B·S·d_out)  (the per-step gate is just a dot product).

    Per the paper, W_V is zero-initialized — combined with the zero-init
    on the conv inside EngramGate, this guarantees the Engram contribution
    is exactly 0 at the start of training. The model bootstraps from its
    pre-Engram behavior and only gradually learns to use the memory.

    Args:
        d_mem:    Engram lookup output width
        d_query:  width of the K projection (matches the gating query)
        d_out:    width of the V projection (final Engram contribution width)
        zero_init_v:  if True, init W_V to 0 for identity-at-init
    """

    def __init__(
        self,
        d_mem: int,
        d_query: int,
        d_out: int,
        zero_init_v: bool = True,
    ):
        super().__init__()
        self.W_K = nn.Linear(d_mem, d_query, bias=False)
        self.W_V = nn.Linear(d_mem, d_out, bias=False)
        self.zero_init_v = zero_init_v
        self._reset_special_inits()

    def _reset_special_inits(self):
        # Standard small init for K — we want a meaningful gating signal
        # from the start so the model can learn to *trust or distrust* memory.
        nn.init.normal_(self.W_K.weight, std=0.02)
        # Zero init for V: makes the Engram contribution exactly 0 at step 0,
        # so the model starts from its pre-Engram solution and grows into
        # using memory rather than being thrown by an uninformed lookup.
        if self.zero_init_v:
            nn.init.zeros_(self.W_V.weight)
        else:
            nn.init.normal_(self.W_V.weight, std=0.02)

    def forward(self, e_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """e_t [B, S, d_mem] → (k [B, S, d_query], v [B, S, d_out])."""
        return self.W_K(e_t), self.W_V(e_t)


# ──────────────────────────────────────────────────────────────────────────
# EngramGate — per-thought-step gating + (optional) depthwise causal conv
# ──────────────────────────────────────────────────────────────────────────

class EngramGate(nn.Module):
    """Context-aware gating + depthwise causal conv (paper §2.3).

    Runs once per thought step inside each ThoughtLayer that uses Engram.
    Cheap: the only matmul-shaped op is a single dot product per position.

    Pipeline (paper Eq. 4–5):
        α_t = σ( <RMSNorm(q_t), RMSNorm(k_t)> / √d_query )       # [B, S, 1]
        ṽ_t = α_t · v_t                                          # [B, S, d_out]
        Y_t = SiLU(Conv1D(RMSNorm(ṽ_t))) + ṽ_t                   # [B, S, d_out]

    The conv is depthwise causal with kernel=4 and dilation=N_max (matches
    the paper's recommended config). It's zero-initialized so Y = ṽ at the
    start of training; combined with W_V zero-init in EngramProjection,
    this means Y_0 = 0 too — the entire Engram path is exactly 0 at init.

    Args:
        d_query:  K-projection / query width (must match q's last dim)
        d_out:    V-projection / output width (must match v's last dim)
        kernel_size: conv kernel (paper: 4)
        dilation:    conv dilation (paper: max N-gram order)
        use_conv:    set False to skip the conv refinement (cheaper, slight
                     loss in expressivity per the paper's ablation)
    """

    def __init__(
        self,
        d_query: int,
        d_out: int,
        kernel_size: int = 4,
        dilation: int = 3,
        use_conv: bool = True,
    ):
        super().__init__()
        self.d_query = d_query
        self.d_out = d_out
        self.use_conv = use_conv

        # RMSNorms on q and k stabilize the dot product magnitude — without
        # them, the gate saturates as the projections grow during training.
        self.q_norm = nn.RMSNorm(d_query)
        self.k_norm = nn.RMSNorm(d_query)
        self.scale = 1.0 / math.sqrt(d_query)

        if use_conv:
            self.conv_norm = nn.RMSNorm(d_out)
            # Depthwise: each channel evolves independently. groups=d_out.
            self.conv = nn.Conv1d(
                d_out, d_out,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=d_out,
                padding=0,            # we'll left-pad manually for causality
                bias=True,
            )
            # Effective receptive field: (kernel - 1) · dilation + 1
            # The left-padding amount is the receptive field minus 1.
            self.causal_pad = (kernel_size - 1) * dilation
            self._reset_special_inits()

    def _reset_special_inits(self):
        if self.use_conv:
            # Zero-init keeps the SiLU(Conv(·)) branch at 0 → Y = ṽ at init.
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        query: torch.Tensor,    # [B, S, d_query]
        k: torch.Tensor,        # [B, S, d_query]
        v: torch.Tensor,        # [B, S, d_out]
    ) -> torch.Tensor:
        """Returns Y [B, S, d_out] — the Engram contribution to add to attn_out."""
        # Per-position scalar gate (paper Eq. 4)
        q_n = self.q_norm(query)
        k_n = self.k_norm(k)
        # Dot product on the d_query axis → [B, S, 1] after keepdim sum
        gate = torch.sigmoid(
            (q_n * k_n).sum(dim=-1, keepdim=True) * self.scale
        )

        v_gated = gate * v                                            # [B, S, d_out]

        if not self.use_conv:
            return v_gated

        # Depthwise causal conv: paper Eq. 5
        v_norm = self.conv_norm(v_gated)
        x = v_norm.transpose(1, 2)                                    # [B, d_out, S]
        x = F.pad(x, (self.causal_pad, 0))                            # left-pad only (causal)
        x = self.conv(x)                                              # [B, d_out, S]
        x = x.transpose(1, 2)                                         # [B, S, d_out]
        return F.silu(x) + v_gated                                    # SiLU branch + residual

# ═════════════════════════════════════════════════════════════════════════
# dssa.py
# ═════════════════════════════════════════════════════════════════════════

class SparseStateExpansion(nn.Module):
    """Linear attention with sparse state expansion.

    Maintains m state partitions S_i ∈ R^{d_key × d_value}. On each token,
    an input-dependent gating vector selects the top-k partitions to update:

        S_i ← S_i + e_i · k^T v    if i ∈ top-k(gating)
        S_i ← S_i                   otherwise

    Output is computed as: o = Σ_i (q^T S_i) · selection_weight_i

    Args:
        d_model: Model dimension (total, split across heads).
        n_heads: Number of attention heads.
        n_partitions: Number of state partitions m.
        top_k: Number of partitions to update per step.
        d_head: Dimension per head (default: d_model // n_heads).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_partitions: int = 32,
        top_k: int = 8,
        d_head: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_partitions = n_partitions
        self.top_k = min(top_k, n_partitions)
        self.d_head = d_head or (d_model // n_heads)

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        # Gating: produces partition selection logits
        self.gate_proj = nn.Linear(d_model, n_partitions, bias=True)

        # Layer norms for stability
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass with sparse state expansion (vectorized).

        Mathematically equivalent to the per-position recurrent scan in
        ``_forward_reference``, but expressed as four standard tensor
        ops (matmul, scatter, matmul, softmax) instead of a Python loop
        over S positions × k partitions × B batch elements.

        Derivation. The recurrent state is
            state[t, h, p, d_k, d_v] = Σ_{t' ≤ t} pw[t', p] · k[t', h, d_k] · v[t', h, d_v]
        where pw is the dense partition-weight tensor (gate_weights at
        the chosen partitions, zero elsewhere). The readout is then
            readout[t, h, p, d_v]
                = Σ_{d_k} q[t, h, d_k] · state[t, h, p, d_k, d_v]
                = Σ_{t' ≤ t} pw[t', p] · (q[t,h] · k[t',h]) · v[t', h, d_v]
        i.e. linear attention with per-partition gated values. We never
        materialize the [B, H, m, dh, dh] state — it cancels out.

        Why the original was slow. The previous implementation had a
        ``for t in range(S)`` Python loop with nested ``for i in range(k)``
        and ``for b in range(B)`` plus a ``part_idx[b].item()`` call that
        forced a CUDA→CPU sync on every iteration: 4096 syncs per call
        at S=512, k=8, B=1. Vectorizing eliminates the sync entirely
        and lets cuBLAS handle the heavy lifting.

        Memory. Peak intermediate is ``weighted_v`` of shape
        [B, S, H, m, dh] (~32 MB at default config in bf16) and the
        attention scores [B, H, S, S] (~8 MB). Total well under 100 MB
        for typical configs; scales linearly with S and quadratically
        with H·dh, so very long contexts (S>4k) may want chunking.
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        m = self.n_partitions
        k = self.top_k

        # Project Q from queries, K/V from text
        q = self.q_proj(x).view(B, S, H, dh)
        kk = self.k_proj(text_kv).view(B, S, H, dh)
        v = self.v_proj(text_kv).view(B, S, H, dh)

        # Normalize Q and K for stable dot products (preserves SSE design)
        q = self.q_norm(q)
        kk = self.k_norm(kk)

        # Top-k partition selection. We use the values returned by topk
        # directly instead of an extra gather (the original did
        # ``gate_logits.gather(-1, topk(...).indices)`` which recomputes
        # the same values).
        gate_logits = self.gate_proj(text_kv)            # [B, S, m]
        top_values, top_indices = gate_logits.topk(k, dim=-1)
        gate_weights = F.softmax(top_values, dim=-1)     # [B, S, k]

        # Scatter the per-position k weights into a dense [B, S, m]
        # tensor. Non-selected partitions stay at zero, so they
        # contribute nothing to the readout — equivalent to "only the
        # top-k partitions get updated" but in a form the rest of the
        # pipeline can consume in vectorized fashion.
        # We use scatter_add_ rather than scatter_ so that the (very
        # rare) case of duplicate top-k indices accumulates exactly
        # like the sequential reference would.
        pw = torch.zeros(B, S, m, device=x.device, dtype=x.dtype)
        pw.scatter_add_(2, top_indices, gate_weights.to(pw.dtype))

        # Linear-attention scores (no softmax). Use matmul (cuBLAS GEMM)
        # rather than einsum for the hot path.
        q_h = q.permute(0, 2, 1, 3)     # [B, H, S_t, dh]
        k_h = kk.permute(0, 2, 3, 1)    # [B, H, dh,  S_u]
        scores = torch.matmul(q_h, k_h)  # [B, H, S_t, S_u]

        # Causal mask — fill with 0 (linear attention has no softmax,
        # so future positions just shouldn't contribute).
        if causal:
            causal_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, 0.0)

        # Per-partition weighted values: gate v by pw, broadcasting over
        # heads and value dimension.
        #   weighted_v[b, u, h, p, d] = pw[b, u, p] · v[b, u, h, d]
        weighted_v = v.unsqueeze(3) * pw.view(B, S, 1, m, 1)  # [B, S, H, m, dh]

        # Per-partition readout. Reshape for a single batched matmul:
        #   scores [B, H, S_t, S_u] @ weighted_v [B, H, S_u, m·dh]
        #     → readout_flat [B, H, S_t, m·dh]
        # then split m·dh back out and permute S_t to dim 1.
        wv_h = weighted_v.permute(0, 2, 1, 3, 4).reshape(B, H, S, m * dh)
        readout_flat = torch.matmul(scores, wv_h)            # [B, H, S, m·dh]
        readout = readout_flat.view(B, H, S, m, dh).permute(0, 2, 1, 3, 4)
        # readout: [B, S, H, m, dh]

        # Per-partition norms → softmax over partitions, with a tiny eps
        # so positions with no contribution yet (e.g. early positions
        # whose top-k partitions never received content) don't NaN.
        partition_norms = readout.norm(dim=-1) + 1e-8        # [B, S, H, m]
        attn_weights = F.softmax(partition_norms, dim=-1)    # [B, S, H, m]

        # Combine partitions. Equivalent to einsum('bthm,bthmd->bthd')
        # but written as a broadcast-and-sum so it's clearly a
        # weighted-mean op rather than a matmul.
        out = (attn_weights.unsqueeze(-1) * readout).sum(dim=3)  # [B, S, H, dh]
        out = out.reshape(B, S, H * dh)
        return self.out_proj(out)

    def _forward_reference(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Sequential reference implementation, kept for numerical
        regression testing. Do NOT use in training — this is the original
        Python-loop version that hits ~4096 CUDA syncs per call. The
        public ``forward`` produces numerically identical output (to bf16
        rounding) at a fraction of the cost.
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        m = self.n_partitions
        k = self.top_k

        q = self.q_proj(x).view(B, S, H, dh)
        kk = self.k_proj(text_kv).view(B, S, H, dh)
        v = self.v_proj(text_kv).view(B, S, H, dh)
        q = self.q_norm(q)
        kk = self.k_norm(kk)

        gate_logits = self.gate_proj(text_kv)
        _, top_indices = gate_logits.topk(k, dim=-1)
        gate_weights = F.softmax(
            gate_logits.gather(-1, top_indices), dim=-1
        )

        state = torch.zeros(B, H, m, dh, dh, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(S):
            kt = kk[:, t]
            vt = v[:, t]
            qt = q[:, t]
            gt = gate_weights[:, t]
            idx_t = top_indices[:, t]

            kv_outer = torch.einsum('bhd,bhe->bhde', kt, vt)

            for i in range(k):
                part_idx = idx_t[:, i]
                weight = gt[:, i]
                for b in range(B):
                    pi = part_idx[b].item()
                    state[b, :, pi] = state[b, :, pi] + weight[b] * kv_outer[b]

            readout = torch.einsum('bhd,bhmde->bhme', qt, state)
            partition_norms = readout.norm(dim=-1) + 1e-8
            attn_weights = F.softmax(partition_norms, dim=-1)
            out_t = torch.einsum('bhm,bhmd->bhd', attn_weights, readout)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, S, H * dh)
        return self.out_proj(output)


class MixtureOfBlockAttention(nn.Module):
    """Block-wise sparse attention (MoBA).

    Divides the KV sequence into blocks of fixed size. For each query,
    scores blocks by their relevance (mean key similarity) and attends
    only to the top-k blocks. This gives O(N·k/B) compute instead of O(N²).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        block_size: Size of each KV block.
        top_k_blocks: Number of blocks to attend to per query.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 64,
        top_k_blocks: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass with block-sparse attention.

        Args:
            x: [B, S, d_model] — query source.
            text_kv: [B, S, d_model] — KV source.
            causal: Whether to apply causal masking.

        Returns:
            output: [B, S, d_model]
        """
        B, S, D = x.shape
        H = self.n_heads
        dh = self.d_head
        bs = self.block_size

        q = self.q_proj(x).view(B, S, H, dh).transpose(1, 2)     # [B, H, S, dh]
        k = self.k_proj(text_kv).view(B, S, H, dh).transpose(1, 2)  # [B, H, S, dh]
        v = self.v_proj(text_kv).view(B, S, H, dh).transpose(1, 2)  # [B, H, S, dh]

        # Number of blocks (pad last block if needed)
        n_blocks = (S + bs - 1) // bs
        pad_len = n_blocks * bs - S

        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        # Reshape K, V into blocks: [B, H, n_blocks, block_size, dh]
        k_blocks = k.view(B, H, n_blocks, bs, dh)
        v_blocks = v.view(B, H, n_blocks, bs, dh)

        # Block-level scoring: mean key per block
        k_block_means = k_blocks.mean(dim=3)  # [B, H, n_blocks, dh]

        # Score each query against block means
        # [B, H, S, dh] @ [B, H, dh, n_blocks] → [B, H, S, n_blocks]
        block_scores = torch.matmul(q, k_block_means.transpose(-2, -1))
        block_scores = block_scores / math.sqrt(dh)

        # Causal masking: block b is valid for query at position t
        # only if block_start <= t (i.e., at least part of the block is visible)
        if causal:
            block_starts = torch.arange(n_blocks, device=q.device) * bs  # [n_blocks]
            query_pos = torch.arange(S, device=q.device)  # [S]
            # [S, n_blocks]: True if block start > query position (future block)
            causal_block_mask = block_starts.unsqueeze(0) > query_pos.unsqueeze(1)
            block_scores = block_scores.masked_fill(
                causal_block_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )

        # Select top-k blocks per query
        k_effective = min(self.top_k_blocks, n_blocks)
        _, top_block_idx = block_scores.topk(k_effective, dim=-1)  # [B, H, S, k]

        # Gather the selected blocks' KV
        # Expand indices for gathering: [B, H, S, k] → gather from [B, H, n_blocks, bs, dh]
        # We need to attend within the selected blocks
        scale = math.sqrt(dh)
        outputs = torch.zeros(B, H, S, dh, device=q.device, dtype=q.dtype)

        for i in range(k_effective):
            block_idx = top_block_idx[:, :, :, i]  # [B, H, S]

            # Gather K and V for this block selection
            # block_idx → index into k_blocks: [B, H, n_blocks, bs, dh]
            bi_expanded = block_idx.unsqueeze(-1).unsqueeze(-1).expand(
                B, H, S, bs, dh
            )
            ki = k_blocks.gather(2, bi_expanded)  # [B, H, S, bs, dh]
            vi = v_blocks.gather(2, bi_expanded)  # [B, H, S, bs, dh]

            # Attention within block
            # [B, H, S, 1, dh] @ [B, H, S, dh, bs] → [B, H, S, 1, bs]
            attn = torch.matmul(
                q.unsqueeze(3), ki.transpose(-2, -1)
            ).squeeze(3) / scale  # [B, H, S, bs]

            # Causal mask within block
            if causal:
                block_starts_sel = block_idx * bs  # [B, H, S]
                positions = torch.arange(bs, device=q.device)  # [bs]
                abs_positions = block_starts_sel.unsqueeze(-1) + positions  # [B, H, S, bs]
                query_positions = torch.arange(S, device=q.device).view(1, 1, S, 1)
                future_mask = abs_positions > query_positions
                attn = attn.masked_fill(future_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            # Handle all-masked rows (position 0 with future-only blocks)
            attn = torch.nan_to_num(attn, nan=0.0)

            # Weighted sum
            # [B, H, S, 1, bs] @ [B, H, S, bs, dh] → [B, H, S, dh]
            block_out = torch.matmul(attn.unsqueeze(3), vi).squeeze(3)
            outputs = outputs + block_out

        # Average over blocks (simple; could use block-score-weighted)
        outputs = outputs / max(k_effective, 1)

        # Reshape and project output
        outputs = outputs.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(outputs)


class DualSpaceSparseAttention(nn.Module):
    """Hybrid attention: SSE for compressed retrieval + MoBA for exact blocks.

    Combines the constant-memory linear attention of Sparse State Expansion
    with periodic exact block attention from Mixture of Block Attention.
    A learnable router determines the mixing ratio per position.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_partitions: SSE state partitions.
        sse_top_k: Active SSE partitions per step.
        block_size: MoBA block size.
        moba_top_k_blocks: MoBA blocks per query.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_partitions: int = 32,
        sse_top_k: int = 8,
        block_size: int = 64,
        moba_top_k_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.sse = SparseStateExpansion(
            d_model=d_model,
            n_heads=n_heads,
            n_partitions=n_partitions,
            top_k=sse_top_k,
        )
        self.moba = MixtureOfBlockAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            top_k_blocks=moba_top_k_blocks,
        )

        # Router: determines mixing ratio between SSE and MoBA
        # Output: 2 logits (SSE weight, MoBA weight) per position
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Hybrid SSE + MoBA attention.

        Args:
            x: [B, S, d_model] — query source.
            text_kv: [B, S, d_model] — KV source.
            causal: Whether to apply causal masking.

        Returns:
            output: [B, S, d_model]
        """
        # Route: decide SSE vs MoBA mixing per position
        route_logits = self.router(x)  # [B, S, 2]
        route_weights = F.softmax(route_logits, dim=-1)  # [B, S, 2]

        # SSE branch (compressed, constant memory)
        sse_out = self.sse(x, text_kv, causal=causal)  # [B, S, D]

        # MoBA branch (exact, block-sparse)
        moba_out = self.moba(x, text_kv, causal=causal)  # [B, S, D]

        # Weighted combination
        w_sse = route_weights[:, :, 0:1]   # [B, S, 1]
        w_moba = route_weights[:, :, 1:2]  # [B, S, 1]

        output = w_sse * sse_out + w_moba * moba_out
        output = self.out_norm(output)
        output = self.dropout(output)

        return output

# ═════════════════════════════════════════════════════════════════════════
# memory.py
# ═════════════════════════════════════════════════════════════════════════

class TemporalMemory(nn.Module):
    """
    FIFO history buffers for pre-activations and post-activations.

    Maintains two rolling buffers of shape [batch, history_len, d_latent].
    Each call to push() shifts the buffer forward (oldest entry dropped)
    and appends the new values at the end.

    Args:
        d_latent: Neuron count (width of each activation snapshot).
        history_len: Depth of the FIFO buffer.
    """

    def __init__(self, d_latent: int, history_len: int):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len

        # Learnable initial states for history buffers
        self.pre_history_init = nn.Parameter(torch.zeros(1, self.history_len, self.d_latent))
        self.post_history_init = nn.Parameter(torch.zeros(1, self.history_len, self.d_latent))

        # Buffers are registered as non-persistent state (not saved with model params)
        # They are initialized lazily on first push() to match batch size and device.
        self.register_buffer("pre_history", None, persistent=False)
        self.register_buffer("post_history", None, persistent=False)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Initialize both history buffers using the learnable initial states.

        Args:
            batch_size: Current batch size.
            device: Target device.
            dtype: Data type for buffers.
        """
        self.pre_history = self.pre_history_init.to(device=device, dtype=dtype).expand(
            batch_size, -1, -1
        ).clone()
        self.post_history = self.post_history_init.to(device=device, dtype=dtype).expand(
            batch_size, -1, -1
        ).clone()

    def push_pre(self, pre_activations: torch.Tensor):
        """
        Push new pre-activations into the FIFO buffer.

        Uses roll + indexed assignment instead of cat. Memory cost is
        equivalent (both allocate a fresh [B, H, D] tensor per call), but
        the autograd graph is cleaner: roll has one parent (the buffer),
        cat has two (the slice + the new entry), and the roll variant
        plays better with the periodic .detach() applied at thought-step
        boundaries to break cross-step graph chaining.

        Args:
            pre_activations: [batch, d_latent] — new pre-activation values.
        """
        # roll returns a NEW tensor (not in-place), so the subsequent
        # index_put modifies that new tensor — never the original leaf.
        # Safe under autograd even when the buffer is a leaf with
        # requires_grad=True.
        self.pre_history = self.pre_history.roll(-1, dims=1)
        self.pre_history[:, -1, :] = pre_activations

    def push_post(self, post_activations: torch.Tensor):
        """
        Push new post-activations into the FIFO buffer.

        Args:
            post_activations: [batch, d_latent] — new post-activation values.
        """
        self.post_history = self.post_history.roll(-1, dims=1)
        self.post_history[:, -1, :] = post_activations

    def get_pre_history(self) -> torch.Tensor:
        """Returns the full pre-activation history buffer [batch, history_len, d_latent]."""
        return self.pre_history

    def get_post_history(self) -> torch.Tensor:
        """Returns the full post-activation history buffer [batch, history_len, d_latent]."""
        return self.post_history


class SynchronizationComputer(nn.Module):
    """
    Computes neural synchronization representations from post-activation history.

    The synchronization matrix S_t = Z_t @ Z_t^T captures the coupling
    structure between neurons based on their recent activation patterns.
    This replaces token-position-based queries with state-based queries.

    Three modes are supported (controlled by `method`):
    - "full": Full S_t flattened → [d_latent^2] (expensive, most expressive)
    - "diag_summary": diag(S_t) + row_means + col_means → [3 * d_latent] (practical)
    - "low_rank": Top-k SVD approximation → [rank * d_latent] (balanced)
    - "sparse_decay": Sparse pairing with learnable exponential decay → [sync_sparse_pairs]

    Args:
        d_latent: Neuron count.
        history_len: FIFO buffer depth.
        method: Synchronization computation method.
        rank: Rank for low_rank method.
        sync_sparse_pairs: Number of sparse pairs for sparse_decay method.
    """

    def __init__(
        self,
        d_latent: int,
        history_len: int,
        method: str = "diag_summary",
        rank: int = 32,
        sync_sparse_pairs: int = 256,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len
        self.method = method
        self.rank = rank
        self.sync_sparse_pairs = sync_sparse_pairs

        if method == "sparse_decay":
            # Pre-choose D_chosen neuron pairs from D total neurons
            idxs_left = torch.randint(low=0, high=d_latent, size=(sync_sparse_pairs,))
            idxs_right = torch.randint(low=0, high=d_latent, size=(sync_sparse_pairs,))
            self.register_buffer("idxs_left", idxs_left)
            self.register_buffer("idxs_right", idxs_right)
            # Define learnable exponential decay scaling factors per neuron pair
            self.r = nn.Parameter(torch.zeros(1, sync_sparse_pairs, 1))

    def compute(self, post_history: torch.Tensor) -> torch.Tensor:
        """
        Compute synchronization representation from post-activation history.

        Args:
            post_history: [batch, history_len, d_latent] — the Z_t matrix.

        Returns:
            sync_repr: [batch, sync_dim] — flattened synchronization representation.
        """
        B, H, D = post_history.shape
        H_safe = max(H, 1)

        # ── Fast path for diag_summary ──────────────────────────────────
        # The naive path materializes S = Zᵀ Z / H of shape [B, D, D],
        # which at D=768, B=512, bf16 is 576 MiB *per layer per thought
        # step* — enough to OOM a 200M model on consumer hardware.
        # Since we only need diag(S), row_means(S), col_means(S), we can
        # compute all three directly from Z in O(B·H·D) memory, never
        # touching the [D, D] matrix.
        #
        # Identities (all exact, no approximation):
        #   diag(S)[b,i]      = (1/H) · Σ_h Z[b,h,i]²
        #   row_means(S)[b,i] = (1/(D·H)) · Σ_h Z[b,h,i] · (Σ_j Z[b,h,j])
        #   col_means(S)[b,j] = row_means(S)[b,j]   (S is symmetric)
        if self.method == "diag_summary":
            # diag: sum of squares along the history axis.
            # einsum fuses the elementwise square with the reduction,
            # avoiding the [B, H, D] intermediate that (post*post) would
            # allocate before .sum(dim=1).
            diag = torch.einsum('bhd,bhd->bd', post_history, post_history) / H_safe  # [B, D]

            # row_means: via bmm against the per-history-step row sums
            # Zsum: [B, H, 1] — cheap, no D×D tensor ever exists
            Zsum = post_history.sum(dim=2, keepdim=True)                # [B, H, 1]
            row_means = torch.bmm(
                post_history.transpose(1, 2),                            # [B, D, H]
                Zsum,                                                    # [B, H, 1]
            ).squeeze(2) / (H_safe * D)                                  # [B, D]

            # col_means == row_means by symmetry of Zᵀ Z
            col_means = row_means

            return torch.cat([diag, row_means, col_means], dim=1)       # [B, 3D]

        # ── Fast path for sparse_decay ──────────────────────────────────
        if self.method == "sparse_decay":
            S_post = post_history # [B, T=H, D]
            # decay BACK in time
            t_back = torch.arange(H - 1, -1, -1, device=post_history.device, dtype=post_history.dtype)
            t_back = t_back.view(1, H, 1) # [1, H, 1]
            
            # Compute per NEURON PAIR exponential decays
            # self.r is [1, D_chosen, 1]
            # r permuted to [1, 1, D_chosen] to match t_back broadcast [1, H, 1] -> [1, H, D_chosen]
            exp_decay = torch.exp(-t_back * self.r.view(1, 1, -1)) # [1, H, D_chosen]
            
            # Subsampled S
            # S[:,:,idxs_left] * exp_decay * S[:,:,idxs_right]
            S_left = S_post[:, :, self.idxs_left] # [B, H, D_chosen]
            S_right = S_post[:, :, self.idxs_right] # [B, H, D_chosen]
            S_multiplied = S_left * exp_decay * S_right # [B, H, D_chosen]
            
            # Sum over the free T (H) dimension and normalise by sqrt of AUC of decays
            synch_representation = S_multiplied.sum(dim=1) / torch.sqrt(exp_decay.sum(dim=1)) # [B, D_chosen]
            return synch_representation

        # ── Full-matrix path (required for "full" and "low_rank") ───────
        # S_t = Z_tᵀ @ Z_t → [batch, d_latent, d_latent]
        # Note: We compute Zᵀ @ Z (not Z @ Zᵀ) to get neuron-neuron coupling
        S = torch.bmm(post_history.transpose(1, 2), post_history)
        S = S / H_safe

        if self.method == "full":
            # Flatten entire matrix
            return S.reshape(B, D * D)

        elif self.method == "low_rank":
            # Top-k eigenvalue approximation via SVD
            # Only compute the top `rank` singular values/vectors
            U, s, _ = torch.linalg.svd(S, full_matrices=False)
            # Take top-rank components: U[:, :, :rank] * s[:, :rank]
            k = min(self.rank, D)
            low_rank = U[:, :, :k] * s[:, :k].unsqueeze(1)  # [batch, d_latent, rank]
            return low_rank.reshape(B, D * k)

        else:
            raise ValueError(f"Unknown sync method: {self.method}")

    @property
    def output_dim(self) -> int:
        """Dimension of the synchronization representation vector."""
        if self.method == "full":
            return self.d_latent * self.d_latent
        elif self.method == "diag_summary":
            return 3 * self.d_latent
        elif self.method == "low_rank":
            return min(self.rank, self.d_latent) * self.d_latent
        elif self.method == "sparse_decay":
            return self.sync_sparse_pairs
        else:
            raise ValueError(f"Unknown sync method: {self.method}")

# ═════════════════════════════════════════════════════════════════════════
# nlm.py
# ═════════════════════════════════════════════════════════════════════════

class NeuronLevelModels(nn.Module):
    """
    Per-neuron MLPs that process temporal history of pre-activations.

    Each of the d_latent neurons has its own small MLP:
        history [batch, history_len] → hidden [batch, nlm_hidden] → output [batch, 1]

    When nlm_groups > 1, neurons are grouped and share MLP parameters
    within each group (reducing parameter count by the group factor).

    Args:
        d_latent: Number of neurons in the latent space.
        history_len: Depth of the FIFO buffer (temporal window).
        nlm_hidden_dim: Hidden dimension of each neuron's MLP.
        nlm_groups: Number of neuron groups (1 = true per-neuron, >1 = grouped).
        dropout: Dropout rate applied after hidden layer.
    """

    def __init__(
        self,
        d_latent: int,
        history_len: int,
        nlm_hidden_dim: int,
        nlm_groups: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.history_len = history_len
        self.nlm_hidden_dim = nlm_hidden_dim
        self.nlm_groups = nlm_groups

        assert d_latent % nlm_groups == 0, \
            f"d_latent ({d_latent}) must be divisible by nlm_groups ({nlm_groups})"
        self.neurons_per_group = d_latent // nlm_groups

        # Each group has its own 2-layer MLP:
        #   Layer 1: [history_len] → [nlm_hidden_dim]
        #   Layer 2: [nlm_hidden_dim] → [1]
        # Packed as [nlm_groups, ...] tensors for batched computation.

        # Layer 1 weights and biases: [nlm_groups, history_len, nlm_hidden_dim]
        self.w1 = nn.Parameter(
            torch.empty(nlm_groups, history_len, nlm_hidden_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(nlm_groups, nlm_hidden_dim))

        # Layer 2 weights and biases: [nlm_groups, nlm_hidden_dim, 1]
        self.w2 = nn.Parameter(
            torch.empty(nlm_groups, nlm_hidden_dim, 1)
        )
        self.b2 = nn.Parameter(torch.zeros(nlm_groups, 1))

        # Learnable output gate — controls how much the NLM output
        # contributes vs. a simple pass-through of the latest pre-activation.
        # Initialized near 0.5 so both the learned dynamics and direct signal
        # contribute equally at initialization.
        self.gate = nn.Parameter(torch.full((d_latent,), 0.0))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize NLM weights for stable training.

        Layer 1: Kaiming init (fan_in = history_len) so that the pre-activation
        history maps into a well-scaled hidden space.
        Layer 2: Small init so initial NLM output ≈ 0, meaning the gated
        output is dominated by the pass-through at the start of training.
        """
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in = self.history_len
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)

        # Small init for layer 2 so NLM starts near identity
        nn.init.normal_(self.w2, std=0.01)
        nn.init.zeros_(self.b2)

    def forward(self, pre_activation_history: torch.Tensor) -> torch.Tensor:
        """
        Process temporal history through per-neuron MLPs.

        Args:
            pre_activation_history: [batch, history_len, d_latent]
                The FIFO buffer of pre-activations, oldest first.

        Returns:
            post_activations: [batch, d_latent]
                The new neuron states after temporal processing.
        """
        B, H, D = pre_activation_history.shape
        assert H == self.history_len and D == self.d_latent

        # Reshape for grouped computation:
        # [batch, history_len, d_latent] → [batch, history_len, nlm_groups, neurons_per_group]
        x = pre_activation_history.view(B, H, self.nlm_groups, self.neurons_per_group)
        # → [batch, nlm_groups, neurons_per_group, history_len]
        x = x.permute(0, 2, 3, 1)

        # Layer 1: [batch, groups, neurons_per_group, history_len] @ [groups, history_len, hidden]
        # Using einsum for the grouped matmul:
        # b=batch, g=groups, n=neurons_per_group, h=history_len, d=hidden
        h = torch.einsum("bgnh,ghd->bgnd", x, self.w1)
        # Add bias: [groups, hidden] broadcasts over [batch, groups, neurons_per_group, hidden]
        h = h + self.b1.unsqueeze(0).unsqueeze(2)

        # Activation: GELU (smooth, avoids dead neurons unlike ReLU)
        h = F.gelu(h)
        h = self.dropout(h)

        # Layer 2: [batch, groups, neurons_per_group, hidden] @ [groups, hidden, 1]
        out = torch.einsum("bgnd,gdo->bgno", h, self.w2)
        out = out + self.b2.unsqueeze(0).unsqueeze(2)
        # → [batch, groups, neurons_per_group, 1] → [batch, groups, neurons_per_group]
        out = out.squeeze(-1)

        # Reshape back: [batch, groups, neurons_per_group] → [batch, d_latent]
        nlm_output = out.reshape(B, self.d_latent)

        # Gated output: blend NLM dynamics with direct pass-through of latest input.
        # gate ∈ (0, 1) via sigmoid; starts at 0.5 so both contribute equally.
        alpha = torch.sigmoid(self.gate)  # [d_latent]
        latest_input = pre_activation_history[:, -1, :]  # [batch, d_latent]

        post_activations = alpha * torch.tanh(nlm_output) + (1 - alpha) * latest_input

        return post_activations

# ═════════════════════════════════════════════════════════════════════════
# matrix_stream.py
# ═════════════════════════════════════════════════════════════════════════

class MatrixResidualStream(nn.Module):
    """Matrix-valued residual stream with manifold-constrained mixing.

    Maintains n parallel streams of dimension d_latent. Each thought layer
    receives the mixed streams, processes them, and returns updated streams
    via post-mixing with residual gating.

    Args:
        d_latent: Width of each stream (matches the model's latent dimension).
        n_streams: Number of parallel streams. More streams = more capacity
            for tracking different aspects of the thought process, but also
            more memory. 4-8 is a good range for 200M-scale models.
        d_model: Text embedding dimension (for query projection output).
        gating: Gating parameterization:
            "diagonal" — each stream has a scalar gate (cheapest, works well)
            "sigmoid"  — sigmoid-gated diagonal (slightly more expressive)
    """

    def __init__(
        self,
        d_latent: int,
        n_streams: int = 4,
        d_model: int = 512,
        gating: str = "diagonal",
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_streams = n_streams
        self.d_model = d_model
        self.gating = gating

        # ── Pre-mixing: mix streams before the layer ────────────────────
        # Input-dependent mixing weights: a small MLP that takes the
        # collapsed stream vector and produces mixing logits.
        self.pre_mix_proj = nn.Linear(d_latent, n_streams * n_streams)
        nn.init.zeros_(self.pre_mix_proj.weight)
        nn.init.zeros_(self.pre_mix_proj.bias)
        # Zero init → initial pre-mixing is identity (softmax of zeros
        # gives uniform weights, but with the residual connection the
        # stream stays at its input value).

        # ── Post-mixing: mix streams after the layer ────────────────────
        self.post_mix_proj = nn.Linear(d_latent, n_streams * n_streams)
        nn.init.zeros_(self.post_mix_proj.weight)
        nn.init.zeros_(self.post_mix_proj.bias)

        # ── Residual gating: per-stream scalar gates ────────────────────
        # H_res: diagonal gating on the residual path.
        # Initialized at 0 → sigmoid(0) = 0.5 so residual and new
        # content contribute equally at init.
        self.res_gate = nn.Parameter(torch.zeros(n_streams))

        # ── Query collapse: project n streams → single query vector ─────
        # Replaces SynchronizationComputer. Collapses the multi-stream
        # state into a single query vector for cross-attention.
        # Attention over streams: query-dependent weighting of which
        # stream(s) contribute to the attention query.
        self.stream_query_weights = nn.Linear(d_latent, n_streams)
        self.query_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Stream initialization ───────────────────────────────────────
        # Learnable per-stream initial state (broadcast to batch/seq).
        self.stream_init = nn.Parameter(
            torch.zeros(n_streams, d_latent)
        )

        # ── Collapse: multi-stream → single latent for synapse/output ──
        # Attention-weighted collapse of streams into a single vector.
        self.collapse_weights = nn.Linear(d_latent, n_streams)
        self.collapse_norm = nn.LayerNorm(d_latent)

    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize the stream state for a new forward pass.

        Returns:
            state: [B, S, n_streams, d_latent]
        """
        # Broadcast learnable init to [B, S, n, D]
        state = self.stream_init.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1, -1
        ).clone().to(dtype=dtype)
        return state

    def mix_pre(
        self,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-layer mixing of streams.

        Takes the current multi-stream state and mixes across streams
        using input-dependent weights. This allows different thought
        layers to "see" different combinations of the stream content.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            mixed: [B, S, n_streams, d_latent]
        """
        B, S, N, D = stream.shape

        # Compute mixing weights from the mean stream state
        mean_state = stream.mean(dim=2)  # [B, S, D]
        mix_logits = self.pre_mix_proj(mean_state)  # [B, S, N*N]
        mix_logits = mix_logits.view(B, S, N, N)  # [B, S, N_out, N_in]

        # Softmax over input streams (convex combination ↔ Birkhoff)
        mix_weights = F.softmax(mix_logits, dim=-1)  # [B, S, N, N]

        # Apply mixing: each output stream is a weighted sum of input streams
        # [B, S, N_out, N_in] @ [B, S, N_in, D] → [B, S, N_out, D]
        mixed = torch.matmul(mix_weights, stream)

        return mixed

    def mix_post(
        self,
        stream: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Post-layer mixing with residual gating.

        After the thought layer processes the collapsed stream into a
        single-vector output, this method updates the multi-stream state
        by combining the layer output with the existing streams via
        learned gating.

        Args:
            stream: [B, S, n_streams, d_latent] — pre-layer stream state
            layer_output: [B, S, d_latent] — output from the thought layer

        Returns:
            updated: [B, S, n_streams, d_latent]
        """
        B, S, N, D = stream.shape

        # Expand layer output to all streams
        # [B, S, D] → [B, S, 1, D] → broadcast to [B, S, N, D]
        expanded = layer_output.unsqueeze(2).expand_as(stream)

        # Post-mixing weights (input-dependent)
        mean_state = stream.mean(dim=2)  # [B, S, D]
        mix_logits = self.post_mix_proj(mean_state)  # [B, S, N*N]
        mix_logits = mix_logits.view(B, S, N, N)
        mix_weights = F.softmax(mix_logits, dim=-1)

        # Mix the expanded output across streams
        new_content = torch.matmul(mix_weights, expanded)  # [B, S, N, D]

        # Residual gating: blend old stream with new content
        # gate ∈ (0, 1) via sigmoid; 0.5 at init (both contribute equally)
        gate = torch.sigmoid(self.res_gate)  # [N]
        gate = gate.view(1, 1, N, 1)  # broadcast shape

        updated = gate * new_content + (1 - gate) * stream

        return updated

    def to_query(
        self,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse multi-stream state into attention query.

        Replaces SynchronizationComputer.compute(). Uses attention-weighted
        collapse over streams to produce a single query vector per position.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            query: [B, S, d_model]
        """
        B, S, N, D = stream.shape

        # Compute stream attention weights
        mean_state = stream.mean(dim=2)  # [B, S, D]
        weights = self.stream_query_weights(mean_state)  # [B, S, N]
        weights = F.softmax(weights, dim=-1)  # [B, S, N]

        # Weighted sum over streams
        collapsed = torch.einsum('bsn,bsnd->bsd', weights, stream)  # [B, S, D]

        # Project to query space
        query = self.query_proj(collapsed)  # [B, S, d_model]
        return query

    def collapse(
        self,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse multi-stream state to single latent vector.

        Used for the synapse model input and output head. Attention-weighted
        combination of streams, normalized.

        Args:
            stream: [B, S, n_streams, d_latent]

        Returns:
            collapsed: [B, S, d_latent]
        """
        B, S, N, D = stream.shape

        mean_state = stream.mean(dim=2)
        weights = self.collapse_weights(mean_state)  # [B, S, N]
        weights = F.softmax(weights, dim=-1)

        collapsed = torch.einsum('bsn,bsnd->bsd', weights, stream)
        return self.collapse_norm(collapsed)

    def extra_repr(self) -> str:
        return (
            f"d_latent={self.d_latent}, n_streams={self.n_streams}, "
            f"d_model={self.d_model}, gating={self.gating}"
        )

# ═════════════════════════════════════════════════════════════════════════
# feec_integrator.py
# ═════════════════════════════════════════════════════════════════════════

class FEECIntegrator(nn.Module):
    """Mixed finite element integrator for the CTM thought loop.

    Maintains a dual state (position z, velocity J) and evolves them
    via a symplectic-like step that preserves discrete energy.

    Args:
        d_latent: Dimension of the latent state.
        n_layers: Number of thought layers (one dt per layer).
        dt_init: Initial step size (small for stability at init).
        damping_init: Initial damping coefficient γ.
            Higher damping → more energy dissipation → faster convergence
            but less expressive dynamics. The model learns to balance.
        clamp_dt: Upper bound on learnable dt to prevent instability.
    """

    def __init__(
        self,
        d_latent: int,
        n_layers: int,
        dt_init: float = 0.1,
        damping_init: float = 0.1,
        clamp_dt: float = 1.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_layers = n_layers
        self.clamp_dt = clamp_dt

        # Per-layer learnable step sizes. Initialized small so the
        # integrator starts near the identity (z_{t+1} ≈ z_t).
        # Using log-parameterization: dt = softplus(raw_dt) to keep dt > 0.
        self.raw_dt = nn.Parameter(
            torch.full((n_layers,), self._inv_softplus(dt_init))
        )

        # Global learnable damping coefficient γ.
        # Parameterized via sigmoid: γ = sigmoid(raw_damping) ∈ (0, 1).
        # This ensures γ stays in a reasonable range — too large and the
        # velocity collapses to zero (no dynamics), too small and energy
        # isn't dissipated (potential oscillation).
        self.raw_damping = nn.Parameter(
            torch.tensor(self._inv_sigmoid(damping_init))
        )

        # Force scaling: learnable per-layer scaling of the force field
        # to match the magnitude of the integrator's natural dynamics.
        # Initialized to 1.0 (identity scaling).
        self.force_scale = nn.Parameter(torch.ones(n_layers))

    @staticmethod
    def _inv_softplus(y: float) -> float:
        """Inverse of softplus: x such that softplus(x) = y."""
        import math
        if y > 20:
            return y
        return math.log(math.exp(y) - 1)

    @staticmethod
    def _inv_sigmoid(y: float) -> float:
        """Inverse of sigmoid: x such that sigmoid(x) = y."""
        import math
        return math.log(y / (1 - y))

    def get_dt(self, layer_idx: int) -> torch.Tensor:
        """Get the step size for a given layer, clamped for stability."""
        dt = F.softplus(self.raw_dt[layer_idx])
        return dt.clamp(max=self.clamp_dt)

    @property
    def damping(self) -> torch.Tensor:
        """Current damping coefficient γ ∈ (0, 1)."""
        return torch.sigmoid(self.raw_damping)

    def step(
        self,
        z: torch.Tensor,
        velocity: torch.Tensor,
        force: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one symplectic integration step.

        Symplectic Euler (velocity-first variant):
            J_{t+1} = (1 - γ·Δt) · J_t + Δt · s_l · F
            u_{t+1} = u_t + Δt · J_{t+1}

        The velocity-first variant has better energy preservation than
        position-first for dissipative systems (our case with γ > 0).

        Args:
            z: [B, S, d_latent] — current position (latent state).
            velocity: [B, S, d_latent] — current velocity.
            force: [B, S, d_latent] — force field from thought layer.
            layer_idx: which layer this step corresponds to.

        Returns:
            z_new: [B, S, d_latent] — updated position.
            velocity_new: [B, S, d_latent] — updated velocity.
        """
        dt = self.get_dt(layer_idx)
        gamma = self.damping
        scale = self.force_scale[layer_idx]

        # Velocity update (with damping)
        velocity_new = (1 - gamma * dt) * velocity + dt * scale * force

        # Position update (using NEW velocity — symplectic)
        z_new = z + dt * velocity_new

        return z_new, velocity_new

    def energy(
        self,
        z: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discrete energy E = ½|J|² + ½|z|².

        Used for monitoring/logging, NOT for the loss function directly.
        A well-behaved thought loop should show non-increasing energy
        after the initial transient.

        Args:
            z: [B, S, d_latent]
            velocity: [B, S, d_latent]

        Returns:
            Scalar energy, averaged over batch and sequence.
        """
        kinetic = 0.5 * (velocity ** 2).sum(dim=-1).mean()
        potential = 0.5 * (z ** 2).sum(dim=-1).mean()
        return kinetic + potential

    def energy_penalty(
        self,
        z: torch.Tensor,
        velocity: torch.Tensor,
        z_prev: torch.Tensor,
        velocity_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Penalty for energy growth between steps.

        Returns ReLU(E_new - E_old) — zero when energy decreases (good),
        positive when energy increases (penalized). Analogous to the
        mono_penalty in the temporal loss but for the dynamical system.

        Args:
            z, velocity: current state
            z_prev, velocity_prev: previous state

        Returns:
            Scalar penalty (≥ 0).
        """
        e_new = self.energy(z, velocity)
        e_old = self.energy(z_prev, velocity_prev)
        return F.relu(e_new - e_old)

    def extra_repr(self) -> str:
        dts = [self.get_dt(i).item() for i in range(self.n_layers)]
        return (
            f"n_layers={self.n_layers}, "
            f"damping={self.damping.item():.4f}, "
            f"dt=[{', '.join(f'{d:.3f}' for d in dts)}]"
        )

# ═════════════════════════════════════════════════════════════════════════
# thought_layer.py
# ═════════════════════════════════════════════════════════════════════════

class UNetSynapse(nn.Module):
    """
    1D U-Net applied across the latent feature dimension to allow 
    complex spatial information sharing between neurons.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        # Input shape: [BS, 1, in_features]
        self.down1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.GELU())
        self.pool1 = nn.MaxPool1d(2)
        
        self.down2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.GELU())
        self.pool2 = nn.MaxPool1d(2)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv_up1 = nn.Sequential(nn.Conv1d(32 + 16, 16, kernel_size=3, padding=1), nn.GELU())
        
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv_up2 = nn.Sequential(nn.Conv1d(16 + 1, 1, kernel_size=3, padding=1), nn.GELU())
        
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BS, in_features]
        x_in = x.unsqueeze(1) # [BS, 1, in_features]
        
        d1 = self.down1(x_in)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        u1 = self.up1(p2)
        if u1.size(2) != d1.size(2):
            u1 = F.interpolate(u1, size=d1.size(2))
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up1(u1)
        if u2.size(2) != x_in.size(2):
            u2 = F.interpolate(u2, size=x_in.size(2))
        u2 = torch.cat([u2, x_in], dim=1)
        u2 = self.conv_up2(u2)
        
        out = u2.squeeze(1) # [BS, in_features]
        return self.proj(out)


class ThoughtLayer(nn.Module):
    """
    A single thought processing block with per-position latent states
    and causally-masked cross-attention.

    Args:
        d_latent: Internal neuron count (state dimension).
        d_model: Text embedding dimension (cross-attention KV dimension).
        n_heads: Number of attention heads.
        nlm_hidden_dim: Hidden dimension per neuron MLP.
        history_len: FIFO buffer depth.
        nlm_groups: Number of neuron groups for NLMs.
        sync_method: Synchronization computation method.
        sync_rank: Rank for low-rank sync.
        dropout: Dropout rate.
        engram_enabled: If True, this layer fuses Engram memory into attn_out.
            The Engram K/V are projected by the parent model (not here) and
            passed in via `engram_kv` at forward time.
        engram_conv_kernel/dilation/use_conv: see EngramGate.
        use_matrix_streams: If True, use MatrixResidualStream instead of NLM+Sync.
        n_streams: Number of parallel streams (when use_matrix_streams=True).
        stream_gating: Gating parameterization for matrix streams.
        use_dssa: If True, use Dual-Space Sparse Attention.
        dssa_n_partitions: SSE state partitions.
        dssa_top_k: Active SSE partitions.
        dssa_block_size: MoBA block size.
        dssa_top_k_blocks: MoBA blocks per query.
        use_triton_attention: Use Triton-accelerated attention.
    """

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        n_heads: int,
        nlm_hidden_dim: int,
        history_len: int,
        nlm_groups: int = 1,
        sync_method: str = "diag_summary",
        sync_rank: int = 32,
        dropout: float = 0.1,
        engram_enabled: bool = False,
        engram_use_conv: bool = True,
        engram_conv_kernel: int = 4,
        engram_conv_dilation: int = 3,
        # v2 features
        use_matrix_streams: bool = False,
        n_streams: int = 4,
        stream_gating: str = "diagonal",
        use_dssa: bool = False,
        dssa_n_partitions: int = 32,
        dssa_top_k: int = 8,
        dssa_block_size: int = 64,
        dssa_top_k_blocks: int = 4,
        use_triton_attention: bool = False,
        synapse_type: str = "mlp",
        # ── Biological extensions (opt-in) ──────────────────────────
        use_hebbian: bool = False,
        hebbian_bottleneck_dim: int = 64,
        hebbian_decay_init: float = 0.9,
        hebbian_lr_init: float = 0.1,
        hebbian_gate_init: float = -3.0,
        hebbian_force_gate: float | None = None,
        hebbian_update_rule: str = "outer_product",
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_matrix_streams = use_matrix_streams
        self.use_dssa = use_dssa
        self.use_triton_attention = use_triton_attention

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads

        # ── v2: Matrix-Valued Residual Stream (replaces Sync+NLM) ────────
        if use_matrix_streams:
            self.stream = MatrixResidualStream(
                d_latent=d_latent,
                n_streams=n_streams,
                d_model=d_model,
                gating=stream_gating,
            )
            # No sync_computer, nlm, or memory needed
            self.sync_computer = None
            self.nlm = None
            self.memory = None
        else:
            self.stream = None
            # ── Synchronization Computer (v1 path) ──────────────────────
            self.sync_computer = SynchronizationComputer(
                d_latent=d_latent,
                history_len=history_len,
                method=sync_method,
                rank=sync_rank,
            )
            sync_dim = self.sync_computer.output_dim

            # ── Query Generation from Synchronization State (v1) ────────
            self.query_proj = nn.Sequential(
                nn.Linear(sync_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

            # ── Neuron-Level Models (v1 path) ───────────────────────────
            self.nlm = NeuronLevelModels(
                d_latent=d_latent,
                history_len=history_len,
                nlm_hidden_dim=nlm_hidden_dim,
                nlm_groups=nlm_groups,
                dropout=dropout,
            )

            # ── Memory Buffers (v1 path) ────────────────────────────────
            self.memory = TemporalMemory(d_latent=d_latent, history_len=history_len)

        # ── v2: DSSA Cross-Attention (replaces manual Q/K/V) ────────────
        if use_dssa:
            self.dssa = DualSpaceSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_partitions=dssa_n_partitions,
                sse_top_k=dssa_top_k,
                block_size=dssa_block_size,
                moba_top_k_blocks=dssa_top_k_blocks,
                dropout=dropout,
            )
            # Don't need the manual attention components
            self.q_norm = None
            self.k_proj = None
            self.v_proj = None
            self.attn_out_proj = None
            self.attn_dropout = None
        else:
            self.dssa = None
            # ── Standard Cross-Attention ────────────────────────────────
            self.q_norm = nn.LayerNorm(d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.attn_out_proj = nn.Linear(d_model, d_model)
            self.attn_dropout = nn.Dropout(dropout)

        # ── Synapse Model ───────────────────────────────────────────────
        self.synapse_norm = nn.LayerNorm(d_model + d_latent)
        if synapse_type == "unet":
            self.synapse = UNetSynapse(d_model + d_latent, d_latent, dropout=dropout)
        else:
            self.synapse = nn.Sequential(
                nn.Linear(d_model + d_latent, d_latent * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_latent * 2, d_latent),
            )
        self.synapse_gate = nn.Linear(d_model + d_latent, d_latent)

        # ── Post-NLM Layer Norm ─────────────────────────────────────────
        self.post_norm = nn.LayerNorm(d_latent)

        # ── Hebbian Fast-Weight Synapse (optional) ──────────────────────
        # Augments the synapse output with a content-addressable readout
        # from a per-(B, S) fast-weight matrix updated online during the
        # thought loop. See biological.HebbianSynapse for details.
        self.use_hebbian = use_hebbian
        if use_hebbian:
            self.hebbian = HebbianSynapse(
                d_latent=d_latent,
                d_model=d_model,
                bottleneck_dim=hebbian_bottleneck_dim,
                decay_init=hebbian_decay_init,
                lr_init=hebbian_lr_init,
                gate_init=hebbian_gate_init,
                force_gate=hebbian_force_gate,
                update_rule=hebbian_update_rule,
            )
        else:
            self.hebbian = None

        # ── Engram Gate (optional) ──────────────────────────────────────
        if engram_enabled:
            self.engram_gate = EngramGate(
                d_query=d_model,
                d_out=d_model,
                kernel_size=engram_conv_kernel,
                dilation=engram_conv_dilation,
                use_conv=engram_use_conv,
            )
        else:
            self.engram_gate = None

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize memory buffers for a new sequence/batch."""
        if self.memory is not None:
            self.memory.reset(batch_size, device, dtype)

    def _standard_attention(
        self,
        queries: torch.Tensor,
        text_keys: torch.Tensor,
        text_values: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard O(N²) cross-attention with causal masking.

        Args:
            queries: [B, S, d_model]
            text_keys: [B, S, d_model]
            text_values: [B, S, d_model]
            key_padding_mask: [B, S] or None

        Returns:
            attn_out: [B, S, d_model]
        """
        B, S, D = queries.shape

        q = self.q_norm(queries)
        k = self.k_proj(text_keys)
        v = self.v_proj(text_values)

        # Check if we should use accelerated attention
        if self.use_triton_attention:
            q_reshaped = q.view(B, S, self.n_heads, self.head_dim)
            k_reshaped = k.view(B, S, self.n_heads, self.head_dim)
            v_reshaped = v.view(B, S, self.n_heads, self.head_dim)
            attn_out = accelerated_causal_attention(
                q_reshaped, k_reshaped, v_reshaped,
                n_heads=self.n_heads,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                training=self.training,
                use_triton=True,
            )
            attn_out = self.attn_out_proj(attn_out)
            return attn_out

        # Standard path
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.attn_out_proj(attn_out)

        return attn_out

    def forward(
        self,
        text_keys: torch.Tensor,
        text_values: torch.Tensor,
        prev_state: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        engram_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        stream_state: torch.Tensor | None = None,
        hebbian_state: torch.Tensor | None = None,
        hebbian_lr_modulator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Execute one thought step with per-position states and causal masking.

        Args:
            text_keys:   [batch, seq_len, d_model] — text embeddings for K projection.
            text_values: [batch, seq_len, d_model] — text embeddings for V projection.
            prev_state:  [batch, seq_len, d_latent] — per-position neuron states z_{t-1}.
            key_padding_mask: [batch, seq_len] — True for padded positions.
            engram_kv: optional (k, v) tuple, each [batch, seq_len, d_model].
            stream_state: [batch, seq_len, n_streams, d_latent] — matrix stream state (v2).
            hebbian_state: [batch, seq_len, m, n] — Hebbian fast-weight matrix, or None.
            hebbian_lr_modulator: [batch, seq_len] or None — per-position
                multiplier applied to Hebbian outer-product update. Supplied
                by the caller as (1 + alpha * uncertainty) from the
                previous tick's certainty. None → uniform lr.

        Returns:
            new_state:   [batch, seq_len, d_latent] — updated per-position states z_t.
            sync_repr:   [batch*seq_len, sync_dim] — sync repr (v1) or dummy (v2).
            new_stream_state: [batch, seq_len, n_streams, d_latent] or None.
            new_hebbian_state: [batch, seq_len, m, n] or None.
            hebbian_lr_eff: scalar tensor (mean effective lr post-modulation),
                or None when Hebbian path not active.
        """
        B, S, D = text_keys.shape
        BS = B * S

        # ── Step 1: Query Generation ────────────────────────────────────
        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            # v2 path: query from matrix stream
            mixed_stream = self.stream.mix_pre(stream_state)
            queries = self.stream.to_query(mixed_stream)  # [B, S, d_model]
            sync_repr = torch.zeros(BS, 1, device=text_keys.device, dtype=text_keys.dtype)
        else:
            # v1 path: query from synchronization
            prev_flat = prev_state.reshape(BS, -1)
            post_history = self.memory.get_post_history()
            sync_repr = self.sync_computer.compute(post_history)
            queries_flat = self.query_proj(sync_repr)
            queries = queries_flat.reshape(B, S, D)

        # ── Step 2: Cross-Attention ─────────────────────────────────────
        if self.use_dssa and self.dssa is not None:
            # v2 DSSA path
            attn_out = self.dssa(queries, text_keys, causal=True)
        else:
            # Standard attention (possibly Triton-accelerated)
            attn_out = self._standard_attention(
                queries, text_keys, text_values, key_padding_mask
            )

        # ── Step 2.5: Engram Fusion (optional) ──────────────────────────
        if self.engram_gate is not None and engram_kv is not None:
            engram_k, engram_v = engram_kv
            engram_out = self.engram_gate(attn_out, engram_k, engram_v)
            attn_out = attn_out + engram_out

        # Flatten attention output
        attn_flat = attn_out.reshape(BS, D)

        # ── Step 3: Synapse Model ───────────────────────────────────────
        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            collapsed = self.stream.collapse(stream_state)  # [B, S, d_latent]
            prev_flat = collapsed.reshape(BS, -1)
        else:
            prev_flat = prev_state.reshape(BS, -1)

        synapse_input = torch.cat([attn_flat, prev_flat], dim=1)
        synapse_input = self.synapse_norm(synapse_input)

        gate = torch.sigmoid(self.synapse_gate(synapse_input))
        candidate = self.synapse(synapse_input)
        pre_activations = gate * candidate + (1 - gate) * prev_flat

        # ── Hebbian Fast-Weight Readout + Update ────────────────────────
        # Reads from M·a, adds (gated) to the synapse pre_activations,
        # then updates M with the outer product of the latent state and
        # the attention output. M persists across ticks within a forward
        # pass; the caller threads it via `hebbian_state`.
        new_hebbian_state = hebbian_state  # passthrough by default
        hebbian_lr_eff = None
        if self.use_hebbian and self.hebbian is not None and hebbian_state is not None:
            # Use post-synapse latents as the "z" of the outer product so
            # the Hebbian trace encodes the current refined state, not
            # the stale prev_state. This is the working-memory write rule:
            # bind the current attention key (a) to the current latent (z).
            pre_act_2d = pre_activations.reshape(B, S, -1)
            heb_readout, new_hebbian_state, hebbian_lr_eff = self.hebbian(
                pre_act_2d, attn_out, hebbian_state,
                lr_modulator=hebbian_lr_modulator,
            )
            pre_activations = pre_activations + heb_readout.reshape(BS, -1)

        # ── Step 4: Post-processing ─────────────────────────────────────
        new_stream_state = None

        if self.use_matrix_streams and self.stream is not None and stream_state is not None:
            # v2 path: update matrix streams (replaces NLM)
            post_activations = self.post_norm(pre_activations)
            new_state_flat = post_activations
            # Update stream state
            layer_output = post_activations.reshape(B, S, -1)
            new_stream_state = self.stream.mix_post(stream_state, layer_output)
        else:
            # v1 path: NLM processing
            self.memory.push_pre(pre_activations)
            pre_history = self.memory.get_pre_history()
            post_activations = self.nlm(pre_history)
            self.memory.push_post(post_activations)
            new_state_flat = self.post_norm(post_activations)

            # Recompute sync for output
            updated_post_history = self.memory.get_post_history()
            sync_repr = self.sync_computer.compute(updated_post_history)

        # Reshape back to [B, S, d_latent]
        new_state = new_state_flat.reshape(B, S, -1)

        return new_state, sync_repr, new_stream_state, new_hebbian_state, hebbian_lr_eff

# ═════════════════════════════════════════════════════════════════════════
# model.py
# ═════════════════════════════════════════════════════════════════════════

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
                # Biological extensions (opt-in)
                use_hebbian=config.use_hebbian_synapse,
                hebbian_bottleneck_dim=config.hebbian_bottleneck_dim,
                hebbian_decay_init=config.hebbian_decay_init,
                hebbian_lr_init=config.hebbian_lr_init,
                hebbian_gate_init=config.hebbian_gate_init,
                hebbian_force_gate=getattr(config, "hebbian_force_gate", None),
                hebbian_update_rule=getattr(config, "hebbian_update_rule", "outer_product"),
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
        # Three configurations, mutually exclusive:
        #   1. per_tick_heads:    8 separate adapters, 1 shared head
        #                         (1.07B params at V=131072 — old default)
        #   2. use_shared_head_film: 1 shared adapter, 1 shared head, with
        #                         per-tick FiLM modulation between them
        #                         (135M params + 16K FiLM — recommended)
        #   3. neither (legacy):  1 shared output_proj used at every tick,
        #                         no tick conditioning — collapses ticks
        if config.per_tick_heads and config.use_shared_head_film:
            raise ValueError(
                "per_tick_heads and use_shared_head_film are mutually "
                "exclusive. Choose one (use_shared_head_film is the "
                "recommended successor)."
            )

        if config.use_shared_head_film:
            T = config.max_thought_steps
            # Shared adapter: latent + text → d_model with norm.
            self.shared_adapter = nn.Sequential(
                nn.Linear(config.d_latent + config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            # Per-tick FiLM parameters. Initialize gamma=1, beta=0 so the
            # initial behavior matches the un-modulated shared head; the
            # network learns to differentiate ticks during training.
            self.film_gamma = nn.Parameter(torch.ones(T, config.d_model))
            self.film_beta = nn.Parameter(torch.zeros(T, config.d_model))
            # Shared LM head — single [d_model, vocab_size] linear.
            # When tie_embeddings is on, the weight is shared with
            # token_embedding (set up further below).
            self.lm_head = nn.Linear(config.d_model, config.vocab_size,
                                     bias=False)
            self.tick_adapters = None
            self.output_proj = None
        elif config.per_tick_heads:
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
            self.shared_adapter = None
            self.film_gamma = None
            self.film_beta = None
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(config.d_latent + config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.vocab_size),
            )
            self.tick_adapters = None
            self.lm_head = None
            self.shared_adapter = None
            self.film_gamma = None
            self.film_beta = None

        # ── Tied Embeddings ─────────────────────────────────────────────
        # Share weights between token_embedding and lm_head. Only meaningful
        # when use_shared_head_film=True (per_tick_heads has its own
        # lm_head we could tie, but the combination is not recommended;
        # with the legacy output_proj path there is no separable head to
        # tie to). The Embedding stores [V, d_model]; nn.Linear stores
        # [V, in_features=d_model] (same shape). Direct assignment of
        # the underlying parameter is the canonical way to tie.
        if config.tie_embeddings:
            if not config.use_shared_head_film:
                raise ValueError(
                    "tie_embeddings requires use_shared_head_film=True "
                    "(no canonical head to tie to otherwise)."
                )
            if self.token_embedding is None:
                raise ValueError(
                    "tie_embeddings requires a token_embedding (cannot "
                    "tie to a feature encoder)."
                )
            # Share the underlying Parameter object. Both modules now point
            # to the same memory; gradients accumulate correctly via
            # autograd's standard handling of shared parameters.
            self.lm_head.weight = self.token_embedding.weight

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

        # ── Cerebellar Readout (Predictive Coding) ──────────────────────
        # Per-tick predictor of the chosen target. See
        # biological.CerebellarReadout for details. Only constructed if
        # the config flag is on; when off, no parameters are added.
        if config.use_predictive_coding:
            pc_hidden = config.pc_hidden_dim if config.pc_hidden_dim > 0 else None
            # target_dim depends on the target mode:
            #   "teacher"    → teacher_d_model (predict teacher hidden state)
            #   "next_tick"  → d_latent       (predict the student's own next-tick z)
            #   "final_tick" → d_latent       (predict the student's own final-tick z)
            pc_target = getattr(config, "pc_target", "teacher")
            if pc_target == "teacher":
                target_dim = config.teacher_d_model
            elif pc_target in ("next_tick", "final_tick"):
                target_dim = config.d_latent
            else:
                raise ValueError(
                    f"Unknown pc_target={pc_target!r}. Must be one of "
                    f"'teacher', 'next_tick', 'final_tick'."
                )
            self.cerebellar_readout = CerebellarReadout(
                d_latent=config.d_latent,
                target_dim=target_dim,
                hidden_dim=pc_hidden,
                normalize=config.pc_normalize,
                loss_type=config.pc_loss_type,
                dropout=config.pc_dropout,
            )
        else:
            self.cerebellar_readout = None

        # Velocity initial state (for FEEC)
        if config.use_feec:
            self.velocity_0 = nn.Parameter(torch.zeros(config.d_latent))
        else:
            self.velocity_0 = None

        # ── CUDA Graph wrapper (v2) ─────────────────────────────────────
        if config.use_cuda_graphs:
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
        """Standard transformer weight initialization.

        Note: LayerNorm with elementwise_affine=False has weight/bias = None,
        so we must guard against that. Several modules in this codebase
        use non-affine LayerNorm intentionally (distill_*_norm,
        biological.HebbianSynapse.readout_norm, biological.CerebellarReadout's
        student/teacher norms) — without these guards, .apply(_init_weights)
        crashes with "NoneType has no attribute fill_".
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
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

        if self.shared_adapter is not None:
            # Shared head + per-tick FiLM modulation.
            # adapter: [B, S, d_latent + d_model] → [B, S, d_model]
            x = self.shared_adapter(combined)
            # FiLM: y = gamma_t * x + beta_t. Per-tick scalars broadcast
            # over the [B, S] dimensions.
            gamma = self.film_gamma[tick]   # [d_model]
            beta = self.film_beta[tick]     # [d_model]
            x = x * gamma + beta
            return self.lm_head(x)

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
        key_padding_mask: torch.Tensor | None = None,
        engram_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        pre_states: list[torch.Tensor] | None = None,
        post_states: list[torch.Tensor] | None = None,
        velocity: torch.Tensor | None = None,
        stream_states: list[torch.Tensor] | None = None,
        targets: torch.Tensor | None = None,
        teacher_log_probs: torch.Tensor | None = None,
        teacher_top_indices: torch.Tensor | None = None,
        distill_temp: float = 1.0,
        hebbian_states: list[torch.Tensor] | None = None,
        teacher_z_for_pc: torch.Tensor | None = None,
        T_total: int = 1,
        prev_certainty: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], torch.Tensor | None, list[torch.Tensor], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        """
        Execute all layers in the sequence for one thought tick.

        Returns:
            z_new: [B, S, d_latent]
            logits_t: [B, S, V] or None (if training)
            new_pre_states, new_post_states: v1 memory
            velocity_new: FEEC velocity
            new_stream_states: v2 streams
            ce_loss_t: Scalar or per-token loss
            kl_loss_t: Scalar KL loss
            certainty_t: [B, S] certainty score
            new_hebbian_states: list of Hebbian fast-weight matrices (one per layer)
            pc_loss_t: Predictive-coding auxiliary loss for this tick (scalar) or None
            hebbian_lr_eff_t: Mean effective Hebbian lr across positions (for logging), or None
        """
        B, S = z.shape[:2]
        layers = self._get_layers_sequence()
        layer_outputs = []
        new_stream_states = []
        new_hebbian_states = []
        hebbian_lr_effs = []  # per-layer effective lr, for diagnostics

        velocity_new = velocity
        if velocity is not None:
            velocity_new = velocity.clone()

        # ── Compute per-position Hebbian lr modulator from prev certainty ──
        # Biological motivation: locus coeruleus releases norepinephrine in
        # response to surprise, and NE gates plasticity for subsequent
        # events. Functionally: uncertain positions write more aggressively
        # to fast-weight memory than confident ones.
        #
        # Modulator shape: [B, S], value (1 + alpha * uncertainty).
        # alpha=0 disables (uniform lr). prev_certainty is None at tick 0
        # → also no modulation. Computed under no_grad: this is a routing
        # signal, not a learnable path, so we don't want grad flowing back
        # through the certainty computation.
        alpha = getattr(self.config, "hebbian_cert_lr_alpha", 0.0)
        hebbian_lr_modulator = None
        if (alpha > 0
                and prev_certainty is not None
                and self.config.use_hebbian_synapse):
            with torch.no_grad():
                # certainty is in [0, 1] for dynamic_aggregate; we use
                # uncertainty = clamp(1 - certainty, 0, 1) to be safe
                # against non-aggregate temporal_loss_types where
                # certainty may be negative (entropy).
                uncertainty = (1.0 - prev_certainty).clamp(min=0.0, max=1.0)
                hebbian_lr_modulator = 1.0 + alpha * uncertainty
                hebbian_lr_modulator = hebbian_lr_modulator.to(z.dtype)

        for l_idx, layer in enumerate(layers):
            # Attention Residuals (Kimi AttnRes)
            # Stack all previous outputs (at least the input z)
            V = torch.stack(layer_outputs + [z], dim=0)
            
            V_norm = V.to(self.attn_res_norm.weight.dtype)
            K = self.attn_res_norm(V_norm) # [N, B, S, D]
            w_l = self.attn_res_queries[l_idx] if l_idx < len(self.attn_res_queries) else self.attn_res_queries[-1]
            scores = torch.einsum('d,nbsd->nbs', w_l, K) # [N, B, S]
            attn_weights = F.softmax(scores, dim=0)
            z_in = torch.einsum('nbs,nbsd->bsd', attn_weights, V) # [B, S, D]

            # Get stream state for this layer
            layer_stream = None
            if self.config.use_matrix_streams and stream_states is not None and l_idx < len(stream_states):
                layer_stream = stream_states[l_idx]

            # Get Hebbian state for this layer
            layer_heb = None
            if hebbian_states is not None and l_idx < len(hebbian_states):
                layer_heb = hebbian_states[l_idx]

            z_out, _sync_repr, new_layer_stream, new_layer_heb, layer_lr_eff = layer(
                text_emb,
                text_emb,
                z_in,
                key_padding_mask,
                engram_kv=engram_kv.get(l_idx) if engram_kv else None,
                stream_state=layer_stream,
                hebbian_state=layer_heb,
                hebbian_lr_modulator=hebbian_lr_modulator,
            )
            if layer_lr_eff is not None:
                hebbian_lr_effs.append(layer_lr_eff)

            if self.config.use_matrix_streams and new_layer_stream is not None:
                    new_stream_states.append(new_layer_stream)
            # Keep new_hebbian_states parallel to `layers`: append None
            # for layers without a Hebbian synapse so indexing by l_idx
            # stays stable across calls. Today the flag is global so
            # either every layer has hebbian or none does, but the
            # robust pattern costs nothing and matches stream_states.
            new_hebbian_states.append(new_layer_heb)

            # ── FEEC Integration ────────────────────────────────────────
            if self.feec is not None and velocity_new is not None:
                force = z_out - z_in  # Force field = layer's contribution
                z_out, velocity_new = self.feec.step(
                    z_in, velocity_new, force, layer_idx=l_idx
                )

            layer_outputs.append(z_out)

        z_new = layer_outputs[-1]
        
        # ── Output + Loss + Certainty (In-Loop) ─────────────────────────
        logits_t = self._output_logits(z_new, text_emb, tick=t)
        
        ce_loss_t = None
        kl_loss_t = None
        certainty_t = None
        
        # Certainty (entropy-based)
        with torch.no_grad():
            probs_t = F.softmax(logits_t, dim=-1)
            entropy_t = -(probs_t * (probs_t + 1e-10).log()).sum(dim=-1)
            if self.config.temporal_loss_type == "dynamic_aggregate":
                max_entropy = math.log(self.config.vocab_size)
                certainty_t = 1.0 - (entropy_t / max_entropy)
            else:
                certainty_t = -entropy_t
        
        if targets is not None:
            V_size = self.config.vocab_size
            logits_flat = logits_t.reshape(-1, V_size)
            targets_flat = targets.reshape(-1)
            
            # CE Loss
            if self.config.temporal_loss_type == "dynamic_aggregate":
                ce_loss_t = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            else:
                ce_loss_t = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
                
            # KL Distillation
            if teacher_log_probs is not None:
                student_logits_flat = logits_flat.float()
                if teacher_top_indices is not None:
                    student_top_logits = student_logits_flat.gather(-1, teacher_top_indices)
                    student_log_probs = F.log_softmax(student_top_logits / distill_temp, dim=-1)
                else:
                    student_log_probs = F.log_softmax(student_logits_flat / distill_temp, dim=-1)
                
                kl_t = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="batchmean",
                    log_target=True,
                ) * (distill_temp * distill_temp)
                kl_loss_t = kl_t
            
            # If we are training and computed losses, we can drop logits_t to save memory.
            if self.training:
                logits_t = None

        # Read out v1 memory state
        new_pre_states = []
        new_post_states = []
        if not self.config.use_matrix_streams:
            for layer in layers:
                if layer.memory is not None:
                    new_pre_states.append(layer.memory.pre_history)
                    new_post_states.append(layer.memory.post_history)

        # ── Predictive-Coding auxiliary loss (per-tick) ─────────────────
        # The cerebellar readout predicts the teacher hidden state from
        # the current student latent. The error is a local, per-tick
        # signal that gives gradient at every tick rather than only at
        # the final-tick LM head. See biological.CerebellarReadout for
        # detail. Gated by pc_tick_aggregation: "all" applies at every
        # tick, "early" only for the first half, "last" only at the end.
        pc_loss_t = None
        if (
            self.cerebellar_readout is not None
            and teacher_z_for_pc is not None
        ):
            agg = getattr(self.config, "pc_tick_aggregation", "all")
            apply_pc = (
                agg == "all"
                or (agg == "early" and t < max(T_total // 2, 1))
                or (agg == "last" and t == T_total - 1)
            )
            if apply_pc:
                # z_new and teacher_z_for_pc should both be [B, S, *].
                # If teacher_z_for_pc is in d_latent space (because the
                # cache was projected upstream), we still try the predict
                # — the predictor maps d_latent → teacher_d_model, so the
                # shapes must match teacher_d_model on the target side.
                # If they don't, the user has misconfigured the cache;
                # skip silently to avoid crashing in-loop.
                if teacher_z_for_pc.shape[-1] == self.cerebellar_readout.teacher_d_model:
                    pc_loss_t = self.cerebellar_readout(z_new, teacher_z_for_pc)

        # Mean effective Hebbian lr for diagnostics (across all layers
        # with an active Hebbian module). None when Hebbian is off.
        hebbian_lr_eff_t = None
        if hebbian_lr_effs:
            hebbian_lr_eff_t = torch.stack(hebbian_lr_effs).mean()

        return z_new, logits_t, new_pre_states, new_post_states, velocity_new, new_stream_states, ce_loss_t, kl_loss_t, certainty_t, new_hebbian_states, pc_loss_t, hebbian_lr_eff_t

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        max_thought_steps: int | None = None,
        teacher_logits: torch.Tensor | None = None,
        teacher_z: torch.Tensor | None = None,
        cached_top_indices: torch.Tensor | None = None,
        cached_top_values: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass with per-position latent states and causal masking.

        Distillation paths (mutually exclusive — supply one OR the other,
        not both):
          - teacher_logits:  full [B, S, V] logits from a live teacher.
              The model computes top-K internally. Used by the legacy
              online-teacher path.
          - cached_top_indices, cached_top_values:
              precomputed top-K indices [B, S, K] (long) and the
              corresponding RAW logit values [B, S, K] (float). Used by
              the offline teacher cache path (scripts/cache_teacher_logits.py).
              When supplied, the model skips the topk computation and
              feeds these straight into the KL loss. cuda:1 can sit
              empty.
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
            # v1: reset FIFO memory buffers. MUST be paired with
            # use_matrix_streams=False (else the NLM path's get_post_history()
            # returns None and ThoughtLayer.forward crashes). This is
            # orthogonal to use_hebbian_synapse — the Hebbian fast-weights
            # are an *additional* working-memory mechanism that augments
            # whichever residual-stream path is active.
            for layer in layers:
                if layer.memory is not None:
                    layer.reset_memory(BS, device, dtype)

        # Hebbian fast-weight states (one matrix per layer, per (B, S))
        # Initialised to zero; updated in-loop by each ThoughtLayer. When
        # use_hebbian_synapse is off, this list is None and the layers
        # short-circuit the read/update path entirely.
        hebbian_states = None
        if self.config.use_hebbian_synapse:
            hebbian_states = []
            for layer in layers:
                if layer.hebbian is not None:
                    hebbian_states.append(
                        layer.hebbian.init_state(B, S, device, dtype)
                    )
                else:
                    hebbian_states.append(None)

        # ── Distillation Prep ───────────────────────────────────────────
        teacher_log_probs = None
        teacher_top_indices = None
        distill_temp = float(self.config.distill_temperature)
        if self.config.use_distillation and targets is not None:
            if teacher_logits is not None or cached_top_indices is not None:
                if cached_top_indices is not None:
                    B_, S_, K_ = cached_top_indices.shape
                    teacher_top_indices = cached_top_indices.reshape(B_ * S_, K_).long()
                    cached_values_flat = cached_top_values.reshape(B_ * S_, K_)
                    teacher_log_probs = F.log_softmax(cached_values_flat.float() / distill_temp, dim=-1)
                else:
                    B_, S_, V_ = teacher_logits.shape
                    teacher_logits_flat = teacher_logits.reshape(B_ * S_, V_)
                    top_k = int(getattr(self.config, "distill_top_k", 0) or 0)
                    if top_k > 0 and top_k < V_:
                        teacher_top_logits, teacher_top_indices = teacher_logits_flat.topk(top_k, dim=-1)
                        teacher_log_probs = F.log_softmax(teacher_top_logits / distill_temp, dim=-1)
                    else:
                        teacher_log_probs = F.log_softmax(teacher_logits_flat / distill_temp, dim=-1)

        # ── Thought Loop ────────────────────────────────────────────────
        all_logits = []
        all_certainties = []
        energy_penalties = []
        per_tick_ce_losses = []
        per_tick_kl_losses = []
        per_tick_pc_losses = []

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

        # Teacher z to pass into the per-tick PC predictor (teacher mode
        # only). For intrinsic PC targets (next_tick / final_tick), the
        # cerebellar loss is computed AFTER the thought loop ends — see
        # below — because we need future z's that aren't available yet
        # at tick t. Setting this to None for intrinsic modes also makes
        # the in-`_thought_step` PC branch a no-op there.
        pc_target_mode = getattr(self.config, "pc_target", "teacher")
        teacher_z_for_pc = (
            teacher_z if (self.cerebellar_readout is not None
                          and pc_target_mode == "teacher")
            else None
        )

        # For intrinsic-PC modes we cache every tick's z (detached for
        # the target side, kept attached for the predictor side). One
        # B·S·d_latent float32 tensor per tick — ~4 MB at B=1, S=512,
        # d_latent=1024, T=8. Negligible vs. activation memory for the
        # main loop.
        intrinsic_pc_active = (
            self.cerebellar_readout is not None
            and pc_target_mode in ("next_tick", "final_tick")
        )
        intrinsic_pc_z_history: list[torch.Tensor] = []

        z_prev = z
        velocity_prev = velocity
        # prev_certainty carries last tick's per-position certainty into
        # the next tick's _thought_step so the Hebbian path can compute
        # a per-position lr modulator. Starts as None at tick 0.
        prev_certainty: torch.Tensor | None = None
        per_tick_hebbian_lr: list[torch.Tensor] = []
        # Per-tick FEEC energy (scalar tensors) for Prospective Configuration.
        # Computed under no_grad — energy is a stability *signal* for
        # weighting the per-tick loss, not a learnable quantity itself.
        # Letting gradient flow through the weight would couple the loss
        # weight to whatever made z and velocity small, which is the
        # opposite of what we want.
        prospective_active = (
            getattr(self.config, "use_prospective_config", False)
            and self.feec is not None
        )
        per_tick_energy: list[torch.Tensor] = []

        for t in range(T):
            if use_per_step_ckpt:
                result = torch_checkpoint.checkpoint(
                    self._thought_step,
                    z, t, text_emb, key_padding_mask, engram_kv,
                    pre_states, post_states, velocity, stream_states,
                    targets, teacher_log_probs, teacher_top_indices, distill_temp,
                    hebbian_states, teacher_z_for_pc, T, prev_certainty,
                    use_reentrant=False,
                )
            else:
                result = self._thought_step(
                    z, t, text_emb, key_padding_mask, engram_kv,
                    pre_states, post_states, velocity, stream_states,
                    targets, teacher_log_probs, teacher_top_indices, distill_temp,
                    hebbian_states, teacher_z_for_pc, T, prev_certainty,
                )

            (z, logits_t, pre_states, post_states, velocity, stream_states,
             ce_loss_t, kl_loss_t, certainty_t, new_hebbian_states, pc_loss_t,
             hebbian_lr_eff_t) = result

            # Per-tick FEEC energy capture for Prospective Configuration.
            # We compute after this tick's state update — so per_tick_energy[t]
            # = E(z_after_tick_t). ΔE_t is then |E_t - E_{t-1}|.
            if prospective_active and velocity is not None:
                with torch.no_grad():
                    per_tick_energy.append(self.feec.energy(z, velocity))

            # Carry certainty forward to next tick's Hebbian lr modulator.
            # We carry it OUTSIDE the checkpointed region (this var lives
            # in the Python loop scope) so it doesn't bloat the saved
            # activations of the checkpointed function.
            if certainty_t is not None:
                prev_certainty = certainty_t

            # Collect z for intrinsic PC. We keep the live tensor (not
            # detached) so gradients flow into the layers that produced
            # it. The *target* side is what gets detached, applied later.
            if intrinsic_pc_active:
                intrinsic_pc_z_history.append(z)

            # Carry forward Hebbian state if any layer updated it.
            if new_hebbian_states and any(h is not None for h in new_hebbian_states):
                hebbian_states = new_hebbian_states

            # Track effective Hebbian lr for logging diagnostics.
            if hebbian_lr_eff_t is not None:
                per_tick_hebbian_lr.append(hebbian_lr_eff_t.detach())

            if ce_loss_t is not None:
                per_tick_ce_losses.append(ce_loss_t)
            if kl_loss_t is not None:
                per_tick_kl_losses.append(kl_loss_t)
            if pc_loss_t is not None:
                per_tick_pc_losses.append(pc_loss_t)
            if certainty_t is not None:
                all_certainties.append(certainty_t)

            # ── FEEC energy penalty ─────────────────────────────────────
            if self.feec is not None and velocity is not None:
                ep = self.feec.energy_penalty(z, velocity, z_prev, velocity_prev)
                energy_penalties.append(ep)
                z_prev = z
                velocity_prev = velocity

            # ── Collect logits for return (if not dropped) ──────────────
            if logits_t is not None:
                all_logits.append(logits_t)

        # Restore v1 memory state
        if not self.config.use_matrix_streams:
            for l_idx, layer in enumerate(layers):
                if layer.memory is not None and l_idx < len(pre_states):
                    layer.memory.pre_history = pre_states[l_idx]
                    layer.memory.post_history = post_states[l_idx]

        # ── Intrinsic Predictive Coding (post-loop computation) ─────────
        # For pc_target in {"next_tick", "final_tick"} the target only
        # exists once the loop has produced future z's. Compute the PC
        # loss here against detached targets (so the gradient flows only
        # through the predictor input, not through the target — this is
        # the standard PC formulation: the prediction error trains the
        # predictor, not the thing being predicted).
        if intrinsic_pc_active and intrinsic_pc_z_history:
            agg = getattr(self.config, "pc_tick_aggregation", "all")
            if pc_target_mode == "next_tick":
                # At tick t (input z_t), target = z_{t+1} (detached).
                # Last tick has no successor so we skip it.
                pairs = [
                    (t, intrinsic_pc_z_history[t], intrinsic_pc_z_history[t + 1].detach())
                    for t in range(len(intrinsic_pc_z_history) - 1)
                ]
            else:  # "final_tick"
                # At tick t (input z_t), target = z_{T-1} (detached).
                # The final tick predicts itself → trivially zero, so skip.
                z_final = intrinsic_pc_z_history[-1].detach()
                pairs = [
                    (t, intrinsic_pc_z_history[t], z_final)
                    for t in range(len(intrinsic_pc_z_history) - 1)
                ]

            for t, z_in_t, z_tgt in pairs:
                apply_pc = (
                    agg == "all"
                    or (agg == "early" and t < max(T // 2, 1))
                    or (agg == "last" and t == T - 2)   # second-to-last is the "last" predicting tick here
                )
                if apply_pc:
                    pc_t = self.cerebellar_readout(z_in_t, z_tgt)
                    per_tick_pc_losses.append(pc_t)

        all_certainties_tensor = torch.stack(all_certainties, dim=0)

        # Restore v1 memory state
        if not self.config.use_matrix_streams:
            for l_idx, layer in enumerate(layers):
                if layer.memory is not None and l_idx < len(pre_states):
                    layer.memory.pre_history = pre_states[l_idx]
                    layer.memory.post_history = post_states[l_idx]

        result = {
            "logits": all_logits[-1] if all_logits else None,
            "certainties": all_certainties_tensor,
            "all_logits": all_logits,
        }

        # FEEC energy diagnostics
        if self.feec is not None and velocity is not None:
            with torch.no_grad():
                result["feec_energy"] = self.feec.energy(z, velocity)

        # ── Hebbian diagnostics ─────────────────────────────────────────
        # Expose two scalars per forward pass so the training log can show
        # whether the fast-weight path is actually being used:
        #   hebbian_gate — mean sigmoid(gate_logit) across all layers.
        #                  At init this is ~sigmoid(-3) ≈ 0.047. If
        #                  training opens the gate (>0.1), the model is
        #                  actively using the Hebbian readout. If it
        #                  stays near init, the LM loss isn't finding
        #                  the fast-weights useful — treat that as a
        #                  signal to either remove the mechanism or
        #                  rethink how it's wired in.
        #   hebbian_M_norm — mean L2 norm of the final-tick fast-weight
        #                  matrix M, averaged over layers and the (B, S)
        #                  positions. Quantifies how much working memory
        #                  is being written. Should grow above zero
        #                  once the loop runs even one tick (M_0 was
        #                  zero, and one outer-product update suffices).
        if self.config.use_hebbian_synapse and hebbian_states is not None:
            with torch.no_grad():
                gate_vals = []
                m_norms = []
                for l_idx, layer in enumerate(layers):
                    if layer.hebbian is None:
                        continue
                    # Report the EFFECTIVE gate value (what actually gets
                    # multiplied into the readout). When force_gate is
                    # set, this is the forced value, not sigmoid(gate_logit).
                    # That keeps the log honest about what the model
                    # is actually doing — the learned gate logit may
                    # diverge from the effective gate in diagnostic mode.
                    if layer.hebbian.force_gate is not None:
                        gate_vals.append(float(layer.hebbian.force_gate))
                    else:
                        gate_vals.append(torch.sigmoid(layer.hebbian.gate_logit).item())
                    M_final = hebbian_states[l_idx]
                    if M_final is not None:
                        # mean Frobenius norm across (B, S) positions
                        # Per-position norm: sqrt(sum of squares of M[b,s])
                        # We mean over positions then over the batch.
                        per_pos_norm = M_final.float().pow(2).sum(dim=(-2, -1)).sqrt()
                        m_norms.append(per_pos_norm.mean().item())
                if gate_vals:
                    result["hebbian_gate"] = sum(gate_vals) / len(gate_vals)
                if m_norms:
                    result["hebbian_M_norm"] = sum(m_norms) / len(m_norms)
                # Effective Hebbian lr (post-modulation). With
                # hebbian_cert_lr_alpha=0 this equals the base lr
                # sigmoid(lr_logit) — useful as a sanity check. With
                # alpha>0 it shows the actual average lr being applied,
                # which should be > base lr (because uncertainty
                # multiplier is >= 1). The ratio of effective_lr to
                # base lr is how much "neurochemical boost" the model
                # is currently receiving.
                if per_tick_hebbian_lr:
                    mean_eff_lr = torch.stack(per_tick_hebbian_lr).mean()
                    result["hebbian_lr_eff"] = mean_eff_lr.item()

        if targets is not None:
            # ── Prospective Configuration: per-tick stability weights ─
            # When use_prospective_config is on and FEEC is active,
            # compute a per-tick weight that downweights ticks where
            # the dynamical system was still in transit. Tick 0 has no
            # predecessor so its weight is 1 (no penalty). Computed
            # under no_grad — see comment in the loop above for why.
            stability_weights = None
            if (getattr(self.config, "use_prospective_config", False)
                    and per_tick_energy
                    and len(per_tick_energy) == T):
                with torch.no_grad():
                    energies = torch.stack(per_tick_energy)  # [T]
                    beta = float(getattr(self.config, "prospective_beta", 2.0))
                    if T > 1 and beta > 0:
                        # relative change: |E_t - E_{t-1}| / (|E_t| + eps)
                        # Using current E as denominator (rather than max
                        # or running mean) so the weight is a property
                        # of "how much did this tick change things,
                        # relative to where we ended up." More invariant
                        # to overall energy scale.
                        e_curr = energies
                        e_prev = torch.cat([energies[:1], energies[:-1]])
                        rel_delta = (e_curr - e_prev).abs() / (e_curr.abs() + 1e-6)
                        # Tick 0: rel_delta is 0 (e_prev == e_curr) → weight = 1.
                        stability_weights = torch.exp(-beta * rel_delta)
                        # Normalize so the weights average to 1 — this
                        # keeps the overall loss scale unchanged, only
                        # redistributing emphasis across ticks. Without
                        # this normalization, large beta would shrink
                        # the loss globally and effectively lower the
                        # learning rate, which is a confound.
                        stability_weights = stability_weights * (T / stability_weights.sum())
                    else:
                        stability_weights = torch.ones_like(energies)

                    # Diagnostics
                    result["stability_w_first"] = float(stability_weights[0].item())
                    result["stability_w_last"] = float(stability_weights[-1].item())
                    result["stability_w_min"] = float(stability_weights.min().item())

            # ── Cross-Entropy Loss ──────────────────────────────────────
            if self.config.temporal_loss_type == "dynamic_aggregate":
                # [B, S, T] unreduced losses (raw, unmodified)
                losses_unreduced = torch.stack(per_tick_ce_losses, dim=1)
                per_tick_loss_tensor = losses_unreduced.mean(dim=0) # [T]

                cert = all_certainties_tensor.view(T, -1).transpose(0, 1) # [BS, T]
                losses_flat = losses_unreduced.reshape(-1, T)

                # Prospective Configuration: stability-weight is used as
                # a SELECTION bias only. We compute argmin/argmax on the
                # weighted losses (so unsettled ticks are less likely to
                # be picked as "best tick" — their weighted loss is
                # larger relative to settled ticks). But the *value*
                # gathered for the final loss is the raw, unweighted
                # CE — we don't want to optimize a rescaled loss that
                # has no calibration meaning. This keeps loss values
                # comparable to runs without PC.
                if stability_weights is not None:
                    # Weighted view for selection. Recall stability_weights
                    # is mean-normalized to 1, so multiplying flips
                    # "less settled = larger loss" — the argmin will then
                    # prefer settled ticks. Wait: we want settled ticks
                    # to win argmin, but multiplying by a weight in [0, T]
                    # where high = settled would make settled ticks have
                    # LARGER weighted loss. So we should INVERT for the
                    # argmin: divide losses by stability (settled →
                    # smaller loss → wins argmin). For consistency with
                    # the ramp path (where multiplying weight*loss
                    # naturally upweights settled ticks in the mean),
                    # think of it as: high weight = "this tick counts
                    # more." For argmin/argmax selection of "best tick",
                    # high-weight ticks should be preferred when the
                    # underlying loss is competitive. The clean way:
                    # divide loss by weight for the selection criterion.
                    eps = 1e-6
                    losses_weighted_for_argmin = losses_flat / (stability_weights.to(losses_flat.dtype) + eps)
                    lowest_idx = losses_weighted_for_argmin.argmin(dim=-1)
                else:
                    lowest_idx = losses_flat.argmin(dim=-1)

                certain_idx = cert.argmax(dim=-1)

                # Gather RAW (unweighted) losses for the actual loss value.
                loss_t1 = losses_flat.gather(1, lowest_idx.unsqueeze(1)).squeeze(1)
                loss_t2 = losses_flat.gather(1, certain_idx.unsqueeze(1)).squeeze(1)
                base_loss = ((loss_t1 + loss_t2) / 2.0).mean()
                per_tick_loss = per_tick_loss_tensor
            else:
                per_tick_loss_tensor = torch.stack(per_tick_ce_losses) # [T]
                ramp = torch.linspace(
                    self.config.tick_ramp_start, self.config.tick_ramp_end, T,
                    device=device, dtype=dtype,
                )
                ramp = ramp * (T / ramp.sum())
                # Prospective Configuration: multiply the ramp by the
                # stability weights. Both vectors are length T and both
                # are normalized to mean 1, so the product is also a
                # length-T weight vector with mean ≈ 1.
                if stability_weights is not None:
                    combined = ramp * stability_weights.to(ramp.dtype)
                    combined = combined * (T / combined.sum())  # re-normalize
                    base_loss = (combined * per_tick_loss_tensor).mean()
                else:
                    base_loss = (ramp * per_tick_loss_tensor).mean()
                per_tick_loss = per_tick_loss_tensor

            # ── Monotonicity Penalty ────────────────────────────────────
            mono_penalty = torch.tensor(0.0, device=device, dtype=dtype)
            if T > 1:
                diffs = per_tick_loss_tensor[1:] - per_tick_loss_tensor[:-1]
                mono_penalty = F.relu(diffs).mean()
                
            decay_until_frac = self.config.mono_penalty_decay_until_frac
            if decay_until_frac > 0 and self.config.max_steps > 0:
                decay_until_step = decay_until_frac * self.config.max_steps
                progress = torch.clamp(self._train_step.float() / max(decay_until_step, 1.0), max=1.0)
                decay_factor = 1.0 - progress * (1.0 - self.config.mono_penalty_min_frac)
            else:
                decay_factor = 1.0
            
            loss = base_loss + (self.config.mono_penalty_weight * decay_factor) * mono_penalty

            # ── FEEC energy penalty ─────────────────────────────────────
            if energy_penalties and self.config.feec_energy_penalty_weight > 0:
                energy_pen = torch.stack(energy_penalties).mean()
                loss = loss + self.config.feec_energy_penalty_weight * energy_pen
                result["feec_energy_penalty"] = energy_pen

            # ── Distillation Loss ───────────────────────────────────────
            distill_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if per_tick_kl_losses:
                per_tick_kl_tensor = torch.stack(per_tick_kl_losses)
                agg = self.config.distill_tick_aggregation
                if agg == "all":
                    distill_loss = per_tick_kl_tensor.mean()
                elif agg == "first":
                    distill_loss = per_tick_kl_tensor[0]
                elif agg == "last":
                    distill_loss = per_tick_kl_tensor[-1]
                elif agg == "decay_ramp":
                    T_local = per_tick_kl_tensor.shape[0]
                    weights = torch.linspace(1.0, 0.1, T_local, device=device, dtype=dtype)
                    weights = weights * (T_local / weights.sum())
                    distill_loss = (weights * per_tick_kl_tensor).mean()
                else:
                    distill_loss = per_tick_kl_tensor.mean()

            # ── Feature Alignment ───────────────────────────────────────
            if self.config.distill_feature_weight > 0 and teacher_z is not None:
                z_aligned = self.distill_proj(z) if hasattr(self, "distill_proj") else z
                method = self.config.distill_feature_method
                if method == "cosine":
                    cos = F.cosine_similarity(z_aligned.float(), teacher_z.float(), dim=-1)
                    f_loss = (1.0 - cos).mean()
                else:
                    f_loss = F.mse_loss(z_aligned, teacher_z)
                distill_loss = distill_loss + self.config.distill_feature_weight * f_loss

            loss = loss + self.config.distill_logit_weight * distill_loss

            # ── Predictive-Coding (Cerebellar) Auxiliary Loss ───────────
            # Per-tick error from the cerebellar readout against the
            # teacher hidden state. Adds a local gradient at every tick.
            # The mean across ticks is taken (matching "all" aggregation
            # in spirit — each tick contributes equally). If
            # pc_tick_aggregation skipped ticks, the mean is over fewer
            # entries, which keeps the per-loss magnitude stable.
            pc_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if per_tick_pc_losses:
                pc_stack = torch.stack(per_tick_pc_losses)
                pc_loss = pc_stack.mean()
                loss = loss + self.config.pc_loss_weight * pc_loss
                result["pc_loss"] = pc_loss.detach()
                result["per_tick_pc_loss"] = pc_stack.detach()

            result["loss"] = loss
            result["per_tick_loss"] = per_tick_loss.detach()
            result["distill_loss"] = distill_loss.detach()

        return result

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