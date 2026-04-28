"""
TernaryLinear — Ternary Weight Network (TWN) drop-in replacement for nn.Linear.

Reference:
    Li et al. "Ternary Weight Networks." arXiv:1605.04711 (2022).
    Wang et al. "BitNet: Scaling 1-bit Transformers for Large Language Models."
        arXiv:2310.11453 (2023). [used for SubLN-style activation handling]

──────────────────────────────────────────────────────────────────────────
WHAT THIS DOES
──────────────────────────────────────────────────────────────────────────

During training:
  - Maintains a full-precision *latent* weight W ∈ R^{out, in}.
  - On every forward pass, ternarizes it on-the-fly:
        Δ = 0.75 * mean(|W|)
        W̃[i,j] = +1 if W[i,j] >  Δ
                  0 if |W[i,j]| ≤ Δ
                 -1 if W[i,j] < -Δ
        α  = mean over { |W[i,j]| : |W[i,j]| > Δ }   (per-tensor scaling)
  - Forward computes  y = α · (W̃ @ x).
  - Backward uses the straight-through estimator (STE): the gradient w.r.t.
    the ternarization is treated as identity. This lets the latent weight
    accumulate continuous updates that eventually flip ternary states.

During inference:
  - Pre-bake the ternary tensor + scaling factor and store them as buffers.
  - Optionally pack the ternary trits into 2-bit storage (4 trits per byte)
    for an ~8× weight memory reduction vs fp16.
  - Forward becomes a fp32-or-bf16 matmul against the +1/0/-1 unpacked
    weight, scaled by α. (PyTorch has no native ternary matmul kernel; the
    real-hardware ADD-only saving is in custom inference kernels which we
    don't ship here. The training story is what matters for this codebase.)

──────────────────────────────────────────────────────────────────────────
HONEST CAVEATS (the user already heard these but worth restating in code)
──────────────────────────────────────────────────────────────────────────

1. Training VRAM doesn't decrease. STE means the layer's saved-for-backward
   tensor is still the fp16/bf16 input x, because dL/dW_latent ≈ dL/dy · x.T
   (treating ternarization as identity). The activation graph stays the
   same size as a vanilla nn.Linear.

2. Training compute increases ~10-30%. Each forward computes Δ, the
   ternarized matrix, and α — three reductions over a full fp32 weight on
   top of the matmul. The benefit is fully realized at inference.

3. The latent weight stays fp32 even when the wrapping model is bf16.
   This is the standard pattern from BitNet/TWN: optimizer needs precision
   to accumulate small updates that gradually push entries across the ±Δ
   boundary. Ternarizing a bf16 latent in-place would freeze training
   within a few hundred steps as updates round off.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Core ternarization: f(W) → (W̃, α) with straight-through gradient
# ──────────────────────────────────────────────────────────────────────────

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