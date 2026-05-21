"""
test_offload_grads.py — Gradient-correctness test for CTMCPUOffloadEngine.

The engine stores ThoughtLayer params on CPU and streams them to a single
GPU template per layer call. The previous `collect_grads` only captured the
LAST recomputed layer's grads (shared-template collapse), silently leaving
layers 0..L-2 untrained. The rewritten engine uses a per-layer recompute
autograd boundary (`_StreamedLayerFn`) that should produce gradients
identical to a normal, non-offloaded model.

This test builds two identical models, runs one normally and one through the
offload engine, and asserts their gradients match — for both the offloaded
layer params and the GPU-resident static params. Requires CUDA.
"""

import copy
import sys

import torch

sys.path.insert(0, ".")

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.extras import CTMCPUOffloadEngine


def make_config(**overrides):
    defaults = dict(
        vocab_size=2048,
        d_model=64,
        d_latent=64,
        n_heads=4,
        n_layers=4,
        nlm_hidden_dim=16,
        history_len=4,
        max_thought_steps=3,
        max_seq_len=32,
        seq_len=16,
        batch_size=2,
        sync_method="diag_summary",
        gradient_checkpointing=False,
        dropout=0.0,
        use_matrix_streams=True,
        n_streams=4,
    )
    defaults.update(overrides)
    return CTMConfig(**defaults)


def _grad_map(model):
    """name -> grad tensor (fp32, cpu), only for params that got a grad."""
    out = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[name] = p.grad.detach().float().cpu()
    return out


def run_case(label, **cfg_overrides):
    print(f"\n{'─' * 64}\n[{label}]\n{'─' * 64}")
    device = "cuda:0"
    config = make_config(**cfg_overrides)

    torch.manual_seed(0)
    ref = CTMTransformer(config).to(device, torch.float32)
    ref.train()

    # Identical twin for the offload path.
    off = copy.deepcopy(ref)
    off.train()

    B, S = config.batch_size, config.seq_len
    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    targets = torch.randint(0, config.vocab_size, (B, S), device=device)

    # ── Reference (no offload) ───────────────────────────────────────────
    torch.manual_seed(7)
    ref_loss = ref(input_ids, targets=targets)["loss"]
    ref_loss.backward()
    ref_grads = _grad_map(ref)

    # ── Offloaded ────────────────────────────────────────────────────────
    engine = CTMCPUOffloadEngine(off, device=device, dtype=torch.float32)
    engine.zero_grad()
    torch.manual_seed(7)
    off_loss = off(input_ids, targets=targets)["loss"]
    off_loss.backward()
    engine.collect_grads()
    off_grads = _grad_map(off)

    # ── Compare ──────────────────────────────────────────────────────────
    loss_diff = abs(ref_loss.item() - off_loss.item())
    print(f"  loss: ref={ref_loss.item():.6f} off={off_loss.item():.6f} "
          f"|Δ|={loss_diff:.2e}")
    assert loss_diff < 1e-3, f"loss mismatch: {loss_diff}"

    layer_names = [n for n in ref_grads if n.startswith("layers.")]
    static_names = [n for n in ref_grads if not n.startswith("layers.")]

    def compare(names, group):
        worst_rel = 0.0
        worst_name = None
        n_missing = 0
        for name in names:
            ga = ref_grads[name]
            gb = off_grads.get(name)
            if gb is None:
                n_missing += 1
                print(f"    MISSING grad in offload path: {name}")
                continue
            denom = ga.norm().item() + 1e-12
            rel = (ga - gb).norm().item() / denom
            if rel > worst_rel:
                worst_rel, worst_name = rel, name
        print(f"  {group}: {len(names)} params, worst rel-err={worst_rel:.2e} "
              f"({worst_name})")
        assert n_missing == 0, f"{n_missing} {group} params missing grad"
        return worst_rel

    worst_layer = compare(layer_names, "layer (offloaded)")
    worst_static = compare(static_names, "static (resident)")

    # Per-layer sanity: every distinct layer index must have a non-zero grad
    # (the old bug zeroed all but the last layer).
    nonzero_by_layer = {}
    for name in layer_names:
        idx = name.split(".")[1]
        s = off_grads[name].abs().sum().item()
        nonzero_by_layer[idx] = nonzero_by_layer.get(idx, 0.0) + s
    print(f"  per-layer |grad| sums: "
          + ", ".join(f"L{k}={v:.2e}" for k, v in sorted(nonzero_by_layer.items())))
    for idx, s in nonzero_by_layer.items():
        assert s > 0, f"layer {idx} has all-zero grad (offload collapse!)"

    assert worst_layer < 2e-2, f"layer grads diverge: {worst_layer}"
    assert worst_static < 2e-2, f"static grads diverge: {worst_static}"

    engine.shutdown()
    print(f"  ✓ PASSED")
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping offload grad test.")
        return
    torch.use_deterministic_algorithms(False)
    ok = True
    ok &= run_case("matrix streams")
    ok &= run_case("matrix streams + hebbian", use_hebbian_synapse=True,
                   hebbian_bottleneck_dim=16)
    print("\n" + ("ALL OFFLOAD GRAD TESTS PASSED ✓" if ok else "FAILED ✗"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
