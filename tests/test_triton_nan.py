import torch
import math
import sys
sys.path.insert(0, ".")

try:
    from ctm_transformer.triton_kernels import triton_causal_attention
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    print("Triton not found")

def test_triton_d_non_pow2():
    if not _HAS_TRITON:
        return
    
    B, H, S, D = 4, 8, 512, 220
    scale = 1.0 / math.sqrt(D)
    device = "cuda"
    dtype = torch.bfloat16
    
    q = torch.randn(B * H, S, D, device=device, dtype=dtype)
    k = torch.randn(B * H, S, D, device=device, dtype=dtype)
    v = torch.randn(B * H, S, D, device=device, dtype=dtype)
    
    print(f"Testing Triton attention with D={D} (non-power of 2)...")
    try:
        out = triton_causal_attention(q, k, v, scale)
        print(f"  Output sum: {out.sum().item():.4f}")
        print(f"  Any NaNs: {torch.isnan(out).any().item()}")
        assert not torch.isnan(out).any(), "Triton output contains NaNs"
        print("  ✓ PASSED")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_triton_d_non_pow2()
    else:
        print("CUDA not available for Triton test")
