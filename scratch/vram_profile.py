
import torch
import torch.nn as nn
from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer

def profile_vram():
    config = CTMConfig(
        d_model=1024,
        d_latent=1024,
        n_layers=24,
        n_heads=16,
        vocab_size=131072,
        use_hyperloop=True,
        hyperloop_n_begin=4,
        hyperloop_n_middle=18,
        hyperloop_n_end=4,
        hyperloop_middle_loops=1,
        max_thought_steps=8,
        use_matrix_streams=True,
        n_streams=4,
        use_triton_attention=True,
        dtype="bfloat16"
    )
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated(device)
    print(f"Initial Mem: {base_mem / 1024**2:.1f} MB")

    print("\n1. Building Model...")
    model = CTMTransformer(config).to(device, dtype)
    model_mem = torch.cuda.memory_allocated(device) - base_mem
    print(f"Model Params + Buffers: {model_mem / 1024**2:.1f} MB")

    # Optimizer
    print("\n2. Building Optimizer (8-bit AdamW)...")
    from bitsandbytes.optim import AdamW8bit
    optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    opt_mem = torch.cuda.memory_allocated(device) - base_mem - model_mem
    print(f"Optimizer State: {opt_mem / 1024**2:.1f} MB")

    # Forward Pass (1 tick)
    print("\n3. Forward Pass (1 tick)...")
    ids = torch.randint(0, 1000, (1, 512), device=device)
    targets = torch.randint(0, 1000, (1, 512), device=device)
    
    result = model(ids, targets=targets, max_thought_steps=1)
    loss = result["loss"]
    
    fwd_mem = torch.cuda.memory_allocated(device) - base_mem - model_mem - opt_mem
    print(f"Forward Activations (1 tick): {fwd_mem / 1024**2:.1f} MB")

    # Backward Pass
    print("\n4. Backward Pass...")
    loss.backward()
    grad_mem = torch.cuda.memory_allocated(device) - base_mem - model_mem - opt_mem - fwd_mem
    print(f"Gradients: {grad_mem / 1024**2:.1f} MB")

    # Full Loop (8 ticks)
    print("\n5. Forward Pass (8 ticks, NO checkpointing)...")
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    config.gradient_checkpointing = False
    result = model(ids, targets=targets, max_thought_steps=8)
    loss = result["loss"]
    
    full_fwd_mem = torch.cuda.memory_allocated(device) - base_mem - model_mem - opt_mem
    print(f"Forward Activations (8 ticks, NO ckpt): {full_fwd_mem / 1024**2:.1f} MB")

    # Full Loop (8 ticks, WITH checkpointing)
    print("\n6. Forward Pass (8 ticks, WITH checkpointing)...")
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    model.config.gradient_checkpointing = True
    model.config.gradient_checkpointing_min_T = 1
    result = model(ids, targets=targets, max_thought_steps=8)
    loss = result["loss"]
    
    # Accumulation Loop (16 steps)
    print("\n7. Simulating 16 Accumulation Steps...")
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device)
    
    for i in range(16):
        result = model(ids, targets=targets, max_thought_steps=8)
        loss = result["loss"] / 16
        loss.backward()
        current = torch.cuda.memory_allocated(device)
        print(f"  Step {i+1}: {current / 1024**2:.1f} MB (diff {(current - start_mem)/1024**2:+.1f})")
    
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated(device)
    print(f"After zero_grad + cache_empty: {final_mem / 1024**2:.1f} MB")

if __name__ == "__main__":
    profile_vram()
