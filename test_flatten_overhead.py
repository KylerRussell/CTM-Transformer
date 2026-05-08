import torch
import time

def test_allocation_overhead():
    # Simulate 400 tensors of various sizes totaling ~492M parameters
    # total ~ 1GB in bfloat16
    tensors = []
    for i in range(400):
        # 1.2M params per tensor on average
        t = torch.randn(1230000, device='cuda', dtype=torch.bfloat16)
        tensors.append(t)
    
    torch.cuda.synchronize()
    
    # Test torch._utils._flatten_dense_tensors
    times = []
    for _ in range(10):
        t0 = time.time()
        flat = torch._utils._flatten_dense_tensors(tensors)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        
    print(f"Flatten avg time: {sum(times)/len(times)*1000:.2f} ms")

    # Test preallocated buffer copy
    flat_buffer = torch.empty(sum(t.numel() for t in tensors), device='cuda', dtype=torch.bfloat16)
    
    times_prealloc = []
    for _ in range(10):
        t0 = time.time()
        offset = 0
        for t in tensors:
            n = t.numel()
            flat_buffer[offset:offset+n].copy_(t.view(-1))
            offset += n
        torch.cuda.synchronize()
        times_prealloc.append(time.time() - t0)

    print(f"Preallocated copy avg time: {sum(times_prealloc)/len(times_prealloc)*1000:.2f} ms")

if __name__ == "__main__":
    test_allocation_overhead()
