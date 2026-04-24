"""
CTM-Transformer Smoke Test

Verifies:
1. Model instantiation with default config
2. Forward pass produces correct output shapes
3. Loss computation works (finite, non-zero)
4. Backward pass succeeds (gradients flow through thought loop)
5. NLM parameters receive gradients
6. Per-tick losses are all populated
"""

import sys
import torch
sys.path.insert(0, ".")

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer


def test_forward_pass():
    """Test basic forward pass with random data."""
    print("=" * 60)
    print("CTM-Transformer Smoke Test")
    print("=" * 60)

    # Small config for fast testing
    config = CTMConfig(
        vocab_size=50257,
        d_model=128,
        d_latent=128,
        n_heads=4,
        n_layers=2,
        nlm_hidden_dim=16,
        history_len=4,
        max_thought_steps=4,
        max_seq_len=64,
        seq_len=32,
        batch_size=2,
        sync_method="diag_summary",
        gradient_checkpointing=False,  # Disable for test clarity
        dropout=0.0,
    )

    device = "cpu"
    model = CTMTransformer(config).to(device)

    n_params = model.get_num_params()
    print(f"\n1. Model instantiated: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    # Random input
    B, S = config.batch_size, config.seq_len
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    targets = torch.randint(0, config.vocab_size, (B, S))

    # Forward pass
    result = model(input_ids, targets=targets)

    logits = result["logits"]
    all_logits = result["all_logits"]
    loss = result["loss"]
    per_tick_loss = result["per_tick_loss"]
    certainties = result["certainties"]

    print(f"\n2. Forward pass shapes:")
    print(f"   logits:         {logits.shape}  (expected: [{B}, {S}, {config.vocab_size}])")
    print(f"   all_logits:     {all_logits.shape}  (expected: [{config.max_thought_steps}, {B}, {S}, {config.vocab_size}])")
    print(f"   certainties:    {certainties.shape}  (expected: [{config.max_thought_steps}, {B}, {S}])")
    print(f"   per_tick_loss:  {per_tick_loss.shape}  (expected: [{config.max_thought_steps}])")

    assert logits.shape == (B, S, config.vocab_size), f"Bad logits shape: {logits.shape}"
    assert all_logits.shape == (config.max_thought_steps, B, S, config.vocab_size)
    assert certainties.shape == (config.max_thought_steps, B, S)
    assert per_tick_loss.shape == (config.max_thought_steps,)
    print("   ✓ All shapes correct")

    print(f"\n3. Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
    print(f"   Per-tick losses: {[f'{l:.4f}' for l in per_tick_loss.tolist()]}")
    print("   ✓ Loss is finite and positive")

    # Backward pass
    loss.backward()
    print(f"\n4. Backward pass completed")

    # Check gradient flow
    has_grad = 0
    no_grad = 0
    nlm_has_grad = False
    sync_has_grad = False
    output_has_grad = False

    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad += 1
            if "nlm" in name:
                nlm_has_grad = True
            if "query_proj" in name:
                sync_has_grad = True
            if "output_head" in name:
                output_has_grad = True
        else:
            no_grad += 1

    print(f"   Parameters with gradients: {has_grad}")
    print(f"   Parameters without gradients: {no_grad}")
    print(f"   NLM gradients:     {'✓' if nlm_has_grad else '✗'}")
    print(f"   Sync gradients:    {'✓' if sync_has_grad else '✗'}")
    print(f"   Output gradients:  {'✓' if output_has_grad else '✗'}")

    assert has_grad > 0, "No parameters received gradients!"
    assert nlm_has_grad, "NLM parameters did not receive gradients!"
    assert output_has_grad, "Output head did not receive gradients!"
    print("   ✓ Gradients flow through thought loop to NLMs")

    # Training step test (few steps)
    print(f"\n5. Mini training test (10 steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_loss = None
    final_loss = None

    for step in range(10):
        result = model(input_ids, targets=targets)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 9:
            final_loss = loss.item()

    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss:   {final_loss:.4f}")
    print(f"   Δ loss:       {final_loss - initial_loss:.4f}")
    if final_loss < initial_loss:
        print("   ✓ Loss decreased — model is learning!")
    else:
        print("   ⚠ Loss did not decrease (may need more steps or tuning)")

    # Generation test
    print(f"\n6. Generation test...")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt_text = "Hello world"
    prompt_tokens = enc.encode(prompt_text)
    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    generated = model.generate(prompt_ids, max_new_tokens=10, temperature=1.0, top_k=50)
    gen_text = enc.decode(generated[0].tolist()[len(prompt_tokens):])
    print(f"   Prompt: '{prompt_text}'")
    print(f"   Generated: '{prompt_text}{gen_text}'")
    print("   ✓ Generation completed")

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_forward_pass()
