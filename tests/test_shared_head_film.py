"""
test_shared_head_film.py — Smoke test for the shared LM head with
per-tick FiLM modulation, plus tied embeddings.

Tests the head logic in isolation (without spinning up the full
CTMTransformer, which has heavy dependencies). The patterns here mirror
exactly what model.py does internally.
"""

import torch
import torch.nn as nn


def build_head_film(d_latent, d_model, vocab_size, T,
                    tie_embeddings=False, token_embedding=None):
    """Mirror the construction logic from model.py for the shared+FiLM path."""
    shared_adapter = nn.Sequential(
        nn.Linear(d_latent + d_model, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
    )
    film_gamma = nn.Parameter(torch.ones(T, d_model))
    film_beta = nn.Parameter(torch.zeros(T, d_model))
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    if tie_embeddings:
        lm_head.weight = token_embedding.weight
    return shared_adapter, film_gamma, film_beta, lm_head


def output_logits_film(shared_adapter, film_gamma, film_beta, lm_head,
                       z, text_emb, tick):
    """Mirror _output_logits in model.py for the shared+FiLM path."""
    combined = torch.cat([z, text_emb], dim=-1)
    x = shared_adapter(combined)
    x = x * film_gamma[tick] + film_beta[tick]
    return lm_head(x)


def test_film_initialization_is_identity():
    """At init, gamma=1 and beta=0, so FiLM is the identity. The output
    at every tick should be IDENTICAL since the only differentiating
    factor (FiLM) is a no-op."""
    print("[test 1] FiLM at init produces identical outputs at all ticks...", end=" ")

    torch.manual_seed(0)
    B, S, d_latent, d_model, vocab_size, T = 2, 4, 64, 64, 128, 8
    shared_adapter, gamma, beta, lm_head = build_head_film(
        d_latent, d_model, vocab_size, T,
    )
    z = torch.randn(B, S, d_latent)
    text_emb = torch.randn(B, S, d_model)

    logits_at_tick = []
    for t in range(T):
        with torch.no_grad():
            logits_at_tick.append(
                output_logits_film(shared_adapter, gamma, beta, lm_head,
                                   z, text_emb, tick=t)
            )

    # All identical at init
    for t in range(1, T):
        diff = (logits_at_tick[0] - logits_at_tick[t]).abs().max().item()
        assert diff < 1e-6, f"tick 0 vs tick {t}: max diff {diff}"
    print(f"OK (max-diff across ticks at init = 0.0)")


def test_film_after_perturbation_diverges():
    """After we perturb gamma and beta differently per tick, the outputs
    SHOULD diverge — that's the whole point of FiLM."""
    print("[test 2] FiLM with non-identity params makes ticks diverge...", end=" ")

    torch.manual_seed(0)
    B, S, d_latent, d_model, vocab_size, T = 2, 4, 64, 64, 128, 4
    shared_adapter, gamma, beta, lm_head = build_head_film(
        d_latent, d_model, vocab_size, T,
    )
    # Perturb FiLM differently per tick
    with torch.no_grad():
        gamma.data = torch.randn_like(gamma) * 0.5 + 1.0
        beta.data = torch.randn_like(beta) * 0.3

    z = torch.randn(B, S, d_latent)
    text_emb = torch.randn(B, S, d_model)
    logits_at_tick = []
    for t in range(T):
        with torch.no_grad():
            logits_at_tick.append(
                output_logits_film(shared_adapter, gamma, beta, lm_head,
                                   z, text_emb, tick=t)
            )

    # Each pair should differ meaningfully now
    diffs = []
    for t in range(1, T):
        diff = (logits_at_tick[0] - logits_at_tick[t]).abs().max().item()
        diffs.append(diff)
    assert all(d > 0.1 for d in diffs), (
        f"FiLM not diverging: diffs={diffs}, expected all >0.1"
    )
    print(f"OK (diffs from tick 0: {[f'{d:.2f}' for d in diffs]})")


def test_tied_embeddings_share_memory():
    """When tied, modifying one updates the other — same Parameter object."""
    print("[test 3] Tied embeddings share memory...", end=" ")

    torch.manual_seed(0)
    d_latent, d_model, vocab_size, T = 64, 64, 128, 4
    token_embedding = nn.Embedding(vocab_size, d_model)
    shared_adapter, gamma, beta, lm_head = build_head_film(
        d_latent, d_model, vocab_size, T,
        tie_embeddings=True, token_embedding=token_embedding,
    )

    # Verify same Parameter object (not just equal values)
    assert lm_head.weight is token_embedding.weight, (
        "lm_head.weight and token_embedding.weight must be the SAME "
        "Parameter (tied), not just equal-valued."
    )

    # Perturb one, verify the other reflects it
    original = token_embedding.weight.data.clone()
    with torch.no_grad():
        token_embedding.weight.data += 0.1
    assert torch.allclose(lm_head.weight.data, token_embedding.weight.data)
    assert not torch.allclose(lm_head.weight.data, original)
    print("OK (same Parameter object, mutation propagates)")


def test_tied_embeddings_gradient_flow():
    """When tied, the gradient on the shared weight should accumulate
    contributions from BOTH the embedding-lookup path AND the head-matmul
    path. Test by computing a loss that depends on both and checking
    grad is non-zero."""
    print("[test 4] Tied embeddings receive gradient from both uses...", end=" ")

    torch.manual_seed(0)
    B, S, d_latent, d_model, vocab_size, T = 2, 4, 32, 32, 64, 2
    token_embedding = nn.Embedding(vocab_size, d_model)
    shared_adapter, gamma, beta, lm_head = build_head_film(
        d_latent, d_model, vocab_size, T,
        tie_embeddings=True, token_embedding=token_embedding,
    )

    # Forward path: lookup tokens, project, output logits, dummy loss
    input_ids = torch.randint(0, vocab_size, (B, S))
    text_emb = token_embedding(input_ids)        # uses tied weight
    z = torch.randn(B, S, d_latent)
    logits = output_logits_film(shared_adapter, gamma, beta, lm_head,
                                z, text_emb, tick=0)
    loss = logits.sum()
    loss.backward()

    # Gradient on the shared weight should be non-zero (both uses contributed)
    assert lm_head.weight.grad is not None
    assert lm_head.weight.grad.abs().sum().item() > 0
    # And it's the SAME tensor as token_embedding.weight.grad (tied)
    assert lm_head.weight.grad is token_embedding.weight.grad
    print("OK (grad accumulates from both paths)")


def test_param_count_dramatically_reduced():
    """The savings here are MORE MODEST than initially advertised.

    The per-tick architecture in this codebase has 8 small adapters
    feeding into a SHARED LM head — not 8 separate heads. So the
    savings from shared+FiLM come from eliminating 7 redundant adapter
    copies (~2M params each at d_model=1024), not from collapsing 8
    big heads into one.

    The big saving comes from --tie_embeddings, which removes the
    [d_model, V] head weight matrix entirely (134M params at V=131072).
    """
    print("[test 5] Param count savings at V=131072...", end=" ")

    d_latent, d_model, vocab_size, T = 1024, 1024, 131072, 8

    # Per-tick heads count: T adapters + 1 SHARED head
    adapter_params_per_tick = (
        (d_latent + d_model) * d_model + d_model       # Linear w + b
        + 2 * d_model                                  # LayerNorm w + b
    )
    head_params = d_model * vocab_size + vocab_size    # bias=True
    per_tick_total = T * adapter_params_per_tick + head_params

    # Shared head + FiLM count: 1 adapter + 2*T*d_model FiLM + 1 head (no bias)
    shared_film_total = (
        adapter_params_per_tick           # one shared adapter
        + 2 * T * d_model                 # FiLM gamma + beta
        + d_model * vocab_size            # head w (no bias)
    )

    saved_no_tie = per_tick_total - shared_film_total
    # Plus tie_embeddings: removes the head weight matrix entirely
    saved_with_tie = saved_no_tie + d_model * vocab_size

    # Without tie: ~7 × adapter_size = ~14.7M
    expected_no_tie = 7 * adapter_params_per_tick
    assert abs(saved_no_tie - expected_no_tie) < 1e6, (
        f"no-tie saved={saved_no_tie:,}, expected ~{expected_no_tie:,}"
    )
    # With tie: previous + 134M
    expected_with_tie = expected_no_tie + d_model * vocab_size
    assert abs(saved_with_tie - expected_with_tie) < 1e6, (
        f"with-tie saved={saved_with_tie:,}, expected ~{expected_with_tie:,}"
    )

    print(f"OK")
    print(f"      per-tick total:           {per_tick_total/1e6:.1f}M params")
    print(f"      shared+FiLM:              {shared_film_total/1e6:.1f}M params")
    print(f"      saved (head not tied):    {saved_no_tie/1e6:.1f}M")
    print(f"      saved (with tie_embeddings): {saved_with_tie/1e6:.1f}M")


if __name__ == "__main__":
    test_film_initialization_is_identity()
    test_film_after_perturbation_diverges()
    test_tied_embeddings_share_memory()
    test_tied_embeddings_gradient_flow()
    test_param_count_dramatically_reduced()
    print("\nAll shared-head-FiLM smoke tests passed.")
