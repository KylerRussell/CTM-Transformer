"""Checks for named CTM variants (reference, attention residuals, contextual K/V)."""
import os

import pytest
import torch

from ctm_transformer.ctm_variants import ContextualKVCTM, build_variant, variant_config
from ctm_transformer.research import load_research_config

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RECIPE = 'research/configs/presentation_control_v1/ctm_shuffled_seed23.json'


def config():
    return load_research_config(RECIPE)[0]


def test_variant_configs_are_explicit():
    c = config()
    assert not c.use_attention_residuals
    assert variant_config(c, 'attn_res').use_attention_residuals
    assert variant_config(c, 'reference') == c and variant_config(c, 'contextual_kv') == c
    with pytest.raises(ValueError, match='Apply variant_config'):
        build_variant(c, 'attn_res')
    with pytest.raises(ValueError, match='disables attention residuals'):
        variant_config(variant_config(c, 'attn_res'), 'reference')
    with pytest.raises(ValueError, match='Unknown'):
        variant_config(c, 'bio')


@needs_cuda
@pytest.mark.parametrize('variant', ['reference', 'attn_res', 'contextual_kv'])
def test_variants_are_causal_train_and_round_trip(variant):
    torch.manual_seed(0)
    c = variant_config(config(), variant)
    model = build_variant(c, variant).to(DEVICE).eval()
    x = torch.randint(3, c.vocab_size, (2, 40), device=DEVICE)
    y = x.clone()
    y[:, 25:] = torch.randint(3, c.vocab_size, (2, 15), device=DEVICE)
    with torch.no_grad():
        a = model(x, max_thought_steps=4)['logits'][:, :25]
        b = model(y, max_thought_steps=4)['logits'][:, :25]
    assert torch.allclose(a, b, atol=1e-5), 'future tokens changed earlier logits'
    model.train()
    targets = torch.full_like(x, -100)
    targets[:, -3:] = x[:, -3:]
    model(x, targets=targets, max_thought_steps=4)['loss'].backward()
    if variant == 'contextual_kv':
        assert isinstance(model, ContextualKVCTM)
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contextual_prelude.parameters())
    if variant == 'attn_res':
        assert len(model.attn_res_queries) > 0 and model.attn_res_queries[0].grad is not None
    clone = build_variant(c, variant).to(DEVICE)
    clone.load_state_dict(model.state_dict(), strict=True)


def test_contextual_kv_parameter_overhead_is_the_prelude():
    c = config()
    reference = sum(p.numel() for p in build_variant(c, 'reference').parameters())
    contextual = build_variant(c, 'contextual_kv')
    prelude = sum(p.numel() for p in contextual.contextual_prelude.parameters())
    assert sum(p.numel() for p in contextual.parameters()) == reference + prelude
    # Attention (4 d^2) plus SwiGLU with hidden 2d (6 d^2) plus two norms.
    assert prelude == 10 * c.d_model ** 2 + 2 * c.d_model
