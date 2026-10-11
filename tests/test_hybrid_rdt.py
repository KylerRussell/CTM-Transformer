"""RDT + CTM hybrids (ctm_transformer/hybrid_rdt.py): each mechanism starts as Dans exactly and trains."""
import pytest
import torch
from dataclasses import replace

from ctm_transformer.hybrid_rdt import DANS, MECHANISMS, hybrid_factory
from ctm_transformer.rdt_recipe import recipe_factory
from tests.test_lm_scale import batch, config

CASES = [(), *[(m,) for m in MECHANISMS], MECHANISMS]


@pytest.mark.parametrize('mechanisms', CASES)
def test_hybrid_is_dans_at_initialization_and_its_mechanisms_get_gradients(mechanisms):
    c = replace(config('rdt'), max_thought_steps=4)
    torch.manual_seed(0)
    hybrid = hybrid_factory(set(mechanisms))(c)
    dans = recipe_factory(DANS)(c)
    dans.load_state_dict({k: v for k, v in hybrid.state_dict().items() if k in dans.state_dict()})
    x, y = batch()
    with torch.no_grad():
        a, b = hybrid(x, return_all_logits=True), dans(x, return_all_logits=True)
        assert torch.allclose(a['logits'], b['logits'], atol=1e-5)
        assert all(torch.allclose(p, q, atol=1e-5) for p, q in zip(a['all_logits'], b['all_logits']))
    loss = hybrid(x, targets=y)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    zero_init = {'sync_readout': 'sync_out.weight', 'neuron_models': 'nlm_scale', 'state_carry': 'carry'}
    params = dict(hybrid.named_parameters())
    for m in mechanisms:
        assert params[zero_init[m]].grad.abs().sum() > 0, m


@pytest.mark.parametrize('mechanism', MECHANISMS)
def test_each_mechanism_changes_the_output_once_its_scale_is_nonzero(mechanism):
    c = replace(config('rdt'), max_thought_steps=4)
    torch.manual_seed(0)
    hybrid = hybrid_factory({mechanism})(c)
    x, _ = batch()
    with torch.no_grad():
        before = hybrid(x)['logits']
        for name in ('sync_out.weight', 'nlm_scale', 'carry'):
            if name in dict(hybrid.named_parameters()):
                dict(hybrid.named_parameters())[name].normal_(std=0.1)
        assert not torch.allclose(before, hybrid(x)['logits'], atol=1e-4)


def test_factory_rejects_unknown_mechanisms_and_requires_the_state_norm():
    with pytest.raises(ValueError):
        hybrid_factory({'tick_memory'})
    with pytest.raises(ValueError):
        hybrid_factory(set(), recipe={**DANS, 'state_norm': False})(config('rdt'))
