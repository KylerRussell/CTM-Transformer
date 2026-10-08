"""RDT recipe variants (ctm_transformer/rdt_recipe.py): the defaults are ScaledBaseline exactly; the switches do what they claim."""
import pytest
import torch
from torch import nn

from ctm_transformer.lm_scale import ScaledBaseline
from ctm_transformer.rdt_recipe import RecipeRDT, recipe_factory
from tests.test_lm_scale import batch, config, grads


@pytest.mark.parametrize('backprop_steps', [None, 2])
def test_default_recipe_is_the_scaled_rdt(backprop_steps):
    torch.manual_seed(0)
    ref = ScaledBaseline(config('rdt'), backprop_steps=backprop_steps)
    torch.manual_seed(0)
    new = RecipeRDT(config('rdt'), backprop_steps=backprop_steps)
    x, y = batch()
    a, b = ref(x, targets=y), new(x, targets=y)
    a['loss'].backward()
    b['loss'].backward()
    ga, gb = grads(ref), grads(new)
    assert torch.equal(a['loss'], b['loss']) and ga.keys() == gb.keys() and all(torch.allclose(ga[k], gb[k]) for k in ga)
    with torch.no_grad():
        ra, rb = ref(x, return_all_logits=True), new(x, return_all_logits=True)
        assert torch.allclose(ra['logits'], rb['logits'], atol=1e-6)
        assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(ra['all_logits'], rb['all_logits']))


def test_norm_switches_remove_post_norms_where_declared():
    core = RecipeRDT(config('rdt'), norm='core_sandwich')
    assert all(isinstance(b.attn_post, nn.Identity) for b in (*core.prelude, *core.coda))
    assert all(isinstance(b.attn_post, nn.RMSNorm) for b in core.core)
    pre = RecipeRDT(config('rdt'), norm='prenorm', normalize=True)
    assert all(isinstance(b.ffn_post, nn.Identity) for b in (*pre.prelude, *pre.core, *pre.coda))
    assert isinstance(pre.inject_norm, nn.RMSNorm) and isinstance(pre.exit_norm, nn.RMSNorm)


def test_random_state_is_drawn_every_pass_and_training_runs():
    torch.manual_seed(0)
    m = RecipeRDT(config('rdt'), norm='prenorm', normalize=True, state_init='random')
    x, y = batch()
    with torch.no_grad():
        assert not torch.equal(m(x)['logits'], m(x)['logits'])
    loss = m(x, targets=y)['loss']
    loss.backward()
    assert torch.isfinite(loss) and m.injection.weight.grad.abs().sum() > 0


def test_factory_rejects_unknown_keys():
    with pytest.raises(ValueError):
        recipe_factory({'norm': 'prenorm', 'warmup': 3})
