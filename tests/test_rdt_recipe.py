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


@pytest.mark.parametrize('state_norm', [False, True])
def test_additive_injection_without_coda_norm_is_the_transformer_at_depth_1(state_norm):
    from dataclasses import replace
    torch.manual_seed(0)
    rdt = RecipeRDT(replace(config('rdt'), max_thought_steps=1), norm='prenorm', injection='add', coda_norm='none', state_norm=state_norm)
    transformer = ScaledBaseline(replace(rdt.config, model_family='transformer', n_layers=len(rdt.prelude) + len(rdt.core) + len(rdt.coda),
                                         prelude_layers=0, core_layers=0, coda_layers=0, max_thought_steps=1, train_depth_min=1, train_depth_max=1))
    blocks = (*rdt.prelude, *rdt.core, *rdt.coda)
    state = {k: v for k, v in rdt.state_dict().items() if not k.startswith(('prelude', 'core', 'coda', 'state_norm'))}
    state.update({f'layers.{i}.{k}': v for i, b in enumerate(blocks) for k, v in b.state_dict().items()})
    transformer.load_state_dict(state)
    x, y = batch()
    with torch.no_grad():
        assert torch.allclose(rdt(x, max_thought_steps=1)['logits'], transformer(x)['logits'], atol=1e-5)
        assert torch.allclose(rdt(x, targets=y, max_thought_steps=1)['loss'], transformer(x, targets=y)['loss'], atol=1e-5)
    assert not any(n.startswith('injection') for n, _ in rdt.named_parameters())


def test_state_norm_bounds_the_additive_state():
    torch.manual_seed(0)
    x, _ = batch()
    growth = {}  # state RMS after 32 iterations over that after 8
    for state_norm in (False, True):
        m = RecipeRDT(config('rdt'), norm='prenorm', injection='add', coda_norm='none', state_norm=state_norm)
        with torch.no_grad():
            e = m._blocks(m.prelude, m.token_embedding(x))
            s, rms = torch.zeros_like(e), {}
            for i in range(1, 33):
                s = m._blocks(m.core, m.injection(torch.cat((m.state_norm(s), e), dim=-1)))
                rms[i] = s.pow(2).mean().sqrt().item()
            growth[state_norm] = rms[32] / rms[8]
    assert growth[True] < 1.5 < 3 < growth[False], growth
