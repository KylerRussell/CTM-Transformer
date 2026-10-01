"""The language-model-scale training paths equal the reference losses and gradients (ctm_transformer/lm_scale.py)."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from ctm_transformer.ctm_lm import ctm_loss, lm_config
from ctm_transformer.lm_scale import ScaledBaseline, ScaledCTMLM, chunked_cross_entropy, ctm_selected_tick_loss, scaled_factory
from ctm_transformer.positions import RoPEBaseline, RoPECTMLM
from ctm_transformer.research import load_research_config

RECIPES = {'transformer': 'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
           'rdt': 'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json',
           'ctm_lm': 'research/configs/presentation_control_v1/ctm_shuffled_seed23.json'}


def config(model, checkpointing=False):
    c = load_research_config(RECIPES[model])[0]
    if model == 'ctm_lm':
        c = lm_config(c)
    c = replace(c, use_positional_encoding=False, dtype='float32', max_thought_steps=1 if model == 'transformer' else 5)
    return replace(c, gradient_checkpointing=checkpointing) if model != 'ctm_lm' else c


def batch(seed=0, B=3, S=11, vocab=71):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, vocab, (B, S), generator=g)
    y = torch.randint(3, vocab, (B, S), generator=g)
    y[0, :4] = -100
    return x, y


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_chunked_cross_entropy_equals_the_mean_cross_entropy():
    torch.manual_seed(0)
    h = torch.randn(4, 9, 16, requires_grad=True)
    w = torch.randn(50, 16, requires_grad=True)
    t = torch.randint(0, 50, (4, 9))
    t[1, :3] = -100
    a = chunked_cross_entropy(h, w, t, chunk=5)
    ga = torch.autograd.grad(a, (h, w))
    b = F.cross_entropy(F.linear(h, w).reshape(-1, 50), t.reshape(-1), ignore_index=-100)
    gb = torch.autograd.grad(b, (h, w))
    assert torch.allclose(a, b, atol=1e-6) and all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ga, gb))


def test_selected_tick_loss_equals_ctm_loss():
    torch.manual_seed(1)
    r = torch.randn(2, 7, 6, 12, requires_grad=True)
    w = torch.randn(40, 12, requires_grad=True)
    t = torch.randint(0, 40, (2, 7))
    t[0, :2] = -100
    a = ctm_selected_tick_loss(r, w, t, chunk=13)
    ga = torch.autograd.grad(a['loss'], (r, w))
    b = ctm_loss(torch.einsum('bstp,vp->bstv', r, w), t)
    gb = torch.autograd.grad(b['loss'], (r, w))
    assert torch.allclose(a['loss'], b['loss'], atol=1e-6)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ga, gb))
    assert torch.allclose(a['per_tick_loss'], b['per_tick_loss'], atol=1e-6) and torch.allclose(a['certainty'], b['certainty'], atol=1e-6)


@pytest.mark.parametrize('model', ['transformer', 'rdt'])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_scaled_baseline_equals_the_rope_baseline(model, checkpointing):
    torch.manual_seed(0)
    ref = RoPEBaseline(config(model))
    torch.manual_seed(0)
    new = ScaledBaseline(config(model, checkpointing))
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), new.state_dict().values()))
    x, y = batch()
    a, b = ref(x, targets=y), new(x, targets=y)
    a['loss'].backward()
    b['loss'].backward()
    ga, gb = grads(ref), grads(new)
    assert torch.allclose(a['loss'], b['loss'], atol=1e-6) and ga.keys() == gb.keys()
    assert all(torch.allclose(ga[k], gb[k], atol=1e-6) for k in ga)
    with torch.no_grad():
        assert torch.equal(ref(x)['logits'], new(x)['logits'])


@pytest.mark.parametrize('checkpointing', [False, True])
def test_scaled_ctm_lm_equals_the_rope_ctm_lm(checkpointing):
    torch.manual_seed(0)
    ref = RoPECTMLM(config('ctm_lm'))
    torch.manual_seed(0)
    new = ScaledCTMLM(config('ctm_lm'), checkpoint_ticks=checkpointing)
    x, y = batch()
    a, b = ref(x, targets=y), new(x, targets=y)
    a['loss'].backward()
    b['loss'].backward()
    ga, gb = grads(ref), grads(new)
    assert torch.allclose(a['loss'], b['loss'], atol=1e-6) and ga.keys() == gb.keys()
    assert all(torch.allclose(ga[k], gb[k], atol=1e-5) for k in ga)
    assert torch.allclose(a['per_tick_loss'], b['per_tick_loss'], atol=1e-6)
    with torch.no_grad():
        assert torch.equal(ref(x)['logits'], new(x)['logits'])


def test_factories_name_the_model_and_checkpointing():
    f = scaled_factory('rdt', checkpointing=True)
    assert f.__qualname__ == 'scaled_factory[rdt,checkpointing]' and f(config('rdt')).config.gradient_checkpointing
    assert scaled_factory('ctm_lm')(config('ctm_lm')).checkpoint_ticks is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('model', ['transformer', 'rdt', 'ctm_lm'])
def test_scaled_models_train_under_bf16_autocast(model):
    m = scaled_factory(model, checkpointing=True)(replace(config(model), dtype='bfloat16')).cuda()
    x, y = (t.cuda() for t in batch())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = m(x, targets=y)['loss']
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_truncated_backpropagation_is_exact_when_it_covers_every_step_and_cuts_early_steps_otherwise():
    torch.manual_seed(0)
    full = ScaledBaseline(config('rdt'))
    torch.manual_seed(0)
    covered = ScaledBaseline(config('rdt'), backprop_steps=5)
    torch.manual_seed(0)
    cut = ScaledBaseline(config('rdt'), backprop_steps=2)
    x, y = batch()
    losses = []
    for m in (full, covered, cut):
        out = m(x, targets=y)
        out['loss'].backward()
        losses.append(out['loss'])
    assert torch.equal(losses[0], losses[1]) and torch.equal(losses[0], losses[2])  # the forward pass is unchanged
    g_full, g_cov, g_cut = grads(full), grads(covered), grads(cut)
    assert all(torch.equal(g_full[k], g_cov[k]) for k in g_full)
    assert not torch.allclose(g_full['core.0.attn.q.weight'], g_cut['core.0.attn.q.weight'])
    assert scaled_factory('rdt', backprop_steps=8).__qualname__ == 'scaled_factory[rdt,plain,backprop8]'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('model', ['transformer', 'rdt', 'ctm_lm'])
def test_compiled_paths_match_eager(model):
    torch.manual_seed(0)
    eager = scaled_factory(model, checkpointing=True)(config(model)).cuda()
    torch.manual_seed(0)
    compiled = scaled_factory(model, checkpointing=True, compile=True)(config(model)).cuda()
    x, y = (t.cuda() for t in batch())
    a, b = eager(x, targets=y), compiled(x, targets=y)
    a['loss'].backward()
    b['loss'].backward()
    ga, gb = grads(eager), grads(compiled)
    assert torch.allclose(a['loss'], b['loss'], atol=1e-4) and all(torch.allclose(ga[k], gb[k], atol=1e-4, rtol=1e-3) for k in ga)
