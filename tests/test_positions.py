"""Checks for the RoPE model factories (ctm_transformer/positions.py)."""
from dataclasses import replace

import pytest
import torch

from ctm_transformer import ctm_lm
from ctm_transformer.baselines import BaselineTransformer
from ctm_transformer.ctm_lm import CTMLM, lm_config
from ctm_transformer.positions import RoPEBaseline, RoPECTMLM, position_factory, rope
from ctm_transformer.research import build_model, load_research_config

RECIPES = {'transformer': 'research/configs/presentation_control_v1/transformer_shuffled_seed23.json',
           'rdt': 'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json',
           'ctm_lm': 'research/configs/presentation_control_v1/ctm_shuffled_seed23.json'}


def config(model, table=False):
    c = load_research_config(RECIPES[model])[0]
    if model == 'ctm_lm':
        c = lm_config(c)
    return replace(c, use_positional_encoding=table, max_thought_steps=4 if model != 'transformer' else 1, dtype='float32')


def pair(model, scheme='rope'):
    torch.manual_seed(0)
    a = position_factory(model, scheme)(config(model))
    torch.manual_seed(0)
    b = CTMLM(config(model)) if model == 'ctm_lm' else BaselineTransformer(config(model))
    return a.eval(), b.eval()


def tokens(length=12, batch=2, seed=3):
    return torch.randint(3, 20, (batch, length), generator=torch.Generator().manual_seed(seed))


def test_rope_matches_ctm_lm_rope_on_even_widths_and_passes_the_odd_channel_through():
    x = torch.randn(2, 3, 7, 16)
    positions = torch.arange(7)
    assert torch.equal(rope(x, positions), ctm_lm.rope(x, positions))
    odd = torch.randn(1, 2, 5, 35)
    out = rope(odd, torch.arange(5))
    assert torch.equal(out[..., 34], odd[..., 34])
    assert torch.allclose(out[..., :34], ctm_lm.rope(odd[..., :34], torch.arange(5)))


@pytest.mark.parametrize('model', ['transformer', 'rdt', 'ctm_lm'])
def test_rope_models_have_the_nope_parameters_and_initialization(model):
    a, b = pair(model)
    assert a.pos_embedding is None and b.pos_embedding is None
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize('model', ['transformer', 'rdt', 'ctm_lm'])
def test_rope_at_position_zero_is_exactly_nope(model):
    a, b = pair(model)
    blocks = a.backbone if model == 'ctm_lm' else [blk.attn for blk in (*a.layers, *a.prelude, *a.core, *a.coda)]
    x = tokens()
    for block in blocks:
        block.positions = torch.zeros(x.shape[1], dtype=torch.long)
    if model == 'ctm_lm':  # the tick attention already uses RoPE in CTM-LM; only the backbone differs
        assert torch.allclose(a.features(x), b.features(x), atol=1e-6)
    else:
        assert torch.allclose(a(x)['logits'], b(x)['logits'], atol=1e-6)
        for block in blocks:
            block.positions = None
        assert not torch.allclose(a(x)['logits'], b(x)['logits'], atol=1e-4)


@pytest.mark.parametrize('model', ['transformer', 'rdt', 'ctm_lm'])
def test_rope_models_are_causal(model):
    a, _ = pair(model)
    x = tokens()
    y = x.clone()
    y[:, 7:] = tokens(seed=9)[:, 7:]
    with torch.no_grad():
        assert torch.allclose(a(x)['logits'][:, :7], a(y)['logits'][:, :7], atol=1e-5)


@pytest.mark.parametrize('model', ['transformer', 'rdt'])
def test_rope_models_see_only_relative_positions(model):
    a, _ = pair(model)
    blocks = [blk.attn for blk in (*a.layers, *a.prelude, *a.core, *a.coda)]
    x = tokens()
    with torch.no_grad():
        base = a(x)['logits']
        for block in blocks:
            block.positions = torch.arange(x.shape[1]) + 17
        assert torch.allclose(a(x)['logits'], base, atol=1e-4)


def test_rope_models_require_the_table_disabled_and_nope_uses_the_frozen_classes():
    with pytest.raises(ValueError):
        RoPEBaseline(config('rdt', table=True))
    with pytest.raises(ValueError):
        RoPECTMLM(config('ctm_lm', table=True))
    with pytest.raises(ValueError):
        position_factory('rdt', 'nope')(config('rdt', table=True))
    assert type(position_factory('rdt', 'nope')(config('rdt'))) is BaselineTransformer
    assert type(position_factory('ctm_lm', 'nope')(config('ctm_lm'))) is CTMLM
    assert position_factory('rdt', 'rope').__qualname__ == 'position_factory[rdt,rope]'


def test_nope_rdt_is_the_frozen_build_model():
    torch.manual_seed(0)
    a = position_factory('rdt', 'nope')(config('rdt'))
    torch.manual_seed(0)
    b = build_model(config('rdt'))
    x = tokens()
    with torch.no_grad():
        assert torch.equal(a(x)['logits'], b(x)['logits'])


@pytest.mark.parametrize('model', ['transformer', 'rdt', 'ctm_lm'])
def test_rope_models_train(model):
    a, _ = pair(model)
    a.train()
    x = tokens()
    out = a(x, targets=x)
    out['loss'].backward()
    grads = [p.grad for p in a.parameters() if p.requires_grad]
    assert torch.isfinite(out['loss']) and all(g is not None and torch.isfinite(g).all() for g in grads)
