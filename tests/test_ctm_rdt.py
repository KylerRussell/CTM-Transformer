"""The CTM-augmented RDT starts as the RDT exactly, and each mechanism acts once its weights move (ctm_transformer/ctm_rdt.py)."""
from dataclasses import replace
from itertools import combinations

import pytest
import torch

from ctm_transformer.ctm_rdt import MECHANISMS, CTMAugmentedRDT, ctm_rdt_factory
from ctm_transformer.lm_scale import ScaledBaseline
from tests.test_lm_scale import batch, config, grads

ALL = [set(c) for n in range(len(MECHANISMS) + 1) for c in combinations(MECHANISMS, n)]


def pair(mechanisms, checkpointing=False):
    torch.manual_seed(0)
    ref = ScaledBaseline(config('rdt', checkpointing))
    torch.manual_seed(0)
    new = CTMAugmentedRDT(config('rdt', checkpointing), **{m: m in mechanisms for m in MECHANISMS}, pairs=24)
    return ref, new


@pytest.mark.parametrize('mechanisms', ALL, ids=lambda m: '+'.join(sorted(m)) or 'none')
def test_starts_as_the_rdt_with_identical_loss_logits_and_shared_gradients(mechanisms):
    ref, new = pair(mechanisms)
    shared = dict(ref.named_parameters())
    assert all(torch.equal(p, shared[n]) for n, p in new.named_parameters() if n in shared)
    x, y = batch()
    a, b = ref(x, targets=y, max_thought_steps=4), new(x, targets=y, max_thought_steps=4)
    a['loss'].backward()
    b['loss'].backward()
    assert torch.allclose(a['loss'], b['loss'], atol=1e-6)
    ga, gb = grads(ref), grads(new)
    assert all(torch.allclose(ga[k], gb[k], atol=1e-5) for k in ga)
    with torch.no_grad():
        assert torch.allclose(ref(x, max_thought_steps=4)['logits'], new(x, max_thought_steps=4)['logits'], atol=1e-5)


@pytest.mark.parametrize('mechanism', MECHANISMS)
def test_each_mechanism_receives_gradient_and_changes_the_output_once_moved(mechanism):
    ref, new = pair({mechanism})
    x, y = batch()
    new(x, targets=y, max_thought_steps=4)['loss'].backward()
    added = {n: p for n, p in new.named_parameters() if n not in dict(ref.named_parameters())}
    assert added and all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in added.items() if 'decay' not in n)
    with torch.no_grad():
        before = new(x, max_thought_steps=4)['logits']
        for p in added.values():
            p.add_(0.1 * torch.randn_like(p))
        assert not torch.allclose(before, new(x, max_thought_steps=4)['logits'], atol=1e-4)


def test_checkpointing_is_exact_with_every_mechanism():
    torch.manual_seed(0)
    plain = CTMAugmentedRDT(config('rdt'), sync_query=True, sync_readout=True, learned_init=True, pairs=24)
    torch.manual_seed(0)
    ckpt = CTMAugmentedRDT(config('rdt', True), sync_query=True, sync_readout=True, learned_init=True, pairs=24)
    with torch.no_grad():
        for (_, p), (_, q) in zip(plain.named_parameters(), ckpt.named_parameters()):
            v = 0.05 * torch.randn_like(p)
            p.add_(v)
            q.add_(v)
    x, y = batch()
    a, b = plain(x, targets=y, max_thought_steps=5), ckpt(x, targets=y, max_thought_steps=5)
    a['loss'].backward()
    b['loss'].backward()
    ga, gb = grads(plain), grads(ckpt)
    assert torch.allclose(a['loss'], b['loss'], atol=1e-6) and ga.keys() == gb.keys()
    assert all(torch.allclose(ga[k], gb[k], atol=1e-5) for k in ga)


def test_factory_names_mechanisms_and_rejects_unknown_ones():
    f = ctm_rdt_factory({'sync_query', 'learned_init'}, checkpointing=True)
    m = f(config('rdt'))
    assert f.__qualname__ == 'ctm_rdt_factory[learned_init+sync_query]' and m.config.gradient_checkpointing
    assert m.mechanisms == {'sync_query': True, 'sync_readout': False, 'learned_init': True}
    with pytest.raises(ValueError):
        ctm_rdt_factory({'history'})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_bf16_training_matches_eager():
    torch.manual_seed(0)
    eager = ctm_rdt_factory(set(MECHANISMS), checkpointing=True, pairs=24)(config('rdt')).cuda()
    torch.manual_seed(0)
    compiled = ctm_rdt_factory(set(MECHANISMS), checkpointing=True, compile=True, pairs=24)(config('rdt')).cuda()
    x, y = (t.cuda() for t in batch())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        a, b = eager(x, targets=y, max_thought_steps=4), compiled(x, targets=y, max_thought_steps=4)
    a['loss'].backward()
    b['loss'].backward()
    assert torch.isfinite(b['loss']) and torch.allclose(a['loss'], b['loss'], atol=2e-2)
