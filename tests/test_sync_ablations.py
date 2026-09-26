"""Checks for the Sync-RDT mechanism ablations."""
import os

import pytest
import torch

from ctm_transformer.research import load_research_config
from ctm_transformer.sync_ablations import KINDS, SyncAblation, ablation_factory
from ctm_transformer.sync_rdt import SyncRDT

DEVICE = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RECIPE = 'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'


def config():
    return load_research_config(RECIPE)[0]


def pair(kind, seed=3):
    torch.manual_seed(seed)
    reference = SyncRDT(config(), history=False, sync=True)
    torch.manual_seed(seed)
    return reference, SyncAblation(config(), kind)


def test_identity_ablation_reproduces_the_sync_cell_exactly():
    reference, ablation = pair('identity')
    a, b = reference.state_dict(), ablation.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    x = torch.randint(3, 71, (2, 20))
    for model in (reference, ablation):
        model.eval()
        with torch.no_grad():
            with torch.no_grad():
                model.sync.r.fill_(0.05)  # Nonzero decay, so the check is not trivially order-free.
            for layer in model.sync_query:
                layer.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(1))
    out_a = reference(x, max_thought_steps=5, return_all_logits=True)
    out_b = ablation(x, max_thought_steps=5, return_all_logits=True)
    assert torch.equal(out_a['logits'], out_b['logits'])
    assert all(torch.equal(p, q) for p, q in zip(out_a['all_logits'], out_b['all_logits']))


def test_ablations_change_only_what_they_declare():
    base = sum(p.numel() for p in pair('identity')[1].parameters())
    counts = {kind: sum(p.numel() for p in SyncAblation(config(), kind).parameters()) for kind in KINDS}
    assert counts['shuffled'] == counts['nodecay'] == counts['current'] == base
    assert counts['linear'] == base + 8 * 96 * 128
    assert counts['state'] == base - 2 * 128 * 96 + 128 * 96
    assert not SyncAblation(config(), 'nodecay').sync.r.requires_grad
    with pytest.raises(ValueError, match='Unknown'):
        ablation_factory('bogus')


@needs_cuda
@pytest.mark.parametrize('kind', [k for k in KINDS if k != 'identity'])
def test_ablations_are_causal_train_and_round_trip(kind):
    torch.manual_seed(0)
    model = SyncAblation(config(), kind).to(DEVICE)
    with torch.no_grad():
        model.sync.r.normal_(0, 0.1) if kind != 'nodecay' else None
        for layer in (model.sync_query or []):
            layer.weight.normal_(0, 0.05)
        if kind == 'state':
            model.state_projection.weight.normal_(0, 0.05)
    model.eval()
    x = torch.randint(3, 71, (2, 30), device=DEVICE)
    y = x.clone()
    y[:, 20:] = torch.randint(3, 71, (2, 10), device=DEVICE)
    with torch.no_grad():
        a = model(x, max_thought_steps=6)['logits']
        assert torch.allclose(a[:, :20], model(y, max_thought_steps=6)['logits'][:, :20], atol=1e-5)
        assert torch.equal(a, model(x, max_thought_steps=6)['logits'])  # Evaluation is reproducible, including shuffles.
    model.train()
    targets = torch.full_like(x, -100)
    targets[:, -3:] = x[:, -3:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(x, targets=targets, max_thought_steps=6)['loss'].backward()
    if kind == 'nodecay':
        assert model.sync.r.grad is None
    elif kind == 'linear':
        assert model.history_features.weight.grad.abs().sum() > 0
    elif kind == 'state':
        assert model.state_projection.weight.grad.abs().sum() > 0
    elif kind == 'current':
        # With one history step the decay term is exp(-r*0) = 1, so r is inert; the queries still learn.
        assert model.sync.r.grad.abs().sum() == 0 and all(q.weight.grad.abs().sum() > 0 for q in model.sync_query)
    else:
        assert model.sync.r.grad is not None and model.sync.r.grad.abs().sum() > 0
    clone = SyncAblation(config(), kind).to(DEVICE)
    clone.load_state_dict(model.state_dict(), strict=True)


@needs_cuda
def test_shuffling_changes_synchronization_only_when_decay_is_nonzero():
    torch.manual_seed(0)
    model = SyncAblation(config(), 'shuffled').to(DEVICE).eval()
    history = torch.randn(4, 8, 96, device=DEVICE)
    model._step = 3
    with torch.no_grad():
        model.sync.r.zero_()
        assert torch.allclose(model._features(history), model.sync.compute(history), atol=1e-5)
        model.sync.r.fill_(0.5)
        assert not torch.allclose(model._features(history), model.sync.compute(history), atol=1e-3)


def test_classification_rule():
    from scripts.summarize_sync_ablation import classify
    from scripts.summarize_group_s3 import exceeds
    sync, rdt, wide = [12, 10, 7, 16, 10], [4, 3, 5, 8, 8], [4, 6, 4, 5, 8]
    assert classify(exceeds(sync, rdt), exceeds(rdt, rdt), exceeds(rdt, wide)) == 'removes'
    assert classify(exceeds(sync, sync), exceeds(sync, rdt), exceeds(sync, wide)) == 'preserves'
    middle = [11, 9, 4, 15, 6]  # sync not larger by the margin; not larger than rdt in 4 seeds
    assert classify(exceeds(sync, middle), exceeds(middle, rdt), exceeds(middle, wide)) == 'partial'
