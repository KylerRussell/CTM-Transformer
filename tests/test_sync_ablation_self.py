"""Checks for ablation A7 (self-pairs only)."""
import os

import pytest
import torch

from ctm_transformer.research import load_research_config
from ctm_transformer.sync_ablation_self import SelfPairAblation
from ctm_transformer.sync_ablations import SyncAblation

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
RECIPE = 'research/configs/presentation_control_v1/recurrent_depth_shuffled_seed23.json'


def test_self_pairs_differ_from_sync_only_in_the_right_indices():
    c = load_research_config(RECIPE)[0]
    torch.manual_seed(4)
    sync = SyncAblation(c, 'identity').state_dict()
    torch.manual_seed(4)
    model = SelfPairAblation(c)
    state = model.state_dict()
    assert sync.keys() == state.keys()
    assert all(torch.equal(sync[k], state[k]) for k in sync if k != 'sync.idxs_right')
    assert torch.equal(state['sync.idxs_right'], state['sync.idxs_left'])
    history = torch.randn(3, 8, c.d_model)
    features = model.sync.compute(history)
    expected = (history[:, :, state['sync.idxs_left']] ** 2).sum(1) / torch.sqrt(torch.tensor(8.0))  # r = 0: equal weights
    assert torch.allclose(features, expected, atol=1e-5) and bool((features >= 0).all())


@needs_cuda
def test_self_pair_ablation_is_causal_trains_and_round_trips():
    device = os.environ.get('CTM_TEST_DEVICE', 'cuda:0')
    c = load_research_config(RECIPE)[0]
    torch.manual_seed(0)
    model = SelfPairAblation(c).to(device)
    with torch.no_grad():
        for layer in model.sync_query:
            layer.weight.normal_(0, 0.05)
    model.eval()
    x = torch.randint(3, 71, (2, 30), device=device)
    y = x.clone()
    y[:, 20:] = torch.randint(3, 71, (2, 10), device=device)
    with torch.no_grad():
        assert torch.allclose(model(x, max_thought_steps=6)['logits'][:, :20], model(y, max_thought_steps=6)['logits'][:, :20], atol=1e-5)
    model.train()
    targets = torch.full_like(x, -100)
    targets[:, -3:] = x[:, -3:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(x, targets=targets, max_thought_steps=6)['loss'].backward()
    assert all(q.weight.grad.abs().sum() > 0 for q in model.sync_query)
    clone = SelfPairAblation(c).to(device)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(clone.sync.idxs_right, clone.sync.idxs_left)
