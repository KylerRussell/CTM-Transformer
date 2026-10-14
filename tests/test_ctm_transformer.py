"""The CTM Transformer (ctm_transformer/ctm_transformer.py): starts as Dans exactly, every mechanism trains and acts."""
import pytest
import torch
from dataclasses import replace

from ctm_transformer.ctm_transformer import DANS, MECHANISMS, ctm_transformer_factory
from ctm_transformer.rdt_recipe import recipe_factory
from tests.test_lm_scale import batch, config, grads

CASES = [(), *[(m,) for m in MECHANISMS], MECHANISMS]
ZERO_INIT = {'neuron_models': 'nlm_scale', 'sync_query': 'sync_queries.0.weight', 'sync_output': 'sync_projection.weight'}


def pair(mechanisms, depth=4, checkpointing=False):
    c = replace(config('rdt'), max_thought_steps=depth)
    torch.manual_seed(0)
    model = ctm_transformer_factory(set(mechanisms), checkpointing=checkpointing)(c)
    dans = recipe_factory(DANS)(c)
    dans.load_state_dict({k: v for k, v in model.state_dict().items() if k in dans.state_dict()})
    return model, dans


@pytest.mark.parametrize('mechanisms', CASES)
@pytest.mark.parametrize('depth', [1, 4])
def test_untrained_ctm_transformer_is_dans_and_every_mechanism_gets_a_gradient(mechanisms, depth):
    model, dans = pair(mechanisms, depth)
    x, y = batch()
    with torch.no_grad():
        a, b = model(x, return_all_logits=True), dans(x, return_all_logits=True)
        assert torch.allclose(a['logits'], b['logits'], atol=1e-5)
        assert all(torch.allclose(p, q, atol=1e-5) for p, q in zip(a['all_logits'], b['all_logits']))
        assert torch.allclose(model(x, targets=y)['loss'], dans(x, targets=y)['loss'], atol=1e-5)
    loss = model(x, targets=y)['loss']
    loss.backward()
    params = dict(model.named_parameters())
    for m in mechanisms:
        if m == 'sync_query' and depth == 1:
            continue  # the query reads synchronization from earlier ticks; there are none at depth 1
        assert params[ZERO_INIT[m]].grad.abs().sum() > 0, m


@pytest.mark.parametrize('mechanism', MECHANISMS)
def test_each_mechanism_changes_the_output_once_its_zero_initialized_parameter_is_set(mechanism):
    model, _ = pair((mechanism,))
    x, _ = batch()
    with torch.no_grad():
        before = model(x)['logits']
        dict(model.named_parameters())[ZERO_INIT[mechanism]].normal_(std=0.1)
        if mechanism == 'sync_query':
            for layer in model.sync_queries:
                layer.weight.normal_(std=0.1)
        assert not torch.allclose(before, model(x)['logits'], atol=1e-4)


def test_checkpointed_execution_matches_plain_execution():
    torch.manual_seed(0)
    plain, _ = pair(MECHANISMS)
    checkpointed, _ = pair(MECHANISMS, checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    with torch.no_grad():
        for m in (plain, checkpointed):
            m.nlm_scale.normal_(std=0.1)
            m.sync_projection.weight.normal_(std=0.1)
            for layer in m.sync_queries:
                layer.weight.normal_(std=0.1)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    x, y = batch()
    a, b = plain(x, targets=y)['loss'], checkpointed(x, targets=y)['loss']
    a.backward()
    b.backward()
    ga, gb = grads(plain), grads(checkpointed)
    assert torch.allclose(a, b, atol=1e-6) and all(torch.allclose(ga[k], gb[k], atol=1e-5) for k in ga)


def test_synchronization_decay_has_a_gradient_and_reads_a_tick_count_independent_average():
    model, _ = pair(('sync_output',))
    sync = model.sync_out
    z = torch.randn(3, model.config.d_model)
    state = None
    for _ in range(5):
        state = sync.step(state, z)  # the same state every tick: the normalized average equals one tick's products
    assert torch.allclose(sync.read(state), z[:, sync.left] * z[:, sync.right], atol=1e-5)
    sync.read(state).sum().backward()
    assert sync.rho.grad is not None


def test_factory_rejects_unknown_mechanisms():
    with pytest.raises(ValueError):
        ctm_transformer_factory({'tick_memory'})


def test_the_fused_neuron_model_kernel_and_the_matmul_pair_products_match_their_definitions():
    from ctm_transformer.ctm_lm import NeuronLevelModels
    from ctm_transformer.ctm_transformer import NLM_HIDDEN, NLM_HISTORY, _neuron_mlp
    torch.manual_seed(0)
    nlm = NeuronLevelModels(12, NLM_HISTORY, NLM_HIDDEN)
    x = torch.randn(5, 12, NLM_HISTORY)
    assert torch.allclose(_neuron_mlp(x, nlm.first.weight, nlm.first.bias, nlm.second.weight, nlm.second.bias), nlm(x), atol=1e-6)
    model, _ = pair(('sync_output',))
    z = torch.randn(3, model.config.d_model)
    s = model.sync_out
    assert torch.allclose(s.products(z), z[:, s.left] * z[:, s.right], atol=1e-6)
