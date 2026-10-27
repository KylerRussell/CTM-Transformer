"""The CTM Transformer (ctm_transformer/ctm_transformer.py): starts as Dans exactly, every mechanism trains and acts."""
import pytest
import torch
from dataclasses import replace

from ctm_transformer.ctm_transformer import DANS, MECHANISMS, ctm_transformer_factory
from ctm_transformer.rdt_recipe import recipe_factory
from tests.test_lm_scale import batch, config, grads

CASES = [(), *[(m,) for m in MECHANISMS], MECHANISMS]
ZERO_INIT = {'neuron_models': 'nlm_scale', 'sync_query': 'sync_queries.0.weight', 'sync_output': 'sync_projection.weight', 'attention_sink': 'sink_gate'}


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
        if mechanism == 'attention_sink':
            model.sink_keys.normal_(std=1.0)
            model.sink_gate.fill_(1.0)
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


def test_attention_sink_with_an_open_gate_is_attention_over_the_sink_and_the_causal_prefix():
    from torch.nn import functional as F
    model, _ = pair(('attention_sink',), depth=1)
    with torch.no_grad():
        model.sink_keys.normal_(std=1.0)
        model.sink_gate.fill_(1.0)
    B, H, S, dh = 2, model.core[0].attn.heads, 5, model.config.d_model // model.core[0].attn.heads
    q, k, v = torch.randn(B, H, S, dh), torch.randn(B, H, S, dh), torch.randn(B, H, S, dh)
    sink = model.sink_keys[0][None, :, None, :].expand(B, -1, 1, -1)
    zero = torch.zeros_like(q[:, :, :1])
    got = F.scaled_dot_product_attention(torch.cat((zero, q), 2), torch.cat((sink, k), 2), torch.cat((zero, v), 2), is_causal=True)[:, :, 1:]
    logits = torch.cat(((q * sink).sum(-1, keepdim=True), q @ k.transpose(-1, -2)), -1) / dh ** 0.5
    mask = torch.cat((torch.ones(S, 1, dtype=torch.bool), torch.ones(S, S, dtype=torch.bool).tril()), -1)
    weights = logits.masked_fill(~mask, float('-inf')).softmax(-1)
    assert torch.allclose(got, weights[..., 1:] @ v, atol=1e-5)


def test_ctm_loss_is_the_mean_of_the_min_loss_and_most_certain_tick_cross_entropy():
    from ctm_transformer.ctm_lm import ctm_loss
    c = replace(config('rdt'), max_thought_steps=4)
    torch.manual_seed(0)
    model = ctm_transformer_factory({'sync_query', 'ctm_loss'})(c)
    with torch.no_grad():
        for layer in model.sync_queries:
            layer.weight.normal_(std=0.1)
    x, y = batch()
    got = model(x, targets=y)['loss']
    with torch.no_grad():
        logits = torch.stack(model(x, return_all_logits=True)['all_logits'], dim=2)
        expected = ctm_loss(logits, y)['loss']
    assert torch.allclose(got, expected, atol=1e-4)
    got.backward()
    assert model.sync_queries[0].weight.grad.abs().sum() > 0


def test_ctm_plus_final_tick_loss_is_the_mean_of_the_two_and_needs_the_ctm_loss():
    from ctm_transformer.ctm_lm import ctm_loss
    c = replace(config('rdt'), max_thought_steps=4)
    torch.manual_seed(0)
    model = ctm_transformer_factory({'sync_query', 'ctm_loss', 'final_tick_loss'})(c)
    x, y = batch()
    got = model(x, targets=y)['loss']
    with torch.no_grad():
        logits = torch.stack(model(x, return_all_logits=True)['all_logits'], dim=2)
        final = torch.nn.functional.cross_entropy(logits[:, :, -1].reshape(-1, logits.shape[-1]), y.reshape(-1))
        expected = (ctm_loss(logits, y)['loss'] + final) / 2
    assert torch.allclose(got, expected, atol=1e-4)
    with pytest.raises(ValueError):
        ctm_transformer_factory({'final_tick_loss'})(c)


def test_recorded_statistics_cover_every_tick_and_the_selection_shares_sum_to_one():
    c = replace(config('rdt'), max_thought_steps=4)
    torch.manual_seed(0)
    model = ctm_transformer_factory({'sync_query', 'ctm_loss'})(c)
    model.record_stats = True
    x, y = batch()
    model(x, targets=y)['loss'].backward()
    s = model.stats
    assert s['state_rms'].shape == (4,) and s['per_tick_ce'].shape == (4,)
    for k in ('min_loss_tick_share', 'most_certain_tick_share'):
        assert torch.allclose(s[k].sum(), torch.tensor(1.0))


def test_detached_readouts_train_the_recurrence_only_through_the_final_tick():
    c = replace(config('rdt'), max_thought_steps=4)
    x, y = batch()
    models = {}
    for name, objectives in (('detached', {'ctm_loss', 'final_tick_loss', 'detached_readouts'}), ('attached', {'ctm_loss', 'final_tick_loss'}), ('final', set())):
        torch.manual_seed(0)
        models[name] = ctm_transformer_factory({'sync_query'} | objectives)(c)
    for m in models.values():
        m.load_state_dict(models['final'].state_dict())
        with torch.no_grad():
            for layer in m.sync_queries:
                torch.manual_seed(1)
                layer.weight.normal_(std=0.1)
    losses = {k: m(x, targets=y)['loss'] for k, m in models.items()}
    assert torch.allclose(losses['detached'], losses['attached'], atol=1e-5)
    for loss in losses.values():
        loss.backward()
    gd, gf = grads(models['detached']), grads(models['final'])
    for k in gd:
        if k.startswith(('token_embedding', 'prelude', 'core', 'sync_queries', 'state_norm')):
            assert torch.allclose(gd[k], gf[k] / 2, atol=1e-6), k
    coda = [k for k in gd if k.startswith('coda')]
    assert any(not torch.allclose(gd[k], gf[k] / 2, atol=1e-6) for k in coda)
    with pytest.raises(ValueError):
        ctm_transformer_factory({'ctm_loss', 'detached_readouts'})(c)


def test_homeostasis_starts_as_dans_and_moves_thresholds_toward_the_target_rate():
    from ctm_transformer.ctm_transformer import HOMEOSTASIS_RATE, HOMEOSTASIS_TARGET
    model, dans = pair(('sync_query', 'homeostasis'))
    x, y = batch()
    with torch.no_grad():
        assert torch.allclose(model(x)['logits'], dans(x)['logits'], atol=1e-5)
    model.train()
    model(x, targets=y)['loss'].backward()
    rates = [a / c for a, c in zip(model._activity, model._activity_count)]
    model.bio_step(0, 100)
    for i, rate in enumerate(rates):
        assert torch.allclose(model.homeostatic_threshold[i], HOMEOSTASIS_RATE * (rate - HOMEOSTASIS_TARGET), atol=1e-6)
    assert all(a is None for a in model._activity)
    assert 'homeostatic_threshold' in model.state_dict()
    with torch.no_grad():
        assert not torch.allclose(model(x)['logits'], dans(x)['logits'], atol=1e-5)  # the shifted thresholds act


def test_homeostatic_activity_is_counted_once_per_forward_with_checkpointing():
    counts = []
    for checkpointing in (False, True):
        torch.manual_seed(0)
        model = ctm_transformer_factory({'homeostasis'}, checkpointing=checkpointing)(replace(config('rdt'), max_thought_steps=3))
        model.train()
        x, y = batch()
        model(x, targets=y)['loss'].backward()
        counts.append((model._activity_count, [a.clone() for a in model._activity]))
    assert counts[0][0] == counts[1][0] == [3] * len(counts[0][0])
    assert all(torch.allclose(a, b) for a, b in zip(counts[0][1], counts[1][1]))


def test_structural_plasticity_masks_to_the_largest_weights_and_refreshes_on_schedule():
    from ctm_transformer.ctm_transformer import SP_DENSITY
    model, dans = pair(('structural_plasticity',))
    for name in ('gate', 'up', 'down'):
        mask = getattr(model, f'sp_mask_{name}')
        assert abs(mask[0].float().mean().item() - SP_DENSITY) < 0.01
    with torch.no_grad():
        for i, block in enumerate(dans.core):
            for name in ('gate', 'up', 'down'):
                getattr(block, name).weight.mul_(getattr(model, f'sp_mask_{name}')[i])
    x, _ = batch()
    with torch.no_grad():
        assert torch.allclose(model(x)['logits'], dans(x)['logits'], atol=1e-5)
        model.core[0].gate.weight.mul_(-3).add_(1.0)
        before = model.sp_mask_gate.clone()
        model.bio_step(0, 300)  # refreshes every 10 steps of 300: not at step 0
        assert torch.equal(model.sp_mask_gate, before)
        model.bio_step(9, 300)
        assert torch.equal(model.sp_mask_gate[0], model._top_k(model.core[0].gate.weight))
