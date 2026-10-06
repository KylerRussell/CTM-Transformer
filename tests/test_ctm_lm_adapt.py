"""CTM-LM adaptations (ctm_transformer/ctm_lm_adapt.py): none is the scaled CTM-LM exactly; A and B do what they claim."""
import pytest
import torch

from ctm_transformer.ctm_lm import Synchronization
from ctm_transformer.ctm_lm_adapt import AdaptedCTMLM, adapted_factory
from ctm_transformer.lm_scale import ScaledCTMLM
from tests.test_lm_scale import batch, config, grads


@pytest.mark.parametrize('checkpointing', [False, True])
def test_without_adaptations_it_is_the_scaled_ctm_lm(checkpointing):
    torch.manual_seed(0)
    ref = ScaledCTMLM(config('ctm_lm'), checkpoint_ticks=checkpointing)
    torch.manual_seed(0)
    new = AdaptedCTMLM(config('ctm_lm'), checkpoint_ticks=checkpointing)
    x, y = batch()
    a, b = ref(x, targets=y), new(x, targets=y)
    a['loss'].backward()
    b['loss'].backward()
    ga, gb = grads(ref), grads(new)
    assert torch.equal(a['loss'], b['loss']) and ga.keys() == gb.keys() and all(torch.equal(ga[k], gb[k]) for k in ga)
    with torch.no_grad():
        ra, rb = ref(x, return_all_logits=True), new(x, return_all_logits=True)
        assert torch.allclose(ra['logits'], rb['logits'], atol=1e-6) and len(ra['all_logits']) == len(rb['all_logits'])


def test_unit_query_scales_the_tick_zero_query_to_unit_rms_and_changes_nothing_else():
    torch.manual_seed(0)
    ref = AdaptedCTMLM(config('ctm_lm'))
    torch.manual_seed(0)
    new = AdaptedCTMLM(config('ctm_lm'), unit_query=True)
    with torch.no_grad():
        q = new.query(Synchronization.read(*new.sync_action.start(new.z_init[None].float())))
    assert q.pow(2).mean().sqrt() == pytest.approx(1.0, rel=1e-4)
    ratio = new.query.weight / ref.query.weight
    assert torch.allclose(ratio, ratio.flatten()[0].expand_as(ratio), rtol=1e-4) and ratio.flatten()[0] > 1
    assert all(torch.equal(p, dict(ref.named_parameters())[n]) for n, p in new.named_parameters() if n != 'query.weight')


def test_observe_token_makes_the_prediction_depend_on_the_current_token_at_initialization():
    torch.manual_seed(0)
    faithful = AdaptedCTMLM(config('ctm_lm'))
    torch.manual_seed(0)
    observing = AdaptedCTMLM(config('ctm_lm'), observe_token=True)
    x, _ = batch(B=1, S=11)
    x2 = x.clone()
    x2[0, -1] = (x[0, -1] + 1) % 71  # change only the last token
    with torch.no_grad():
        def change(m):
            return float((m(x)['logits'][0, -1] - m(x2)['logits'][0, -1]).abs().max())
        assert change(observing) > 5 * change(faithful)  # at 11 tokens the faithful prefix average dilutes the token about 11-fold
    assert observing.synapse.down1.in_features == 2 * observing.config.d_model + observing.config.d_latent
    x, y = batch()
    loss = observing(x, targets=y)['loss']
    loss.backward()
    assert torch.isfinite(loss) and observing.synapse.down1.weight.grad.abs().sum() > 0


def test_mean_tick_loss_is_the_mean_cross_entropy_over_ticks():
    torch.manual_seed(0)
    m = AdaptedCTMLM(config('ctm_lm'), mean_tick_loss=True)
    x, y = batch()
    loss = m(x, targets=y)['loss']
    with torch.no_grad():
        logits = m(x, return_all_logits=True)['all_logits']
    keep = y.ne(-100)
    expected = torch.stack([torch.nn.functional.cross_entropy(l[keep].float(), y[keep]) for l in logits]).mean()
    assert torch.allclose(loss, expected, atol=1e-5)
    loss.backward()
    assert m.query.weight.grad is not None


def test_factory_rejects_unknown_adaptations():
    assert adapted_factory({'unit_query'})(config('ctm_lm')).adaptations['unit_query']
    with pytest.raises(ValueError):
        adapted_factory({'token_shortcut'})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_checkpointed_bf16_training_runs():
    m = adapted_factory({'unit_query', 'observe_token'}, checkpointing=True, compile=True)(config('ctm_lm')).cuda()
    x, y = (t.cuda() for t in batch())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = m(x, targets=y)['loss']
    loss.backward()
    assert torch.isfinite(loss)


def test_query_feature_lets_the_tick_attention_learn():
    # A random query projection of a token's own feature is not aligned with its key, so it does not make the
    # output token-dependent at initialization; it gives the attention a non-negligible query, so keys get gradient.
    norms = []
    for kw in ({}, {'query_feature': True}):
        torch.manual_seed(0)
        m = AdaptedCTMLM(config('ctm_lm'), mean_tick_loss=True, **kw)
        x, y = batch()
        m(x, targets=y)['loss'].backward()
        norms.append(float(m.key.weight.grad.norm()))
    assert norms[1] > 50 * norms[0]


@pytest.mark.parametrize('adaptation', ['token_start', 'token_history'])
def test_token_start_and_history_make_the_prediction_depend_on_the_current_token(adaptation):
    torch.manual_seed(0)
    faithful = AdaptedCTMLM(config('ctm_lm'))
    torch.manual_seed(0)
    adapted = AdaptedCTMLM(config('ctm_lm'), **{adaptation: True})
    assert all(torch.equal(p, dict(faithful.named_parameters())[n]) for n, p in adapted.named_parameters() if n in dict(faithful.named_parameters()))
    x, _ = batch(B=1, S=11)
    x2 = x.clone()
    x2[0, -1] = (x[0, -1] + 1) % 71
    with torch.no_grad():
        def change(m):
            return float((m(x)['logits'][0, -1] - m(x2)['logits'][0, -1]).abs().max())
        assert change(adapted) > 5 * change(faithful)


def test_final_tick_and_certainty_losses_equal_their_definitions():
    x, y = batch()
    keep = y.ne(-100)
    torch.manual_seed(0)
    final = AdaptedCTMLM(config('ctm_lm'), final_tick_loss=True)
    torch.manual_seed(0)
    certain = AdaptedCTMLM(config('ctm_lm'), certainty_loss=True)
    with torch.no_grad():
        logits = torch.stack(final(x, return_all_logits=True)['all_logits'], dim=2)[keep].float()  # [n, T, V]
    logp = logits.log_softmax(-1)
    ce = -logp.gather(-1, y[keep][:, None, None].expand(-1, logits.shape[1], 1)).squeeze(-1)
    most_certain = (logp.exp() * logp).sum(-1).argmax(-1)
    assert torch.allclose(final(x, targets=y)['loss'], ce[:, -1].mean(), atol=1e-5)
    assert torch.allclose(certain(x, targets=y)['loss'], ce.gather(1, most_certain[:, None]).mean(), atol=1e-5)


def test_state_readout_adds_a_trained_projection_of_the_state_and_losses_are_exclusive():
    m = AdaptedCTMLM(config('ctm_lm'), state_readout=True, mean_tick_loss=True)
    x, y = batch()
    m(x, targets=y)['loss'].backward()
    assert m.state_readout.weight.grad is not None and m.state_readout.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        AdaptedCTMLM(config('ctm_lm'), mean_tick_loss=True, final_tick_loss=True)


def test_sparse_tick_loss_is_the_mean_over_every_quarter_tick():
    from ctm_transformer.ctm_lm_adapt import sparse_ticks
    assert sparse_ticks(16) == [3, 7, 11, 15] and sparse_ticks(5) == [1, 2, 3, 4] and sparse_ticks(2) == [0, 1]
    torch.manual_seed(0)
    m = AdaptedCTMLM(config('ctm_lm'), sparse_tick_loss=True)
    x, y = batch()
    keep = y.ne(-100)
    with torch.no_grad():
        logits = m(x, return_all_logits=True)['all_logits']
    T = len(logits)
    expected = torch.stack([torch.nn.functional.cross_entropy(logits[t][keep].float(), y[keep]) for t in sparse_ticks(T)]).mean()
    assert torch.allclose(m(x, targets=y)['loss'], expected, atol=1e-5)


def test_readout_evaluation_reports_the_most_certain_trained_tick_for_sparse_tick_models():
    from ctm_transformer.pretrain import ctm_readout_losses
    torch.manual_seed(0)
    m = AdaptedCTMLM(config('ctm_lm'), token_start=True, sparse_tick_loss=True).eval()
    x, y = batch()
    y = y.clamp_min(0)
    with torch.no_grad():
        losses = ctm_readout_losses(m, x, y)
    assert set(losses) == {'final_tick', 'most_certain_tick', 'most_certain_trained_tick'} and all(v > 0 for v in losses.values())
    with torch.no_grad():
        assert set(ctm_readout_losses(AdaptedCTMLM(config('ctm_lm')).eval(), x, y)) == {'final_tick', 'most_certain_tick'}


def test_decay_fixes_keep_a_gradient_and_start_where_intended():
    import math
    x, y = batch()
    torch.manual_seed(0)
    soft = AdaptedCTMLM(config('ctm_lm'), token_start=True, sparse_tick_loss=True, decay_softplus=True)
    assert torch.allclose(torch.nn.functional.softplus(soft.sync_out.decay), torch.full_like(soft.sync_out.decay, 0.01), atol=1e-6)
    with torch.no_grad():
        soft.sync_out.decay.fill_(-3.0)  # where the clamped version would have no gradient
    soft(x, targets=y)['loss'].backward()
    assert soft.sync_out.decay.grad.abs().sum() > 0
    torch.manual_seed(0)
    spread = AdaptedCTMLM(config('ctm_lm'), token_start=True, sparse_tick_loss=True, decay_spread=True)
    r = spread.sync_out.decay
    assert 0 <= float(r.min()) and float(r.max()) <= 3 and float(r.std()) > 0.5
    torch.manual_seed(0)
    reference = AdaptedCTMLM(config('ctm_lm'), token_start=True, sparse_tick_loss=True)
    shared = dict(reference.named_parameters())
    assert all(torch.equal(p, shared[n]) for n, p in spread.named_parameters() if 'decay' not in n)
