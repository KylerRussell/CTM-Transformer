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


def test_factory_rejects_unknown_adaptations():
    assert adapted_factory({'unit_query'})(config('ctm_lm')).adaptations == {'unit_query': True, 'observe_token': False}
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
