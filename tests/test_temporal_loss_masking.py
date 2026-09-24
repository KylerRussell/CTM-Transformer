"""Check actual CTM temporal objectives against masked token-loss oracles."""
from dataclasses import replace
import os
import pytest
import torch
import torch.nn.functional as F
from ctm_transformer.model import CTMTransformer
from ctm_transformer.research import load_research_config


def model_and_data(mode,mono=0.):
    config,_=load_research_config('research/configs/ctm_ordered_long_v1.json')
    config=replace(config,d_model=32,d_latent=32,n_heads=4,n_layers=1,history_len=4,
        nlm_groups=32,nlm_hidden_dim=8,sync_sparse_pairs=16,max_thought_steps=3,
        temporal_loss_type=mode,mono_penalty_weight=mono,mono_penalty_decay_until_frac=0.)
    device=os.environ.get('CTM_TEST_DEVICE','cuda:0');torch.manual_seed(27);torch.set_num_threads(4)
    model=CTMTransformer(config).to(device).eval()
    ids=torch.randint(3,config.vocab_size,(2,9),device=device)
    targets=torch.full_like(ids,-100);targets[0,-2:]=torch.tensor([4,2],device=device)
    # Second example is entirely ignored; the first has only two labels.
    return model,ids,targets


@pytest.mark.parametrize('mode,mono',[('final_ce',0.),('ramp_mono',0.2),('dynamic_aggregate',0.),('dynamic_aggregate',0.2)])
def test_masked_objective_and_gradients(mode,mono):
    model,ids,targets=model_and_data(mode,mono)
    result=model(ids,targets=targets);logits=result['all_logits'];valid=targets.flatten()!=-100
    losses=torch.stack([F.cross_entropy(x.reshape(-1,x.shape[-1]),targets.flatten(),reduction='none') for x in logits],dim=1)
    per_tick=losses[valid].mean(0)
    if mode=='final_ce':base=per_tick[-1]
    elif mode=='ramp_mono':
        weights=torch.linspace(.5,1.5,len(logits),device=ids.device);base=(per_tick*weights/weights.sum()).sum()
    else:
        certainty=result['certainties'].reshape(len(logits),-1).T
        chosen=certainty.argmax(1)
        base=((losses.min(1).values+losses.gather(1,chosen[:,None]).squeeze(1))/2)[valid].mean()
    oracle=base+mono*F.relu(per_tick[1:]-per_tick[:-1]).mean()
    torch.testing.assert_close(result['loss'],oracle,rtol=1e-5,atol=1e-6)
    weight=model.output_proj[-1].weight
    actual_grad=torch.autograd.grad(result['loss'],weight,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(oracle,weight)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-4,atol=1e-6)


def test_dynamic_rejects_all_ignored_batch():
    model,ids,targets=model_and_data('dynamic_aggregate');targets.fill_(-100)
    with pytest.raises(ValueError,match='supervised token'):model(ids,targets=targets)
