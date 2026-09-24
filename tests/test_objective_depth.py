from dataclasses import replace
import os
import pytest
import torch
from torch.nn import functional as F
from ctm_transformer.research import load_research_config,build_model
from ctm_transformer.experiment import validate_training_config,objective_metadata
from ctm_transformer.confidence_readout import ConfidenceReadout

@pytest.mark.parametrize('objective',['uniform','dynamic'])
@pytest.mark.parametrize('depth',[4,16])
def test_actual_masked_objectives_and_temporal_gradients(objective,depth):
    device=os.environ.get('CTM_TEST_DEVICE','cuda:0');torch.set_num_threads(4);torch.manual_seed(91)
    c,_=load_research_config(f'research/configs/ctm_{objective}_t{depth}_v1.json')
    c=replace(c,d_model=32,d_latent=32,n_heads=4,n_layers=1,nlm_groups=32,nlm_hidden_dim=8,sync_sparse_pairs=16)
    validate_training_config(c);model=build_model(c).to(device).eval()
    x=torch.randint(3,71,(2,7),device=device);y=torch.full_like(x,-100);y[:,-2:]=x[:,-2:]
    result=model(x,targets=y)
    loss=torch.stack([F.cross_entropy(z.reshape(-1,71),y.flatten(),reduction='none') for z in result['all_logits']],1)
    valid=y.flatten()!=-100
    if objective=='uniform':expected=loss[valid].mean()
    else:
        chosen=result['certainties'].reshape(depth,-1).T.argmax(1)
        expected=((loss.min(1).values+loss.gather(1,chosen[:,None]).squeeze(1))/2)[valid].mean()
        assert objective_metadata(c)['direct_ce_tick_weights_at_max_depth'] is None
    torch.testing.assert_close(result['loss'],expected,atol=1e-6,rtol=1e-5)
    actual_grad=torch.autograd.grad(result['loss'],model.layers[0].nlm.w1,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,model.layers[0].nlm.w1)[0]
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-6,rtol=1e-4)
    with torch.no_grad():
        native=model(x);wrapped=ConfidenceReadout(model)(x)
        for b in range(2):
            for s in range(7):
                z=torch.stack([t[b,s] for t in native['all_logits']]).float()
                logp=z.log_softmax(-1)
                chosen=int((logp.exp()*logp).sum(-1).argmax())
                assert torch.equal(wrapped['logits'][b,s],native['all_logits'][chosen][b,s])
    with pytest.raises(ValueError,match='inference-only'):ConfidenceReadout(model)(x,targets=y)
