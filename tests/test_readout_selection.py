from dataclasses import replace
import json
from pathlib import Path
import torch
import pytest
from torch.nn import functional as F
from ctm_transformer.baselines import BaselineConfig,BaselineTransformer
from ctm_transformer.confidence_readout import ConfidenceReadout
from ctm_transformer.research import load_research_config,build_model
from ctm_transformer.algorithmic import AnswerDataset,AlgorithmicTokenizer
from ctm_transformer.experiment import train_experiment,validate_training_config


def test_recurrent_tick_logits_equal_separate_truncations():
    torch.manual_seed(12)
    c=BaselineConfig(model_family='recurrent_depth',d_model=16,n_heads=4,ffn_hidden_dim=32,vocab_size=13,
        n_layers=0,prelude_layers=1,core_layers=1,coda_layers=1,max_thought_steps=4,max_seq_len=8,seq_len=8)
    m=BaselineTransformer(c).eval();x=torch.randint(13,(2,5))
    output=m(x,return_all_logits=True)
    assert len(output['all_logits'])==4
    for t,logits in enumerate(output['all_logits'],1):torch.testing.assert_close(logits,m(x,max_thought_steps=t)['logits'],rtol=0,atol=0)
    torch.testing.assert_close(output['logits'],output['all_logits'][-1],rtol=0,atol=0)
    reader=ConfidenceReadout(m);chosen=reader(x)
    stack=torch.stack(output['all_logits'],2).float();logp=stack.log_softmax(-1)
    index=(-(logp.exp()*logp).sum(-1)).argmin(2)
    expected=stack.gather(2,index[...,None,None].expand(-1,-1,1,13)).squeeze(2)
    torch.testing.assert_close(chosen['logits'],expected)


@pytest.mark.parametrize('step,factor',[(0,1.0),(2,.55),(4,.1)])
def test_dynamic_monotonic_penalty_and_gradients(step,factor):
    torch.set_num_threads(4);torch.manual_seed(12)
    c,_=load_research_config('research/configs/ctm_dynamic_t4_v1.json')
    c=replace(c,d_model=32,d_latent=32,nlm_groups=32,n_layers=1,nlm_hidden_dim=8,sync_sparse_pairs=16,
        max_steps=10,warmup_steps=0,mono_penalty_weight=.5,mono_penalty_decay_until_frac=.4,mono_penalty_min_frac=.1)
    validate_training_config(c);m=build_model(c).cuda().eval();m._train_step.fill_(step)
    x=torch.tensor([[1,4,5,6]],device='cuda');y=torch.tensor([[-100,-100,7,2]],device='cuda')
    output=m(x,targets=y)
    stack=torch.stack(output['all_logits'],2)
    ce=torch.stack([F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none') for z in output['all_logits']],1)
    mask=y.flatten().ne(-100);ce=ce[mask]
    cert=output['certainties'].view(4,-1).T[mask]
    base=(ce.min(1).values+ce.gather(1,cert.argmax(1,keepdim=True)).squeeze(1)).mean()/2
    mean=ce.mean(0);oracle=base+.5*factor*F.relu(mean[1:]-mean[:-1]).mean()
    torch.testing.assert_close(output['loss'],oracle)
    param=m.layers[0].nlm.w1
    a=torch.autograd.grad(output['loss'],param,retain_graph=True)[0];b=torch.autograd.grad(oracle,param)[0]
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)


def test_policy_selection_checkpoints_and_counters(tmp_path):
    torch.set_num_threads(4)
    paths=[]
    for split in ('train','validation'):
        p=tmp_path/f'{split}.jsonl';p.write_text('\n'.join(Path(f'research/data/ordered_pointer_v1/pointer/{split}.jsonl').read_text().splitlines()[:4])+'\n');paths.append(p)
    c,identity=load_research_config('research/configs/ctm_dynamic_t4_v1.json')
    c=replace(c,d_model=32,d_latent=32,nlm_groups=32,n_layers=1,nlm_hidden_dim=8,sync_sparse_pairs=16,
        batch_size=2,max_steps=2,warmup_steps=0,eval_interval=1,log_interval=1,device='cuda:0',
        data_path=str(paths[0]),eval_data_path=str(paths[1]),checkpoint_dir=str(tmp_path/'run'))
    train_experiment(c,identity,17,data_format='algorithmic',selection_readouts=('final','confidence'),save_validation_checkpoints=True)
    rows=[json.loads(s) for s in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    expected=min(((row['validation_readouts'][policy]['loss'],row['step'],i,policy) for row in rows for i,policy in enumerate(('final','confidence'))))
    checkpoint=torch.load(tmp_path/'run/best.pt',weights_only=False)
    assert checkpoint['step']==expected[1] and checkpoint['readout_policy']==expected[3]
    assert checkpoint['supervised_tokens_seen']==checkpoint['step']*4
    for row in rows:
        assert row['validation']['loss']==min(row['validation_readouts'][p]['loss'] for p in ('final','confidence'))
        for p in ('final','confidence'):
            assert row['validation_readouts'][p]['target_tokens']==8
            assert len(row['validation_readouts'][p]['predictions'])==4
        assert sum(row['validation_readouts']['diagnostics']['confidence_tick_histogram'])==8
    assert (tmp_path/'run/step_000001.pt').exists() and (tmp_path/'run/step_000002.pt').exists()
