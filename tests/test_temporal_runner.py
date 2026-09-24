"""Verify weighted temporal training and objective-independent final-logit validation."""
from dataclasses import replace
import json
import os
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from ctm_transformer.algorithmic import AlgorithmicTokenizer,AnswerDataset
from ctm_transformer.experiment import validate_training_config,objective_metadata,evaluate_loss,train_experiment
from ctm_transformer.research import load_research_config,build_model

DEVICE=os.environ.get('CTM_TEST_DEVICE','cuda:0')

def small(recipe,**kwargs):
    cfg,identity=load_research_config(f'research/configs/ctm_temporal_{recipe}_v1.json')
    cfg=replace(cfg,d_model=32,d_latent=32,n_heads=4,n_layers=1,nlm_groups=32,nlm_hidden_dim=8,
                sync_sparse_pairs=16,batch_size=2,seq_len=64,max_seq_len=64,max_steps=2,warmup_steps=0,
                eval_interval=1,log_interval=1,dtype='float32',device=DEVICE,**kwargs)
    return cfg,identity

@pytest.fixture
def datasets(tmp_path):
    paths=[]
    for split in ('train','validation'):
        source=Path(f'research/data/ordered_pointer_v1/pointer/{split}.jsonl')
        path=tmp_path/f'{split}.jsonl';path.write_text('\n'.join(source.read_text().splitlines()[:4])+'\n');paths.append(path)
    return paths

@pytest.mark.parametrize('recipe,weights',[('final',[0,0,0,1]),('uniform',[.25]*4),('late',[0,1/6,1/3,1/2])])
def test_actual_losses_and_gradients(recipe,weights,datasets):
    torch.set_num_threads(4);torch.manual_seed(18)
    config,_=small(recipe);validate_training_config(config)
    model=build_model(config).to(DEVICE).eval()
    ds=AnswerDataset(datasets[0],AlgorithmicTokenizer(),64)
    x,y,_=ds.batch(slice(0,2));x,y=x.to(DEVICE),y.to(DEVICE)
    result=model(x,targets=y)
    independent=torch.stack([F.cross_entropy(logits.reshape(-1,config.vocab_size),y.flatten(),ignore_index=-100) for logits in result['all_logits']])
    oracle=(independent*torch.tensor(weights,device=DEVICE)).sum()
    torch.testing.assert_close(result['loss'],oracle)
    parameters=[model.output_proj[-1].weight,model.layers[0].nlm.w1]
    actual=torch.autograd.grad(result['loss'],parameters,retain_graph=True)
    expected=torch.autograd.grad(oracle,parameters)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
    torch.testing.assert_close(torch.tensor(objective_metadata(config)['direct_ce_tick_weights_at_max_depth']),torch.tensor(weights,dtype=torch.float32))


def test_validation_is_final_ce_for_all_objectives(datasets):
    config,_=small('final');model=build_model(config).to(DEVICE).train()
    ds=AnswerDataset(datasets[1],AlgorithmicTokenizer(),64)
    scores=[]
    for recipe in ('final','uniform','late'):
        cfg,_=small(recipe);model.config=cfg
        scores.append(evaluate_loss(model,ds,cfg,torch.device(DEVICE))['loss'])
        assert model.training
    assert scores[0]==scores[1]==scores[2]
    model.eval();x,y,_=ds.batch(slice(0,4))
    oracle=F.cross_entropy(model(x.to(DEVICE))['logits'].reshape(-1,cfg.vocab_size),y.to(DEVICE).flatten()).item()
    assert scores[-1]==pytest.approx(oracle,rel=1e-6)

@pytest.mark.parametrize('recipe',['final','uniform','late'])
def test_shared_runner_trains_and_records_recipe(recipe,datasets,tmp_path):
    config,identity=small(recipe,data_path=str(datasets[0]),eval_data_path=str(datasets[1]),checkpoint_dir=str(tmp_path/recipe))
    summary=train_experiment(config,identity,17,data_format='algorithmic')
    assert summary['complete'] and summary['steps']==2
    record=json.loads((tmp_path/recipe/'research_run.json').read_text())
    weights=record['training_objective']['direct_ce_tick_weights_at_max_depth']
    for row in map(json.loads,(tmp_path/recipe/'metrics.jsonl').read_text().splitlines()):
        assert row['loss']==pytest.approx(sum(w*v for w,v in zip(weights,row['per_tick_supervised_ce'])),rel=1e-5)
    assert (tmp_path/recipe/'best.pt').exists()

@pytest.mark.parametrize('start,end',[(-1.,1.),(0.,0.),(float('nan'),1.)])
def test_rejects_invalid_temporal_weights(start,end):
    config,_=small('uniform');config=replace(config,tick_ramp_start=start,tick_ramp_end=end)
    with pytest.raises(ValueError,match='Temporal CE weights'):validate_training_config(config)


def test_rejects_invalid_monotonic_penalty():
    config,_=small('uniform')
    for cfg in (replace(config,mono_penalty_weight=-.1),):
        with pytest.raises(ValueError,match='mono_penalty_weight must'):validate_training_config(cfg)
