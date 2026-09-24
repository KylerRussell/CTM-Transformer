"""No-update audit of final-loss temporal gradients and inactive CTM loss reductions."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.query_diagnostic import validate_query_suite
from ctm_transformer.experiment import file_hash
from scripts.eval_harness import _load_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:1')
    p.add_argument('--output',type=Path,default=Path('research/results/attention_temporal_v1/ctm_temporal.json'))
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh output')
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    path='research/runs/ordered_long_v1/seed17/ctm/best.pt'
    model,config=_load_checkpoint(path);model.to(args.device).eval()
    data=validate_query_suite('research/data/query_diagnostic_v1','research/data/ordered_pointer_v1')
    rows=[data[f'train_probe_start_{n}'].records[i] for i in range(4) for n in 'ABCDEFGH']
    tok=AlgorithmicTokenizer();xs=[];ys=[]
    for r in rows:
        prefix=[tok.bos_token]+tok.encode(r['prompt']);answer=tok.encode(r['answer'])+[tok.eot_token]
        xs.append((prefix+answer)[:-1]);ys.append([-100]*(len(prefix)-1)+answer)
    ids=torch.tensor(xs,device=args.device);targets=torch.tensor(ys,device=args.device)
    states=[];histories=[];handles=[];original_step=model._thought_step
    def step(*a,**k):
        result=original_step(*a,**k);result[0].retain_grad();states.append(result[0]);return result
    model._thought_step=step
    for name,module in model.named_modules():
        if name.endswith('.nlm'):
            def hook(module,inputs,name=name):
                inputs[0].retain_grad();histories.append((name,inputs[0]))
            handles.append(module.register_forward_pre_hook(hook))
    output=model(ids,targets=targets);output['loss'].backward()
    state_norms=[x.grad.float().norm().item() if x.grad is not None else None for x in states]
    assert all(v is not None and v>0 and torch.isfinite(torch.tensor(v)) for v in state_norms)
    history_gradients=[{'module':n,'gradient_norm_by_slot_oldest_first':x.grad.float().square().sum((0,2)).sqrt().tolist()} for n,x in histories]
    parameter_gradients={n:p.grad.float().norm().item() if p.grad is not None else None for n,p in model.named_parameters()
        if any(s in n for s in ('history_init','sync_computer.r','nlm.w1','nlm.w2'))}
    for h in handles:h.remove()
    model._thought_step=original_step;model.zero_grad(set_to_none=True)
    measurements={}
    with torch.no_grad():
        for mode in ('final_ce','ramp_mono','dynamic_aggregate'):
            model.config.temporal_loss_type=mode
            result=model(ids,targets=targets)
            logits=result['all_logits'];losses=torch.stack([F.cross_entropy(x.float().reshape(-1,config.vocab_size),targets.flatten(),reduction='none') for x in logits],dim=1)
            valid=targets.flatten()!=-100
            per_tick=losses[valid].mean(0)
            if mode=='final_ce':oracle=per_tick[-1]
            elif mode=='ramp_mono':
                w=torch.linspace(config.tick_ramp_start,config.tick_ramp_end,len(logits),device=args.device)
                oracle=(w/w.sum()*per_tick).sum()
            else:
                cert=result['certainties'].reshape(len(logits),-1).T
                certain=cert.argmax(1)
                selected=(losses.min(1).values+losses.gather(1,certain[:,None]).squeeze(1))/2
                oracle=selected[valid].mean()
            measurements[mode]={'implementation_loss':result['loss'].item(),'masked_oracle_loss':oracle.item(),
                'implementation_over_masked_oracle':(result['loss']/oracle).item(),
                'per_tick_masked_ce':per_tick.tolist(),
                'matches_masked_oracle':bool(torch.isclose(result['loss'],oracle,rtol=1e-5,atol=1e-6))}
        model.config.temporal_loss_type='final_ce'
    report={'complete':True,'checkpoint_sha256':file_hash(path),'precision':'FP32; gradients only; no optimizer update',
        'training_batch':[{'id':r['id'],'start':r['start']} for r in rows],
        'supervised_tokens':int(valid.sum()),'total_input_positions':int(valid.numel()),
        'tick_output_gradient_norms':state_norms,'history_input_gradients':history_gradients,
        'temporal_parameter_gradient_norms':parameter_gradients,'loss_measurements':measurements,
        'history_slots':{'length':config.history_len,'ticks':config.max_thought_steps,
            'new_observations_at_nlm_by_tick':list(range(1,config.max_thought_steps+1)),
            'new_observations_at_attention_by_tick':list(range(config.max_thought_steps))},
        'source_sha256':{p:file_hash(p) for p in ('scripts/audit_ctm_temporal.py','ctm_transformer/model.py','ctm_transformer/config.py','ctm_transformer/experiment.py')}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gradient_norms':state_norms,'loss_measurements':measurements},indent=2))

if __name__=='__main__':main()
