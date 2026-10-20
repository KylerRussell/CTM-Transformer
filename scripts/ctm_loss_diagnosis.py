"""Why CTsqL (CTsq trained with the CTM loss) failed (2026-10-20).

Both seeds stalled at train loss ~7.2 from step ~50 with gradient norms of 1e2-1e9 (CTsq: 5.1 at step 256,
gradient norm ~0.3). For a checkpoint, on a few held-out windows at a fixed tick count, this reports:
- per-tick cross-entropy and certainty, and which ticks the CTM loss selects (min-loss, most certain);
- the gradient norm per module group under the CTM loss, under the final-tick loss, and under the
  cross-entropy of single ticks, to locate where the gradient blows up.

A healthy reference: Dans at step 400 loaded into the CTsqL model (sync queries zero, so it is Dans exactly).

Usage: python -m scripts.ctm_loss_diagnosis --gpu N --checkpoint PATH|init [--windows 2] [--depth 16]
"""
import argparse,json,os
from pathlib import Path

OUT=Path('research/results/ctm_transformer')
CTSQL=Path('research/runs/ctm_transformer/CTsqL_lr0.002')


def build(checkpoint,device):
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    run=json.loads((CTSQL/'config.json').read_text());data=json.loads(Path(run['data']).read_text())
    model=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe'])(build_config(run,data['vocab_size']))
    if checkpoint=='init':return model.to(device),run,data,0,[]
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    missing,unexpected=model.load_state_dict(state['model'],strict=False)
    assert not unexpected and all(k.startswith('sync_') for k in missing),(missing,unexpected)
    return model.to(device),run,data,state.get('step'),missing


def group(name):
    head=name.split('.')[0]
    return f'{head}.{name.split(".")[1]}' if head in ('prelude','core','coda','sync_queries') else head


def grad_norms(model):
    import torch
    out={}
    for n,p in model.named_parameters():
        if p.grad is not None:out[group(n)]=out.get(group(n),0.0)+float(p.grad.float().pow(2).sum())
    total=sum(out.values())**0.5
    return {'total':total,**{k:v**0.5 for k,v in sorted(out.items())}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--checkpoint',required=True)
    p.add_argument('--windows',type=int,default=2);p.add_argument('--depth',type=int,default=16);a=p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    from ctm_transformer.pretrain import TokenWindows
    from ctm_transformer.lm_scale import ctm_selected_tick_loss
    device=torch.device('cuda');model,run,data,step,missing=build(a.checkpoint,device);model.train()
    t=run['train'];valid=TokenWindows([Path(run['data']).parent/s for s in data['validation']],t['seq_len'],0)
    x,y=valid.batch(list(range(t['eval_offset'],t['eval_offset']+a.windows)));x,y=x.to(device),y.to(device)
    result={'checkpoint':a.checkpoint,'step':step,'depth':a.depth,'windows':a.windows,'sync_parameters_missing':bool(missing)}
    # One autocast block per pass: inside a shared block, weight casts cached during the no-grad pass are reused by
    # the gradient passes and cut the gradient to every autocast-cast weight (core, prelude, coda).
    with torch.autocast('cuda',dtype=torch.bfloat16):
        with torch.no_grad():
            hidden=[];model._hidden_readouts=True
            model._recur(model.token_embedding(x),a.depth,hidden);model._hidden_readouts=False
            out=ctm_selected_tick_loss(torch.stack(hidden,2),model.lm_head.weight,y)
            sel=out['selected_ticks'].cpu()
            result['per_tick_ce']=[round(float(v),3) for v in out['per_tick_loss']]
            result['per_tick_certainty']=[round(float(v),3) for v in out['certainty']]
            result['hidden_rms_by_tick']=[round(float(h.float().pow(2).mean().sqrt()),3) for h in hidden]
            for i,name in enumerate(('min_loss_tick','most_certain_tick')):
                counts=torch.bincount(sel[:,i],minlength=a.depth)
                result[name+'_share']=[round(float(c)/len(sel),3) for c in counts]
            del hidden
    def gradient(depth):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=y,max_thought_steps=depth)['loss']
        loss.backward();return {'loss':float(loss),**grad_norms(model)}
    grads={'ctm_loss':gradient(a.depth)}
    model.ctm_loss=False
    grads['final_tick']=gradient(a.depth)
    for k in sorted({1,2,a.depth//2,a.depth}):grads[f'single_tick_{k}']=gradient(k)
    model.ctm_loss=True
    result['gradients']=grads
    tag='init' if a.checkpoint=='init' else Path(a.checkpoint).parent.name+'_'+Path(a.checkpoint).stem
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'ctm_loss_diagnosis_{tag}_d{a.depth}.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='gradients'}))
    for k,v in grads.items():print(k,{n:round(g,3) for n,g in v.items()})


if __name__=='__main__':main()
