"""One evaluation set for every 400M-token language model (2026-11-21, after external review, concern 5).

The paper reported Dans at 16 ticks as 3.574 (training's final evaluation, 1,024 windows), 3.519 (64 windows) and
about 3.534 (128 windows). This script scores every final checkpoint on the 1,024 held-out windows of the training
runs' final evaluation (validation windows 40,000-41,023, 1,024 tokens each), so every LM number in the paper comes
from one set.

For a recurrent model, one forward pass of 96 ticks gives:
- the held-out loss at every tick count in TICKS (the readout at tick t equals a t-tick run: the recurrence is causal
  in ticks and nothing depends on the total);
- the state's relative change per tick, ||x_t - x_{t-1}|| / ||x_{t-1}|| per token, averaged over tokens, for
  t = 1..96 (x_0 is the state before the first tick), a direct measure of the fixed point.
A Transformer is scored once. Output: research/results/lm_unified/<run>.json; --summarize writes LM_UNIFIED.md.

Usage: python -m scripts.lm_unified_eval --gpu N --runs RUN_DIR... | --summarize
"""
import argparse,json,os,re
from pathlib import Path

OUT=Path('research/results/lm_unified');MAX_TICKS=96
TICKS=(1,2,3,4,5,6,8,10,12,16,20,24,32,48,64,96)
RUNS=[f'research/runs/rdt_recipe/{n}' for n in ('Dans_rand_lr0.002','Dans_rand_lr0.002_s2','T8_lr0.002','T8_lr0.002_s2','T20_lr0.002')]+\
     [f'research/runs/ctm_transformer/{v}_lr0.002{s}' for v in ('CTsq','CTsqS','CTfull','CTnlm','CTso','CTsqH','CTsqLD') for s in ('','_s2')]


def load(run_dir,device):
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.rdt_recipe import recipe_factory
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    from ctm_transformer.lm_scale import scaled_factory
    run=json.loads((run_dir/'config.json').read_text());data=json.loads(Path(run['data']).read_text())
    if run.get('ctm_transformer') is not None:factory=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe'])
    elif run.get('rdt_recipe') is not None:factory=recipe_factory(run['rdt_recipe'])
    else:factory=scaled_factory(run['family'])
    model=factory(build_config(run,data['vocab_size']));t=run['train']
    model.load_state_dict(torch.load(run_dir/f'step_{t["total_steps"]:07d}.pt',map_location='cpu',weights_only=False)['model'])
    return model.to(device).eval(),run,data


class TickLosses(list):
    """Stands in for the model's per-tick readout list: scores each readout as it is produced and keeps only sums."""
    def __init__(self,y):super().__init__();self.y=y.reshape(-1);self.tick=0;self.sums={}
    def append(self,logits):
        import torch.nn.functional as F
        self.tick+=1
        if self.tick in TICKS:self.sums[self.tick]=float(F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),self.y,reduction='sum'))


def evaluate(run_dir,device,windows=1024,micro=4):
    import torch
    import torch.nn.functional as F
    from ctm_transformer.pretrain import TokenWindows
    out=OUT/f'{run_dir.name}.json'
    if out.exists():return
    model,run,data=load(run_dir,device);t=run['train'];recurrent=run['family']=='rdt'
    valid=TokenWindows([Path(run['data']).parent/s for s in data['validation']],t['seq_len'],0);offset=t.get('eval_offset',0)
    assert offset==40000 and t.get('final_eval_windows')==windows,(offset,t.get('final_eval_windows'))
    loss={};change=[0.0]*MAX_TICKS;tokens=0
    if recurrent:
        states=[]
        def grab(module,inputs):
            x=inputs[0]
            if not states or states[-1] is not x:states.append(x)
        hook=model.state_norm.register_forward_pre_hook(grab)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(offset,offset+windows,micro):
            x,y=valid.batch(list(range(i,i+micro)));x,y=x.to(device),y.to(device);tokens+=y.numel()
            if not recurrent:
                logits=model(x)['logits'];loss[1]=loss.get(1,0.0)+float(F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum'));continue
            readouts=TickLosses(y);states.clear()
            final=model._recur(model.token_embedding(x),MAX_TICKS+1,readouts)  # one extra tick so the hook sees x_96
            for k,v in readouts.sums.items():loss[k]=loss.get(k,0.0)+v
            for k in range(MAX_TICKS):
                a,b=states[k].float(),states[k+1].float();change[k]+=float(((b-a).norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-6)).sum())
    if recurrent:hook.remove()
    result={'run':run_dir.name,'family':run['family'],'windows':windows,'offset':offset,'tokens':tokens,
            'loss':{str(k):v/tokens for k,v in sorted(loss.items())}}
    if recurrent:result['relative_state_change']={str(k+1):change[k]/tokens for k in range(MAX_TICKS)}
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n')
    print(run_dir.name,{k:round(v,4) for k,v in result['loss'].items() if k in ('1','4','8','16','96')},flush=True)


def summarize():
    rs={p.stem:json.loads(p.read_text()) for p in sorted(OUT.glob('*.json'))};mean=lambda v:sum(v)/len(v)
    groups={}
    for n,r in rs.items():groups.setdefault(re.sub(r'(_rand)?_lr[0-9.]+(_s\d+)?$','',n),[]).append(r)  # seeds of one arm together
    show=('1','2','4','8','16','32','96')
    L=['# Language models on one evaluation set','','1,024 held-out windows (validation 40,000-41,023), 1,048,576 tokens; `scripts/lm_unified_eval.py`.',
       'Loss in nats per token, mean over seeds (seeds in brackets); Transformers in the 1-tick column.','',
       '| Model | Seeds | '+' | '.join(f'{t} ticks' for t in show)+' | Change per tick at 8 / 16 / 96 |','|---|---:|'+'---:|'*len(show)+'---|']
    for g,v in sorted(groups.items(),key=lambda kv:mean([r['loss'].get('16',r['loss']['1']) for r in kv[1]])):
        cells=[f"{mean([r['loss'][t] for r in v]):.4f}" if all(t in r['loss'] for r in v) else '-' for t in show]
        ch=' / '.join(f"{mean([r['relative_state_change'][t] for r in v]):.1e}" for t in ('8','16','96')) if 'relative_state_change' in v[0] else '-'
        L.append(f'| {g} | {len(v)} | '+' | '.join(cells)+f' | {ch} |')
    (OUT/'LM_UNIFIED.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='*',default=[])
    p.add_argument('--all',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    for r in (RUNS if a.all else a.runs):evaluate(Path(r),torch.device('cuda'))


# 2026-11-26: the five-seed runs of scripts/lm_revision.py (seeds 3-5; T20 seeds 2-5)
REVISION=[f'research/runs/revision/{a}_lr0.002_s{s}' for a in ('Dans','CTsq','CTsqS','T8') for s in (3,4,5)]+[f'research/runs/revision/T20_lr0.002_s{s}' for s in (2,3,4,5)]


if __name__=='__main__':main()
