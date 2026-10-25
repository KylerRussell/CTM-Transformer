"""Capability breakdown of held-out loss for trained checkpoints (option B, 2026-10-25).

The CTM Transformer ties Dans on average held-out loss. This asks whether the tie hides differences on
the kinds of tokens where the CTM and CMM papers claim strengths: recall of earlier context, retention
over distance, more ticks than trained, and in-context key-value recall (CMM: associative recall, copy,
priority sort with length generalization; arXiv 2610.07907). No training; every model is evaluated
identically, recurrent models at 16 ticks unless stated.

Measures:
- M0 overall: mean cross-entropy over 256 held-out windows of 1,024 tokens (offset 40,000, as the final eval).
- M1 recall split of each target y_i = x_{i+1}:
  repeat = the pair (x_i, y_i) occurred earlier in the window (an induction / copy target);
  novel = y_i occurs nowhere in x_0..x_i; other = the rest.
- M2 retention: repeat targets by distance to the most recent earlier occurrence of the pair:
  1-32, 33-128, 129-512, 513-1023 tokens.
- M3 position: targets by position in the window: 0-63, 64-255, 256-511, 512-1023.
- M4 ticks (recurrent models): loss at 1, 2, 4, 8, 16, 32, 48, 64 and 96 ticks on 64 of the windows.
  Training sampled tick counts up to 48 (mean 15); 64 and 96 are beyond anything trained.
- M5 in-context recall: synthetic episodes "<eot> k1: v1, k2: v2, ..., kN: vN, k3: v3, k1: v1, ..."
  (N pairs of random single-token words, then all N keys queried in a random order). Loss and accuracy
  (argmax) on the queried values only; N = 8, 32, 96; 64 episodes each.

- M6 copy (added 2026-10-25 after M5 showed a floor: Dans recalls 3% of values at N = 8, T8 11%, both
  under 2% at N >= 32): "<eot> w1 ... wL w1 ... wL" with L random single-token words; loss and accuracy on the
  second copy, excluding its first token (L = 16, 64, 256; 64 episodes each). This is the CMM's copy task in
  language-model form. Its rule is the general one below, fixed before any M6 result.

Rules, fixed before the evaluation. Comparisons use two-seed means against Dans (T20 has one seed and
is reported only); the Transformer baseline T8 is compared with Dans by the same rule.
- A model differs from Dans on a measure if its mean differs by at least 0.02 nats (2 points for
  accuracy) and both its seeds lie on the same side of both Dans seeds.
- Capability-specific (M1-M3): per seed, excess = (measure - overall) - (Dans's mean measure - Dans's mean
  overall). A model has a specific advantage (disadvantage) on a measure if both seeds' excess is at most
  -0.02 (at least +0.02): the measure moves beyond what its overall loss explains.
- M4: a model generalizes past its trained ticks if its loss at 64 and at 96 ticks is no more than 0.01
  above its 16-tick loss. CMM's claim that alpha/beta synchronization generalizes to more ticks is
  supported here if CTsq and CTso (which read alpha/beta) each have both seeds' 96-tick drift at least
  0.01 below both Dans seeds'.

Usage: python -m scripts.capability_eval --gpu N --runs NAME... [--copy] | --summarize
"""
import argparse,json,math,os,random
from pathlib import Path

OUT=Path('research/results/capability_eval')
RUN_DIRS=[Path('research/runs/rdt_recipe'),Path('research/runs/ctm_transformer')]
WINDOWS,TICK_WINDOWS,OFFSET=256,64,40000
TICKS=(1,2,4,8,16,32,48,64,96)
DIST_BINS=((1,32),(33,128),(129,512),(513,1023));POS_BINS=((0,63),(64,255),(256,511),(512,1023))
RECALL_N=(8,32,96);EPISODES=64;COPY_L=(16,64,256)
GROUPS={'T8':['T8_lr0.002','T8_lr0.002_s2'],'T20':['T20_lr0.002'],'Dans':['Dans_rand_lr0.002','Dans_rand_lr0.002_s2'],
        **{v:[f'{v}_lr0.002',f'{v}_lr0.002_s2'] for v in ('CTfull','CTnlm','CTsq','CTso','CTsqS','CTsqLD')}}


def run_dir(name):
    return next(d/name for d in RUN_DIRS if (d/name/'config.json').exists())


def load(name,device):
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.rdt_recipe import recipe_factory
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    from ctm_transformer.lm_scale import scaled_factory
    d=run_dir(name);run=json.loads((d/'config.json').read_text());data=json.loads(Path(run['data']).read_text())
    config=build_config(run,data['vocab_size'])
    if run.get('ctm_transformer') is not None:factory=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe'])
    elif run.get('rdt_recipe'):factory=recipe_factory(run['rdt_recipe'])
    else:factory=scaled_factory(run['family'])
    model=factory(config);t=run['train']
    model.load_state_dict(torch.load(d/f'step_{t["total_steps"]:07d}.pt',map_location='cpu',weights_only=False)['model'])
    return model.to(device).eval(),run,data


def token_ce(model,x,y,depth,recurrent):
    """Per-token cross-entropy [B, S] at the given tick count."""
    import torch
    from torch.nn import functional as F
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x,max_thought_steps=depth if recurrent else None)['logits']
    return torch.stack([F.cross_entropy(l.float(),t,reduction='none') for l,t in zip(logits,y)]),logits.argmax(-1)


def categories(x,y):
    """M1/M2 labels for one window: 'repeat' with distance, 'novel', or 'other'."""
    last_pair,seen,labels=dict(),set(),[]
    for i,(a,b) in enumerate(zip(x,y)):
        seen.add(a)
        j=last_pair.get((a,b))
        labels.append(('repeat',i-j) if j is not None else ('novel',0) if b not in seen else ('other',0))
        last_pair[(a,b)]=i
    return labels


def word_tokens(tokenizer_path,count=2000,seed=0):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_path));words=[]
    for piece,i in tok.get_vocab().items():
        text=tok.decode([i])
        if text.startswith(' ') and text[1:].isalpha() and text[1:].islower() and 3<=len(text)-1<=8:words.append(i)
    words.sort();random.Random(seed).shuffle(words)
    colon,comma=tok.encode(':').ids,tok.encode(',').ids
    assert len(colon)==1 and len(comma)==1
    return words[:count],colon[0],comma[0]


def recall_episodes(words,colon,comma,eot,n,episodes,seed):
    rng=random.Random(seed);seqs,value_positions=[],[]
    for _ in range(episodes):
        chosen=rng.sample(words,2*n);keys,vals=chosen[:n],chosen[n:]
        seq=[eot];pos=[]
        for k,v in zip(keys,vals):seq+=[k,colon,v,comma]
        for q in rng.sample(range(n),n):
            seq+=[keys[q],colon];pos.append(len(seq)-1);seq+=[vals[q],comma]  # the target at pos is the value
        seqs.append(seq);value_positions.append(pos)
    return seqs,value_positions


def copy_episodes(words,eot,length,episodes,seed):
    rng=random.Random(seed)
    return [[eot,*span,*span] for span in (rng.sample(words,length) for _ in range(episodes))]


def evaluate_copy(name,device):
    """M6, added to an existing result file."""
    import torch
    path=OUT/f'{name}.json';result=json.loads(path.read_text())
    if 'copy' in result:return
    model,run,data=load(name,device);recurrent=run['family']=='rdt'
    words,_,_=word_tokens(Path(run['data']).parent/data['tokenizer']);copy={}
    for n in COPY_L:
        seqs=copy_episodes(words,data['eot_id'],n,EPISODES,seed=1000+n);losses,hits=[],[]
        for b in range(0,EPISODES,8):
            batch=torch.tensor(seqs[b:b+8]);x,y=batch[:,:-1].to(device),batch[:,1:].to(device)
            ce,pred=token_ce(model,x,y,16,recurrent)
            second=slice(n+1,2*n)  # targets y[p] = x[p+1]: positions n+1..2n-1 predict the second copy after its first token
            losses.append(float(ce[:,second].mean()));hits.append(float((pred[:,second]==y[:,second]).float().mean()))
        copy[str(n)]={'loss':sum(losses)/len(losses),'accuracy':100*sum(hits)/len(hits)}
    result['copy']=copy;path.write_text(json.dumps(result,indent=1)+'\n')
    print(name,'copy',json.dumps({k:(round(v['loss'],3),round(v['accuracy'],1)) for k,v in copy.items()}),flush=True)


def evaluate(name,device):
    import torch
    from ctm_transformer.pretrain import TokenWindows
    model,run,data=load(name,device);recurrent=run['family']=='rdt';t=run['train']
    valid=TokenWindows([Path(run['data']).parent/s for s in data['validation']],t['seq_len'],0)
    sums={};count={}
    def add(key,ce):sums[key]=sums.get(key,0.0)+ce;count[key]=count.get(key,0)+1
    for b in range(0,WINDOWS,4):
        x,y=valid.batch(list(range(OFFSET+b,OFFSET+b+4)));ce,_=token_ce(model,x.to(device),y.to(device),16,recurrent);ce=ce.cpu().tolist()
        for xs,ys,cs in zip(x.tolist(),y.tolist(),ce):
            for i,((kind,dist),c) in enumerate(zip(categories(xs,ys),cs)):
                add('overall',c);add(kind,c)
                if kind=='repeat':add('dist_'+next(f'{lo}-{hi}' for lo,hi in DIST_BINS if lo<=dist<=hi),c)
                add('pos_'+next(f'{lo}-{hi}' for lo,hi in POS_BINS if lo<=i<=hi),c)
    result={'run':name,'windows':WINDOWS,'loss':{k:sums[k]/count[k] for k in sums},'tokens':count}
    if recurrent:
        ticks={}
        for d in TICKS:
            total=0.0
            for b in range(0,TICK_WINDOWS,4):
                x,y=valid.batch(list(range(OFFSET+b,OFFSET+b+4)));ce,_=token_ce(model,x.to(device),y.to(device),d,recurrent);total+=float(ce.mean())
            ticks[str(d)]=total/(TICK_WINDOWS//4)
        result['ticks']=ticks
    words,colon,comma=word_tokens(Path(run['data']).parent/data['tokenizer'])
    recall={}
    for n in RECALL_N:
        seqs,positions=recall_episodes(words,colon,comma,data['eot_id'],n,EPISODES,seed=n)
        losses,hits=[],[]
        for b in range(0,EPISODES,8):
            batch=torch.tensor(seqs[b:b+8]);x,y=batch[:,:-1].to(device),batch[:,1:].to(device)
            ce,pred=token_ce(model,x,y,16,recurrent)
            for e,pos in enumerate(positions[b:b+8]):
                losses+=[float(ce[e,p]) for p in pos];hits+=[int(pred[e,p]==y[e,p]) for p in pos]
        recall[str(n)]={'loss':sum(losses)/len(losses),'accuracy':100*sum(hits)/len(hits)}
    result['recall']=recall
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{name}.json').write_text(json.dumps(result,indent=1)+'\n')
    print(name,json.dumps({'overall':round(result['loss']['overall'],4),'repeat':round(result['loss']['repeat'],4),'novel':round(result['loss']['novel'],4),
          'recall':{k:round(v['accuracy'],1) for k,v in recall.items()},'ticks':{k:round(v,3) for k,v in result.get('ticks',{}).items()}}),flush=True)


def measures(r):
    m={'overall':r['loss']['overall'],**{k:r['loss'][k] for k in ('repeat','novel','other')},
       **{k:r['loss'][k] for k in r['loss'] if k.startswith(('dist_','pos_'))}}
    for n,v in r['recall'].items():m[f'recall{n}_loss']=v['loss'];m[f'recall{n}_acc']=v['accuracy']
    for n,v in r.get('copy',{}).items():m[f'copy{n}_loss']=v['loss'];m[f'copy{n}_acc']=v['accuracy']
    if 'ticks' in r:
        for d in ('32','48','64','96'):m[f'drift{d}']=r['ticks'][d]-r['ticks']['16']
    return m


def summarize():
    res={n:measures(json.loads((OUT/f'{n}.json').read_text())) for g in GROUPS.values() for n in g if (OUT/f'{n}.json').exists()}
    have={g:[res[n] for n in ns if n in res] for g,ns in GROUPS.items()};dans=have['Dans'];mean=lambda vs:sum(vs)/len(vs)
    keys=list(dans[0].keys())
    L=['# Capability breakdown of held-out loss','','Protocol and rules: `scripts/capability_eval.py` (docstring). Two-seed means; T20 has one seed.','',
       '| Measure | '+' | '.join(g for g in GROUPS if have[g])+' |','|---|'+'---:|'*sum(1 for g in GROUPS if have[g])]
    for k in keys:
        cells=[]
        for g in GROUPS:
            if not have[g]:continue
            if k not in have[g][0]:cells.append('');continue
            cells.append(f'{mean([r[k] for r in have[g]]):.{1 if k.endswith("_acc") else 3}f}')
        L.append(f'| {k} | '+' | '.join(cells)+' |')
    L+=['','## Rule outcomes against Dans','']
    for g,rs in have.items():
        if g in ('Dans','T20') or len(rs)<2:continue
        notes=[]
        for k in keys:
            if k not in rs[0] or k.startswith('drift'):continue
            acc=k.endswith('_acc');thr=2.0 if acc else 0.02;sign=-1 if acc else 1  # sign*diff < 0 is better
            d=mean([r[k] for r in rs])-mean([r[k] for r in dans])
            better=sign*d<=-thr and all(sign*r[k]<min(sign*q[k] for q in dans) for r in rs)
            worse=sign*d>=thr and all(sign*r[k]>max(sign*q[k] for q in dans) for r in rs)
            if better or worse:notes.append(f'{k}: {"better" if better else "worse"} ({d:+.3f})')
            if k in ('repeat','novel','other') or k.startswith(('dist_','pos_')):
                ex=[(r[k]-r['overall'])-(mean([q[k] for q in dans])-mean([q['overall'] for q in dans])) for r in rs]
                if all(e<=-0.02 for e in ex):notes.append(f'{k}: specific advantage (excess {ex[0]:+.3f} / {ex[1]:+.3f})')
                if all(e>=0.02 for e in ex):notes.append(f'{k}: specific disadvantage (excess {ex[0]:+.3f} / {ex[1]:+.3f})')
        L.append(f'- **{g}:** '+('; '.join(notes) if notes else 'no measure differs from Dans by the rule'))
    L+=['','## Ticks beyond training (M4)','']
    for g,rs in have.items():
        if rs and 'drift96' in rs[0]:
            gen=all(r['drift64']<=0.01 and r['drift96']<=0.01 for r in rs)
            L.append(f'- **{g}:** drift at 32/48/64/96 ticks vs 16: '+' / '.join(f'{mean([r[f"drift{d}"] for r in rs]):+.3f}' for d in ('32','48','64','96'))
                     +' (per seed at 96: '+', '.join(f'{r["drift96"]:+.3f}' for r in rs)+f'); generalizes: {"yes" if gen else "no"}')
    if all(len(have[g])==2 for g in ('CTsq','CTso','Dans')):
        ok=all(r['drift96']<=min(q['drift96'] for q in dans)-0.01 for g in ('CTsq','CTso') for r in have[g])
        L.append(f'- **CMM alpha/beta claim (CTsq and CTso drift at 96 ticks at least 0.01 below both Dans seeds):** {"supported" if ok else "not supported"}')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CAPABILITY_EVAL.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='*')
    p.add_argument('--summarize',action='store_true');p.add_argument('--copy',action='store_true',help='add M6 to existing results');a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    for n in a.runs or []:
        if a.copy:evaluate_copy(n,torch.device('cuda'))
        elif not (OUT/f'{n}.json').exists():evaluate(n,torch.device('cuda'))


if __name__=='__main__':main()
