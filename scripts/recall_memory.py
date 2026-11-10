"""Synchronization as memory: associative recall trained until the binding is learned (2026-11-10).

The CTM paper's future work names synchronization as a memory mechanism. In the CMM-task grid (scripts/cmm_tasks.py)
no model learned associative recall in 5,000 updates. Every model, the Transformers included, sat at about 32% of
queries right, which is choosing among the values present without binding keys to values. So that test could not
compare memory. This one first finds a recipe in which the Transformer learns the binding, then compares the CTM
Transformer with Dans on it.

Task: the grid's recall (BOS k1 v1 .. kN vN, then QUERY k -> v; 64 keys, 16 values). Train N 2-16, ID N 2-16,
OOD N 17-32. Models and optimiser are the grid's at width 128 (AdamW lr 1e-3, warmup 200, cosine to 10%, batch 128,
clip 1, bf16). Recurrent models train with the LM tick sampler (log-normal Poisson, mean 15, max 48) and are
evaluated at 16 ticks (OOD also at 32 and 64). Primary measure: query accuracy (answers right / queries), on 512
fixed examples per split. Resume checkpoints every 500 updates.

Recipes (training changes only; evaluation uses the recipe's query format):
- long: the grid's task (4 queries per example), 40,000 updates.
- cur: as long, with a curriculum: N drawn from 2..min(16, 2 + step // 700) (full range from update 9,800).
- dense: every key is queried once, in random order (N answers per example), 40,000 updates.

Phase 1, screen (T8, seeds 0 and 1, each recipe; only T8 decides, and a recurrent run costs about 10x T8's
1.3-1.5 s per update alone against 0.12, so Dans first appears in phase 2). ID query accuracy on 256 examples every 1,000 updates.
A run stops early once that is at least 95% at two evaluations in a row. Rules, fixed before any run:
- A recipe makes recall learnable if T8 reaches 90% ID query accuracy on both seeds.
- The phase-2 recipe is the first learnable one in the order long, cur, dense (closest to the CMM task first).
  Its budget is the larger of 10,000 and twice the largest number of updates T8 needed to reach 90%, rounded up to
  a multiple of 5,000, at most 40,000.
- If no recipe is learnable, synchronization as memory cannot be tested on recall at this width, and that is
  reported.

Phase 2, comparison (the chosen recipe and budget, no early stop; seeds 0-4): T8, Dans, CTsq (sync query), CTsqS
(sync query + attention sink), CTfull (neuron models + sync query + sync output), CTnlmA (neuron models with their
activation). Rules, fixed before any run:
- A beats B on a measure if its 5-seed mean is at least 5 points higher and it is higher on at least 4 of 5 seeds
  (paired by seed). Measures: ID and OOD query accuracy at 16 ticks.
- Learning speed: updates until ID query accuracy (the 256-example check every 1,000 updates) first reaches 90%;
  a run that never does counts as the budget plus 1,000. A beats B on speed if its mean is at least 20% lower and it
  is lower on at least 4 of 5 seeds.
- A CTM win: a CT variant beats Dans on OOD accuracy, ID accuracy or speed. A CTM cost: Dans beats it.
- Ticks pay on OOD for a recurrent model if its mean OOD accuracy at 64 ticks is at least 5 points above 16.
- Reported: Dans against T8 by the same rules, accuracy by N, curves.

Usage: python -m scripts.recall_memory --phase screen|compare --gpu N --runs CELL:RECIPE:SEED... | --time CELL:RECIPE
       | --summarize
"""
import argparse,json,math,os,random,time
from pathlib import Path

from scripts.cmm_tasks import BATCH,BOS,PAD,PRI,QUERY,SYM,build

OUT=Path('research/results/recall_memory');CKPT=Path('research/runs/recall_memory')
RECIPES=('long','cur','dense');SCREEN_CELLS=('T8',);SCREEN_SEEDS=(0,1);MAX_STEPS=40000
COMPARE_CELLS=('T8','Dans','CTsq','CTsqS','CTfull','CTnlmA');COMPARE_SEEDS=(0,1,2,3,4)
SPLITS={'train':(2,16),'id':(2,16),'ood':(17,32)};EVAL=512;CHECK=256;CHECK_EVERY=1000;RESUME_EVERY=500


def example(size,dense,rng):
    keys=rng.sample(PRI,size);vals=[rng.choice(SYM) for _ in keys];seq=[BOS]
    for k,v in zip(keys,vals):seq+=[k,v]
    mask=[False]*len(seq)
    order=rng.sample(range(size),size) if dense else (rng.sample(range(size),4) if size>=4 else [rng.randrange(size) for _ in range(4)])
    for q in order:seq+=[QUERY,keys[q],vals[q]];mask+=[False,True,False]
    return seq,mask


def batch(lo,hi,dense,n,rng):
    import torch
    rows=[example(rng.randint(lo,hi),dense,rng) for _ in range(n)];length=max(len(s) for s,_ in rows)-1
    x=torch.full((n,length),PAD);y=torch.full((n,length),-100)
    for i,(s,m) in enumerate(rows):
        x[i,:len(s)-1]=torch.tensor(s[:-1])
        for p in range(len(s)-1):
            if m[p]:y[i,p]=s[p+1]
    return x,y


def accuracy(model,recurrent,dense,split,ticks,n,device,by_size=False):
    """Query accuracy (%) on n fixed examples; with by_size, also per N."""
    import torch
    rng=random.Random(f'recall-memory-{split}-{dense}');lo,hi=SPLITS[split];hits=total=0;sizes={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(n//64):
            ns=[rng.randint(lo,hi) for _ in range(64)];rows=[example(k,dense,rng) for k in ns];length=max(len(s) for s,_ in rows)-1
            x=torch.full((64,length),PAD);y=torch.full((64,length),-100)
            for i,(s,m) in enumerate(rows):
                x[i,:len(s)-1]=torch.tensor(s[:-1])
                for p in range(len(s)-1):
                    if m[p]:y[i,p]=s[p+1]
            x,y=x.to(device),y.to(device);pred=model(x,max_thought_steps=ticks if recurrent else None)['logits'].argmax(-1)
            mask=y.ne(-100);right=((pred==y)&mask).sum(1).tolist();count=mask.sum(1).tolist()
            for k,r,c in zip(ns,right,count):hits+=r;total+=c;h=sizes.setdefault(k,[0,0]);h[0]+=r;h[1]+=c
    acc=100*hits/total
    return (acc,{str(k):round(100*h/c,1) for k,(h,c) in sorted(sizes.items())}) if by_size else acc


def train(cell,recipe,seed,phase,device,steps=None,timing=False):
    import torch
    from ctm_transformer.depth_sampling import lognormal_poisson
    from ctm_transformer.pretrain import lr_at,set_lr,step_depth
    name=f'{phase}_{cell}_{recipe}_seed{seed}';out=OUT/f'{name}.json'
    if out.exists() and not timing:return
    budget=steps or (MAX_STEPS if phase=='screen' else json.loads((OUT/'CHOICE.json').read_text())['budget'])
    dense=recipe=='dense';early=phase=='screen'
    torch.manual_seed(seed);model,recurrent=build(cell);model=model.to(device)
    sampler=lognormal_poisson(15,0.5,48) if recurrent else None
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    rng=random.Random(f'recall-memory-train-{seed}');start=time.time();curve=[];first=0
    resume=CKPT/f'{name}.resume.pt'
    if resume.exists() and not timing:
        st=torch.load(resume,map_location=device,weights_only=False);model.load_state_dict(st['model']);opt.load_state_dict(st['opt'])
        first=st['step'];curve=st['curve'];rng.setstate(st['rng']);torch.set_rng_state(st['torch_rng'].cpu());torch.cuda.set_rng_state(st['cuda_rng'].cpu())
        start-=st['seconds'];print(name,'resumed at step',first,flush=True)
    stop=budget
    for step in range(first,budget):
        if step and step%RESUME_EVERY==0 and not timing:
            CKPT.mkdir(parents=True,exist_ok=True);tmp=resume.with_suffix('.tmp')
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'step':step,'curve':curve,'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(),'seconds':time.time()-start},tmp);os.replace(tmp,resume)
        set_lr(opt,lr_at(step,1e-3,200,budget,0.1))
        hi=min(16,2+step//700) if recipe=='cur' else 16
        x,y=batch(2,hi,dense,BATCH,rng);x,y=x.to(device),y.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=y,max_thought_steps=step_depth(sampler,seed,step) if recurrent else None)['loss']
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        if timing and step+1==first+200:
            torch.cuda.synchronize();print(cell,recipe,f'{(time.time()-start)/200:.3f} s/update, peak {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB',flush=True);return
        if (step+1)%CHECK_EVERY==0:
            model.eval();acc=accuracy(model,recurrent,dense,'id',16,CHECK,device);model.train()
            curve.append({'step':step+1,'loss':float(loss),'id_check':acc})
            if early and len(curve)>=2 and min(c['id_check'] for c in curve[-2:])>=95:stop=step+1;break
    model.eval();reached=next((c['step'] for c in curve if c['id_check']>=90),None)
    ticks=(16,32,64) if recurrent else (None,)
    result={'cell':cell,'recipe':recipe,'seed':seed,'phase':phase,'budget':budget,'stopped':stop,'seconds':time.time()-start,
            'parameters':sum(p.numel() for p in model.parameters()),'updates_to_90':reached,'curve':curve}
    result['id'],result['id_by_n']=accuracy(model,recurrent,dense,'id',16 if recurrent else None,EVAL,device,True)
    result['ood']={str(t):accuracy(model,recurrent,dense,'ood',t,EVAL,device) for t in ticks}
    _,result['ood_by_n']=accuracy(model,recurrent,dense,'ood',16 if recurrent else None,EVAL,device,True)
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n');resume.unlink(missing_ok=True)
    print(name,f"{result['seconds']:.0f}s stopped {stop} to90 {reached} id {result['id']:.1f} ood",{k:round(v,1) for k,v in result['ood'].items()},flush=True)


def results(phase):
    return {(r['cell'],r['recipe'],r['seed']):r for r in (json.loads(f.read_text()) for f in OUT.glob(f'{phase}_*.json'))}


def choose():
    res=results('screen');L=['# Recall memory: phase 1 screen','','| Cell | Recipe | Seed | Updates to 90% | Stopped | ID | OOD (16 ticks) |','|---|---|---:|---:|---:|---:|---:|']
    for (c,r,s),v in sorted(res.items()):
        L.append(f"| {c} | {r} | {s} | {v['updates_to_90'] or '-'} | {v['stopped']} | {v['id']:.1f} | {v['ood'].get('16',v['ood'].get('None')):.1f} |")
    choice=None
    for r in RECIPES:
        t8=[res.get(('T8',r,s)) for s in SCREEN_SEEDS]
        if all(t8) and all(v['updates_to_90'] for v in t8):
            need=max(v['updates_to_90'] for v in t8);choice={'recipe':r,'budget':min(MAX_STEPS,max(10000,5000*math.ceil(2*need/5000))),'t8_updates_to_90':need};break
    L+=['',f'Choice: {choice}' if choice else 'No recipe makes recall learnable for T8.']
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'SCREEN.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))
    if choice:(OUT/'CHOICE.json').write_text(json.dumps(choice)+'\n')


def compare():
    res=results('compare');recipe=json.loads((OUT/'CHOICE.json').read_text())['recipe'];S=COMPARE_SEEDS
    cells=[c for c in COMPARE_CELLS if all((c,recipe,s) in res for s in S)]
    get=lambda c,m:[(res[(c,recipe,s)]['id'] if m=='id' else res[(c,recipe,s)]['ood'].get('16',res[(c,recipe,s)]['ood'].get('None')) if m=='ood'
                     else (res[(c,recipe,s)]['updates_to_90'] or res[(c,recipe,s)]['budget']+1000)) for s in S]
    mean=lambda v:sum(v)/len(v)
    def beats(a,b,m):
        va,vb=get(a,m),get(b,m)
        if m=='speed':return mean(va)<=0.8*mean(vb) and sum(x<y for x,y in zip(va,vb))>=4
        return mean(va)-mean(vb)>=5 and sum(x>y for x,y in zip(va,vb))>=4
    L=['# Recall memory: phase 2 comparison',f'Recipe {recipe}; protocol and rules: `scripts/recall_memory.py`.','',
       '| Cell | ID | OOD at 16 | OOD at 64 | Updates to 90% (per seed) |','|---|---:|---:|---:|---|']
    for c in cells:
        o64=[res[(c,recipe,s)]['ood'].get('64') for s in S]
        L.append(f"| {c} | {mean(get(c,'id')):.1f} | {mean(get(c,'ood')):.1f} | {mean(o64):.1f} | "+', '.join(str(res[(c,recipe,s)]['updates_to_90'] or '-') for s in S)+' |' if o64[0] is not None else
                 f"| {c} | {mean(get(c,'id')):.1f} | {mean(get(c,'ood')):.1f} | - | "+', '.join(str(res[(c,recipe,s)]['updates_to_90'] or '-') for s in S)+' |')
    L+=['','## Rule outcomes','']
    pairs=[('Dans','T8'),('T8','Dans')]+[(c,'Dans') for c in cells if c.startswith('CT')]+[('Dans',c) for c in cells if c.startswith('CT')]
    for a,b in pairs:
        if a in cells and b in cells:
            w=[m for m in ('id','ood','speed') if beats(a,b,m)]
            if w:L.append(f'- **{a} beats {b}:** '+', '.join(w))
    for c in cells:
        if c!='T8':
            o16,o64=mean(get(c,'ood')),mean([res[(c,recipe,s)]['ood']['64'] for s in S]);L.append(f'- {c}: OOD 16 -> 64 ticks {o16:.1f} -> {o64:.1f}'+(' (ticks pay)' if o64-o16>=5 else ''))
    (OUT/'COMPARE.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('screen','compare'),default='screen');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--time');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:
        choose()
        if (OUT/'CHOICE.json').exists() and any(OUT.glob('compare_*.json')):compare()
        return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    if a.time:c,r=a.time.split(':');train(c,r,0,'screen',device,timing=True);return
    for x in a.runs:c,r,s=x.split(':');train(c,r,int(s),a.phase,device)


if __name__=='__main__':main()
