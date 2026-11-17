"""Lifelong learning: a stream of interfering tasks, learned one after another (2026-11-17).

The CTM paper names lifelong learning as future work. Our only test was one domain shift at d704 (stage 0B: after
100M tokens of math and code, forgetting of web text was the same for every model, +0.37 to +0.40). This tests the
usual continual-learning setting at width 128, where the models have differed (scripts/recall_memory.py).

Tasks: one input format, BOS_t s1..sL SEP -> output (16 symbols, L 4-16), the task named by its start token t:
- copy: s1..sL;  reverse: sL..s1;  sort: the symbols in increasing order;  increment: (s_i + 1) mod 16 for each i.
They share inputs and output vocabulary and conflict in their outputs, so learning one interferes with the others.
Loss on output tokens only; sequence accuracy (all output tokens right) on 512 fixed examples per task.

Models: the CMM grid's (scripts/cmm_tasks.py) at width 128: T8, Dans, CTsqS (best CTM Transformer on recall), CTnlmA
(neuron-level models with their activation). Recurrent models train with the LM tick sampler and are evaluated at 16
ticks. AdamW, betas (0.9, 0.95), weight decay 0.1, batch 128, clip 1, bf16; lr 1e-3 after a 200-update warmup,
constant (no decay across the stream). Resume checkpoints every 500 updates.

Pilot (`--phase pilot`): T8 and Dans, seed 0, each of the six tasks alone from scratch for 3,000 updates, accuracy every
250 updates. Rule, fixed before the pilot: the phase length P is the largest number of updates either model needs to
reach 90% on any task, rounded up to a multiple of 500, at least 1,000; if some task is not reached by both in 3,000
updates it is replaced by the first of rotate (s2..sL s1) and swap (adjacent pairs swapped) that both learn.

Main stream (`--phase stream`), and its rules, are added after the pilot and before any stream run.

Usage: python -m scripts.lifelong --phase pilot|stream --gpu N --runs CELL:TASK_OR_STREAM:SEED... | --summarize
"""
import argparse,json,math,os,random,time
from pathlib import Path

from scripts.cmm_tasks import BATCH,PAD,SEP,SYM,build

OUT=Path('research/results/lifelong');CKPT=Path('research/runs/lifelong')
TASKS=('copy','reverse','sort','increment','rotate','swap');START={t:20+i for i,t in enumerate(TASKS)};LENGTHS=(4,16)
EVAL=512;RESUME_EVERY=500;PILOT_STEPS=3000;PILOT_CHECK=250;PILOT_CELLS=('T8','Dans')


def output(task,s):
    if task=='copy':return list(s)
    if task=='reverse':return s[::-1]
    if task=='sort':return sorted(s)
    if task=='increment':return [SYM[(SYM.index(x)+1)%len(SYM)] for x in s]
    if task=='rotate':return s[1:]+s[:1]
    out=list(s)
    for i in range(0,len(s)-1,2):out[i],out[i+1]=out[i+1],out[i]
    return out


def batch(task,n,rng):
    import torch
    rows=[]
    for _ in range(n):
        s=[rng.choice(SYM) for _ in range(rng.randint(*LENGTHS))];rows.append(([START[task],*s,SEP,*output(task,s)],len(s)+2))
    length=max(len(r) for r,_ in rows)-1;x=torch.full((n,length),PAD);y=torch.full((n,length),-100)
    for i,(r,first) in enumerate(rows):
        x[i,:len(r)-1]=torch.tensor(r[:-1]);y[i,first-1:len(r)-1]=torch.tensor(r[first:])
    return x,y


def accuracy(model,recurrent,task,device):
    import torch
    rng=random.Random(f'lifelong-eval-{task}');hits=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(EVAL//64):
            x,y=batch(task,64,rng);x,y=x.to(device),y.to(device);pred=model(x,max_thought_steps=16 if recurrent else None)['logits'].argmax(-1)
            mask=y.ne(-100);hits+=int((((pred==y)|~mask).all(1)).sum())
    return 100*hits/EVAL


def train(cell,schedule,seed,phase,device,check_every,tag):
    """schedule: list of (task, updates). Returns the result dict; evaluates every task at each check and phase end."""
    import torch
    from ctm_transformer.depth_sampling import lognormal_poisson
    from ctm_transformer.pretrain import set_lr,step_depth
    name=f'{phase}_{cell}_{tag}_seed{seed}';out=OUT/f'{name}.json'
    if out.exists():return
    torch.manual_seed(seed);model,recurrent=build(cell);model=model.to(device)
    sampler=lognormal_poisson(15,0.5,48) if recurrent else None
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    rng=random.Random(f'lifelong-train-{seed}');start=time.time();log=[];first=0;total=sum(u for _,u in schedule)
    tasks=sorted({t for t,_ in schedule},key=TASKS.index);resume=CKPT/f'{name}.resume.pt'
    if resume.exists():
        st=torch.load(resume,map_location=device,weights_only=False);model.load_state_dict(st['model']);opt.load_state_dict(st['opt']);first=st['step'];log=st['log']
        rng.setstate(st['rng']);torch.set_rng_state(st['torch_rng'].cpu());torch.cuda.set_rng_state(st['cuda_rng'].cpu());start-=st['seconds'];print(name,'resumed at',first,flush=True)
    bounds=[];acc=0
    for t,u in schedule:bounds.append((acc,acc+u,t));acc+=u
    for step in range(first,total):
        if step and step%RESUME_EVERY==0:
            CKPT.mkdir(parents=True,exist_ok=True);tmp=resume.with_suffix('.tmp')
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'step':step,'log':log,'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(),'seconds':time.time()-start},tmp);os.replace(tmp,resume)
        task=next(t for a,b,t in bounds if a<=step<b);set_lr(opt,1e-3*min(1,(step+1)/200))
        x,y=batch(task,BATCH,rng);x,y=x.to(device),y.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=y,max_thought_steps=step_depth(sampler,seed,step) if recurrent else None)['loss']
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        if (step+1)%check_every==0 or any(step+1==b for _,b,_ in bounds):
            model.eval();log.append({'step':step+1,'task':task,'loss':float(loss),'acc':{t:accuracy(model,recurrent,t,device) for t in tasks}});model.train()
    result={'cell':cell,'tag':tag,'seed':seed,'phase':phase,'schedule':schedule,'seconds':time.time()-start,'log':log}
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n');resume.unlink(missing_ok=True)
    print(name,f"{result['seconds']:.0f}s final",log[-1]['acc'],flush=True)


def pilot_summary():
    L=['# Lifelong learning: pilot (each task alone, from scratch)','','| Cell | Task | Updates to 90% | Accuracy at 3,000 |','|---|---|---:|---:|'];need={}
    for f in sorted(OUT.glob('pilot_*.json')):
        r=json.loads(f.read_text());t=r['tag'];hit=next((e['step'] for e in r['log'] if e['acc'][t]>=90),None);need[(r['cell'],t)]=hit
        L.append(f"| {r['cell']} | {t} | {hit or '-'} | {r['log'][-1]['acc'][t]:.1f} |")
    learned=[t for t in TASKS if all(need.get((c,t)) for c in PILOT_CELLS)]
    if learned:L+=['',f'Learned by both: {learned}; phase length if these are used: {max(1000,500*math.ceil(max(need[(c,t)] for c in PILOT_CELLS for t in learned)/500))}']
    (OUT/'PILOT.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('pilot','stream'),default='pilot');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:pilot_summary();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    for x in a.runs:
        c,t,s=x.split(':')
        if a.phase=='pilot':train(c,[(t,PILOT_STEPS)],int(s),'pilot',device,PILOT_CHECK,t)


if __name__=='__main__':main()
