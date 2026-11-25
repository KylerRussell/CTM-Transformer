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

Pilot outcome (2026-11-17): T8 and Dans each reach 90% on every task within 500-1,000 updates, so P = 1,000 and the
stream uses copy, reverse, sort, increment.

Main stream (`--phase stream`): copy, reverse, sort, increment, then the same again (two cycles, 8 phases of 1,000
updates, 8,000 in all, no replay); every task evaluated every 250 updates and at each phase end. T8, Dans, CTsqS,
CTnlmA; seeds 0-4. A[p][t] is task t's accuracy at the end of phase p.
Measures (per run):
- Forgetting F: for each task except the last of a cycle, its accuracy at the end of its own phase minus its accuracy
  at the end of that cycle; the mean over the 6 (task, cycle) pairs.
- Final accuracy: the mean over the four tasks at the end of the stream.
- Relearning: updates within its cycle-2 phase until a task reaches 90% (checks every 250; 1,250 if never), mean
  over the four tasks.
Rules, fixed before any stream run:
- A beats B on forgetting if its 5-seed mean F is at least 10 points lower and lower on at least 4 of 5 seeds
  (paired by seed); on final accuracy if at least 5 points higher and higher on at least 4 of 5; on relearning if at
  least 20% lower and lower on at least 4 of 5.
- A CTM win: CTsqS or CTnlmA beats Dans on any measure; a CTM cost: Dans beats it. A win counts only once confirmed
  on fresh seeds 5-14 by the same rule scaled to 10 seeds (higher or lower on at least 8 of 10), rules unchanged.
- Reported: T8 against every cell; each run's accuracy matrix; Dans against T8.

Stream outcome (2026-11-18): forgetting is total for every model (98-99 points; each task 0-1% after the next phase)
and relearning is at the floor (250 updates): no model beats another. Without replay nothing makes a model use the task
token, so the test saturates.

Replay (2026-11-19, `--phase replay`): the same stream, but each training example comes from a uniformly chosen
earlier-seen task (not the current one) with probability r, so that forgetting is partial and models can differ.
Pilot, rule fixed before it: T8 and Dans, seed 0, r = 0.01, 0.03, 0.1; the chosen r is the one whose mean forgetting
over the two models is closest to 50 points among those where both models' forgetting is between 10 and 90; if none
qualifies, r = 0.3 (all above 90) or 0.003 (all below 10) is added and the rule re-applied.
Main replay stream: T8, Dans, CTsqS, CTnlmA at the chosen r on seeds 1-5 (seed 0 chose r). Rules: the stream's,
unchanged (forgetting at least 10 points lower and lower on 4 of 5; final accuracy at least 5 higher and higher on 4 of
5; relearning at least 20% lower and lower on 4 of 5; a CTM win is CTsqS or CTnlmA beating Dans, counted only once
confirmed on fresh seeds 6-15, at least 8 of 10).

Usage: python -m scripts.lifelong --phase pilot|stream|replay --gpu N --runs CELL:TASK_OR_STREAM:SEED... | --summarize
"""
import argparse,json,math,os,random,time
from pathlib import Path

from scripts.cmm_tasks import BATCH,PAD,SEP,SYM,build

OUT=Path('research/results/lifelong');CKPT=Path('research/runs/lifelong')
TASKS=('copy','reverse','sort','increment','rotate','swap');STREAM=('copy','reverse','sort','increment');PHASE=1000;CHECK=250
STREAM_CELLS=('T8','Dans','CTsqS','CTnlmA');STREAM_SEEDS=(0,1,2,3,4);START={t:20+i for i,t in enumerate(TASKS)};LENGTHS=(4,16)
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


def batch(task,n,rng,replay=None):
    import torch
    rows=[]
    for _ in range(n):
        t=task if replay is None or not replay[1] or rng.random()>=replay[0] else rng.choice(replay[1])
        s=[rng.choice(SYM) for _ in range(rng.randint(*LENGTHS))];rows.append(([START[t],*s,SEP,*output(t,s)],len(s)+2))
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


def train(cell,schedule,seed,phase,device,check_every,tag,replay=0.0):
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
                        'cuda_rng':torch.cuda.get_rng_state(),'seconds':time.time()-start},tmp);__import__('scripts.pretrain',fromlist=['durable_replace']).durable_replace(tmp,resume)  # fsynced (2026-11-25)
        task=next(t for a,b,t in bounds if a<=step<b);set_lr(opt,1e-3*min(1,(step+1)/200))
        seen=sorted({t for a,b,t in bounds if b<=step and t!=task})
        x,y=batch(task,BATCH,rng,(replay,seen) if replay else None);x,y=x.to(device),y.to(device)
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


def measures(r):
    ends=[e for e in r['log'] if e['step']%PHASE==0];A=[e['acc'] for e in ends];n=len(STREAM)
    F=[A[c*n+i][t]-A[c*n+n-1][t] for c in (0,1) for i,t in enumerate(STREAM[:-1])]
    rel=[]
    for i,t in enumerate(STREAM):
        lo=(n+i)*PHASE;hit=next((e['step']-lo for e in r['log'] if lo<e['step']<=lo+PHASE and e['acc'][t]>=90),None);rel.append(hit if hit is not None else PHASE+CHECK)
    return {'forgetting':sum(F)/len(F),'final':sum(A[-1][t] for t in STREAM)/n,'relearning':sum(rel)/n,'matrix':A}


def stream_summary(seeds=STREAM_SEEDS,title='seeds 0-4',need=4,prefix='stream_',name=None):
    res={(r['cell'],r['seed']):measures(r) for r in (json.loads(f.read_text()) for f in OUT.glob(prefix+'*.json')) if r['seed'] in seeds}
    cells=[c for c in STREAM_CELLS if all((c,s) in res for s in seeds)];mean=lambda v:sum(v)/len(v)
    L=[f'# Lifelong learning: task stream ({title})','','Protocol and rules: `scripts/lifelong.py`.','','| Cell | Forgetting | Final accuracy | Relearning (updates) |','|---|---:|---:|---:|']
    for c in cells:L.append(f'| {c} | '+' | '.join(f"{mean([res[(c,s)][m] for s in seeds]):.1f}" for m in ('forgetting','final','relearning'))+' |')
    def beats(a,b,m):
        va,vb=[res[(a,s)][m] for s in seeds],[res[(b,s)][m] for s in seeds]
        if m=='forgetting':return mean(vb)-mean(va)>=10 and sum(x<y for x,y in zip(va,vb))>=need
        if m=='final':return mean(va)-mean(vb)>=5 and sum(x>y for x,y in zip(va,vb))>=need
        return mean(va)<=0.8*mean(vb) and sum(x<y for x,y in zip(va,vb))>=need
    L+=['','## Rule outcomes','']
    for a,b in [(x,'Dans') for x in ('CTsqS','CTnlmA')]+[('Dans',x) for x in ('CTsqS','CTnlmA')]+[('Dans','T8'),('T8','Dans'),('T8','CTsqS'),('CTsqS','T8'),('T8','CTnlmA'),('CTnlmA','T8')]:
        if a in cells and b in cells:
            w=[m for m in ('forgetting','final','relearning') if beats(a,b,m)]
            L.append(f"- {a} beats {b}: {', '.join(w) if w else 'none'}")
    L+=['','## Accuracy at each phase end (seed means; rows = phase, columns = '+', '.join(STREAM)+')','']
    for c in cells:
        L.append(f'- **{c}:** '+' | '.join('/'.join(f"{mean([res[(c,s)]['matrix'][p][t] for s in seeds]):.0f}" for t in STREAM) for p in range(2*len(STREAM))))
    name=name or ('STREAM.md' if seeds==STREAM_SEEDS else 'STREAM_CONFIRM.md');(OUT/name).write_text('\n'.join(L)+'\n');print('\n'.join(L))


REPLAY_RATES=(0.01,0.03,0.1);REPLAY_SEEDS=(1,2,3,4,5)


def replay_summary():
    L=['# Lifelong learning with replay: pilot (seed 0)','','| r | T8 forgetting | Dans forgetting | T8 final | Dans final |','|---:|---:|---:|---:|---:|'];F={}
    for r in sorted({json.loads(f.read_text())['tag'] for f in OUT.glob('replay_*_seed0.json')}):
        m={c:measures(json.loads((OUT/f'replay_{c}_{r}_seed0.json').read_text())) for c in PILOT_CELLS if (OUT/f'replay_{c}_{r}_seed0.json').exists()}
        if len(m)==2:F[r]=m;L.append(f"| {r[6:]} | {m['T8']['forgetting']:.1f} | {m['Dans']['forgetting']:.1f} | {m['T8']['final']:.1f} | {m['Dans']['final']:.1f} |")
    ok={r:m for r,m in F.items() if all(10<=v['forgetting']<=90 for v in m.values())}
    if ok:
        r=min(ok,key=lambda k:abs(sum(v['forgetting'] for v in ok[k].values())/2-50));L+=['',f'Chosen: {r}'];(OUT/'REPLAY_CHOICE.json').write_text(json.dumps({'tag':r,'rate':float(r[6:])})+'\n')
    else:L+=['','No rate qualifies yet.']
    (OUT/'REPLAY_PILOT.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))
    if ok:
        tag=json.loads((OUT/'REPLAY_CHOICE.json').read_text())['tag']
        stream_summary(REPLAY_SEEDS,f'replay {tag[6:]}, seeds 1-5',4,f'replay_*_{tag}_seed','REPLAY.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('pilot','stream','replay'),default='pilot');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:
        pilot_summary()
        if any(OUT.glob('stream_*.json')):stream_summary()
        if any(OUT.glob('replay_*.json')):replay_summary()
        return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    for x in a.runs:
        c,t,s=x.split(':')
        if a.phase=='pilot':train(c,[(t,PILOT_STEPS)],int(s),'pilot',device,PILOT_CHECK,t)
        elif a.phase=='stream':train(c,[(x,PHASE) for _ in range(2) for x in STREAM],int(s),'stream',device,CHECK,'stream')
        else:train(c,[(x,PHASE) for _ in range(2) for x in STREAM],int(s),'replay',device,CHECK,f'replay{float(t):g}',replay=float(t))


if __name__=='__main__':main()
