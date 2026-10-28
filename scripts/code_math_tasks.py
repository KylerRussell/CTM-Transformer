"""Stage 0C of the ladder plan: synthetic arithmetic and program tracing, trained from scratch (2026-10-28).

Code and math need multi-step computation whose depth grows with the input. These tasks isolate that,
with out-of-distribution (OOD) splits deeper or longer than anything trained, where recurrence (more
ticks) should matter. Tokens: PAD 0, BOS 1, SEP 2, QUERY 3 ('?'), digits 4-13, '+' 14, '*' 15, '(' 16,
')' 17, '=' 18, ';' 19, variables 20-45. The loss is on answer tokens only.

| Task | Sequence -> answer | Train | ID test | OOD test |
|---|---|---|---|---|
| addition | BOS a (n digits, least significant first) + b (n digits) SEP -> a + b (n + 1 digits, least significant first) | n 1-10 | n 1-10 | n 11-20 |
| expression | BOS e = -> e mod 7, e a nested expression of depth d over the digits 1-6 with + and * | d 1-3 | d 1-3 | d 4-6 |
| tracking | BOS statements ; ? v -> the value of v mod 7 | k 1-4 | k 1-4 | k 5-8 |

- expression: a depth-d expression is (A op B) with one operand of depth d - 1 and the other of depth 0
  or 1 (random order); depth 0 is a digit 1-6; the outermost parentheses are dropped. Example, depth 3:
  (3+(4*2))*5 = 5 (mod 7). Arithmetic is mod 7 with nonzero leaves so that no product collapses to 0
  (mod 10, a factor 0, or 2 times 5, would make many deep expressions 0 and the task shallow).
- tracking: two interleaved chains of k assignments over distinct variables. A chain starts "v=c" (c 0-6);
  each later statement is "w=v+c" (c 0-6), "w=v*c" (c 1-6) or "w=v" for the chain's previous variable v;
  values mod 7. The query asks for the last variable of one of the two chains, so k is the number of hops
  to follow.

Models: T8, T20, Dans, CTsq, CTnlm and CTsqH at width 128, exactly as in scripts/cmm_tasks.py (its
model builder, training settings and evaluation: 5,000 updates of batch 128, AdamW lr 1e-3, recurrent
models with the LM's tick sampler, mean 15, at most 48; 512 fixed examples per split). Seeds 0, 1, 2.
Recurrent models are evaluated at 16 ticks, and on the OOD split also at 32 and 64.

Rules, fixed before any run (as in scripts/cmm_tasks.py):
- Primary measure: sequence accuracy (every answer token right) on each task's ID and OOD split
  (6 task-splits), at 16 ticks for recurrent models.
- Model A beats model B on a task-split if its 3-seed mean is at least 5 points higher and it is higher
  on all 3 seeds (paired by seed).
- Comparisons: every model against Dans (T8 against Dans is the RDT against its 1-tick twin), and
  CTnlm and CTsqH against CTsq. The wins and losses against CTsq enter the selection of the ladder's
  CTM Transformer arm (scripts/code_math_probe.py). T20 is reported, and judged against Dans like the others.
- Ticks: a recurrent model extrapolates with ticks on a task if its OOD 3-seed mean at 64 ticks is at
  least 10 points above its mean at 16 ticks.

Usage: python -m scripts.code_math_tasks --gpu N --cells CELL... [--slice k/n] | --summarize | --smoke CELL [--steps S]
"""
import argparse,json,os,random,time
from pathlib import Path

from scripts.cmm_tasks import BATCH,EVAL_EXAMPLES,PAD,BOS,SEP,QUERY,STEPS,build

OUT=Path('research/results/code_math_tasks')
DIGIT=list(range(4,14));PLUS,TIMES,OPEN,CLOSE,EQ,SEMI=14,15,16,17,18,19;VAR=list(range(20,46))
TASKS=('addition','expression','tracking')
SPLITS={'addition':{'train':(1,10),'id':(1,10),'ood':(11,20)},'expression':{'train':(1,3),'id':(1,3),'ood':(4,6)},
        'tracking':{'train':(1,4),'id':(1,4),'ood':(5,8)}}
CELLS=('T8','T20','Dans','CTsq','CTnlm','CTsqH');VARIANTS=('CTnlm','CTsqH')


def expression(depth,rng):
    """(tokens, value mod 7) of a depth-`depth` expression, parenthesized."""
    if depth==0:v=rng.randint(1,6);return [DIGIT[v]],v
    a,va=expression(depth-1,rng);b,vb=expression(rng.randint(0,min(1,depth-1)),rng)
    if rng.random()<0.5:a,va,b,vb=b,vb,a,va
    op=rng.choice((PLUS,TIMES));v=(va+vb)%7 if op==PLUS else (va*vb)%7
    return [OPEN,*a,op,*b,CLOSE],v


def example(task,size,rng):
    """(tokens, answer mask): the mask marks the positions whose next token is an answer token."""
    if task=='addition':
        a=[rng.randrange(10) for _ in range(size)];b=[rng.randrange(10) for _ in range(size)]
        total=sum(d*10**i for i,d in enumerate(a))+sum(d*10**i for i,d in enumerate(b))
        s=[(total//10**i)%10 for i in range(size+1)]
        seq=[BOS,*[DIGIT[d] for d in a],PLUS,*[DIGIT[d] for d in b],SEP,*[DIGIT[d] for d in s]];first=2*size+2
        return seq,[first<=i<len(seq)-1 for i in range(len(seq))]
    if task=='expression':
        e,v=expression(size,rng);e=e[1:-1]
        seq=[BOS,*e,EQ,DIGIT[v]];return seq,[i==len(seq)-2 for i in range(len(seq))]
    names=rng.sample(VAR,2*size);chains=[names[:size],names[size:]];values=[None,None];statements=[[],[]]
    for c in (0,1):
        for j,w in enumerate(chains[c]):
            if j==0:values[c]=rng.randrange(7);statements[c].append([w,EQ,DIGIT[values[c]]]);continue
            v=chains[c][j-1];kind=rng.randrange(3);d=rng.randrange(7) if kind==0 else rng.randint(1,6)
            if kind==0:values[c]=(values[c]+d)%7;statements[c].append([w,EQ,v,PLUS,DIGIT[d]])
            elif kind==1:values[c]=(values[c]*d)%7;statements[c].append([w,EQ,v,TIMES,DIGIT[d]])
            else:statements[c].append([w,EQ,v])
    order=[0]*size+[1]*size;rng.shuffle(order);seq=[BOS];taken=[0,0]  # interleave, keeping each chain's order
    for c in order:seq+=statements[c][taken[c]]+[SEMI];taken[c]+=1
    q=rng.randrange(2);seq+=[QUERY,chains[q][-1],DIGIT[values[q]]]
    return seq,[i==len(seq)-2 for i in range(len(seq))]


def batch(task,split,n,rng):
    import torch
    lo,hi=SPLITS[task][split];rows=[example(task,rng.randint(lo,hi),rng) for _ in range(n)]
    length=max(len(s) for s,_ in rows)-1
    x=torch.full((n,length),PAD);y=torch.full((n,length),-100)
    for i,(s,m) in enumerate(rows):
        x[i,:len(s)-1]=torch.tensor(s[:-1])
        for p in range(len(s)-1):
            if m[p]:y[i,p]=s[p+1]
    return x,y


def evaluate(model,recurrent,task,split,ticks,device):
    import torch
    rng=random.Random(f'eval-{task}-{split}');seq_hits=tok_hits=toks=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(EVAL_EXAMPLES//64):
            x,y=batch(task,split,64,rng);x,y=x.to(device),y.to(device)
            pred=model(x,max_thought_steps=ticks if recurrent else None)['logits'].argmax(-1)
            mask=y.ne(-100);right=(pred==y)&mask
            seq_hits+=int((right.sum(1)==mask.sum(1)).sum());tok_hits+=int(right.sum());toks+=int(mask.sum())
    return {'sequence_accuracy':100*seq_hits/EVAL_EXAMPLES,'token_accuracy':100*tok_hits/toks}


def train(cell,task,seed,device,steps=STEPS):
    import torch
    from ctm_transformer.depth_sampling import lognormal_poisson
    from ctm_transformer.pretrain import lr_at,set_lr,step_depth
    out=OUT/f'{cell}_{task}_seed{seed}.json'
    if out.exists() and steps==STEPS:return
    torch.manual_seed(seed);model,recurrent=build(cell);model=model.to(device)
    sampler=lognormal_poisson(15,0.5,48) if recurrent else None
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    rng=random.Random(f'train-{task}-{seed}');start=time.time();curve=[]
    for step in range(steps):
        set_lr(opt,lr_at(step,1e-3,200,steps,0.1));x,y=batch(task,'train',BATCH,rng);x,y=x.to(device),y.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x,targets=y,max_thought_steps=step_depth(sampler,seed,step) if recurrent else None)['loss']
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        if hasattr(model,'bio_step'):model.bio_step(step,steps)
        if (step+1)%500==0:curve.append({'step':step+1,'loss':float(loss),'grad_norm':float(norm)})
    model.eval();result={'cell':cell,'task':task,'seed':seed,'steps':steps,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated(device)/2**30,'curve':curve}
    result['id']=evaluate(model,recurrent,task,'id',16,device)
    result['ood']={str(t):evaluate(model,recurrent,task,'ood',t,device) for t in ((16,32,64) if recurrent else (1,))}
    if steps==STEPS:OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n')
    ood=result['ood']['16' if recurrent else '1']
    print(cell,task,seed,f"{result['seconds']:.0f}s id {result['id']['sequence_accuracy']:.1f} ood {ood['sequence_accuracy']:.1f} final loss {curve[-1]['loss'] if curve else float(loss):.3f}",flush=True)
    return result


def summarize():
    res={}
    for f in OUT.glob('*_seed*.json'):r=json.loads(f.read_text());res[(r['cell'],r['task'],r['seed'])]=r
    acc=lambda r,s,t='16':r['id']['sequence_accuracy'] if s=='id' else r['ood'][t if t in r['ood'] else '1']['sequence_accuracy']
    cells=[c for c in CELLS if any(k[0]==c for k in res)];splits=[(t,s) for t in TASKS for s in ('id','ood')]
    L=['# Stage 0C: synthetic arithmetic and program tracing','','Protocol and rules: `scripts/code_math_tasks.py` (docstring). Sequence accuracy (%), mean of 3 seeds (per seed in brackets).','',
       '| Task-split | '+' | '.join(cells)+' |','|---|'+'---:|'*len(cells)]
    for t,s in splits:
        row=[]
        for c in cells:
            v=[acc(res[(c,t,k)],s) for k in (0,1,2) if (c,t,k) in res]
            row.append(f'{sum(v)/len(v):.1f} ({"/".join(f"{a:.0f}" for a in v)})' if v else '')
        L.append(f'| {t} {s} | '+' | '.join(row)+' |')
    def beats(a,b,t,s):
        if not all((c,t,k) in res for c in (a,b) for k in (0,1,2)):return None
        va,vb=[acc(res[(a,t,k)],s) for k in (0,1,2)],[acc(res[(b,t,k)],s) for k in (0,1,2)]
        return sum(va)/3-sum(vb)/3>=5 and all(x>y for x,y in zip(va,vb))
    L+=['','## Rule outcomes','']
    pairs=[(c,'Dans') for c in cells if c!='Dans']+[('Dans',c) for c in cells if c!='Dans']+[(v,'CTsq') for v in VARIANTS]+[('CTsq',v) for v in VARIANTS]
    for a,b in pairs:
        wins=[f'{t} {s}' for t,s in splits if beats(a,b,t,s)]
        if wins:L.append(f'- **{a} beats {b}:** '+', '.join(wins))
    complete=all((c,t,k) in res for c in ('CTsq',*VARIANTS) for t in TASKS for k in (0,1,2))
    selection={v:{'wins':[f'{t} {s}' for t,s in splits if beats(v,'CTsq',t,s)],'losses':[f'{t} {s}' for t,s in splits if beats('CTsq',v,t,s)]} for v in VARIANTS}
    if complete:(OUT/'selection.json').write_text(json.dumps(selection,indent=1)+'\n')
    L+=['','## OOD sequence accuracy at more ticks (recurrent models; 16 / 32 / 64)','']
    for c in cells:
        rows=[]
        for t in TASKS:
            v=[res[(c,t,k)]['ood'] for k in (0,1,2) if (c,t,k) in res and '32' in res[(c,t,k)]['ood']]
            if v:
                m={d:sum(o[d]['sequence_accuracy'] for o in v)/len(v) for d in ('16','32','64')}
                rows.append(f'{t} '+' / '.join(f'{m[d]:.1f}' for d in ('16','32','64'))+(' (extrapolates)' if len(v)==3 and m['64']-m['16']>=10 else ''))
        if rows:L.append(f'- **{c}:** '+'; '.join(rows))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CODE_MATH_TASKS.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--cells',nargs='*',default=list(CELLS))
    p.add_argument('--seeds',nargs='*',type=int,default=[0,1,2]);p.add_argument('--tasks',nargs='*',default=list(TASKS))
    p.add_argument('--summarize',action='store_true');p.add_argument('--smoke');p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--slice',help='k/n: run every n-th run of the grid, starting at k');a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    if a.smoke:
        for t in a.tasks:train(a.smoke,t,0,device,steps=a.steps)
        return
    grid=[(c,t,seed) for seed in a.seeds for t in a.tasks for c in a.cells]
    if a.slice:k,n=map(int,a.slice.split('/'));random.Random(0).shuffle(grid);grid=grid[k::n]
    for c,t,seed in grid:train(c,t,seed,device)


if __name__=='__main__':main()
