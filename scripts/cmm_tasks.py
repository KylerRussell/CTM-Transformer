"""The CMM paper's tasks, trained from scratch: does the CTM Transformer inherit the CTM/CMM strengths? (option C, 2026-10-25)

Continuous Memory Machines (Regan et al., arXiv 2610.07907) reports gains on algorithmic memory tasks
(copy, associative recall, priority sort with length generalization, parity), in-context regression,
maze solving and ImageNet-100. The language models here sit near the floor on synthetic recall and copy
(research/results/capability_eval), so this trains small models on the tasks themselves.

Tasks (discrete tokens; the loss is on answer tokens only; CMM's vector inputs become symbol tokens):

| Task | Sequence -> answer | Train | Test in-distribution (ID) | Test out-of-distribution (OOD) |
|---|---|---|---|---|
| copy | BOS s1..sL SEP -> s1..sL (16 symbols) | L 1-32 | L 1-32 | L 33-64 |
| recall | BOS k1 v1 .. kN vN, then 4 x (QUERY k -> v) (64 keys, 16 values) | N 2-16 | N 2-16 | N 17-32 |
| sort | BOS p1 v1 .. pN vN SEP -> the values in increasing priority (64 priorities) | N 4-16 | N 4-16 | N 17-50 |
| parity | BOS b1..bn SEP -> parity of the 1 bits | n 1-40 | n 1-40 | n 41-80 |

Adaptations: CMM's copy uses 100 real vectors and its recall queries one item; priority sort outputs
all N values (CMM: the 16 lowest of 20); parity follows CMM (train 1-40, its test at 40), with an OOD
split added. Few-shot regression (continuous), maze and ImageNet are not included.

Models: the language-model architectures at width 128 (4 heads, SwiGLU 336, RoPE, max length 256):
T8 (8 layers, Dans's one-tick twin), T20 (20 layers, compute-matched to 4 ticks), Dans (2 + 4 thinking
+ 2, the LM recipe), and the CTM Transformer variants CTsq, CTnlm, CTso, CTfull and CTsqS (the CMM
sink), exactly as in ctm_transformer/ctm_transformer.py. Recurrent models train with the LM's tick
sampler (log-normal Poisson, mean 15, at most 48) and are evaluated at 16 ticks (OOD also at 32 and 64).

Training: AdamW lr 1e-3, betas (0.9, 0.95), weight decay 0.1, warmup 200, cosine to 10%, batch 128,
5,000 updates (640,000 examples), gradient clip 1, bf16 autocast, activation checkpointing for recurrent
models; examples drawn fresh each step from a seeded generator. At width 128 the runs are bound by
kernel launches (a smoke test: Dans 0.8 s per update at 6% GPU use), so several runs share each GPU
(`--slice k/n` takes every n-th run of the grid).
Seeds 0, 1, 2; seed s gives every model the same data. Evaluation: 512 fixed examples per split
(the same for every model and seed); sequence accuracy (all answer tokens right) and token accuracy.

Rules, fixed before any run:
- Primary measure: sequence accuracy on each task's ID and OOD split (8 task-splits).
- Model A beats model B on a task-split if its 3-seed mean is at least 5 points higher and it is
  higher on all 3 seeds (paired by seed).
- The CTM Transformer inherits a CTM/CMM strength if a variant beats Dans on a task-split; it costs
  something if Dans beats it. Dans against T8 is judged the same way. T20 is reported.
- Ticks: for recurrent models, OOD sequence accuracy at 32 and 64 ticks against 16 is reported.

Usage: python -m scripts.cmm_tasks --gpu N --cells CELL... [--seeds 0 1 2] [--tasks ...] [--slice k/n] | --summarize | --smoke CELL [--steps S]
"""
import argparse,json,math,os,random,time
from pathlib import Path

OUT=Path('research/results/cmm_tasks')
PAD,BOS,SEP,QUERY=0,1,2,3
SYM=list(range(4,20));PRI=list(range(20,84));BIT=[84,85];VOCAB=96
TASKS=('copy','recall','sort','parity')
SPLITS={'copy':{'train':(1,32),'id':(1,32),'ood':(33,64)},'recall':{'train':(2,16),'id':(2,16),'ood':(17,32)},
        'sort':{'train':(4,16),'id':(4,16),'ood':(17,50)},'parity':{'train':(1,40),'id':(1,40),'ood':(41,80)}}
STEPS,BATCH,EVAL_EXAMPLES,MAX_LEN=5000,128,512,256
CELLS={'T8':('transformer',8),'T20':('transformer',20),'Dans':('dans',()),'CTsq':('ct',('sync_query',)),'CTnlm':('ct',('neuron_models',)),
       'CTso':('ct',('sync_output',)),'CTfull':('ct',('neuron_models','sync_query','sync_output')),'CTsqS':('ct',('sync_query','attention_sink'))}


def example(task,size,rng):
    """(tokens, answer mask): the mask marks the positions whose next token is an answer token."""
    if task=='copy':
        s=[rng.choice(SYM) for _ in range(size)];seq=[BOS,*s,SEP,*s];first=size+1
        return seq,[i>=first and i<len(seq)-1 for i in range(len(seq))]
    if task=='recall':
        keys=rng.sample(PRI,size);vals=[rng.choice(SYM) for _ in keys];seq=[BOS]
        for k,v in zip(keys,vals):seq+=[k,v]
        mask=[False]*len(seq)
        for q in rng.sample(range(size),min(4,size)) if size>=4 else [rng.randrange(size) for _ in range(4)]:
            seq+=[QUERY,keys[q]];mask+=[False,True];seq.append(vals[q]);mask.append(False)
        return seq,mask
    if task=='sort':
        pri=rng.sample(PRI,size);vals=[rng.choice(SYM) for _ in pri];seq=[BOS]
        for p,v in zip(pri,vals):seq+=[p,v]
        seq.append(SEP);first=len(seq)-1;seq+=[v for _,v in sorted(zip(pri,vals))]
        return seq,[i>=first and i<len(seq)-1 for i in range(len(seq))]
    bits=[rng.randrange(2) for _ in range(size)];seq=[BOS,*[BIT[b] for b in bits],SEP,BIT[sum(bits)%2]]
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


def build(cell):
    from dataclasses import replace
    from scripts.pretrain import build_config
    from ctm_transformer.lm_scale import scaled_factory
    from ctm_transformer.rdt_recipe import recipe_factory
    from ctm_transformer.ctm_transformer import DANS,ctm_transformer_factory
    kind,arg=CELLS[cell];train={'seq_len':MAX_LEN,'micro_batch':BATCH,'lr':1e-3}
    if kind=='transformer':
        run={'family':'transformer','train':train,'model':{'d_model':128,'n_layers':arg,'ffn_hidden_dim':336,'init_std':0.02,'n_heads':4}};factory=scaled_factory('transformer')
    else:
        run={'family':'rdt','train':train,'eval_depth':16,'model':{'d_model':128,'prelude_layers':2,'core_layers':4,'coda_layers':2,'ffn_hidden_dim':336,'init_std':0.02,'n_heads':4}}
        factory=recipe_factory(DANS,checkpointing=True) if kind=='dans' else ctm_transformer_factory(set(arg),checkpointing=True)
    return factory(build_config(run,VOCAB)),kind!='transformer'


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
        if (step+1)%500==0:curve.append({'step':step+1,'loss':float(loss),'grad_norm':float(norm)})
    model.eval();result={'cell':cell,'task':task,'seed':seed,'steps':steps,'seconds':time.time()-start,'peak_gib':torch.cuda.max_memory_allocated(device)/2**30,'parameters':sum(p.numel() for p in model.parameters()),'curve':curve}
    result['id']=evaluate(model,recurrent,task,'id',16,device)
    result['ood']={str(t):evaluate(model,recurrent,task,'ood',t,device) for t in ((16,32,64) if recurrent else (1,))}
    if recurrent:result['id_by_ticks']={str(t):evaluate(model,recurrent,task,'id',t,device) for t in (1,4)}
    if steps==STEPS:OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n')
    ood=result['ood']['16' if recurrent else '1']
    print(cell,task,seed,f"{result['seconds']:.0f}s id {result['id']['sequence_accuracy']:.1f} ood {ood['sequence_accuracy']:.1f} final loss {curve[-1]['loss'] if curve else float(loss):.3f}",flush=True)
    return result


def summarize():
    res={}
    for f in OUT.glob('*_seed*.json'):r=json.loads(f.read_text());res[(r['cell'],r['task'],r['seed'])]=r
    acc=lambda r,s:r['id']['sequence_accuracy'] if s=='id' else r['ood']['16' if '16' in r['ood'] else '1']['sequence_accuracy']
    cells=[c for c in CELLS if any(k[0]==c for k in res)]
    L=['# CMM tasks trained from scratch','','Protocol and rules: `scripts/cmm_tasks.py` (docstring). Sequence accuracy (%), mean of 3 seeds (per seed in brackets).','',
       '| Task-split | '+' | '.join(cells)+' |','|---|'+'---:|'*len(cells)]
    for t in TASKS:
        for s in ('id','ood'):
            cells_txt=[]
            for c in cells:
                v=[acc(res[(c,t,k)],s) for k in (0,1,2) if (c,t,k) in res]
                cells_txt.append(f'{sum(v)/len(v):.1f} ({"/".join(f"{a:.0f}" for a in v)})' if v else '')
            L.append(f'| {t} {s} | '+' | '.join(cells_txt)+' |')
    def beats(a,b,t,s):
        if not all((c,t,k) in res for c in (a,b) for k in (0,1,2)):return None
        va,vb=[acc(res[(a,t,k)],s) for k in (0,1,2)],[acc(res[(b,t,k)],s) for k in (0,1,2)]
        return sum(va)/3-sum(vb)/3>=5 and all(x>y for x,y in zip(va,vb))
    L+=['','## Rule outcomes','']
    for a,b in [('Dans','T8'),('T8','Dans')]+[(c,'Dans') for c in cells if c not in ('T8','T20','Dans')]+[('Dans',c) for c in cells if c not in ('T8','T20','Dans')]:
        wins=[f'{t} {s}' for t in TASKS for s in ('id','ood') if beats(a,b,t,s)]
        if wins:L.append(f'- **{a} beats {b}:** '+', '.join(wins))
    L+=['','## OOD sequence accuracy at more ticks (recurrent models; 16 / 32 / 64)','']
    for c in cells:
        rows=[]
        for t in TASKS:
            v=[res[(c,t,k)]['ood'] for k in (0,1,2) if (c,t,k) in res and '32' in res[(c,t,k)]['ood']]
            if v:rows.append(f'{t} '+' / '.join(f'{sum(o[d]["sequence_accuracy"] for o in v)/len(v):.1f}' for d in ('16','32','64')))
        if rows:L.append(f'- **{c}:** '+'; '.join(rows))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CMM_TASKS.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--cells',nargs='*',default=[])
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
    if a.slice:k,n=map(int,a.slice.split('/'));random.Random(0).shuffle(grid);grid=grid[k::n]  # shuffled so slow cells spread over workers
    for c,t,seed in grid:train(c,t,seed,device)


if __name__=='__main__':main()
