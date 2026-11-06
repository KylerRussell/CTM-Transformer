"""Reasoning tasks embedded in text, at our language-model size (2026-11-05).

Recurrent-depth language models show test-time gains on reasoning, not on web-text perplexity. Here every model
(Dans, the CTM Transformer) is flat on web text from about 8 ticks. Stage A (scripts/parity_in_text.py, width 128)
showed that parity problems inside text, given ticks that scale with their length, are solved beyond the trained
length by thinking longer. This asks the same at d704, with four reasoning tasks written into ordinary text, and
whether the CTM Transformer does better than Dans there.

Data (`--build`, research/data/pretrain/reasoning_32k/): continued-pretraining text from FineWeb-Edu train_001.bin
(the 400M-token runs drew windows from all 14 shards; their overlap with this text is about 4%) with a reasoning
problem inserted after every G text tokens, G ~ 1 + Geometric(1/128). Problems are written with spaced tokens of the
FineWeb-Edu tokenizer (every digit and symbol is one token) and end with " .":
- add: " Add : a + b = s ." with a, b of n digits and s of n + 1 digits, all least significant digit first
  (n 1-10 in training; out of distribution (OOD) 11-20);
- track: " Track : v = c ; w = v + c ; ... ; ? x = c ." two interleaved chains of k assignments (v = c, w = v + c,
  w = v * c or w = v; values mod 7), the query asks for the last variable of one chain (k 1-4; OOD 5-8);
- parity: " Parity : b1 .. bn = odd ." (n 1-40; OOD 41-80);
- eval: " Eval : expression = c ." nested + and * over digits 1-6, depth d, mod 7 (d 1-3; OOD 4-6), as
  scripts/code_math_tasks.py.
A side array marks every problem token with the problem's question length in tokens (up to and including "="),
for span budgets. 100M training tokens; problems are about 15% of them (the manifest records the share).

Training (scripts/pretrain.py): from each 400M-token checkpoint of Dans and CTsq (replicate 1 from seed 1, 2 from seed
2), 100M tokens (381 updates of 262,144; warmup 50; cosine to 10%; peak lr 1e-3, as 0B in scripts/code_math_probe.py),
the ordinary language-model loss on every token. Arms:
- sampler: the usual tick sampler for every position (mean 15, at most 48);
- span: problem positions get ceil(0.5 q) + U{0..8} ticks for question length q (input-based, label-free); text
  positions the step's sampler count (`span_budgets`).
CTsq is compiled at full micro-batch (scripts/ctm_speed_check.py).

Evaluation: for each task, split (ID: the training range; OOD) and model, 256 problems after 96 tokens of held-out
text (validation_000.bin), scored by teacher-forced exact match of the answer tokens (every answer token the
argmax). Readouts: fixed ticks 4, 8, 16, 32, 64 for every position; span budgets (as trained, text positions 16) at
scale 1 and 2 (ceil(s 0.5 q) + 4). Text loss: 128 held-out windows without problems at 16 ticks.

Rules, fixed before any run (two replicates; "both" means both replicates):
- A task-split is learned if some arm's mean accuracy is at least 80% on its ID split (at 16 ticks, or with span
  budgets for span arms). Only learned task-splits are judged.
- Ticks pay on a task-split for an arm: OOD accuracy at 64 fixed ticks at least 10 points above 16 ticks (both), or,
  for span arms, at span scale 2 at least 10 points above scale 1 (both).
- A CTM win: in the same training mode, CTsq's accuracy (sampler arms at 16 ticks; span arms with span budgets at
  scale 1, or at the better of scale 1 and 2 if ticks pay) beats Dans's by at least 5 points on at least two
  learned task-splits (both replicates, paired), and Dans beats CTsq on none.
- Text cost (reported): each arm's text loss against its model's sampler arm.

Usage: python -m scripts.reasoning_in_text --build | --queue k/2 | --evaluate RUN... | --summarize
"""
import argparse,json,math,os,random,subprocess,sys
from pathlib import Path
import numpy as np

DATA=Path('research/data/pretrain/reasoning_32k');SRC=Path('research/data/pretrain/fineweb_edu_32k')
RUNS=Path('research/runs/reasoning_in_text');OUT=Path('research/results/reasoning_in_text')
TASKS=('add','track','parity','eval');SPLITS={'add':((1,10),(11,20)),'track':((1,4),(5,8)),'parity':((1,40),(41,80)),'eval':((1,3),(4,6))}
TRAIN_TOKENS=100_000_000;GAP=128;BASES={'Dans':('research/runs/rdt_recipe/Dans_rand_lr0.002','research/runs/rdt_recipe/Dans_rand_lr0.002_s2'),
    'CTsq':('research/runs/ctm_transformer/CTsq_lr0.002','research/runs/ctm_transformer/CTsq_lr0.002_s2')}
ARMS=('Dans_sampler','CTsq_sampler','Dans_span','CTsq_span');REPLICATES=(1,2);SPAN={'scale':0.5,'jitter':8}
HOURS={'Dans_sampler':2.6,'CTsq_sampler':2.9,'Dans_span':5.0,'CTsq_span':5.8};FIXED=(4,8,16,32,64);EVAL_N,PREFIX=256,96
_TOK=None


def vocab():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        t=Tokenizer.from_file(str(SRC/'tokenizer.json'));ids={}
        for w in [*[str(d) for d in range(10)],*'abcdefghijklmnopqrstuvwxyz','+','*','=',';','?','(',')',':','.','Add','Track','Eval','odd','even']:
            e=t.encode(' '+w).ids;assert len(e)==1,w;ids[w]=e[0]
        ids['Parity']=t.encode(' Parity').ids;_TOK=ids
    return _TOK


def expression(depth,rng):
    if depth==0:v=rng.randint(1,6);return [str(v)],v
    a,va=expression(depth-1,rng);b,vb=expression(rng.randint(0,min(1,depth-1)),rng)
    if rng.random()<0.5:a,va,b,vb=b,vb,a,va
    op=rng.choice('+*');v=(va+vb)%7 if op=='+' else (va*vb)%7
    return ['(',*a,op,*b,')'],v


def problem(task,size,rng):
    """(token ids, question length): the question runs up to and including '='; the answer follows, then ' .'."""
    V=vocab()
    if task=='add':
        a=[rng.randrange(10) for _ in range(size)];b=[rng.randrange(10) for _ in range(size)]
        total=sum(d*10**i for i,d in enumerate(a))+sum(d*10**i for i,d in enumerate(b));s=[(total//10**i)%10 for i in range(size+1)]
        q=['Add',':',*map(str,a),'+',*map(str,b),'='];ans=list(map(str,s))
    elif task=='parity':
        bits=[rng.randrange(2) for _ in range(size)];q=['Parity',':',*map(str,bits),'='];ans=['odd' if sum(bits)%2 else 'even']
    elif task=='eval':
        e,v=expression(size,rng);q=['Eval',':',*e[1:-1],'='];ans=[str(v)]
    else:
        names=rng.sample('abcdefghijklmnopqrstuvwxyz',2*size);chains=[names[:size],names[size:]];values=[0,0];stmts=[[],[]]
        for c in (0,1):
            for j,w in enumerate(chains[c]):
                if j==0:values[c]=rng.randrange(7);stmts[c].append([w,'=',str(values[c])]);continue
                v=chains[c][j-1];kind=rng.randrange(3);d=rng.randrange(7) if kind==0 else rng.randint(1,6)
                if kind==0:values[c]=(values[c]+d)%7;stmts[c].append([w,'=',v,'+',str(d)])
                elif kind==1:values[c]=(values[c]*d)%7;stmts[c].append([w,'=',v,'*',str(d)])
                else:stmts[c].append([w,'=',v])
        order=[0]*size+[1]*size;rng.shuffle(order);q=['Track',':'];taken=[0,0]
        for c in order:q+=stmts[c][taken[c]]+[';'];taken[c]+=1
        k=rng.randrange(2);q+=['?',chains[k][-1],'='];ans=[str(values[k])]
    ids=[]
    for w in q:ids+=V['Parity'] if w=='Parity' else [V[w]]
    return ids,[V[w] for w in ans]+[V['.']],len(ids)


def build():
    DATA.mkdir(parents=True,exist_ok=True);text=np.memmap(SRC/'train_001.bin',dtype=np.uint16,mode='r');rng=random.Random('reasoning-build')
    tokens=np.zeros(TRAIN_TOKENS,dtype=np.uint16);side=np.zeros(TRAIN_TOKENS,dtype=np.uint16);pos=0;src=0;counts={t:0 for t in TASKS}
    while pos<TRAIN_TOKENS:
        g=1;p=1/GAP
        while rng.random()>p:g+=1
        g=min(g,TRAIN_TOKENS-pos);tokens[pos:pos+g]=text[src:src+g];pos+=g;src+=g
        if pos>=TRAIN_TOKENS:break
        task=rng.choice(TASKS);lo,hi=SPLITS[task][0];q,a,qlen=problem(task,rng.randint(lo,hi),rng);ids=(q+a)[:TRAIN_TOKENS-pos]
        tokens[pos:pos+len(ids)]=ids;side[pos:pos+len(ids)]=qlen;pos+=len(ids);counts[task]+=1
    tokens.tofile(DATA/'train_000.bin');side.tofile(DATA/'side_000.bin')
    src_manifest=json.loads((SRC/'manifest.json').read_text())
    manifest={'name':'reasoning_32k','tokenizer':'../fineweb_edu_32k/tokenizer.json','tokenizer_sha256':src_manifest['tokenizer_sha256'],'vocab_size':32768,
              'eot_id':0,'dtype':'uint16','train':['train_000.bin'],'side':['side_000.bin'],'validation':['../fineweb_edu_32k/validation_000.bin'],
              'text_source':'fineweb_edu_32k/train_001.bin, first %d tokens'%src,'problems':counts,'problem_token_share':float((side>0).mean()),'gap_mean':GAP}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ('problems','problem_token_share')}))


# ---------------------------------------------------------------- training

def config(arm,replicate):
    model,mode=arm.split('_');base=Path(BASES[model][replicate-1]);run=json.loads((base/'config.json').read_text());t=run['train']
    run.update(name=f'{arm}_r{replicate}',seed=replicate,data=str(DATA/'manifest.json'),run_directory=str(RUNS/f'{arm}_r{replicate}'),
               init_from={'checkpoint':str(base/f'step_{t["total_steps"]:07d}.pt'),'prefixes':['']})
    t.update(total_steps=381,warmup_steps=50,lr=1e-3,eval_interval=381,eval_windows=64,final_eval_windows=128,eval_offset=40000,
             checkpoint_interval=50,snapshot_interval=10**12,delete_checkpoint_on_complete=False)
    if model=='CTsq':t['compile_ctm']=True;t.pop('micro_batch_by_depth',None)
    if mode=='span':run['span_budgets']=SPAN
    return run


def train(arm,replicate,gpu):
    d=RUNS/f'{arm}_r{replicate}'
    if (d/'complete.json').exists():return
    d.mkdir(parents=True,exist_ok=True);(d/'config.json').write_text(json.dumps(config(arm,replicate),indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    with open(d/'train.log','a') as f:subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env)
    if not (d/'complete.json').exists():raise RuntimeError(f'{arm}_r{replicate}: training did not complete')


# ---------------------------------------------------------------- evaluation

def eval_batch(task,split,rng,text):
    """x [N, L], answer targets [N, L] (-100 elsewhere), question span (first, qlen) per example."""
    import torch
    lo,hi=SPLITS[task][split];rows=[]
    for _ in range(EVAL_N):
        start=rng.randrange(len(text)-PREFIX);q,a,qlen=problem(task,rng.randint(lo,hi),rng);seq=[*text[start:start+PREFIX].tolist(),*q,*a];rows.append((seq,PREFIX,qlen,len(a)))
    L=max(len(r[0]) for r in rows)-1;x=torch.zeros((EVAL_N,L),dtype=torch.long);y=torch.full((EVAL_N,L),-100)
    for i,(seq,first,qlen,alen) in enumerate(rows):
        t=torch.tensor(seq);x[i,:len(seq)-1]=t[:-1];end=first+qlen  # positions end-1 .. end+alen-2 predict the answer tokens (the final ' .' excluded)
        y[i,end-1:end+alen-2]=t[end:end+alen-1]
    return x,y,[(r[1],r[2],len(r[0])-1) for r in rows]


def accuracy(model,x,y,device,ticks=None,budgets=None):
    import torch
    hits=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(x),32):
            xb,yb=x[i:i+32].to(device),y[i:i+32].to(device)
            out=model(xb,tick_budgets=budgets[i:i+32].to(device)) if budgets is not None else model(xb,max_thought_steps=ticks)
            pred=out['logits'].argmax(-1);mask=yb.ne(-100);hits+=int((((pred==yb)|~mask).all(1)).sum())
    return 100*hits/len(x)


def span_budgets(spans,shape,scale):
    import torch
    b=torch.full(shape,16,dtype=torch.long)
    for i,(first,qlen,length) in enumerate(spans):b[i,first:length]=math.ceil(scale*SPAN['scale']*qlen)+SPAN['jitter']//2
    return b


def evaluate(run):
    import torch
    from scripts.code_math_probe import load,window_loss
    from ctm_transformer.pretrain import TokenWindows
    out=OUT/f'{run}.json'
    if out.exists():return
    device=torch.device('cuda');model,_=load(RUNS/run,device);text=np.memmap(SRC/'validation_000.bin',dtype=np.uint16,mode='r')
    r={'run':run,'accuracy':{}}
    for task in TASKS:
        for split,label in ((0,'id'),(1,'ood')):
            x,y,spans=eval_batch(task,split,random.Random(f'reasoning-eval-{task}-{label}'),text)
            acc={f'fixed{t}':accuracy(model,x,y,device,ticks=t) for t in FIXED}
            for s in (1,2):acc[f'span{s}']=accuracy(model,x,y,device,budgets=span_budgets(spans,x.shape,s))
            r['accuracy'][f'{task}_{label}']=acc
    r['text_loss_16']=window_loss(model,True,TokenWindows([SRC/'validation_000.bin'],1024,0),list(range(40000,40128)),16,device)
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=1)+'\n')
    print(run,'text',round(r['text_loss_16'],4),{k:(round(v['fixed16']),round(v['fixed64']),round(v['span1']),round(v['span2'])) for k,v in r['accuracy'].items()},flush=True)
    del model;torch.cuda.empty_cache()


def queue(k,n):
    jobs=sorted(((a,r) for a in ARMS for r in REPLICATES),key=lambda j:HOURS[j[0]],reverse=True);loads=[0.0]*n;mine=[]
    for j in jobs:
        q=loads.index(min(loads));loads[q]+=HOURS[j[0]]
        if q==k:mine.append(j)
    return mine


# ---------------------------------------------------------------- summary

def summarize(title='Reasoning tasks embedded in text (d704, 100M tokens of continued pretraining)',script='reasoning_in_text',report='REASONING_IN_TEXT.md'):
    res={(a,r):json.loads((OUT/f'{a}_r{r}.json').read_text()) for a in ARMS for r in REPLICATES if (OUT/f'{a}_r{r}.json').exists()}
    mean=lambda v:sum(v)/len(v);splits=[f'{t}_{s}' for t in TASKS for s in ('id','ood')]
    read=lambda a,r,ts,key:res[(a,r)]['accuracy'][ts][key]
    def main_read(a,r,ts,pays=False):
        if a.endswith('sampler'):return read(a,r,ts,'fixed16')
        return max(read(a,r,ts,'span1'),read(a,r,ts,'span2')) if pays else read(a,r,ts,'span1')
    L=[f'# {title}','',f'Protocol and rules: `scripts/{script}.py` (docstring). Exact-match accuracy (%), mean of two replicates.','',
       '| Task-split | '+' | '.join(f'{a} 16 / 64 / span1 / span2' for a in ARMS)+' |','|---|'+'---|'*len(ARMS)]
    for ts in splits:
        row=[]
        for a in ARMS:
            v=[res[(a,r)]['accuracy'][ts] for r in REPLICATES if (a,r) in res]
            row.append(' / '.join(f'{mean([x[k] for x in v]):.0f}' for k in ('fixed16','fixed64','span1','span2')) if v else '')
        L.append(f'| {ts} | '+' | '.join(row)+' |')
    tl=[f'{a} {mean([res[(a,r)]["text_loss_16"] for r in REPLICATES if (a,r) in res]):.3f}' for a in ARMS if any((a,r) in res for r in REPLICATES)]
    L+=['','Text loss at 16 ticks: '+', '.join(tl),'','## Rule outcomes','']
    complete=all((a,r) in res for a in ARMS for r in REPLICATES)
    if not complete:L.append('- incomplete');(OUT/report).write_text('\n'.join(L)+'\n');print('\n'.join(L));return
    learned={t:any(mean([read(a,r,f'{t}_id','span1' if a.endswith('span') else 'fixed16') for r in REPLICATES])>=80 for a in ARMS) for t in TASKS}
    L.append('- **Learned tasks (some arm at least 80% ID):** '+', '.join(f'{t} {"yes" if learned[t] else "no"}' for t in TASKS))
    pays={}
    for a in ARMS:
        for ts in splits:
            if not ts.endswith('ood') or not learned[ts.split('_')[0]]:continue
            fixed=all(read(a,r,ts,'fixed64')>=read(a,r,ts,'fixed16')+10 for r in REPLICATES)
            span=a.endswith('span') and all(read(a,r,ts,'span2')>=read(a,r,ts,'span1')+10 for r in REPLICATES)
            pays[(a,ts)]=fixed or span
            if fixed or span:L.append(f'- **Ticks pay:** {a} on {ts} ('+('fixed 16 -> 64' if fixed else 'span scale 1 -> 2')+')')
    for mode in ('sampler','span'):
        c,d=f'CTsq_{mode}',f'Dans_{mode}';wins,losses=[],[]
        for ts in splits:
            if not learned[ts.split('_')[0]]:continue
            vc=[main_read(c,r,ts,pays.get((c,ts),False)) for r in REPLICATES];vd=[main_read(d,r,ts,pays.get((d,ts),False)) for r in REPLICATES]
            if mean(vc)-mean(vd)>=5 and all(x>y for x,y in zip(vc,vd)):wins.append(ts)
            if mean(vd)-mean(vc)>=5 and all(y>x for x,y in zip(vc,vd)):losses.append(ts)
        L.append(f'- **{mode} arms:** CTsq beats Dans on {wins or "none"}; Dans beats CTsq on {losses or "none"}; CTM win: {"yes" if len(wins)>=2 and not losses else "no"}')
    (OUT/report).write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');p.add_argument('--queue')
    p.add_argument('--evaluate',nargs='*');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.build:build();return
    if a.summarize:summarize();return
    if a.evaluate:
        for r in a.evaluate:evaluate(r)
        return
    k,n=map(int,a.queue.split('/'));os.environ['CUDA_VISIBLE_DEVICES']=str(k);jobs=queue(k,n);print('queue',k,[f'{x}_r{y}' for x,y in jobs],flush=True)
    for arm,rep in jobs:train(arm,rep,k);evaluate(f'{arm}_r{rep}')


if __name__=='__main__':main()
