"""Associative recall inside text at d704: does the CTM Transformer's recall advantage reach the language model? (2026-11-16)

At width 128 (scripts/recall_memory.py), with a curriculum over the number of pairs, CTsqS (queries from neuron
synchronization + the CMM attention sink) learned key-value binding in 19 of 25 runs and Dans in 4 of 25, both at
their tuned learning rate, confirmed on two sets of fresh seeds. The 400M-token language models sit near chance on
synthetic recall (research/results/capability_eval: 4-9% at 8 pairs). This asks whether continued pretraining on text
with recall problems in it, under the same curriculum, makes CTsqS learn in-context recall where Dans does not.

Data (`--build`, research/data/pretrain/recall_32k/): as scripts/reasoning_mix.py (FineWeb-Edu train_002.bin text, a
problem after every G text tokens, G ~ 1 + Geometric(1/24), four 25M-token stages presented in order), with one task:
" Recall : k1 v1 ; k2 v2 ; ... ; kN vN ; ? k = v ." Keys are N distinct letters from a-z and A-Z (52), values digits
0-9, the query one of the N keys; every symbol is one token (" Recall" is two). Train N 2-16 (stage s = 1..4: N
uniform over [2, 2 + ceil(14 s / 4)], so the full range in stage 4); OOD N 17-32.

Training (scripts/pretrain.py, as scripts/reasoning_in_text.py's sampler arms): from each 400M-token checkpoint of
Dans and CTsqS (replicate 1 from seed 1, 2 from seed 2), 100M tokens (381 updates of 262,144; warmup 50; cosine to
10%; peak lr 1e-3), the ordinary language-model loss on every token, the usual tick sampler (mean 15, at most 48).

Evaluation: per split 512 problems after 96 tokens of held-out text (validation_000.bin), each scored by whether the
argmax at the "=" position is the value; at fixed ticks 4, 8, 16, 32, 64; accuracy by N at 16 ticks. Chance is 10%
(about 1/distinct values present for a model that picks among the values in context). Text loss: 128 held-out windows
without problems at 16 ticks.

Rules, fixed before any run (two replicates, paired by seed):
- A run binds if its ID accuracy at 16 ticks is at least 80%.
- A CTM win: CTsqS's accuracy at 16 ticks is at least 10 points above Dans's on the ID or the OOD split on both
  replicates, and Dans is not 10 points above CTsqS on either split on any replicate. A CTM cost: the reverse.
- Ticks pay for a run if its OOD accuracy at 64 ticks is at least 10 points above 16 (reported per arm; "pays" if on
  both replicates).
- Two replicates of an all-or-nothing outcome are thin: if the outcome is a win or a split (the arms bind on different
  numbers of replicates), the follow-up is more continued-training replicates, with rules fixed before them.
- Reported: accuracy by N, ticks, text loss of each arm.

Usage: python -m scripts.recall_in_text --build | --queue k/2 | --summarize
"""
import argparse,json,math,os,random
from pathlib import Path
import numpy as np

import scripts.reasoning_in_text as R

DATA=Path('research/data/pretrain/recall_32k');RUNS=Path('research/runs/recall_in_text');OUT=Path('research/results/recall_in_text')
R.DATA,R.RUNS=DATA,RUNS
BASES={'Dans':R.BASES['Dans'],'CTsqS':('research/runs/ctm_transformer/CTsqS_lr0.002','research/runs/ctm_transformer/CTsqS_lr0.002_s2')}
R.BASES=BASES
ARMS=('Dans_sampler','CTsqS_sampler');REPLICATES=(1,2);SPLITS={'id':(2,16),'ood':(17,32)}
STAGES,STAGE_TOKENS,GAP=4,25_000_000,24;TICKS=(4,8,16,32,64);EVAL_N,PREFIX=512,96
KEYS=[*'abcdefghijklmnopqrstuvwxyz',*'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];_V=None


def vocab():
    global _V
    if _V is None:
        from tokenizers import Tokenizer
        t=Tokenizer.from_file(str(R.SRC/'tokenizer.json'));_V={}
        for w in [*KEYS,*map(str,range(10)),';','?','=',':','.']:e=t.encode(' '+w).ids;assert len(e)==1,w;_V[w]=e[0]
        _V['Recall']=t.encode(' Recall').ids
    return _V


def problem(size,rng):
    """(question ids up to and including '=', value id, ' .' id)."""
    V=vocab();keys=rng.sample(KEYS,size);vals=[rng.randrange(10) for _ in keys];q=[*V['Recall'],V[':']]
    for k,v in zip(keys,vals):q+=[V[k],V[str(v)],V[';']]
    i=rng.randrange(size);q+=[V['?'],V[keys[i]],V['=']]
    return q,V[str(vals[i])],V['.']


def build():
    DATA.mkdir(parents=True,exist_ok=True);text=np.memmap(R.SRC/'train_002.bin',dtype=np.uint16,mode='r');rng=random.Random('recall-in-text-build')
    src=0;count=0;share=[]
    for stage in range(1,STAGES+1):
        tokens=np.zeros(STAGE_TOKENS,dtype=np.uint16);side=np.zeros(STAGE_TOKENS,dtype=np.uint16);pos=0
        while pos<STAGE_TOKENS:
            g=1
            while rng.random()>1/GAP:g+=1
            g=min(g,STAGE_TOKENS-pos);tokens[pos:pos+g]=text[src:src+g];pos+=g;src+=g
            if pos>=STAGE_TOKENS:break
            q,v,end=problem(rng.randint(2,2+math.ceil(14*stage/STAGES)),rng);ids=[*q,v,end][:STAGE_TOKENS-pos]
            tokens[pos:pos+len(ids)]=ids;side[pos:pos+len(ids)]=len(q);pos+=len(ids);count+=1
        tokens.tofile(DATA/f'train_{stage-1:03d}.bin');side.tofile(DATA/f'side_{stage-1:03d}.bin');share.append(float((side>0).mean()))
    m=json.loads((R.SRC/'manifest.json').read_text())
    manifest={'name':'recall_32k','tokenizer':'../fineweb_edu_32k/tokenizer.json','tokenizer_sha256':m['tokenizer_sha256'],'vocab_size':32768,'eot_id':0,'dtype':'uint16',
              'train':[f'train_{i:03d}.bin' for i in range(STAGES)],'side':[f'side_{i:03d}.bin' for i in range(STAGES)],'validation':['../fineweb_edu_32k/validation_000.bin'],
              'text_source':'fineweb_edu_32k/train_002.bin, first %d tokens'%src,'problems':count,'problem_token_share_by_stage':share,'gap_mean':GAP,
              'curriculum':'stage s: N uniform over [2, 2 + ceil(14 s / 4)]'}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ('problems','problem_token_share_by_stage')}))


def config(arm,replicate):
    model,_=arm.split('_');base=Path(BASES[model][replicate-1]);run=json.loads((base/'config.json').read_text());t=run['train']
    run.update(name=f'{arm}_r{replicate}',seed=replicate,data=str(DATA/'manifest.json'),run_directory=str(RUNS/f'{arm}_r{replicate}'),
               init_from={'checkpoint':str(base/f'step_{t["total_steps"]:07d}.pt'),'prefixes':['']})
    t.update(total_steps=381,warmup_steps=50,lr=1e-3,eval_interval=381,eval_windows=64,final_eval_windows=128,eval_offset=40000,
             checkpoint_interval=50,snapshot_interval=10**12,delete_checkpoint_on_complete=False,stagewise=True)
    if model!='Dans':t['compile_ctm']=True;t.pop('micro_batch_by_depth',None)
    return run


R.config=config


def evaluate(run):
    import torch
    from scripts.code_math_probe import load,window_loss
    from ctm_transformer.pretrain import TokenWindows
    out=OUT/f'{run}.json'
    if out.exists():return
    device=torch.device('cuda');model,_=load(RUNS/run,device);text=np.memmap(R.SRC/'validation_000.bin',dtype=np.uint16,mode='r');r={'run':run}
    for split,(lo,hi) in SPLITS.items():
        rng=random.Random(f'recall-in-text-eval-{split}');rows=[]
        for _ in range(EVAL_N):
            start=rng.randrange(len(text)-PREFIX);n=rng.randint(lo,hi);q,v,_=problem(n,rng);rows.append((n,[*text[start:start+PREFIX].tolist(),*q],v))
        L=max(len(s) for _,s,_ in rows);x=torch.zeros((EVAL_N,L),dtype=torch.long);at=torch.tensor([len(s)-1 for _,s,_ in rows]);val=torch.tensor([v for _,_,v in rows])
        for i,(_,s,_) in enumerate(rows):x[i,:len(s)]=torch.tensor(s)
        acc={};right16=None
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for t in TICKS:
                hit=[]
                for i in range(0,EVAL_N,32):
                    logits=model(x[i:i+32].to(device),max_thought_steps=t)['logits'];pred=logits[torch.arange(len(logits)),at[i:i+32].to(device)].argmax(-1)
                    hit+=(pred.cpu()==val[i:i+32]).tolist()
                acc[str(t)]=100*sum(hit)/EVAL_N
                if t==16:right16=hit
        by={}
        for (n,_,_),h in zip(rows,right16):b=by.setdefault(n,[0,0]);b[0]+=h;b[1]+=1
        r[split]=acc;r[f'{split}_by_n']={str(n):round(100*h/c,1) for n,(h,c) in sorted(by.items())}
    r['text_loss_16']=window_loss(model,True,TokenWindows([R.SRC/'validation_000.bin'],1024,0),list(range(40000,40128)),16,device)
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=1)+'\n')
    print(run,'text',round(r['text_loss_16'],4),'id',{k:round(v,1) for k,v in r['id'].items()},'ood',{k:round(v,1) for k,v in r['ood'].items()},flush=True)
    del model;torch.cuda.empty_cache()


def summarize():
    res={(a,k):json.loads((OUT/f'{a}_r{k}.json').read_text()) for a in ARMS for k in REPLICATES if (OUT/f'{a}_r{k}.json').exists()}
    L=['# Associative recall inside text (d704, 100M tokens of continued pretraining)','','Protocol and rules: `scripts/recall_in_text.py`. Accuracy (%) of the value at the "=" position; chance 10%.','',
       '| Run | ID at 4 / 8 / 16 / 32 / 64 ticks | OOD at 4 / 8 / 16 / 32 / 64 ticks | Text loss | Binds |','|---|---|---|---:|---|']
    for (a,k),r in sorted(res.items()):
        L.append(f"| {a}_r{k} | "+' / '.join(f"{r['id'][str(t)]:.1f}" for t in TICKS)+' | '+' / '.join(f"{r['ood'][str(t)]:.1f}" for t in TICKS)+f" | {r['text_loss_16']:.3f} | {'yes' if r['id']['16']>=80 else 'no'} |")
    if len(res)==len(ARMS)*len(REPLICATES):
        a,b='CTsqS_sampler','Dans_sampler';d={s:[res[(a,k)][s]['16']-res[(b,k)][s]['16'] for k in REPLICATES] for s in SPLITS}
        win=any(all(x>=10 for x in d[s]) for s in SPLITS) and not any(x<=-10 for s in SPLITS for x in d[s])
        cost=any(all(x<=-10 for x in d[s]) for s in SPLITS) and not any(x>=10 for s in SPLITS for x in d[s])
        bind={x:sum(res[(x,k)]['id']['16']>=80 for k in REPLICATES) for x in ARMS}
        L+=['','## Rule outcomes','',f"- **CTM win:** {'yes' if win else 'no'}; CTM cost: {'yes' if cost else 'no'} (CTsqS minus Dans at 16 ticks: "+'; '.join(f"{s} {', '.join(f'{x:+.1f}' for x in d[s])}" for s in SPLITS)+')',
            f"- Binding: CTsqS {bind[a]}/2, Dans {bind[b]}/2"+(' (a split: more replicates follow)' if bind[a]!=bind[b] else ''),
            '- Ticks pay (OOD 64 vs 16, at least +10): '+', '.join(f"{x} {'yes' if all(res[(x,k)]['ood']['64']-res[(x,k)]['ood']['16']>=10 for k in REPLICATES) else 'no'}" for x in ARMS)]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'RECALL_IN_TEXT.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');p.add_argument('--queue');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.build:build();return
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));os.environ['CUDA_VISIBLE_DEVICES']=str(k)
    jobs=[(arm,rep) for rep in REPLICATES for arm in ARMS][k::n];print('queue',k,[f'{x}_r{y}' for x,y in jobs],flush=True)
    for arm,rep in jobs:R.train(arm,rep,k);evaluate(f'{arm}_r{rep}')


if __name__=='__main__':main()
