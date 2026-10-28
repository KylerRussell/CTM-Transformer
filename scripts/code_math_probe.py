"""Stage 0A and 0B of the ladder plan: do the CTM Transformer's additions differ on code and math? (2026-10-28)

The CTM Transformer ties Dans (the RDT) on FineWeb-Edu, uniformly across the token categories tested
(research/results/capability_eval). Code and math need multi-step structure, so differences between the
additions may show there instead. Data: scripts/code_math_data.py (FineMath-4+, codeparrot-clean Python,
GSM8K, MBPP, HumanEval; our FineWeb-Edu 32k tokenizer).

Models: the 400M-token language models T8, T20, Dans, CTsq, CTnlm, CTso and CTsqH (d704; two seeds each,
T20 one), at their final checkpoints.

Measures (mean cross-entropy per token, nats; recurrent models at 16 ticks):
- math: 1,024 held-out FineMath-4+ windows of 1,024 tokens; code: 1,024 held-out codeparrot-clean windows;
- gsm8k: the reference solution's tokens given "Question: ...\\nAnswer:" (GSM8K test, 1,319 problems);
- mbpp: the reference program's tokens given its docstring prompt (MBPP test, 500 problems);
- humaneval: the canonical solution's tokens given the prompt (HumanEval, 164 problems);
- fineweb: 256 held-out FineWeb-Edu windows (offset 40,000, as the capability evaluation's M0).
Solution measures pool every solution token; a sequence longer than 1,024 tokens keeps its last 1,024.
Reported, not judged: loss at 1, 4, 8, 16, 32 and 64 ticks on 64 math and 64 code windows.

0A, zero-shot: the measures on the 400M-token checkpoints, without training.
0B, continued pretraining: each checkpoint trains on 100M more tokens of a 50/50 FineMath-4+ /
codeparrot-clean mix (381 updates of 262,144 tokens; warmup 50; cosine to 10%; peak lr 1e-3, half the
pretraining peak; every other setting as in its pretraining; data order by replicate, the same for every
model), then the measures again. Forgetting (fineweb after minus before) is reported.

Rules, fixed before any run. Comparisons use two-seed means; T20 has one seed and is reported only.
- A model differs from a reference on a measure if its mean differs by at least 0.02 nats and both its
  seeds lie on the same side of both reference seeds. References: Dans for every model (T8 against
  Dans is the RDT against its 1-tick twin), and CTsq for the other CTM Transformer variants.
- Capability-specific: per seed, excess = (measure - fineweb) - (reference's mean measure - reference's
  mean fineweb). A model has a specific advantage (disadvantage) on a code / math measure if both
  seeds' excess is at most -0.02 (at least +0.02): the difference goes beyond what its web-text loss explains.
- Both 0A and 0B are judged by these rules. 0A is out of domain (none of the models saw code or
  math); the selection below uses 0B and 0C.
- Selection of the ladder's CTM Transformer arm: a variant (CTnlm, CTso, CTsqH) replaces CTsq if it beats
  CTsq on at least two of the 0B code / math measures (math, code, gsm8k, mbpp, humaneval) and the 0C
  task-splits (scripts/code_math_tasks.py) together, and loses on none. If several qualify, the one with
  the most wins; ties go to the lower 0B math + code loss. Otherwise CTsq stays.

Usage: python -m scripts.code_math_probe --gpu N --queue k/2 | --summarize
"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path

DATA=Path('research/data/pretrain/code_math_32k');OUT=Path('research/results/code_math');RUNS=Path('research/runs/code_math')
SOURCE_DIRS=[Path('research/runs/rdt_recipe'),Path('research/runs/ctm_transformer')]
GROUPS={'T8':['T8_lr0.002','T8_lr0.002_s2'],'T20':['T20_lr0.002'],'Dans':['Dans_rand_lr0.002','Dans_rand_lr0.002_s2'],
        **{v:[f'{v}_lr0.002',f'{v}_lr0.002_s2'] for v in ('CTsq','CTnlm','CTso','CTsqH')}}
VARIANTS=('CTnlm','CTso','CTsqH');MEASURES=('math','code','gsm8k','mbpp','humaneval');CPT_STEPS=381
WINDOWS,FINEWEB_WINDOWS,FINEWEB_OFFSET,TICK_WINDOWS,TICKS=1024,256,40000,64,(1,4,8,16,32,64)


def source(name):return next(d/name for d in SOURCE_DIRS if (d/name/'config.json').exists())


def cpt_config(name):
    run=json.loads((source(name)/'config.json').read_text());t=run['train']
    run.update(name=f'{name}_cm',seed=2 if name.endswith('_s2') else 1,data=str(DATA/'manifest.json'),run_directory=str(RUNS/name),
               init_from={'checkpoint':str(source(name)/f'step_{t["total_steps"]:07d}.pt'),'prefixes':['']})
    t.update(total_steps=CPT_STEPS,warmup_steps=50,lr=1e-3,eval_interval=CPT_STEPS,eval_windows=64,final_eval_windows=256,eval_offset=0,
             checkpoint_interval=50,snapshot_interval=10**12,delete_checkpoint_on_complete=False)
    return run


def load(run_dir,device):
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.rdt_recipe import recipe_factory
    from ctm_transformer.ctm_transformer import ctm_transformer_factory
    from ctm_transformer.lm_scale import scaled_factory
    run=json.loads((run_dir/'config.json').read_text());config=build_config(run,32768)
    if run.get('ctm_transformer') is not None:factory=ctm_transformer_factory(set(run['ctm_transformer']),run['rdt_recipe'])
    elif run.get('rdt_recipe'):factory=recipe_factory(run['rdt_recipe'])
    else:factory=scaled_factory(run['family'])
    model=factory(config)
    model.load_state_dict(torch.load(run_dir/f'step_{run["train"]["total_steps"]:07d}.pt',map_location='cpu',weights_only=False)['model'])
    return model.to(device).eval(),run['family']=='rdt'


def window_loss(model,recurrent,windows,indices,ticks,device):
    import torch
    from torch.nn import functional as F
    total=0.0;count=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(indices),4):
            x,y=windows.batch(indices[i:i+4]);x,y=x.to(device),y.to(device)
            logits=model(x,max_thought_steps=ticks if recurrent else None)['logits'].float()
            total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum'));count+=y.numel()
    return total/count


def solution_loss(model,recurrent,pairs,tok,device):
    """Pooled cross-entropy of the solution tokens given <eot> + prompt."""
    import torch
    from torch.nn import functional as F
    rows=[]
    for p in pairs:
        prompt=[0]+tok.encode(p['prompt']).ids;sol=tok.encode(p['solution']).ids;seq=(prompt+sol)[-1025:]
        rows.append((seq,min(len(sol),len(seq)-1)))
    rows.sort(key=lambda r:len(r[0]));total=0.0;count=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(rows),8):
            chunk=rows[i:i+8];L=max(len(s) for s,_ in chunk)-1
            x=torch.zeros((len(chunk),L),dtype=torch.long);y=torch.full((len(chunk),L),-100)
            for j,(s,n) in enumerate(chunk):
                x[j,:len(s)-1]=torch.tensor(s[:-1]);y[j,len(s)-1-n:len(s)-1]=torch.tensor(s[len(s)-n:])
            x,y=x.to(device),y.to(device);logits=model(x,max_thought_steps=16 if recurrent else None)['logits'].float()
            total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='sum'));count+=int(y.ne(-100).sum())
    return total/count


def evaluate(name,stage,device):
    """stage 'zero' (0A, the 400M-token checkpoint) or 'cpt' (0B, after continued pretraining)."""
    from tokenizers import Tokenizer
    from ctm_transformer.pretrain import TokenWindows
    out=OUT/f'{stage}_{name}.json'
    if out.exists():return
    model,recurrent=load(source(name) if stage=='zero' else RUNS/name,device)
    tok=Tokenizer.from_file('research/data/pretrain/fineweb_edu_32k/tokenizer.json');tasks=json.loads((DATA/'tasks.json').read_text())
    sets={k:TokenWindows([DATA/f'{k}_valid.bin'],1024,0) for k in ('math','code')}
    fineweb=TokenWindows([Path('research/data/pretrain/fineweb_edu_32k/validation_000.bin')],1024,0)
    r={'run':name,'stage':stage,'loss':{k:window_loss(model,recurrent,w,list(range(WINDOWS)),16,device) for k,w in sets.items()}}
    r['loss'].update({k:solution_loss(model,recurrent,tasks[k],tok,device) for k in ('gsm8k','mbpp','humaneval')})
    r['loss']['fineweb']=window_loss(model,recurrent,fineweb,list(range(FINEWEB_OFFSET,FINEWEB_OFFSET+FINEWEB_WINDOWS)),16,device)
    if recurrent:r['ticks']={k:{str(t):window_loss(model,recurrent,w,list(range(TICK_WINDOWS)),t,device) for t in TICKS} for k,w in sets.items()}
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=1)+'\n')
    print(stage,name,json.dumps({k:round(v,4) for k,v in r['loss'].items()}),flush=True)


def train(name,gpu):
    d=RUNS/name
    if (d/'complete.json').exists():return
    d.mkdir(parents=True,exist_ok=True);(d/'config.json').write_text(json.dumps(cpt_config(name),indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    print('train',name,flush=True)
    with open(d/'train.log','a') as f:code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
    print('train',name,'exit',code,flush=True)
    if not (d/'complete.json').exists():raise RuntimeError(f'{name}: training did not complete (exit {code})')


def hours(name):
    rows=[json.loads(r) for r in (source(name)/'metrics.jsonl').read_text().splitlines() if r.strip()]
    return sum(r['seconds'] for r in rows)


def queue(k,n):
    """Longest-first assignment of the runs to n queues, balanced by their pretraining time."""
    names=sorted((x for g in GROUPS.values() for x in g),key=hours,reverse=True);loads=[0.0]*n;mine=[]
    for x in names:
        q=loads.index(min(loads));loads[q]+=hours(x)
        if q==k:mine.append(x)
    return mine


# ---------------------------------------------------------------- summary

def results(stage):
    return {n:json.loads((OUT/f'{stage}_{n}.json').read_text()) for g in GROUPS.values() for n in g if (OUT/f'{stage}_{n}.json').exists()}


def compare(rs,ref,k,specific=True):
    """'better' / 'worse' / None for the seeds rs against the reference seeds, by the rules; plus specific advantage."""
    mean=lambda v:sum(v)/len(v);d=mean([r[k] for r in rs])-mean([q[k] for q in ref]);notes=[]
    if d<=-0.02 and all(r[k]<min(q[k] for q in ref) for r in rs):notes.append('better')
    if d>=0.02 and all(r[k]>max(q[k] for q in ref) for r in rs):notes.append('worse')
    if specific and k!='fineweb':
        ex=[(r[k]-r['fineweb'])-(mean([q[k] for q in ref])-mean([q['fineweb'] for q in ref])) for r in rs]
        if all(e<=-0.02 for e in ex):notes.append('specific advantage')
        if all(e>=0.02 for e in ex):notes.append('specific disadvantage')
    return d,notes


def outcomes(stage):
    res=results(stage);have={g:[res[n]['loss'] for n in ns if n in res] for g,ns in GROUPS.items()};out={}
    for g in GROUPS:
        if len(have[g])<2:continue
        for ref in ('Dans',)+(('CTsq',) if g in VARIANTS else ()):
            if g==ref or len(have[ref])<2:continue
            out[(g,ref)]={k:compare(have[g],have[ref],k) for k in MEASURES+('fineweb',)}
    return have,out


def summarize():
    mean=lambda v:sum(v)/len(v);L=['# Stage 0A / 0B: code and math','','Protocol and rules: `scripts/code_math_probe.py` (docstring). Loss in nats per token; two-seed means (T20 one seed).','']
    for stage,title in (('zero','0A: zero-shot (400M-token checkpoints)'),('cpt','0B: after 100M tokens of math + code')):
        have,out=outcomes(stage);groups=[g for g in GROUPS if have[g]]
        if not groups:continue
        L+=[f'## {title}','','| Measure | '+' | '.join(groups)+' |','|---|'+'---:|'*len(groups)]
        for k in MEASURES+('fineweb',):L.append(f'| {k} | '+' | '.join(f'{mean([r[k] for r in have[g]]):.3f}' for g in groups)+' |')
        L+=['','Rule outcomes:','']
        for (g,ref),ms in out.items():
            notes=[f'{k} {", ".join(n)} ({d:+.3f})' for k,(d,n) in ms.items() if n]
            L.append(f'- **{g} vs {ref}:** '+('; '.join(notes) if notes else 'no measure differs'))
        L.append('')
    zero,cpt=results('zero'),results('cpt')
    if cpt:
        L+=['## Forgetting and gains (0B minus 0A)','','| Model | fineweb | math | code |','|---|---:|---:|---:|']
        for g,ns in GROUPS.items():
            ns=[n for n in ns if n in zero and n in cpt]
            if ns:L.append(f'| {g} | '+' | '.join(f'{mean([cpt[n]["loss"][k]-zero[n]["loss"][k] for n in ns]):+.3f}' for k in ('fineweb','math','code'))+' |')
        L+=['','## Loss by ticks after 0B (64 windows; reported only)','']
        for g,ns in GROUPS.items():
            ns=[n for n in ns if n in cpt and 'ticks' in cpt[n]]
            if ns:L.append(f'- **{g}:** '+'; '.join(f'{k} '+' / '.join(f'{mean([cpt[n]["ticks"][k][str(t)] for n in ns]):.3f}' for t in TICKS) for k in ('math','code'))+f' (ticks {"/".join(map(str,TICKS))})')
        L+=['','## Selection of the ladder arm (0B measures + 0C task-splits against CTsq)','']
        _,out=outcomes('cpt');tasks=Path('research/results/code_math_tasks/selection.json')
        task_votes=json.loads(tasks.read_text()) if tasks.exists() else None
        qualified=[]
        for v in VARIANTS:
            if (v,'CTsq') not in out:continue
            wins=[k for k,(d,n) in out[(v,'CTsq')].items() if k in MEASURES and 'better' in n];losses=[k for k,(d,n) in out[(v,'CTsq')].items() if k in MEASURES and 'worse' in n]
            if task_votes:wins+=task_votes[v]['wins'];losses+=task_votes[v]['losses']
            ok=len(wins)>=2 and not losses;qualified+=[(v,len(wins))] if ok else []
            L.append(f'- **{v}:** wins {wins or "none"}; losses {losses or "none"}; qualifies: {"yes" if ok else "no"}')
        if task_votes is None:L.append('- 0C results not yet in; the selection is provisional.')
        if qualified:
            best=max(qualified,key=lambda q:(q[1],-mean([cpt[n]['loss']['math']+cpt[n]['loss']['code'] for n in GROUPS[q[0]]])))
            L.append(f'- **Ladder arm:** {best[0]} replaces CTsq')
        else:L.append('- **Ladder arm:** CTsq stays')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CODE_MATH.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--queue',help='k/n')
    p.add_argument('--summarize',action='store_true');p.add_argument('--zero-only',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));names=queue(k,n);print('queue',k,names,flush=True)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    for x in names:evaluate(x,'zero',device)
    if a.zero_only:return
    torch.cuda.empty_cache()
    for x in names:train(x,a.gpu);evaluate(x,'cpt',device);torch.cuda.empty_cache()


if __name__=='__main__':main()
