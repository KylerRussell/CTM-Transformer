"""Reasoning tasks in text with a fair chance to be learned: a 50% problem share and a size curriculum (2026-11-06).

In scripts/reasoning_in_text.py (100M tokens of continued pretraining, problems 15% of the tokens, sizes uniform
over the training range) only reversed addition was learned; tracking, parity and nested expressions stayed near
chance on their training sizes, so whether thinking longer pays on them could not be tested. At width 128, parity was
fitted by 2-5 of 10 seeds with uniform lengths and by 9-10 of 10 with a length curriculum (scripts/parity_ticks.py,
_cur). This repeats the experiment with the problems made learnable.

Data (`--build`, research/data/pretrain/reasoning_mix_32k/): as reasoning_in_text, with two changes:
- a problem after every G text tokens, G ~ 1 + Geometric(1/24): problems are about half of the tokens;
- a size curriculum: four shards of 25M tokens, presented in order (pretrain `stagewise`). In stage s = 1..4 each
  task's size is uniform over [lo, lo + ceil((hi - lo) s / 4)] of its training range, so stage 4 is the full range.
Text from FineWeb-Edu train_002.bin. The tasks, formats, OOD ranges and side arrays are reasoning_in_text's.

Training and evaluation: exactly reasoning_in_text (arms Dans/CTsq x sampler/span, replicates 1-2 from the
400M-token checkpoints of seeds 1-2, 100M tokens, peak lr 1e-3; the same evaluation problems and text windows).

Rules, fixed before any run: reasoning_in_text's rules (learned task-splits, ticks pay, a CTM win), unchanged.
Also reported: which tasks became learned that were not in reasoning_in_text, and each arm's text loss against the
reasoning_in_text arm of the same name (the cost of less text).

Usage: python -m scripts.reasoning_mix --build | --queue k/2 | --summarize
"""
import argparse,json,math,os,random
from pathlib import Path
import numpy as np

import scripts.reasoning_in_text as R

R.DATA=Path('research/data/pretrain/reasoning_mix_32k');R.RUNS=Path('research/runs/reasoning_mix');R.OUT=Path('research/results/reasoning_mix')
STAGES,STAGE_TOKENS,GAP=4,25_000_000,24
_config=R.config


def config(arm,replicate):
    run=_config(arm,replicate);run['train']['stagewise']=True;return run


R.config=config


def build():
    R.DATA.mkdir(parents=True,exist_ok=True);text=np.memmap(R.SRC/'train_002.bin',dtype=np.uint16,mode='r');rng=random.Random('reasoning-mix-build')
    src=0;counts={t:0 for t in R.TASKS};share=[]
    for stage in range(1,STAGES+1):
        tokens=np.zeros(STAGE_TOKENS,dtype=np.uint16);side=np.zeros(STAGE_TOKENS,dtype=np.uint16);pos=0
        while pos<STAGE_TOKENS:
            g=1
            while rng.random()>1/GAP:g+=1
            g=min(g,STAGE_TOKENS-pos);tokens[pos:pos+g]=text[src:src+g];pos+=g;src+=g
            if pos>=STAGE_TOKENS:break
            task=rng.choice(R.TASKS);lo,hi=R.SPLITS[task][0];q,a,qlen=R.problem(task,rng.randint(lo,lo+math.ceil((hi-lo)*stage/STAGES)),rng)
            ids=(q+a)[:STAGE_TOKENS-pos];tokens[pos:pos+len(ids)]=ids;side[pos:pos+len(ids)]=qlen;pos+=len(ids);counts[task]+=1
        tokens.tofile(R.DATA/f'train_{stage-1:03d}.bin');side.tofile(R.DATA/f'side_{stage-1:03d}.bin');share.append(float((side>0).mean()))
    src_manifest=json.loads((R.SRC/'manifest.json').read_text())
    manifest={'name':'reasoning_mix_32k','tokenizer':'../fineweb_edu_32k/tokenizer.json','tokenizer_sha256':src_manifest['tokenizer_sha256'],'vocab_size':32768,
              'eot_id':0,'dtype':'uint16','train':[f'train_{i:03d}.bin' for i in range(STAGES)],'side':[f'side_{i:03d}.bin' for i in range(STAGES)],
              'validation':['../fineweb_edu_32k/validation_000.bin'],'text_source':'fineweb_edu_32k/train_002.bin, first %d tokens'%src,
              'problems':counts,'problem_token_share_by_stage':share,'gap_mean':GAP,'curriculum':'stage s: sizes uniform over [lo, lo + ceil((hi - lo) s / 4)]'}
    (R.DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ('problems','problem_token_share_by_stage')}))


def summarize():
    R.summarize('Reasoning tasks in text, 50% problems and a size curriculum (d704, 100M tokens of continued pretraining)','reasoning_mix','REASONING_MIX.md')
    before=Path('research/results/reasoning_in_text');L=['','## Against reasoning_in_text (reported)','']
    mean=lambda v:sum(v)/len(v)
    for a in R.ARMS:
        now=[R.OUT/f'{a}_r{r}.json' for r in R.REPLICATES];old=[before/f'{a}_r{r}.json' for r in R.REPLICATES]
        if not all(f.exists() for f in now+old):continue
        get=lambda fs,k:mean([json.loads(f.read_text())[k] for f in fs])
        acc=lambda fs,ts:mean([json.loads(f.read_text())['accuracy'][ts]['span1' if a.endswith('span') else 'fixed16'] for f in fs])
        L.append(f'- **{a}:** text loss {get(now,"text_loss_16"):.3f} (was {get(old,"text_loss_16"):.3f}); ID accuracy '+
                 ', '.join(f'{t} {acc(now,t+"_id"):.0f} (was {acc(old,t+"_id"):.0f})' for t in R.TASKS))
    f=R.OUT/'REASONING_MIX.md';f.write_text(f.read_text()+'\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');p.add_argument('--queue')
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.build:build();return
    if a.summarize:summarize();return
    k,n=map(int,a.queue.split('/'));os.environ['CUDA_VISIBLE_DEVICES']=str(k);jobs=R.queue(k,n);print('queue',k,[f'{x}_r{y}' for x,y in jobs],flush=True)
    for arm,rep in jobs:R.train(arm,rep,k);R.evaluate(f'{arm}_r{rep}')


if __name__=='__main__':main()
