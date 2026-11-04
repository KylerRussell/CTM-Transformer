"""Training-rate parity: does a compiled CTsq train at about Dans's rate with the same loss? (2026-11-04)

At 400M tokens CTsq took 15.6 GPU-hours against Dans's 10.2. A benchmark (d704, micro-batch 16, 1,024 tokens)
found two causes:
- the CTM Transformer was never compiled. ctm_transformer_factory took no compile option, while Dans's blocks are
  compiled. Uncompiled, CTsq is 1.39x Dans per sequence at 15 ticks; compiled (prelude, coda and thinking layer),
  1.08x.
- CTsq halved its micro-batch on steps deeper than 28 ticks (micro_batch_by_depth [[28, 8]]), which is slower per
  sequence, while the compiled model at micro-batch 16 peaks at 17.4 GB at 48 ticks.
Compiled and uncompiled CTsq agree to 1e-6 in loss and gradients (float32, every mechanism).

Arm: CTsq_fast = the pilot's CTsq_sampler arm (scripts/per_token_pilot.py: 100M tokens, seeds 1 and 2, the same
data) with `compile_ctm` and no micro_batch_by_depth.

Rules, fixed before the runs:
- Same loss: CTsq_fast's two-seed mean loss at 16 ticks (the pilot's 128-window evaluation) within 0.01 of
  CTsq_sampler's (4.755).
- Similar rate: CTsq_fast's mean seconds per update at most 1.15 times Dans_sampler's (both on one GPU, run alone).
- If both hold, every later CTM Transformer run uses `compile_ctm` without micro_batch_by_depth.

Usage: python -m scripts.ctm_speed_check --seed S --gpu N | --summarize
"""
import argparse,json,os,subprocess,sys
from pathlib import Path

import scripts.per_token_pilot as pilot

ARM='CTsq_fast'


def config(seed):
    run=pilot.config('CTsq_sampler',seed);t=run['train']
    run.update(name=f'{ARM}_s{seed}',run_directory=str(pilot.RUNS/f'{ARM}_s{seed}'));t['compile_ctm']=True;t.pop('micro_batch_by_depth',None)
    return run


def train(seed,gpu):
    d=pilot.RUNS/f'{ARM}_s{seed}'
    if not (d/'complete.json').exists():
        d.mkdir(parents=True,exist_ok=True);(d/'config.json').write_text(json.dumps(config(seed),indent=2)+'\n')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
        with open(d/'train.log','a') as f:subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(d/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env)
        if not (d/'complete.json').exists():raise RuntimeError(f'{ARM}_s{seed}: training did not complete')
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    from scripts.exit_calibration import analyse
    analyse(f'{ARM}_s{seed}',torch.device('cuda'),run_dir=d,out_dir=pilot.OUT,t_max=pilot.T_EVAL)


def seconds(name):
    rows=[json.loads(r) for r in (pilot.RUNS/name/'metrics.jsonl').read_text().splitlines() if r.strip()]
    return sum(r['seconds'] for r in rows)/len(rows)


def summarize():
    mean=lambda v:sum(v)/len(v);loss=lambda n:json.loads((pilot.OUT/f'{n}.json').read_text())['fixed_ce']['16']
    names={a:[f'{a}_s{s}' for s in pilot.SEEDS] for a in ('Dans_sampler','CTsq_sampler',ARM)}
    L=['# Training-rate parity (100M tokens)','','Protocol and rules: `scripts/ctm_speed_check.py` (docstring).','','| Arm | Loss at 16 ticks | Seconds per update | vs Dans |','|---|---:|---:|---:|']
    dans=mean([seconds(n) for n in names['Dans_sampler']]);row={}
    for a,ns in names.items():
        l=[loss(n) for n in ns];s=mean([seconds(n) for n in ns]);row[a]=(mean(l),s)
        L.append(f'| {a} | {mean(l):.3f} ({"/".join(f"{x:.3f}" for x in l)}) | {s:.1f} | {s/dans:.2f}x |')
    same=abs(row[ARM][0]-row['CTsq_sampler'][0])<=0.01;rate=row[ARM][1]<=1.15*dans
    L+=['',f'- **Same loss (within 0.01 of CTsq_sampler):** {"yes" if same else "no"} ({row[ARM][0]-row["CTsq_sampler"][0]:+.3f})',
        f'- **Similar rate (at most 1.15x Dans):** {"yes" if rate else "no"} ({row[ARM][1]/dans:.2f}x)',
        f'- **Outcome:** '+('later CTM Transformer runs use compile_ctm without micro_batch_by_depth' if same and rate else 'the change is not adopted as is')]
    (pilot.OUT/'CTM_SPEED.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    train(a.seed,a.gpu)


if __name__=='__main__':main()
