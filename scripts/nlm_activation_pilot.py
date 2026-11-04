"""The neuron-level activation in the language model: a 100M-token pilot (2026-11-04).

In trained CTnlm (400M tokens) the CTM's neuron-level models, added to the SwiGLU activation through a
zero-initialized scale, stayed small (RMS 0.02-0.03 against 0.35-0.47) and removing them cost 0.009 nats, so the
CTM's core mechanism was barely used. CTnlmA (ctm_transformer `nlm_activation`) makes them the gate activation of
the thinking layers, as in the CTM: each neuron's activation is its private model's output over its last 4 ticks of
pre-activation. Every model so far reaches a fixed point by about 8 ticks; neuron-level dynamics are the CTM's own
candidate for a state that keeps changing usefully.

Arm: CTnlmA_sampler, the pilot's sampler arm (scripts/per_token_pilot.py: d704, 100M tokens, seeds 1 and 2, the
same data order, the usual tick sampler) with mechanisms neuron_models + nlm_activation, compiled (`compile_ctm`),
micro-batch 8 above 28 ticks as CTnlm. Compared with the pilot's Dans_sampler (4.761) and CTsq_sampler (4.755).
Evaluation as the pilot: 128 held-out windows, every fixed tick count 1-96.

Rules, fixed before the runs (two-seed means; both seeds must agree):
- Ticks pay: loss at 32 and at 64 ticks each at least 0.01 below the loss at 16 ticks.
- A CTM win: loss at 16 ticks at least 0.02 below Dans_sampler's, both seeds below both Dans seeds.
- No cost: loss at 16 ticks no more than 0.01 above Dans_sampler's.
- Decision: if "ticks pay" or "a CTM win" holds, CTnlmA goes to 400M tokens with two seeds (and Dans is the reference).

Usage: python -m scripts.nlm_activation_pilot --seed S --gpu N | --summarize
"""
import argparse,json,os,subprocess,sys
from pathlib import Path

import scripts.per_token_pilot as pilot

ARM='CTnlmA_sampler'


def config(seed):
    run=pilot.config('CTsq_sampler',seed);t=run['train']
    run.update(name=f'{ARM}_s{seed}',run_directory=str(pilot.RUNS/f'{ARM}_s{seed}'),ctm_transformer=['neuron_models','nlm_activation'])
    t['compile_ctm']=True;t['micro_batch_by_depth']=[[28,8]]
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


def summarize():
    mean=lambda v:sum(v)/len(v);get=lambda n:json.loads((pilot.OUT/f'{n}.json').read_text())['fixed_ce']
    arms=('Dans_sampler','CTsq_sampler',ARM);r={a:[get(f'{a}_s{s}') for s in pilot.SEEDS] for a in arms}
    L=['# Neuron-level activation: language-model pilot (100M tokens)','','Protocol and rules: `scripts/nlm_activation_pilot.py` (docstring).','',
       '| Arm | '+' | '.join(f'{t} ticks' for t in (1,2,4,8,16,32,64,96))+' |','|---|'+'---:|'*8]
    for a in arms:L.append(f'| {a} | '+' | '.join(f'{mean([x[str(t)] for x in r[a]]):.3f}' for t in (1,2,4,8,16,32,64,96))+' |')
    x,d=r[ARM],r['Dans_sampler']
    pays=all(s[str(t)]<=s['16']-0.01 for s in x for t in (32,64))
    diff=mean([s['16'] for s in x])-mean([s['16'] for s in d])
    win=diff<=-0.02 and all(s['16']<min(q['16'] for q in d) for s in x)
    L+=['',f'- **Ticks pay:** {"yes" if pays else "no"}',f'- **A CTM win (at least 0.02 below Dans at 16 ticks):** {"yes" if win else "no"} ({diff:+.3f})',
        f'- **No cost (within 0.01 of Dans):** {"yes" if diff<=0.01 else "no"}',
        f'- **Decision:** '+('CTnlmA goes to 400M tokens with two seeds' if pays or win else 'CTnlmA does not go to 400M tokens')]
    (pilot.OUT/'NLM_ACTIVATION.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    train(a.seed,a.gpu)


if __name__=='__main__':main()
