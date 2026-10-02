"""Development probes: does CTM-LM learn to use its input with fix A or fix B? (research/CTM_LM_DESIGN.md, 2026-10-02)

The CTM-heavy arm at the sweep's narrowest width (d 448), trained for 300 steps of
32 sequences of 1,024 tokens (9.8M tokens) on one GPU:

| Probe | Model | Learning rate |
|---|---|---|
| faithful | faithful CTM-LM | 5e-4 |
| faithful_low_lr | faithful CTM-LM | 1.25e-4 |
| A_unit_query | fix A: tick-0 query rescaled to unit rms (initialization only) | 5e-4 |
| B_observe_token | fix B: each position's backbone feature enters the synapse (adaptation) | 5e-4 |
| transformer_reference | the Transformer arm at d 384, for scale | 5e-4 |
| rdt_aware_baseline | the compute-aware RDT at d 384 | 5e-4 |
| rdt_aware_low_lr | the same | 1.25e-4 |
| rdt_aware_unit_embedding | the same with input embeddings initialized at std 1 (from about 0.03) | 5e-4 |

The RDT probes were added after the compute-aware RDT also plateaued at the
unigram level in the sweep, with its prelude collapsing token information. The
same "works" rule applies to them; they decide whether the RDT recipe changes.

Measures, on 64 held-out windows from the sweep's selection range (offset 40,000):
* the held-out loss of the final tick (CTM-LM) or the output (Transformer);
* context use: the loss with every input token replaced by a random token
  (same targets), minus the real loss. A model that ignores its input scores 0.

Decision rule, fixed before the runs:
* a probe **works** if its held-out loss is at most 7.0 nats (unigram: 7.62)
  and its context use is at least 0.5 nats;
* CTM-LM then uses the smallest change that works: A if it works, otherwise B
  (reported as an adaptation), otherwise neither, and the failure is the result.

    python -m scripts.ctm_lm_probes --gpu 1   # run the missing probes in order, then summarize
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F

RUNS=Path('research/runs/ctm_lm_probes');OUT=Path('research/results/ctm_lm_probes')
PROBES={'faithful':('ctm_heavy',448,[],5e-4),'faithful_low_lr':('ctm_heavy',448,[],1.25e-4),
        'A_unit_query':('ctm_heavy',448,['unit_query'],5e-4),'B_observe_token':('ctm_heavy',448,['observe_token'],5e-4),
        'transformer_reference':('transformer',384,None,5e-4),
        # RDT probes (added 2026-10-02): the compute-aware RDT at d 384 also plateaued at the unigram level in the sweep.
        'rdt_aware_baseline':('rdt_aware',384,None,5e-4),'rdt_aware_low_lr':('rdt_aware',384,None,1.25e-4),
        'rdt_aware_unit_embedding':('rdt_aware',384,None,5e-4,{'embedding_init_std':1.0})}
STEPS,MICRO,ACCUMULATION,WINDOWS,OFFSET=300,8,4,64,40_000
WORKS_LOSS,WORKS_CONTEXT=7.0,0.5


def probe_config(name):
    from scripts.lr_sweep import sweep_config
    arm,d,adaptations,lr,*extra=PROBES[name];run=sweep_config(arm,d,0,1);run.update(*extra)
    run['name']=name;run['run_directory']=str(RUNS/name)
    if adaptations is not None:run['ctm_adaptations']=adaptations
    run['train'].update(lr=lr,total_steps=STEPS,micro_batch=MICRO,accumulation=ACCUMULATION,warmup_steps=30,eval_interval=100,eval_windows=64,
                        final_eval_windows=256,checkpoint_interval=10**9,delete_checkpoint_on_complete=False)
    return run


def context_use(name,device):
    """Held-out loss with real inputs and with random inputs (same targets), from the final checkpoint."""
    from scripts.pretrain import build_config
    from ctm_transformer.ctm_lm_adapt import adapted_factory
    from ctm_transformer.lm_scale import scaled_factory
    from ctm_transformer.pretrain import TokenWindows
    run=json.loads((RUNS/name/'config.json').read_text());data=json.loads(Path(run['data']).read_text());root=Path(run['data']).parent
    config=build_config(run,data['vocab_size'])
    model=(adapted_factory(set(run['ctm_adaptations'])) if 'ctm_adaptations' in run else scaled_factory(run['family']))(config).to(device)
    kwargs={'max_thought_steps':run['eval_depth']} if run['family']=='rdt' else {}
    model.load_state_dict(torch.load(RUNS/name/'latest.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    valid=TokenWindows([root/s for s in data['validation']],1024,0);g=torch.Generator().manual_seed(0);real=shuffled=0.0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(OFFSET,OFFSET+WINDOWS):
            x,y=valid.batch([i]);x,y=x.to(device),y.to(device);xr=torch.randint(0,data['vocab_size'],x.shape,generator=g).to(device)
            for inputs,acc in ((x,'real'),(xr,'random')):
                loss=float(F.cross_entropy(model(inputs,**kwargs)['logits'].float().reshape(-1,data['vocab_size']),y.reshape(-1)))
                if acc=='real':real+=loss/WINDOWS
                else:shuffled+=loss/WINDOWS
    return real,shuffled


def summarize(device):
    rows=[]
    for name in PROBES:
        if not (RUNS/name/'complete.json').exists():continue
        real,rand=context_use(name,device);metrics=[json.loads(r) for r in (RUNS/name/'metrics.jsonl').read_text().splitlines() if r.strip()]
        rows.append({'probe':name,'held_out_loss':real,'random_input_loss':rand,'context_use':rand-real,
                     'final_validation':metrics[-1].get('validation'),'works':real<=WORKS_LOSS and rand-real>=WORKS_CONTEXT})
    ok={r['probe'] for r in rows if r['works']}
    ok-={r for r in ok if not r.startswith(('faithful','A_','B_'))}
    choice='A_unit_query' if 'A_unit_query' in ok else 'B_observe_token' if 'B_observe_token' in ok else None
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'probes.json').write_text(json.dumps({'probes':rows,'choice':choice,'rule':f'works: loss <= {WORKS_LOSS} and context use >= {WORKS_CONTEXT}; A before B'},indent=2)+'\n')
    lines=['# CTM-LM input-use probes — results','','Protocol: `scripts/ctm_lm_probes.py` (docstring). 300 steps × 32 sequences × 1,024 tokens (9.8M tokens), one GPU.',
           f'Held-out loss on {WINDOWS} windows from offset {OFFSET}; unigram model: 7.62 nats.','',
           '| Probe | Held-out loss | Loss with random inputs | Context use | Works |','|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['probe']} | {r['held_out_loss']:.3f} | {r['random_input_loss']:.3f} | {r['context_use']:+.3f} | {'yes' if r['works'] else 'no'} |")
    lines+=['',f'**CTM-LM choice by the rule:** {choice or "neither fix works"}.']
    (OUT/'PROBES.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


def main():
    from scripts.lr_sweep import gpu_free
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--summarize-only',action='store_true');p.add_argument('--only',nargs='*',help='run only these probes');a=p.parse_args()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    if not a.summarize_only:
        for name in (a.only or PROBES):
            out=RUNS/name
            if (out/'complete.json').exists() or (out/'diverged.json').exists():continue
            while not gpu_free(a.gpu):time.sleep(30)  # e.g. a sweep run still finishing on the reserved GPU
            out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(probe_config(name),indent=2)+'\n')
            print(f'probe {name}',flush=True)
            with open(out/'train.log','a') as f:
                code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(out/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
            print(f'probe {name} exit {code}',flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);summarize(torch.device('cuda:0'))

if __name__=='__main__':main()
