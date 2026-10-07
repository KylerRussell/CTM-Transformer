"""Training tests of CTM-LM tick use (research/CTM_TICK_DIAGNOSTICS.md, 2026-10-06).

Every test is the CTM-aware arm at d 384 trained on 100M tokens with the learning-rate
sweep's exact settings (381 steps of 262,144 tokens, warmup 100, cosine to 10%, one GPU,
final evaluation on 1,024 held-out windows), so the sweep's run at the best rate,
ctm_aware_d384_h1_k+2 (lr 4e-3, D + sparse), is the control.

| Test | Change from the control |
|---|---|
| E7_improvement | improvement_loss in place of sparse_tick_loss; tick count drawn per step (log-normal Poisson, mean 15, sigma 0.5, at most 32) |
| E8_cross_position | cross_position: keys and values also read every causal position's current state |
| E4c_feature_head | backbone, embedding and final norm copied from the 400M-token CTM-aware model and frozen; a linear head on f_i is trained in place of the CTM (lr 2e-3) |
| E4a_fresh_ctm | the same frozen backbone; a freshly initialized CTM is trained on it (lr 2e-3) |
| S1_shallow | backbone cut from 24 layers to 2, so the ticks must integrate context through their attention reads (lr 2e-3) |
| S2_shallow_cross | S1 with cross_position: the ticks become a recurrent-depth core with CTM dynamics (lr 2e-3) |
| R1_rdt_heavy_depth1 | the RDT-heavy sweep run at d 704 (lr 1e-3) trained at a fixed depth of 1 in place of randomized depth: does recurrence help the RDT here? |
| S2_T1, S2_T4 | S2 trained and evaluated with 1 or 4 ticks (every tick trained): what the ticks are worth, with parameters fixed |
| S3_shallow_cross_random | S2 with the tick count drawn per step as in the RDT (log-normal Poisson, mean 15, sigma 0.5, at most 32), against drift past the answer |

S1 and S2 (added 2026-10-06, after E8) test whether the ticks contribute once the
backbone no longer does all the work. Their learning-rate control is the sweep's
ctm_aware_d384_h1_k+1 (lr 2e-3, held-out 5.591); the compute-aware RDT at the same
width and rate (rdt_aware_d384_h1_k+1) is the architectural reference.

After the runs, the per-tick held-out loss (ticks 1-32, 32 windows) of each CTM model
and of the control is measured with scripts/ctm_tick_diagnostics.ctm_ticks.

Usage: python -m scripts.tick_use_tests --gpu N [--only names] [--summarize]
"""
import argparse,json,os,subprocess,sys
from pathlib import Path

RUNS=Path('research/runs/tick_use');OUT=Path('research/results/tick_use')
CONTROL=Path('research/runs/lr_sweep/ctm_aware_d384_h1_k+2')
BACKBONE=('research/runs/lr_sweep/ctm_aware_d384_h4_k+1/step_0001526.pt',['token_embedding.','backbone.','backbone_norm.'])
FROZEN=['token_embedding.','backbone.','backbone_norm.']
TESTS={
    'E7_improvement':{'ctm_adaptations':['token_start','improvement_loss'],'depth_sampler':{'kind':'lognormal_poisson','mean':15,'sigma':0.5,'maximum':32},
                      'micro_batch':4},
    'E8_cross_position':{'ctm_adaptations':['token_start','sparse_tick_loss','cross_position']},
    'E4c_feature_head':{'ctm_adaptations':['feature_head'],'lr':2e-3,'init_from':True},
    'E4a_fresh_ctm':{'ctm_adaptations':['token_start','sparse_tick_loss'],'lr':2e-3,'init_from':True},
    'S1_shallow':{'ctm_adaptations':['token_start','sparse_tick_loss'],'lr':2e-3,'model':{'n_layers':2}},
    'S2_shallow_cross':{'ctm_adaptations':['token_start','sparse_tick_loss','cross_position'],'lr':2e-3,'model':{'n_layers':2}},
    'S3_shallow_cross_random':{'ctm_adaptations':['token_start','sparse_tick_loss','cross_position'],'lr':2e-3,'model':{'n_layers':2},
                               'depth_sampler':{'kind':'lognormal_poisson','mean':15,'sigma':0.5,'maximum':32}},
    'R1_rdt_heavy_depth1':{'base':'research/runs/lr_sweep/rdt_heavy_d704_h1_k+0','rdt_depth':1},
    'S2_T1':{'ctm_adaptations':['token_start','sparse_tick_loss','cross_position'],'lr':2e-3,'model':{'n_layers':2,'max_thought_steps':1}},
    'S2_T4':{'ctm_adaptations':['token_start','sparse_tick_loss','cross_position'],'lr':2e-3,'model':{'n_layers':2,'max_thought_steps':4}},
}


def config(name):
    spec=TESTS[name]
    if 'rdt_depth' in spec:  # an RDT sweep run retrained at one fixed depth
        run=json.loads((Path(spec['base'])/'config.json').read_text());run.pop('depth_sampler')
        run.update(name=name,run_directory=str(RUNS/name),eval_depth=spec['rdt_depth'],eval_depths=[spec['rdt_depth']])
        run['train']['delete_checkpoint_on_complete']=False
        return run
    run=json.loads((CONTROL/'config.json').read_text())
    run.update(name=name,run_directory=str(RUNS/name),ctm_adaptations=spec['ctm_adaptations'])
    t=run['train'];t['delete_checkpoint_on_complete']=False
    if 'lr' in spec:t['lr']=spec['lr']
    if 'micro_batch' in spec:t['accumulation']=t['accumulation']*t['micro_batch']//spec['micro_batch'];t['micro_batch']=spec['micro_batch']
    if 'depth_sampler' in spec:run['depth_sampler']=spec['depth_sampler']
    if 'model' in spec:run['model'].update(spec['model'])
    run['eval_depth']=run['model']['max_thought_steps']  # the trainer passes eval_depth as the tick count of every step
    if spec.get('init_from'):run['init_from']={'checkpoint':BACKBONE[0],'prefixes':BACKBONE[1]};run['freeze_prefixes']=FROZEN
    return run


def final_validation(directory):
    rows=[json.loads(r) for r in (Path(directory)/'metrics.jsonl').read_text().splitlines() if r.strip()]
    return [r for r in rows if 'validation' in r][-1]['validation']


def summarize(device):
    import torch
    from scripts.ctm_tick_diagnostics import EVAL_OFFSET,ctm_ticks,load
    from ctm_transformer.pretrain import TokenWindows
    manifest=Path('research/data/pretrain/fineweb_edu_32k/manifest.json');data=json.loads(manifest.read_text())
    valid=TokenWindows([manifest.parent/s for s in data['validation']],1024,0)
    batches=[tuple(t.to(device) for t in valid.batch([i])) for i in range(EVAL_OFFSET,EVAL_OFFSET+32)]
    models={'control (D + sparse, lr 4e-3)':(CONTROL,'step_0000381.pt'),
            'E4 backbone source (400M tokens)':(Path(BACKBONE[0]).parent,Path(BACKBONE[0]).name)}
    models.update({n:(RUNS/n,'step_0000381.pt') for n in TESTS})
    rows={}
    for label,(directory,checkpoint) in models.items():
        if not (directory/checkpoint).exists():continue
        model,run=load(directory,checkpoint,device);entry={'final_validation':final_validation(directory)}
        if run['family']=='ctm_lm' and 'feature_head' not in run['ctm_adaptations']:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                ce=torch.cat([ctm_ticks(model,x,y,32)[0] for x,y in batches])
            entry['per_tick']=[float(v) for v in ce.mean(0)]
        rows[label]=entry;del model;torch.cuda.empty_cache()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'tests.json').write_text(json.dumps(rows,indent=1)+'\n')
    show=[1,2,3,4,8,12,16,24,32]
    metric=lambda e:e['final_validation'].get('final_tick',next(iter(e['final_validation'].values())))
    L=['# CTM-LM tick-use training tests','','Protocol: [CTM_TICK_DIAGNOSTICS.md](../../CTM_TICK_DIAGNOSTICS.md). Held-out loss is the final tick on 1,024 windows; per-tick loss is on 32 windows (ticks beyond 16 extrapolate).','',
       '| Model | Held-out loss | '+' | '.join(f'tick {t}' for t in show)+' |','|---|---:|'+'---:|'*len(show)]
    for label,e in rows.items():
        L.append(f'| {label} | {metric(e):.3f} | '+' | '.join(f'{e["per_tick"][t-1]:.3f}' for t in show) +' |' if 'per_tick' in e else
                 f'| {label} | {metric(e):.3f} | '+' | '.join('' for _ in show)+' |')
    (OUT/'TESTS.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);p.add_argument('--only',nargs='*');p.add_argument('--summarize',action='store_true')
    a=p.parse_args()
    if a.summarize:
        import torch
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);summarize(torch.device('cuda:0'));return
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    for name in a.only or list(TESTS):
        out=RUNS/name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        (out/'config.json').write_text(json.dumps(config(name),indent=2)+'\n')
        print('test',name,flush=True)
        with open(out/'train.log','a') as f:
            code=subprocess.run([sys.executable,'-u','-m','scripts.pretrain','--config',str(out/'config.json')],stdout=f,stderr=subprocess.STDOUT,env=env).returncode
        print('test',name,'exit',code,flush=True)


if __name__=='__main__':main()
