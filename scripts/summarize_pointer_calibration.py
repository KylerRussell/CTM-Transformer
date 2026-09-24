"""Summarize the Transformer pointer calibration and apply its declared decision rules."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/pointer_calibration_v1')
HOPS=range(1,9);GATE_MEAN,GATE_EACH,SPEED=0.90,0.80,0.90


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.2f}%'
def key(h):return f'nodes=12,steps={h}'


def accuracy(evaluation,hop):
    split='eval_id' if hop<=4 else 'eval_depth'
    return evaluation['results'][split]['1']['metrics']['generation']['by_difficulty'][key(hop)]['exact_match']


def first_reaching(curve,hop,threshold=SPEED):
    return next((c['step'] for c in curve if c['by_hop'].get(key(hop),0)>=threshold),None)


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    seeds=registry['seeds'];lrs=sorted({e['learning_rate'] for e in registry['entries']});chance=registry['chance_accuracy']
    cell=lambda task,lr,s:next(e['cell'] for e in registry['entries'] if (e['task'],e['learning_rate'],e['seed'])==(task,lr,s))
    onehop={lr:[accuracy(ev[cell('onehop',lr,s)],1) for s in seeds] for lr in lrs}
    # D1: highest mean final hop-1 accuracy; ties go to the lower learning rate.
    selected=max(lrs,key=lambda lr:(round(statistics.mean(onehop[lr]),6),-lr))
    gate={'selected_learning_rate':selected,'mean':statistics.mean(onehop[selected]),'min':min(onehop[selected]),
          'passed':statistics.mean(onehop[selected])>=GATE_MEAN and min(onehop[selected])>=GATE_EACH}
    speed={lr:[first_reaching(models[cell('onehop',lr,s)]['validation_curve'],1) for s in seeds] for lr in lrs}
    multihop={lr:{h:[accuracy(ev[cell('multihop',lr,s)],h) for s in seeds] for h in HOPS} for lr in lrs}
    report={'complete':True,'no_test_evaluation':True,'chance_accuracy':chance,
        'decision_rules':{'D1':'Select the learning rate with the highest mean final hop-1 accuracy on one-hop training (ties: lower LR).',
                          'D2':f'The format is learnable if the selected LR has mean >= {GATE_MEAN:.0%} and every seed >= {GATE_EACH:.0%}; otherwise stop this format.'},
        'onehop_hop1_accuracy':onehop,'retrieval_gate':gate,'first_validation_step_hop1_at_least_90pct':speed,
        'multihop_accuracy_by_hop':multihop,
        'training':{c:{k:t[k] for k in ('final_training_loss_last100','training_seconds','peak_allocated_bytes')} for c,t in models.items()},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in ev}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    L=['# Transformer pointer calibration (Stage 1a)','',
       f'All {len(ev)} declared Transformer runs completed and were evaluated from checkpoints frozen at exactly {registry["primary_checkpoint_step"]:,} updates. '
       f'Every training map was presented once. Chance is 1/11 = {pct(chance)}. Development splits only; no test set exists for this study.','',
       '## One-hop learnability by learning rate','','Hop-1 exact accuracy on 1,024 held-out `eval_id` maps (256 per hop), final checkpoint.','',
       '| Peak LR | '+' | '.join(f'seed {s}' for s in seeds)+' | Mean | First validation update with hop-1 ≥ 90% | Final train CE |','|---:|'+'---:|'*(len(seeds)+3)]
    for lr in lrs:
        ce=statistics.mean(models[cell('onehop',lr,s)]['final_training_loss_last100'] for s in seeds)
        L.append(f'| {lr:g} | '+' | '.join(pct(x) for x in onehop[lr])+f' | {pct(statistics.mean(onehop[lr]))} | '+
                 ', '.join('never' if x is None else f'{x:,}' for x in speed[lr])+f' | {ce:.4f} |')
    L+=['',f"**D1 selected LR: {selected:g}.** **D2: format {'learnable' if gate['passed'] else 'NOT learnable'}** "
        f"(mean {pct(gate['mean'])}, lowest seed {pct(gate['min'])}; gate ≥ {GATE_MEAN:.0%} mean, ≥ {GATE_EACH:.0%} each).",'',
        '## Multi-hop training (hops 1–4), accuracy by hop','','Hops 5–8 are unseen depths. Mean of both seeds (per-seed values in `summary.json`).','',
        '| Peak LR | '+' | '.join(f'hop {h}' for h in HOPS)+' |','|---:|'+'---:|'*len(HOPS)]
    for lr in lrs:L.append(f'| {lr:g} | '+' | '.join(pct(statistics.mean(multihop[lr][h])) for h in HOPS)+' |')
    L+=['','## Training','','| Cell | Final train CE (last 100) | Train min | Peak GiB |','|---|---:|---:|---:|']
    for c,t in models.items():L.append(f"| {c} | {t['final_training_loss_last100']:.4f} | {t['training_seconds']/60:.1f} | {t['peak_allocated_bytes']/2**30:.3f} |")
    L+=['','![One-hop learning curves](onehop_curves.png)','','![Multi-hop accuracy by hop](multihop_by_hop.png)','',
        'See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-cell evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')

    fig,ax=plt.subplots(figsize=(7,4))
    for i,lr in enumerate(lrs):
        for j,s in enumerate(seeds):
            curve=models[cell('onehop',lr,s)]['validation_curve']
            ax.plot([c['step'] for c in curve],[c['by_hop'][key(1)] for c in curve],color=f'C{i}',ls='-' if j==0 else '--',label=f'LR {lr:g}, seed {s}')
    ax.axhline(chance,color='gray',lw=1,ls=':');ax.set_xlabel('update');ax.set_ylabel('validation hop-1 accuracy');ax.set_ylim(0,1.02)
    ax.legend(fontsize=7);ax.set_title('One-hop training, fresh maps');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'onehop_curves.{ext}',dpi=180)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,lr in enumerate(lrs):ax.plot(list(HOPS),[statistics.mean(multihop[lr][h]) for h in HOPS],marker='o',color=f'C{i}',label=f'LR {lr:g}')
    ax.axvspan(4.5,8.5,color='gray',alpha=.1);ax.axhline(chance,color='gray',lw=1,ls=':');ax.set_ylim(0,1.02)
    ax.set_xlabel('hops (shaded: unseen)');ax.set_ylabel('exact accuracy');ax.legend(fontsize=8);ax.set_title('Multi-hop training, final checkpoint');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'multihop_by_hop.{ext}',dpi=180)
    print('\n'.join(L[:16]),flush=True)

if __name__=='__main__':main()
