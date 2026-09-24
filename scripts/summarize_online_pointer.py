"""Summarize the fresh-map pointer study from frozen evaluations."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/online_pointer_v1')
FAMILIES=('ctm','transformer','recurrent_depth');CONDITIONS=('repeated','fresh','multihop')
HOPS=range(1,9);GATE_MEAN,GATE_EACH=0.90,0.80


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.2f}%'
def msd(xs):return (statistics.mean(xs),statistics.stdev(xs) if len(xs)>1 else 0.0)
def fmt(xs):m,s=msd(xs);return f'{100*m:.2f}% ± {100*s:.2f}'


def accuracy(evaluation,hop,depth=None):
    depth=str(depth or evaluation['primary_thought_steps']);split='eval_id' if hop<=3 else 'eval_depth'
    return evaluation['results'][split][depth]['metrics']['generation']['by_difficulty'][f'nodes=12,steps={hop}']['exact_match']


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    ev={}
    for t in frozen['models']:
        e=read(ROOT/f"{t['cell']}.evaluation.json")
        assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
        ev[t['cell']]=e
    cell=lambda f,c,s:f'{f}_{c}_seed{s}';seeds=registry['seeds'];chance=registry['chance_accuracy']
    hop1={f:{c:[accuracy(ev[cell(f,c,s)],1) for s in seeds] for c in CONDITIONS} for f in FAMILIES}
    gate={f:{'mean':statistics.mean(hop1[f]['fresh']),'min':min(hop1[f]['fresh']),
             'passed':statistics.mean(hop1[f]['fresh'])>=GATE_MEAN and min(hop1[f]['fresh'])>=GATE_EACH} for f in FAMILIES}
    paired={f:[a-b for a,b in zip(hop1[f]['fresh'],hop1[f]['repeated'])] for f in FAMILIES}
    byhop={f:{c:{h:[accuracy(ev[cell(f,c,s)],h) for s in seeds] for h in HOPS} for c in CONDITIONS} for f in FAMILIES}
    sweep={f:{c:{T:{h:[accuracy(ev[cell(f,c,s)],h,T) for s in seeds] for h in HOPS} for T in models[cell(f,c,seeds[0])]['evaluation_thought_steps']}
              for c in CONDITIONS} for f in FAMILIES if f!='transformer'}
    report={'complete':True,'no_test_evaluation':True,'chance_accuracy':chance,'gate_rule':{'mean_at_least':GATE_MEAN,'every_seed_at_least':GATE_EACH},
        'hop1_accuracy':hop1,'fresh_minus_repeated_hop1':paired,'retrieval_gate':gate,'accuracy_by_hop':byhop,'tick_sweep':sweep,
        'training':{c:{k:models[c][k] for k in ('unique_training_maps','mean_presentations_per_map','final_training_loss_last100','training_seconds','peak_allocated_bytes')} for c in models},
        'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in ev}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    L=['# Fresh-map pointer study','',
       f'All {len(ev)} declared runs completed and were evaluated from checkpoints frozen at exactly 3,000 updates. '
       f'Maps are single 12-cycles, so chance is 1/11 = {pct(chance)}. Development splits only; no test set exists for this study. '
       'Mean ± sample SD across seeds 23/29/31; three seeds support no significance claim.','',
       '## Primary 1: does fresh data fix one-hop retrieval?','',
       'Hop-1 exact accuracy on 256 held-out maps (`eval_id`), declared readout at the trained depth.','',
       '| Family | Repeated (2,048 maps, ~47×) | Fresh (96,000 maps, 1×) | Fresh − repeated, paired | Multi-hop-trained |','|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        m,s=msd(paired[f])
        L.append(f"| {f} | {fmt(hop1[f]['repeated'])} | {fmt(hop1[f]['fresh'])} | {100*m:+.2f} ± {100*s:.2f} points | {fmt(hop1[f]['multihop'])} |")
    L+=['',f'## Primary 2: retrieval gate (fresh one-hop mean ≥ {GATE_MEAN:.0%} and every seed ≥ {GATE_EACH:.0%})','','| Family | Mean | Lowest seed | Gate |','|---|---:|---:|---|']
    for f in FAMILIES:L.append(f"| {f} | {pct(gate[f]['mean'])} | {pct(gate[f]['min'])} | {'passed' if gate[f]['passed'] else 'not passed'} |")
    L+=['','## Primary 3: multi-hop training, accuracy by hop','','Trained on hops 1–3; hops 4–8 are unseen depths (`eval_depth`, 256 maps per hop). Trained thought depth.','',
        '| Family | '+' | '.join(f'hop {h}' for h in HOPS)+' |','|---|'+'---:|'*len(HOPS)]
    for f in FAMILIES:L.append(f'| {f} | '+' | '.join(fmt(byhop[f]['multihop'][h]) for h in HOPS)+' |')
    L+=['','## Secondary: inference tick budget (CTM and recurrent depth)','','Mean accuracy; T16 is the trained depth. Other budgets change only the number of thought ticks at inference.','']
    for f in sweep:
        for c in ('fresh','multihop'):
            L+=[f'**{f}, {c} training**','','| T | '+' | '.join(f'hop {h}' for h in HOPS)+' |','|---:|'+'---:|'*len(HOPS)]
            for T,row in sweep[f][c].items():L.append(f'| {T} | '+' | '.join(pct(statistics.mean(row[h])) for h in HOPS)+' |')
            L.append('')
    L+=['## Training','','| Cell | Unique maps | Final train CE (last 100) | Train min | Peak GiB |','|---|---:|---:|---:|---:|']
    for c,t in models.items():
        L.append(f"| {c} | {t['unique_training_maps']:,} | {t['final_training_loss_last100']:.4f} | {t['training_seconds']/60:.1f} | {t['peak_allocated_bytes']/2**30:.3f} |")
    L+=['','![Hop-1 accuracy by training regime](hop1_accuracy.png)','','![Accuracy by hop](accuracy_by_hop.png)','',
        'See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-cell evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')

    fig,ax=plt.subplots(figsize=(7,4));width=0.26
    for i,c in enumerate(CONDITIONS):
        xs=[j+(i-1)*width for j in range(len(FAMILIES))]
        ax.bar(xs,[statistics.mean(hop1[f][c]) for f in FAMILIES],width,label=c,alpha=.8)
        for x,f in zip(xs,FAMILIES):ax.scatter([x]*len(seeds),hop1[f][c],color='k',s=10,zorder=3)
    ax.axhline(chance,ls='--',color='gray',lw=1,label='chance');ax.set_xticks(range(len(FAMILIES)),FAMILIES);ax.set_ylim(0,1.02)
    ax.set_ylabel('hop-1 exact accuracy');ax.legend(fontsize=8);ax.set_title('One-hop retrieval by training regime');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'hop1_accuracy.{ext}',dpi=180)
    fig,axes=plt.subplots(1,len(FAMILIES),figsize=(12,3.6),sharey=True)
    for ax,f in zip(axes,FAMILIES):
        curves=sweep[f]['multihop'] if f in sweep else {1:byhop[f]['multihop']}
        for T,row in curves.items():ax.plot(list(HOPS),[statistics.mean(row[h]) for h in HOPS],marker='o',ms=3,label=f'T={T}')
        ax.axvspan(3.5,8.5,color='gray',alpha=.1);ax.axhline(chance,ls='--',color='gray',lw=1)
        ax.set_title(f'{f} (multi-hop trained)');ax.set_xlabel('hops (shaded: unseen)');ax.legend(fontsize=7)
    axes[0].set_ylabel('exact accuracy');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_hop.{ext}',dpi=180)
    print('\n'.join(L[:24]),flush=True)

if __name__=='__main__':main()
