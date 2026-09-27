"""Summarize the Sync-RDT mechanism ablations against the frozen S3 controls."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctm_transformer.experiment import file_hash
from scripts.summarize_group_s3 import correct_prefix,exceeds

ROOT=Path('research/results/sync_ablation_v1')


def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.1f}%'


def classify(sync_vs,vs_rdt,vs_wide):
    """removes: sync exceeds the ablation; preserves: the ablation exceeds both RDT controls; otherwise partial."""
    if sync_vs['first_exceeds']:return 'removes'
    if vs_rdt['first_exceeds'] and vs_wide['first_exceeds']:return 'preserves'
    return 'partial'


def main():
    registry=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');models={t['cell']:t for t in frozen['models']}
    assert file_hash(Path(registry['control_study'])/'checkpoints.json')==registry['control_checkpoints_sha256']
    ev={c:read(ROOT/f'{c}.evaluation.json') for c in models}
    for e in ev.values():assert e['complete'] and e['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    for name,c in registry['controls'].items():
        assert file_hash(c['evaluation'])==c['sha256'];ev[name]=read(c['evaluation'])
    seeds=registry['seeds'];cells=['sync','rdt','rdt_wide']+registry['ablations']
    positions=lambda cell,s,T:ev[f'{cell}_seed{s}']['results'][T]['by_position']
    prefix={cell:{T:[correct_prefix(positions(cell,s,T)) for s in seeds] for T in ('4','8','16','32')} for cell in cells}
    primary={cell:prefix[cell]['16'] for cell in cells}
    mid={cell:[statistics.mean(positions(cell,s,'16')[8:16]) for s in seeds] for cell in cells}
    outcomes={}
    for kind in registry['ablations']:
        sync_vs=exceeds(primary['sync'],primary[kind]);vs_rdt=exceeds(primary[kind],primary['rdt']);vs_wide=exceeds(primary[kind],primary['rdt_wide'])
        outcomes[kind]={'classification':classify(sync_vs,vs_rdt,vs_wide),'sync_vs_ablation':sync_vs,'ablation_vs_rdt':vs_rdt,
                        'ablation_vs_rdt_wide':vs_wide,'uses_steps':exceeds(prefix[kind]['16'],prefix[kind]['4'])}
    report={'complete':True,'no_test_evaluation':True,'correct_prefix':prefix,'primary_correct_prefix':primary,
        'mean_accuracy_positions_9_16':mid,'outcomes':outcomes,
        'parameters':{k:models[f'{k}_seed{seeds[0]}']['parameters']['total'] for k in registry['ablations']},
        'training_minutes':{k:statistics.mean(models[f'{k}_seed{s}']['training_seconds'] for s in seeds)/60 for k in registry['ablations']},
        'control_checkpoints_sha256':registry['control_checkpoints_sha256'],'checkpoint_freeze_sha256':file_hash(ROOT/'checkpoints.json'),
        'analysis_source_sha256':file_hash(__file__),
        'evaluations':{c:{'path':str(ROOT/f'{c}.evaluation.json'),'sha256':file_hash(ROOT/f'{c}.evaluation.json')} for c in models}}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

    med=statistics.median
    L=['# Sync-RDT mechanism ablations (S3 word problem)','',
       f'{len(models)} ablation runs ({len(registry["ablations"])} ablations × {len(seeds)} seeds {seeds}) on the S3 study data, paired with its frozen `sync`, `rdt` and `rdt_wide` runs. '
       'Correct prefix = leading positions with at least 90% held-out accuracy on length-32 words, at T=16. Development data; no test set.','',
       '| Cell | Correct prefix per seed | Median | Mean accuracy, positions 9–16 | Median prefix at T=4/8/16/32 |','|---|---|---:|---:|---|']
    for cell in cells:
        L.append(f"| {cell} | {', '.join(map(str,primary[cell]))} | {med(primary[cell]):g} | {pct(statistics.mean(mid[cell]))} | "
                 +'/'.join(f"{med(prefix[cell][T]):g}" for T in ('4','8','16','32'))+' |')
    L+=['','## Declared classification','','Removes: `sync` exceeds the ablation. Preserves: the ablation exceeds both `rdt` and `rdt_wide`. Otherwise partial. Rule: larger in ≥ 4 of 5 seeds, median difference ≥ 2 positions.','',
        '| Ablation | sync vs ablation (larger seeds, median) | ablation vs rdt | ablation vs rdt_wide | Uses steps (T16 vs T4) | Classification |','|---|---|---|---|---|---|']
    for kind,o in outcomes.items():
        f=lambda x:f"{x['larger_seeds']}/{x['smaller_seeds']}, {x['median_difference']:+g}"
        L.append(f"| {kind} | {f(o['sync_vs_ablation'])} | {f(o['ablation_vs_rdt'])} | {f(o['ablation_vs_rdt_wide'])} | "
                 f"{'yes' if o['uses_steps']['first_exceeds'] else 'no'} | **{o['classification']}** |")
    for kind,d in registry.get('diverged',{}).items():
        steps=', '.join(f"{seed}: step {v['last_completed_step']}" for seed,v in d['evidence'].items())
        L+=['',f"**{kind}: diverged at every seed** ({d['error']}; last completed updates {steps}). Recorded, not retried; see [amendment 1](AMENDMENT_1.md)."]
        report['diverged']=registry['diverged']
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    L+=['','![Accuracy by position](accuracy_by_position.png)','','See [frozen protocol](PLAN_BEFORE_RUNS.md), `registry.json`, `checkpoints.json` and the per-run evaluation files.']
    (ROOT/'RESULTS.md').write_text('\n'.join(L)+'\n')

    fig,ax=plt.subplots(figsize=(9,5))
    for cell in cells:
        ax.plot(range(1,33),[statistics.mean(positions(cell,s,'16')[i] for s in seeds) for i in range(32)],label=cell,lw=2 if cell in ('sync','rdt') else 1)
    ax.axvline(16.5,color='gray',ls='--',lw=1);ax.axhline(1/6,color='gray',ls=':',lw=1);ax.set_ylim(0,1.02)
    ax.set_xlabel('position (dashed: end of trained lengths)');ax.set_ylabel('accuracy at T=16 (mean over seeds)');ax.legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'accuracy_by_position.{ext}',dpi=180)
    print('\n'.join(L[:20]),flush=True)

if __name__=='__main__':main()
