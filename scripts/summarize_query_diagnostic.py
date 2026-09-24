"""Audit exhaustive paired queries and report accuracy, sensitivity, and source confusion."""
import argparse
import hashlib
import json
from pathlib import Path
from ctm_transformer.query_diagnostic import validate_query_suite,GROUPS,NODES
from ctm_transformer.query_analysis import summarize_grid

FAMILIES=('transformer','recurrent_depth','ctm')
NAMES={'transformer':'Transformer','recurrent_depth':'Recurrent depth','ctm':'CTM'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_root',type=Path,default=Path('research/results/query_diagnostic_v1'))
    p.add_argument('--dataset_root',type=Path,default=Path('research/data/query_diagnostic_v1'))
    p.add_argument('--source_root',type=Path,default=Path('research/data/ordered_pointer_v1'))
    p.add_argument('--previous_results',type=Path,default=Path('research/results/ordered_long_v1'))
    args=p.parse_args()
    def read(path):return json.loads(path.read_text())
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    data=validate_query_suite(args.dataset_root,args.source_root)
    reports={};audits={}
    for family in FAMILIES:
        evaluation=read(args.results_root/f'{family}_seed17.eval.json')
        previous_path=args.previous_results/f'ordered_{family}_seed17.eval.json'
        previous=read(previous_path)
        audit=read(args.results_root/f'{family}_seed17.audit.json')
        assert evaluation['complete'] and audit['complete']
        assert evaluation['model_family']==family
        assert digest(previous_path)==audit['previous_evaluation_sha256']
        assert digest(args.dataset_root/'manifest.json')==evaluation['dataset_manifest_sha256']==audit['dataset_manifest_sha256']
        assert digest(evaluation['checkpoint']['path'])==evaluation['checkpoint']['sha256']==previous['checkpoint']['sha256']==audit['checkpoint_sha256']
        assert evaluation['depths']==previous['depths']
        reports[family]={};audits[family]={'checkpoint':evaluation['checkpoint'],'wall_seconds':evaluation['wall_seconds'],'original_query_agreement':{}}
        def predictions(record,split):return next(iter(record['results'][split]['depths'].values()))['predictions']
        for group in GROUPS:
            rows=[]
            for node in NODES:
                split=f'{group}_start_{node}'
                preds=predictions(evaluation,split)
                original=data[split].records
                assert len(preds)==len(original)
                for r,pred in zip(original,preds):
                    assert r['id']==pred['id'] and r['answer']==pred['answer']
                    rows.append({**r,**pred})
            prior={r['id']:r for r in predictions(previous,group)}
            originals=[r for r in rows if r['start']==r['original_start']]
            assert {r['id'] for r in originals}==set(prior)
            mismatches=[r['id'] for r in originals if any(r[k]!=prior[r['id']][k] for k in ('prediction','answer','correct','terminated'))]
            audits[family]['original_query_agreement'][group]={'examples':len(originals),'mismatches':mismatches}
            assert not mismatches, f'Investigate original-query reproduction: {family}/{group}'
            reports[family][group]=summarize_grid(rows)
    source_paths=('ctm_transformer/query_analysis.py','scripts/summarize_query_diagnostic.py')
    output={'purpose':'descriptive paired all-query diagnostic; correlated queries within maps; validation influenced checkpoint selection',
            'complete':True,'runs':reports,'audits':audits,'analysis_source_sha256':{p:digest(p) for p in source_paths}}
    (args.results_root/'query_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# All-start query-selection results','',
           'Frozen checkpoints from the 3,000-update experiment. Each model answers every start A–H on the same 256 training-probe maps and 128 validation maps. Original queries reproduce the prior outputs exactly. No training or checkpoint reselection occurred.','',
           '| Model | Training maps, all starts | Validation maps, all starts | Training maps: all 8 correct | Validation maps: all 8 correct |',
           '|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        tr,va=reports[f]['train_probe'],reports[f]['validation']
        lines.append(f"| {NAMES[f]} | {tr['overall']['exact_match']:.1%} | {va['overall']['exact_match']:.1%} | {tr['all_queries_correct_maps']}/256 | {va['all_queries_correct_maps']}/128 |")
    lines.extend(['','All-start columns contain 2,048 training-map and 1,024 validation-map queries per model. Queries on the same map are correlated; the all-eight metric uses maps as the unit. Validation maps were used for checkpoint selection and are not a new independent test set.','',
                  '## Query exposure','',
                  '| Model | Original trained query (256) | Changed query, training map (1,792) | Original validation query (128) | Changed validation query (896) |',
                  '|---|---:|---:|---:|---:|'])
    for f in FAMILIES:
        tr,va=reports[f]['train_probe']['by_exposure'],reports[f]['validation']['by_exposure']
        values=[tr['trained_query'],tr['changed_training_map_query'],va['original_validation_query'],va['changed_validation_query']]
        lines.append('| '+NAMES[f]+' | '+' | '.join(f"{r['exact_match']:.1%}" for r in values)+' |')
    lines.extend(['','## Validation accuracy by requested start','',
                  '| Start | Transformer | Recurrent depth | CTM |','|---|---:|---:|---:|'])
    for n in NODES:
        lines.append('| '+n+' | '+' | '.join(f"{reports[f]['validation']['by_start'][n]['correct']}/128 ({reports[f]['validation']['by_start'][n]['exact_match']:.1%})" for f in FAMILIES)+' |')
    lines.extend(['','## Output sensitivity on validation maps','',
                  '| Model | Identical-output start pairs (of 3,584) | Changed queries returning original answer (of 896) | Invalid outputs (of 1,024) |',
                  '|---|---:|---:|---:|'])
    for f in FAMILIES:
        va=reports[f]['validation']
        lines.append(f"| {NAMES[f]} | {va['same_output_pairs']} ({va['same_output_pair_rate']:.1%}) | {va['changed_queries_output_original_answer']} ({va['original_answer_persistence_rate']:.1%}) | {va['invalid_outputs']} |")
    lines.extend(['','Pairs compare the full generated string and EOS status. Original-answer persistence requires exact generation with EOS. Neither metric measures internal attention.','',
                  '![Per-start accuracy and output-implied source confusion](query_patterns.png)','',
                  'The lower panels invert each graph to identify which source has the predicted value as its successor. Rows are requested starts; columns are output-implied sources. The diagonal is correct. “?” covers invalid labels or missing EOS. This is a behavioral association, not evidence of which edge received attention. Source labels and positions are confounded by canonical ordering.','',
                  'See [protocol and interpretation](../../QUERY_DIAGNOSTIC.md), the immutable [pre-evaluation plan](PLAN_BEFORE_RUNS.md), and `query_summary.json` for per-map counts, exposure breakdowns, original-query reproduction audits, and hashes. These one-seed results use unmatched model sizes/compute.'])
    (args.results_root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    heatmaps=[]
    for col,f in enumerate(FAMILIES):
        for group,label in [('train_probe','Training maps'),('validation','Validation maps')]:
            axes[0,col].plot(list(NODES),[100*reports[f][group]['by_start'][n]['exact_match'] for n in NODES],marker='o',label=label)
        axes[0,col].set(title=NAMES[f],xlabel='Requested start',ylabel='Exact match (%)',ylim=(0,105))
        axes[0,col].grid(alpha=.2);axes[0,col].legend(fontsize=8)
        matrix=np.array([[reports[f]['validation']['output_implied_source_counts'][n][k]/128 for k in NODES+'?'] for n in NODES])
        heatmaps.append(axes[1,col].imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto'))
        axes[1,col].set(xticks=range(9),xticklabels=list(NODES+'?'),yticks=range(8),yticklabels=list(NODES),
                        xlabel='Source whose successor matches output',ylabel='Requested start',title='Validation output-implied source')
        for i in range(8):
            for j in range(9):
                if matrix[i,j]>=.05:axes[1,col].text(j,i,f'{matrix[i,j]:.0%}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.55 else 'black')
    fig.colorbar(heatmaps[-1],ax=list(axes[1,:]),label='Fraction of queries in row',shrink=.85)
    fig.suptitle('All-start query diagnostic; frozen seed-17 checkpoints; validation used in selection')
    fig.savefig(args.results_root/'query_patterns.png',dpi=180)
    fig.savefig(args.results_root/'query_patterns.pdf');plt.close(fig)
    print('\n'.join(lines[:12]))

if __name__=='__main__':main()
