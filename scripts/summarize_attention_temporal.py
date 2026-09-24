"""Summarize measured answer-position attention and CTM temporal diagnostics."""
import json
from pathlib import Path
import numpy as np
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/attention_temporal_v1')
FAMILIES=('transformer','recurrent_depth','ctm')
NAMES={'transformer':'Transformer','recurrent_depth':'Recurrent depth','ctm':'CTM'}


def main():
    reports={};arrays={};summary={}
    for f in FAMILIES:
        report=json.loads((ROOT/f'{f}.json').read_text());reports[f]=report
        assert report['complete'] and report['max_instrumented_logit_difference']==0
        assert file_hash(ROOT/f'{f}.npz')==report['arrays_sha256']
        archive=np.load(ROOT/f'{f}.npz');a=archive['attention'];rows=report['examples']
        arrays[f]=archive
        assert a.shape[0]==3072==len(rows) and a.shape[1]==len(report['events'])
        values=np.stack([a[i][:,:,r['value_positions']] for i,r in enumerate(rows)])
        starts=np.array(['ABCDEFGH'.index(r['start']) for r in rows])
        correct_value=np.take_along_axis(values,starts[:,None,None,None],axis=3).squeeze(-1)
        query=np.stack([a[i][:,:,r['query_position']] for i,r in enumerate(rows)])
        source=np.stack([a[i][:,:,r['source_positions']] for i,r in enumerate(rows)])
        metrics={'correct_value_mass':correct_value,'query_label_mass':query,
                 'all_values_mass':values.sum(-1),'all_source_labels_mass':source.sum(-1)}
        # Among value positions only; not the overall attention argmax.
        top_value=(values.argmax(-1)==starts[:,None,None]).astype(float)
        metrics['correct_top_value_fraction']=top_value
        summary[f]={'events':report['events'],'groups':{}}
        correct=np.array([r['correct'] for r in rows]);validation=np.array([r['split'].startswith('validation_') for r in rows])
        for group,mask in [('train_probe',~validation),('validation',validation),('validation_correct',validation&correct),('validation_wrong',validation&~correct)]:
            summary[f]['groups'][group]={'examples':int(mask.sum()),'metrics':{k:v[mask].mean(0).tolist() for k,v in metrics.items()}}
        summary[f]['by_start']={}
        for j,node in enumerate('ABCDEFGH'):
            mask=validation&(starts==j)
            summary[f]['by_start'][node]={'examples':int(mask.sum()),'accuracy':float(correct[mask].mean()),
                'correct_value_mass':correct_value[mask].mean(0).tolist(),'query_label_mass':query[mask].mean(0).tolist()}
        if f=='ctm':
            tok=AlgorithmicTokenizer();gold=np.array([tok.encode(r['answer'])[0] for r in rows])
            summary[f]['tick_readout']={}
            for group,mask in [('train_probe',~validation),('validation',validation)]:
                summary[f]['tick_readout'][group]={'answer_token_accuracy':(archive['tick_prediction_ids'][mask]==gold[mask,None]).mean(0).tolist(),
                    'mean_gold_probability':archive['tick_gold_probability'][mask].mean(0).tolist()}
    temporal=json.loads((ROOT/'ctm_temporal.json').read_text())
    after_fix=json.loads((ROOT/'ctm_temporal_after_fix.json').read_text())
    assert all(m['matches_masked_oracle'] for m in after_fix['loss_measurements'].values())
    assert temporal['tick_output_gradient_norms']==after_fix['tick_output_gradient_norms']
    result={'complete':True,'attention':summary,'temporal':temporal,'temporal_after_fix':after_fix,
            'analysis_source_sha256':{p:file_hash(p) for p in ('scripts/summarize_attention_temporal.py',)}}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Attention and CTM temporal audit results','',
        'All 9,216 instrumented query forwards preserved full native logits exactly and reproduced the earlier generated answer token. No optimizer update or checkpoint reselection occurred.','',
        '## Validation attention by event (averaged over heads)','',
        '| Model / event | Correct-value mass, correct answers | Correct-value mass, wrong answers | Query-label mass |',
        '|---|---:|---:|---:|']
    for f in FAMILIES:
        s=summary[f]
        for e,event in enumerate(s['events']):
            right=np.mean(s['groups']['validation_correct']['metrics']['correct_value_mass'][e])
            wrong=np.mean(s['groups']['validation_wrong']['metrics']['correct_value_mass'][e])
            query=np.mean(s['groups']['validation']['metrics']['query_label_mass'][e])
            lines.append(f'| {NAMES[f]} / {event} | {right:.3f} | {wrong:.3f} | {query:.3f} |')
    lines.extend(['','Attention to the literal answer value is a narrow diagnostic. Source-label tokens and hidden representations can carry useful information. Correct/wrong differences are associations, not interventions. Per-head measurements and other token-category masses are retained in `summary.json`.','',
        '## CTM readout across trained ticks','',
        '| Tick | Training-map answer-token accuracy | Validation answer-token accuracy | Validation mean correct-answer probability |',
        '|---|---:|---:|---:|'])
    tick=summary['ctm']['tick_readout']
    for t in range(4):
        lines.append(f"| {t+1} | {tick['train_probe']['answer_token_accuracy'][t]:.1%} | {tick['validation']['answer_token_accuracy'][t]:.1%} | {tick['validation']['mean_gold_probability'][t]:.3f} |")
    lines.extend(['','Intermediate tick scores are answer-token readouts, without EOS evaluation. Only tick 4 was directly supervised; all ticks participate in backpropagation.','',
        '## No-update temporal gradient and loss audit','',
        'FP32 gradients of final CE to tick output states have norms: '+', '.join(f'{v:.6f}' for v in temporal['tick_output_gradient_norms'])+'. This confirms connectivity on the selected 32-example training batch; it does not establish optimal gradient balance.','',
        '| Objective | Before fix | Masked oracle | Ratio |', '|---|---:|---:|---:|'])
    for mode,m in temporal['loss_measurements'].items():
        lines.append(f"| {mode} | {m['implementation_loss']:.6f} | {m['masked_oracle_loss']:.6f} | {m['implementation_over_masked_oracle']:.6f} |")
    lines.extend(['','The inactive `dynamic_aggregate` path divides by all token positions, including ignored prompts. Here it scales the intended answer/EOS loss by 2/54 = 1/27. Final CE and ramp loss with zero monotonicity weight agree with the masked oracle. This issue does not affect the existing final-CE pilot results. The inactive dynamic-loss path is now corrected: its loss is 0.841108, exactly matching the masked oracle. Five model-level loss/gradient tests pass, including a wholly ignored example and rejection of an entirely unsupervised batch. The final-CE loss and audited tick gradients are unchanged. Before/after source versions and measurements are archived separately.','',
        '![Attention and CTM ticks](attention_ticks.png)','',
        'See [protocol, interpretation, and proposed ablations](../../ATTENTION_TEMPORAL.md) and the immutable [pre-measurement plan](PLAN_BEFORE_RUNS.md). Raw attention arrays and per-query metadata are retained.'])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    s=summary['ctm']
    for group,label in [('validation_correct','Correct answers'),('validation_wrong','Wrong answers')]:
        y=np.array(s['groups'][group]['metrics']['correct_value_mass']).mean(1)
        axes[0].plot(range(1,len(y)+1),y,marker='o',label=label)
    axes[0].set(title='CTM: attention on correct value',xlabel='Layer calls: tick 1 L1/L2, tick 2 L1/L2, …',ylabel='Attention mass (head average)',xticks=range(1,9))
    axes[0].legend();axes[0].grid(alpha=.2)
    for group,label in [('train_probe','Training maps'),('validation','Validation maps')]:
        axes[1].plot(range(1,5),100*np.array(tick[group]['answer_token_accuracy']),marker='o',label=label)
    axes[1].set(title='CTM shared-head answer readout',xlabel='Thought tick',ylabel='Answer-token accuracy (%)',xticks=range(1,5),ylim=(0,100))
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.suptitle('Frozen checkpoints; descriptive measurements; intermediate ticks not directly supervised')
    fig.savefig(ROOT/'attention_ticks.png',dpi=180);fig.savefig(ROOT/'attention_ticks.pdf');plt.close(fig)
    print('\n'.join(lines[-17:]))

if __name__=='__main__':main()
