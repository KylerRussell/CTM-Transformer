"""Validation-only tick trajectories at frozen CTM recipe checkpoints."""
import gc,json
from pathlib import Path
import torch
from ctm_transformer.algorithmic import load_algorithmic_split,AlgorithmicTokenizer
from ctm_transformer.experiment import get_batch,file_hash
from scripts.eval_harness import _load_checkpoint
from scripts.diagnose_tick_selection import trajectories

ROOT=Path('research/results/readout_comparison_v1')


@torch.inference_mode()
def main():
    output=ROOT/'tick_dynamics.json'
    if output.exists():raise ValueError('Fresh dynamics output required')
    selection=json.loads((ROOT/'selection.json').read_text());torch.set_num_threads(4);torch.cuda.set_device('cuda:0')
    results={}
    for trial in selection['trials']:
        if trial['model_family']!='ctm':continue
        assert file_hash(trial['checkpoint'])==trial['checkpoint_sha256']
        model,config=_load_checkpoint(trial['checkpoint']);model.cuda().eval()
        data=load_algorithmic_split('research/data/ordered_pointer_v1','pointer','validation',config.seq_len)
        logits=[];targets=[]
        for offset in range(0,len(data),config.batch_size):
            x,y,_=get_batch(data,slice(offset,offset+config.batch_size),'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model(x,max_thought_steps=config.max_thought_steps)
            mask=y.ne(-100);logits.append(torch.stack(result['all_logits'],2)[mask].float().cpu());targets.append(y[mask].cpu())
        logits=torch.cat(logits);targets=torch.cat(targets);answers=targets.ne(AlgorithmicTokenizer.eot_token)
        answer=trajectories(logits[answers],targets[answers]);both=trajectories(logits,targets)
        violation={}
        for key,direction in [('raw_tick_ce',1),('confidence_ce',1),('raw_tick_accuracy',-1),('confidence_accuracy',-1)]:
            values=[r[key] for r in answer['curve']];changes=[direction*(b-a) for a,b in zip(values,values[1:])]
            violation[key]={'regressing_transitions':sum(v>1e-7 for v in changes),'transitions':len(changes),'largest_regression':max(0,max(changes))}
        results[trial['cell']]={'trial':trial,'dataset':data.metadata,'answer_only':answer,'answer_and_eos':both,'answer_curve_regressions':violation}
        del model,result;gc.collect();torch.cuda.empty_cache()
    report={'complete':True,'selection_sha256':file_hash(ROOT/'selection.json'),'split':'validation',
        'scope':'Descriptive per-tick diagnostics at each trial own frozen selected checkpoint. Answer-only is next token from prompt; EOS scoring is teacher-forced. No recipe/checkpoint reselection.',
        'source_sha256':{p:file_hash(p) for p in ('scripts/evaluate_comparison_dynamics.py','scripts/diagnose_tick_selection.py','ctm_transformer/model.py','ctm_transformer/algorithmic.py')},'results':results}
    output.write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    lines=['# Tick dynamics at frozen CTM checkpoints','',
        'Validation-only answer-token diagnostics. Each recipe uses its own selected checkpoint; these are not answer-plus-EOS generation scores. A fixed checkpoint is unrolled once and read at every tick. No additional tuning follows this analysis.','',
        '| Recipe | CE regressions: raw / confidence | Accuracy regressions: raw / confidence | Confidence answer accuracy at T16 |','|---|---:|---:|---:|']
    for col,(cell,r) in enumerate(results.items()):
        rows=r['answer_only']['curve'];x=[v['available_ticks'] for v in rows]
        for key,label,style in [('raw_tick_ce','Individual tick','-'),('confidence_ce','Confidence best so far','--'),('oracle_min_ce','Gold-aware minimum',':')]:axes[0,col].plot(x,[v[key] for v in rows],style,label=label)
        for key,label,style in [('raw_tick_accuracy','Individual tick','-'),('confidence_accuracy','Confidence best so far','--'),('oracle_any_correct_accuracy','Gold-aware any correct',':')]:axes[1,col].plot(x,[100*v[key] for v in rows],style,label=label)
        axes[0,col].set(title=cell,ylabel='Answer-token CE');axes[1,col].set(xlabel='Available thought ticks',ylabel='Answer-token accuracy (%)',ylim=(0,105))
        for row in (0,1):axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=7)
        v=r['answer_curve_regressions']
        lines.append(f"| {cell} | {v['raw_tick_ce']['regressing_transitions']} / {v['confidence_ce']['regressing_transitions']} | {v['raw_tick_accuracy']['regressing_transitions']} / {v['confidence_accuracy']['regressing_transitions']} | {rows[-1]['confidence_accuracy']:.1%} |")
    fig.suptitle('CTM temporal recipes: raw ticks, deployable confidence selection, and gold-aware envelopes')
    fig.savefig(ROOT/'tick_dynamics.png',dpi=180);fig.savefig(ROOT/'tick_dynamics.pdf');plt.close(fig)
    lines.extend(['','Regressions count adjacent transitions among15transitions, using tolerance1e-7. The soft training penalty acts on training batch-mean raw CE; it does not guarantee validation monotonicity. Confidence refers to minimum entropy among available ticks, not known correctness. Gold-aware envelopes improve by construction and cannot be treated as deployment results. Any-correct coverage can also rise simply by offering more diverse candidates among the eight answer characters; it does not by itself establish iterative reasoning.','', '![CTM tick dynamics](tick_dynamics.png)'])
    (ROOT/'TICK_DYNAMICS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:10]))

if __name__=='__main__':main()
