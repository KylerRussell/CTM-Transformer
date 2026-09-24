"""Post-hoc tick diagnostics; gold-aware envelopes are never deployment scores."""
import json
from pathlib import Path
import torch
from ctm_transformer.algorithmic import load_algorithmic_split, AlgorithmicTokenizer
from ctm_transformer.experiment import get_batch, file_hash
from scripts.eval_harness import _load_checkpoint


def trajectories(logits, targets):
    """Summarize [supervised tokens, ticks, vocabulary] logits."""
    logp = logits.float().log_softmax(-1)
    ce = -logp.gather(-1, targets[:, None, None].expand(-1, logits.shape[1], 1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    correct = logits.argmax(-1).eq(targets[:, None])
    rows = []
    for t in range(1, logits.shape[1] + 1):
        # argmin gives the first index for ties.
        oracle = ce[:, :t].argmin(-1, keepdim=True)
        confidence = entropy[:, :t].argmin(-1, keepdim=True)
        rows.append({'available_ticks': t, 'raw_tick_ce': ce[:, t-1].mean().item(),
            'raw_tick_accuracy': correct[:, t-1].float().mean().item(),
            'oracle_min_ce': ce.gather(1, oracle).mean().item(),
            'oracle_min_ce_tick_accuracy': correct.gather(1, oracle).float().mean().item(),
            'oracle_any_correct_accuracy': correct[:, :t].any(-1).float().mean().item(),
            'confidence_ce': ce.gather(1, confidence).mean().item(),
            'confidence_accuracy': correct.gather(1, confidence).float().mean().item(),
            'confidence_tick_histogram': torch.bincount(confidence.flatten(), minlength=logits.shape[1]).tolist()})
    return {'tokens': len(targets), 'curve': rows,
            'per_token_ce': ce.tolist(), 'per_token_correct': correct.tolist(),
            'per_token_entropy': entropy.tolist()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.cuda.set_device('cuda:0')
    root = Path('research/results/objective_depth_v1')
    output = root/'tick_selection_diagnostic.json'
    if output.exists():raise ValueError('Fresh diagnostic output required')
    results = {}
    for cell in ('uniform_t4','dynamic_t4','uniform_t16','dynamic_t16'):
        run = Path('research/runs/temporal_ablation_v1/seed17/uniform') if cell == 'uniform_t4' else Path('research/runs/objective_depth_v1/seed17')/cell
        model, config = _load_checkpoint(run/'best.pt');model.to('cuda:0').eval()
        dataset = load_algorithmic_split('research/data/ordered_pointer_v1','pointer','validation',config.seq_len)
        all_logits=[];all_targets=[]
        for offset in range(0,len(dataset),config.batch_size):
            inputs, targets, _ = get_batch(dataset,slice(offset,offset+config.batch_size),'cuda:0')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction=model(inputs,max_thought_steps=config.max_thought_steps)
            ticks=torch.stack(prediction['all_logits'],dim=2)
            supervised=targets.ne(-100)
            all_logits.append(ticks[supervised].float().cpu());all_targets.append(targets[supervised].cpu())
        logits=torch.cat(all_logits);targets=torch.cat(all_targets)
        answer=targets.ne(AlgorithmicTokenizer.eot_token)
        assert len(targets)==256 and answer.sum().item()==128
        results[cell]={'checkpoint':model._checkpoint_metadata,'dataset':dataset.metadata,
            'answer_and_eos':trajectories(logits,targets),
            'answer_only':trajectories(logits[answer],targets[answer])}
        del model,prediction,ticks;torch.cuda.empty_cache()
    report={'complete':True,'post_hoc':True,
        'selection':'Existing checkpoints selected by final-tick validation CE; no reselection',
        'scoring':'Teacher-forced supervised-token diagnostics. Answer-only uses the original prompt; EOS uses the gold answer. These are not answer-plus-EOS generation scores.',
        'oracle_warning':'Minimum CE and any-correct envelopes use gold labels, are diagnostics only, and improve monotonically by construction. Minimum-CE-selected accuracy need not be monotonic. Confidence selection uses no labels and has no accuracy monotonicity guarantee.',
        'source_sha256':{p:file_hash(p) for p in ('scripts/diagnose_tick_selection.py','ctm_transformer/model.py','ctm_transformer/algorithmic.py','scripts/eval_harness.py')},'results':results}
    output.write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax, (cell,result) in zip(axes.flat,results.items()):
        rows=result['answer_only']['curve'];x=[r['available_ticks'] for r in rows]
        for key,label,style in [('raw_tick_ce','Individual tick','-'),('confidence_ce','Confidence-selected','--'),('oracle_min_ce','Gold-aware best so far',':')]:
            ax.plot(x,[r[key] for r in rows],style,label=label)
        ax.set(title=cell,xlabel='Available thought ticks',ylabel='Answer-token CE');ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Post-hoc readout diagnosis: final-CE-selected checkpoints, original validation')
    fig.savefig(root/'tick_selection_diagnostic.png',dpi=180);fig.savefig(root/'tick_selection_diagnostic.pdf');plt.close(fig)
    for cell,result in results.items():print(cell,result['answer_only']['curve'][-1])

if __name__=='__main__':main()
