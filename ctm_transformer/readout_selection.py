"""Label-free readout policies and explicit validation selection metrics."""
import torch
from torch.nn import functional as F
from ctm_transformer.confidence_readout import ConfidenceReadout
from ctm_transformer.research import model_family


@torch.inference_mode()
def evaluate_readouts(model, dataset, config, device, policies=('final','confidence')):
    from ctm_transformer.experiment import get_batch
    from ctm_transformer.algorithmic_eval import greedy_answers, summarize_predictions
    if not policies or any(p not in ('final','confidence') for p in policies):
        raise ValueError('Unknown or empty readout policy')
    was_training=model.training;model.eval()
    totals={p:0.0 for p in policies};correct={p:0 for p in policies}
    count=0;raw_ce=None;raw_correct=None;oracle_ce=None;oracle_correct=None;hist=None
    try:
        for offset in range(0,len(dataset),config.batch_size):
            inputs,targets,_=get_batch(dataset,slice(offset,offset+config.batch_size),device)
            kw={'return_all_logits':True} if model_family(config)!='ctm' else {}
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.dtype=='bfloat16'):
                output=model(inputs,max_thought_steps=config.max_thought_steps,**kw)
            ticks=torch.stack(output['all_logits'],dim=2)
            mask=targets.ne(-100);gold=targets[mask];logits=ticks[mask].float()
            logp=logits.log_softmax(-1)
            ce=-logp.gather(-1,gold[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
            entropy=-(logp.exp()*logp).sum(-1)
            chosen=entropy.argmin(-1);indices=torch.arange(len(gold),device=gold.device)
            accuracy=logits.argmax(-1).eq(gold[:,None])
            values={'final':(ce[:,-1],accuracy[:,-1]),'confidence':(ce[indices,chosen],accuracy[indices,chosen])}
            for policy in policies:
                totals[policy]+=values[policy][0].sum().item();correct[policy]+=values[policy][1].sum().item()
            batch_hist=torch.bincount(chosen,minlength=logits.shape[1])
            additions=(ce.sum(0),accuracy.sum(0),ce.cummin(1).values.sum(0),accuracy.cumsum(1).gt(0).sum(0),batch_hist)
            if raw_ce is None:raw_ce,raw_correct,oracle_ce,oracle_correct,hist=additions
            else:
                raw_ce+=additions[0];raw_correct+=additions[1];oracle_ce+=additions[2];oracle_correct+=additions[3];hist+=additions[4]
            count+=len(gold)
        result={}
        for policy in policies:
            reader=model if policy=='final' else ConfidenceReadout(model)
            predictions=[]
            for offset in range(0,len(dataset),config.batch_size):
                predictions.extend(greedy_answers(reader,dataset.records[offset:offset+config.batch_size],config,device,config.max_thought_steps))
            result[policy]={'loss':totals[policy]/count,'target_tokens':count,'thought_steps':config.max_thought_steps,
                'token_accuracy':correct[policy]/count,'generation':summarize_predictions(predictions),'predictions':predictions}
        result['diagnostics']={'raw_tick_ce':(raw_ce/count).tolist(),'raw_tick_token_accuracy':(raw_correct/count).tolist(),
            'gold_aware_min_ce_so_far':(oracle_ce/count).tolist(),'gold_aware_any_correct_so_far':(oracle_correct/count).tolist(),
            'confidence_tick_histogram':hist.tolist(),
            'note':'Supervised answer/EOS token diagnostics; gold-aware envelopes are not deployment scores and do not select checkpoints.'}
        return result
    finally:
        model.train(was_training)
