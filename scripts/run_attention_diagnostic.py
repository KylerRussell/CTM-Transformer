"""Measure answer-position attention and CTM tick readouts without modifying outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from ctm_transformer.attention_diagnostic import AttentionCapture,prompt_positions
from ctm_transformer.query_diagnostic import validate_query_suite
from ctm_transformer.experiment import file_hash
from scripts.eval_harness import _load_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--families',nargs='+',required=True,choices=['ctm','transformer','recurrent_depth'])
    p.add_argument('--device',required=True)
    p.add_argument('--results_root',type=Path,default=Path('research/results/attention_temporal_v1'))
    args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    data=validate_query_suite('research/data/query_diagnostic_v1','research/data/ordered_pointer_v1')
    tokenizer=AlgorithmicTokenizer();args.results_root.mkdir(parents=True,exist_ok=True)
    for family in args.families:
        output=args.results_root/f'{family}.json'
        if output.exists() or output.with_suffix('.npz').exists():raise ValueError('Fresh outputs required')
        prior_path=Path(f'research/results/query_diagnostic_v1/{family}_seed17.eval.json')
        prior=json.loads(prior_path.read_text());checkpoint=prior['checkpoint']
        assert file_hash(checkpoint['path'])==checkpoint['sha256']
        model,config=_load_checkpoint(checkpoint['path']);model.to(args.device).eval()
        report={'complete':False,'family':family,'checkpoint':checkpoint,'previous_evaluation_sha256':file_hash(prior_path),
            'dataset_manifest_sha256':file_hash('research/data/query_diagnostic_v1/manifest.json'),
            'capture':'native CTM probability tensor; FP32 last-row reconstruction for baseline SDPA; native outputs retained',
            'source_sha256':{f:file_hash(f) for f in ('ctm_transformer/attention_diagnostic.py','scripts/run_attention_diagnostic.py','ctm_transformer/model.py','ctm_transformer/baselines.py','ctm_transformer/config.py')},
            'examples':[],'max_instrumented_logit_difference':0.,'max_sdpa_reconstruction_error':0.}
        arrays=[];event_names=None;tick_probs=[];tick_predictions=[]
        with torch.inference_mode(),AttentionCapture(model,family) as capture:
            for split,dataset in data.items():
                old_preds=next(iter(prior['results'][split]['depths'].values()))['predictions']
                for offset in range(0,len(dataset),config.batch_size):
                    rows=dataset.records[offset:offset+config.batch_size]
                    prompts=[[tokenizer.bos_token]+tokenizer.encode(r['prompt']) for r in rows]
                    assert len({len(r) for r in prompts})==1
                    ids=torch.tensor(prompts,device=args.device)
                    capture.begin()
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model(ids)
                    captured=capture.records
                    names=[n for n,_ in captured]
                    if event_names is None:event_names=names
                    assert event_names==names
                    # Disable capture entirely for an independent native forward.
                    # The outer context is resumed after this batch using only unchanged hooks.
                    for h in capture.handles:h.remove()
                    capture.handles=[]
                    if hasattr(capture,'native'):torch.nn.functional.scaled_dot_product_attention=capture.native
                    with torch.autocast('cuda',dtype=torch.bfloat16):reference=model(ids)['logits']
                    difference=(result['logits'].float()-reference.float()).abs().max().item()
                    assert difference==0.,f'Instrumentation changed logits: {difference}'
                    report['max_instrumented_logit_difference']=max(report['max_instrumented_logit_difference'],difference)
                    report['max_sdpa_reconstruction_error']=max(report['max_sdpa_reconstruction_error'],capture.max_reconstruction_error)
                    capture.__enter__()
                    attention=torch.stack([a for _,a in captured],dim=1)
                    arrays.append(attention.numpy())
                    logits=result.get('all_logits',[result['logits']])
                    readouts=torch.stack([x[:,-1,:].float().cpu() for x in logits],dim=1)
                    labels=torch.tensor([tokenizer.encode(r['answer'])[0] for r in rows])
                    probabilities=readouts.softmax(-1).gather(-1,labels[:,None,None].expand(-1,len(logits),1)).squeeze(-1)
                    tick_probs.append(probabilities.numpy());tick_predictions.append(readouts.argmax(-1).numpy())
                    for i,row in enumerate(rows):
                        old=old_preds[offset+i]
                        assert old['id']==row['id']
                        predicted=tokenizer.decode([int(readouts[i,-1].argmax())])
                        assert old['terminated'] and predicted==old['prediction']
                        sources,values,query=prompt_positions(row)
                        report['examples'].append({'id':row['id'],'split':split,'start':row['start'],
                            'answer':row['answer'],'prediction':predicted,'correct':old['correct'],
                            'source_positions':sources,'value_positions':values,'query_position':query})
        report['events']=event_names;report['complete']=True
        np.savez_compressed(output.with_suffix('.npz'),attention=np.concatenate(arrays),
            tick_gold_probability=np.concatenate(tick_probs),tick_prediction_ids=np.concatenate(tick_predictions))
        report['arrays_sha256']=file_hash(output.with_suffix('.npz'))
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(f'{family}: {len(report["examples"])} queries, {len(event_names)} attention events, exact logits preserved',flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
