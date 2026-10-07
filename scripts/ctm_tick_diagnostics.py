"""Evaluation-only diagnostics of CTM-LM tick use (research/CTM_TICK_DIAGNOSTICS.md).

Runs on trained checkpoints; nothing is trained.

* E1 tick ablation and extrapolation: per-tick held-out cross-entropy for ticks 1-32
  (trained with 16), with the tick attention switched off from tick k+1 on
  (k = 1, 3, 4, 8: the values are zeroed, so the attention output is 0), and with the
  token-conditioned start state removed (z_0 = z_init).
* E2 tick benefit by token difficulty: delta = CE(tick 4) - CE(tick 16) per token,
  binned by unigram surprisal, by the Transformer's per-token loss and by token type.
* E3 oracle-gap null: for a set of predictors, the per-token minimum cross-entropy
  (an oracle that sees the label) against the mean-probability ensemble. The CTM's
  ticks 4-16 are compared with 13 MC-dropout passes of the Transformer (dropout on
  each block's residual update) at several rates.

Usage: python -m scripts.ctm_tick_diagnostics [--windows 32] [--gpu 0]
"""
import argparse,json,math,os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

OUT=Path('research/results/ctm_tick_diagnostics')
SWEEP=Path('research/runs/lr_sweep');PROBES=Path('research/runs/ctm_lm_probes')
# name -> (run directory, checkpoint file)
CTMS={'ctm_aware_400M':(SWEEP/'ctm_aware_d384_h4_k+1','step_0001526.pt'),
      'ctm_heavy_100M':(SWEEP/'ctm_heavy_d448_h1_k+2','step_0000381.pt'),
      'D_C_mean_tick_10M':(PROBES/'D_C','latest.pt')}
TRANSFORMER=(SWEEP/'transformer_d384_h4_k+2','step_0001526.pt')
EVAL_OFFSET=40000
EXTRAPOLATE=32
ATTENTION_UNTIL=(1,3,4,8)
DROPOUT=(0.02,0.05,0.1,0.2)


def load(directory,checkpoint,device):
    from scripts.pretrain import build_config
    from ctm_transformer.ctm_lm_adapt import adapted_factory
    from ctm_transformer.lm_scale import scaled_factory
    run=json.loads((directory/'config.json').read_text());config=build_config(run,32768)
    model=adapted_factory(set(run['ctm_adaptations']),unet_width=run.get('unet_width'))(config) if run['family']=='ctm_lm' else scaled_factory(run['family'])(config)
    model.load_state_dict(torch.load(directory/checkpoint,map_location='cpu',weights_only=False)['model'])
    return model.to(device).eval(),run


def token_stats(logits,y):
    """Per-token cross-entropy and certainty (1 - normalized entropy) of logits [N, V]."""
    logp=logits.float().log_softmax(-1)
    ce=-logp.gather(-1,y[:,None]).squeeze(-1);certainty=1+(logp.exp()*logp).sum(-1)/math.log(logp.shape[-1])
    return ce,certainty


@torch.no_grad()
def ctm_ticks(model,x,y,T,attention_until=None,token_start=True):
    """Per-tick [N, T] cross-entropy and certainty, mirroring AdaptedCTMLM.forward (evaluation path)."""
    from ctm_transformer.positions import rope
    B,S=x.shape;d,D,H=model.config.d_model,model.config.d_latent,model.config.n_heads;N=B*S
    feats=model.features(x);positions=torch.arange(S,device=x.device)
    heads=lambda t:t.view(B,S,H,d//H).transpose(1,2)
    keys=rope(heads(model.key(feats)),positions);values=heads(model.value(feats));silent=torch.zeros_like(values)
    observed=feats.reshape(N,d) if model.adaptations['observe_token'] else None
    query_extra=model.query_feature(feats).reshape(N,d) if model.query_feature is not None else None
    z=model.z_init.expand(N,D).float()
    if model.start_projection is not None and token_start:z=z+model.start_projection(feats).reshape(N,D).float()
    history=model.history_init.expand(N,D,model.config.history_len)
    if model.history_projection is not None:history=history+model.history_projection(feats).reshape(N,D,1).to(history.dtype)
    alpha_a,beta_a=model.sync_action.start(z);alpha_o,beta_o=model.sync_out.start(z)
    ce,cert=[],[]
    for t in range(T):
        v=values if attention_until is None or t<attention_until else silent
        z,history,alpha_a,beta_a,alpha_o,beta_o,readout=model._tick(z,history,alpha_a,beta_a,alpha_o,beta_o,keys,v,positions,observed,query_extra)
        c,k=token_stats(model.lm_head(readout.to(torch.bfloat16)),y.reshape(N));ce.append(c);cert.append(k)
    return torch.stack(ce,1),torch.stack(cert,1)


def add_residual_dropout(model,p):
    """Dropout on each block's residual update (output - input), active at evaluation."""
    def hook(module,inputs,output):return inputs[0]+F.dropout(output-inputs[0],p,True)
    return [b.register_forward_hook(hook) for b in model.layers]


def oracle_summary(ce):
    """ce: [n, k] cross-entropy of k predictors. Mean, oracle (min over predictors), mean-probability ensemble, per-token spread."""
    k=ce.shape[1]
    return {'predictors':k,'mean_ce':float(ce.mean()),'oracle_ce':float(ce.min(1).values.mean()),
            'ensemble_ce':float((-torch.logsumexp(-ce,1)+math.log(k)).mean()),'per_token_std':float(ce.std(1).mean())}


def token_types(tokenizer_path):
    """Class of each vocabulary id: word_start (leading space + letter), word_piece (letters, no space), number, punctuation, whitespace, other."""
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_path));types=[]
    for i in range(32768):
        s=tok.decode([i])
        if not s.strip():types.append('whitespace')
        elif s[0]==' ' and s.strip()[0].isalpha():types.append('word_start')
        elif s.strip().isalpha():types.append('word_piece')
        elif any(ch.isdigit() for ch in s):types.append('number')
        elif all(not ch.isalnum() for ch in s.strip()):types.append('punctuation')
        else:types.append('other')
    return np.array(types)


def binned(delta,key,edges_or_labels,categorical=False):
    rows=[];total=float(delta.sum())
    if categorical:
        groups=[(label,key==label) for label in edges_or_labels]
    else:
        q=np.quantile(key,np.linspace(0,1,11));groups=[(f'{q[i]:.2f}-{q[i+1]:.2f}',(key>=q[i])&((key<q[i+1]) if i<9 else (key<=q[i+1]))) for i in range(10)]
    for label,m in groups:
        if m.sum()==0:continue
        rows.append({'bin':label,'tokens':int(m.sum()),'mean_delta':float(delta[m].mean()),'share_of_gain':float(delta[m].sum()/total) if total else float('nan')})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--windows',type=int,default=32);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--memory-fraction',type=float,default=0.35)
    p.add_argument('--calibration-only',action='store_true',help='add the E2 temperature control to an existing diagnostics.json')
    a=p.parse_args();device=torch.device(f'cuda:{a.gpu}');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
    from ctm_transformer.pretrain import TokenWindows
    data_manifest=Path('research/data/pretrain/fineweb_edu_32k/manifest.json');data=json.loads(data_manifest.read_text());root=data_manifest.parent
    valid=TokenWindows([root/s for s in data['validation']],1024,0);indices=list(range(EVAL_OFFSET,EVAL_OFFSET+a.windows))
    batches=[tuple(t.to(device) for t in valid.batch([i])) for i in indices]
    targets=torch.cat([y.reshape(-1) for _,y in batches]).cpu().numpy()
    OUT.mkdir(parents=True,exist_ok=True);results={'windows':a.windows,'tokens':int(targets.size),'eval_offset':EVAL_OFFSET}
    if a.calibration_only:
        results=json.loads((OUT/'diagnostics.json').read_text());model,_=load(*TRANSFORMER,device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            transformer_ce=torch.cat([token_stats(model(x)['logits'][0],y[0])[0] for x,y in batches]).cpu().numpy()
        del model;torch.cuda.empty_cache()
        results['calibrated']={name:calibrated_benefit(name,batches,transformer_ce,device) for name in CTMS}
        (OUT/'diagnostics.json').write_text(json.dumps(results,indent=1)+'\n');write_report(results);return

    # Transformer: per-token loss (difficulty) and the MC-dropout null.
    model,_=load(*TRANSFORMER,device);tr={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        tr_ce=torch.cat([token_stats(model(x)['logits'][0],y[0])[0] for x,y in batches])
        tr['clean_ce']=float(tr_ce.mean());tr['dropout']={}
        for rate in DROPOUT:
            hooks=add_residual_dropout(model,rate);torch.manual_seed(0)
            passes=torch.stack([torch.cat([token_stats(model(x)['logits'][0],y[0])[0] for x,y in batches]) for _ in range(13)],1)
            for h in hooks:h.remove()
            tr['dropout'][str(rate)]=oracle_summary(passes)
    results['transformer']=tr;transformer_ce=tr_ce.cpu().numpy();del model;torch.cuda.empty_cache()
    print(json.dumps(tr),flush=True)

    # Unigram surprisal of each target from the first training shard; token classes from the tokenizer.
    counts=np.bincount(np.memmap(root/data['train'][0],dtype=np.uint16,mode='r'),minlength=32768).astype(np.float64)+1
    unigram=-np.log(counts/counts.sum())[targets];types=token_types(root/data['tokenizer'])[targets]

    results['ctm']={}
    for name,(directory,checkpoint) in CTMS.items():
        model,run=load(directory,checkpoint,device);r={}
        def run_all(T=EXTRAPOLATE,attention_until=None,token_start=True):
            parts=[ctm_ticks(model,x,y,T,attention_until,token_start) for x,y in batches]
            return torch.cat([c for c,_ in parts]),torch.cat([k for _,k in parts])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ce,cert=run_all()
            r['per_tick']=[float(v) for v in ce.mean(0)]
            r['most_certain_of_16']=float(ce[:,:16].gather(1,cert[:,:16].argmax(1,keepdim=True)).mean())
            r['ticks_1_16']=oracle_summary(ce[:,:16]);r['ticks_4_16']=oracle_summary(ce[:,3:16])
            r['attention_until']={str(k):[float(v) for v in run_all(T=16,attention_until=k)[0].mean(0)] for k in ATTENTION_UNTIL}
            if model.start_projection is not None:r['no_token_start']=[float(v) for v in run_all(T=16,token_start=False)[0].mean(0)]
        delta=(ce[:,3]-ce[:,15]).cpu().numpy();r['delta_4_16']=float(delta.mean())
        r['by_unigram_surprisal']=binned(delta,unigram,None);r['by_transformer_loss']=binned(delta,transformer_ce,None)
        r['by_token_type']=binned(delta,types,['word_start','word_piece','number','punctuation','whitespace','other'],categorical=True)
        r['tick16_minus_transformer']=float(ce[:,15].mean()-tr_ce.mean())
        results['ctm'][name]=r;print(name,json.dumps({k:r[k] for k in ('per_tick','delta_4_16','ticks_4_16')}),flush=True)
        del model;torch.cuda.empty_cache()
    results['calibrated']={name:calibrated_benefit(name,batches,transformer_ce,device) for name in CTMS}
    (OUT/'diagnostics.json').write_text(json.dumps(results,indent=1)+'\n');write_report(results)


def write_report(res):
    L=['# CTM-LM tick-use diagnostics (evaluation only)','',f'Protocol: [CTM_TICK_DIAGNOSTICS.md](../../CTM_TICK_DIAGNOSTICS.md). {res["windows"]} held-out windows from offset {res["eval_offset"]} ({res["tokens"]:,} tokens).','']
    show=[1,2,3,4,8,12,16,20,24,32]
    L+=['## E1: per-tick loss, extrapolation beyond 16, attention off, start state off','','| Model | Readout | '+' | '.join(f'{t}' for t in show)+' |','|---|---|'+'---:|'*len(show)]
    for name,r in res['ctm'].items():
        L.append(f'| {name} | all ticks (trained 16) | '+' | '.join(f'{r["per_tick"][t-1]:.3f}' for t in show)+' |')
        for k,v in r['attention_until'].items():L.append(f'| {name} | attention off after tick {k} | '+' | '.join(f'{v[t-1]:.3f}' if t<=16 else '' for t in show)+' |')
        if 'no_token_start' in r:L.append(f'| {name} | no token start state | '+' | '.join(f'{r["no_token_start"][t-1]:.3f}' if t<=16 else '' for t in show)+' |')
    L+=['','## E3: oracle gap against its null','','| Predictor set | Count | Mean CE | Ensemble CE | Oracle CE | Oracle gap (mean − oracle) | Per-token CE std |','|---|---:|---:|---:|---:|---:|---:|']
    row=lambda label,s:f'| {label} | {s["predictors"]} | {s["mean_ce"]:.3f} | {s["ensemble_ce"]:.3f} | {s["oracle_ce"]:.3f} | {s["mean_ce"]-s["oracle_ce"]:.3f} | {s["per_token_std"]:.3f} |'
    for name,r in res['ctm'].items():L+=[row(f'{name} ticks 4–16',r['ticks_4_16']),row(f'{name} ticks 1–16',r['ticks_1_16'])]
    L.append(f'| Transformer, no dropout | 1 | {res["transformer"]["clean_ce"]:.3f} | | | | |')
    for rate,s in res['transformer']['dropout'].items():L.append(row(f'Transformer MC dropout {rate}',s))
    L+=['','## E2: tick benefit (CE tick 4 − CE tick 16) by token difficulty','']
    for name,r in res['ctm'].items():
        L+=[f'**{name}**: mean benefit {r["delta_4_16"]:+.4f} nats; tick 16 minus Transformer {r["tick16_minus_transformer"]:+.3f}.','']
        for key,title in (('by_unigram_surprisal','Unigram surprisal decile'),('by_transformer_loss','Transformer loss decile'),('by_token_type','Token type')):
            L+=[f'| {title} | Tokens | Mean benefit | Share of total benefit |','|---|---:|---:|---:|']+[f'| {b["bin"]} | {b["tokens"]} | {b["mean_delta"]:+.4f} | {b["share_of_gain"]:.0%} |' for b in r[key]]+['']
    if 'calibrated' in res:
        L+=['## E2 control: benefit after fitting one temperature per tick','','| Model | Tick | Temperature | Mean entropy | CE | Calibrated CE |','|---|---:|---:|---:|---:|---:|']
        for name,c in res['calibrated'].items():
            for t in c['ticks']:t=str(t) if str(t) in c['temperature'] else t;L.append(f'| {name} | {t} | {c["temperature"][t]:.2f} | {c["entropy"][t]:.3f} | {c["raw_ce"][t]:.3f} | {c["calibrated_ce"][t]:.3f} |')
        for name,c in res['calibrated'].items():
            L+=['',f'**{name}**: calibrated benefit (tick {c["ticks"][0]} − tick {c["ticks"][-1]}) {c["delta_calibrated"]:+.4f} nats.','','| Transformer loss decile | Tokens | Mean calibrated benefit |','|---|---:|---:|']
            L+=[f'| {b["bin"]} | {b["tokens"]} | {b["mean_delta"]:+.4f} |' for b in c['by_transformer_loss']]
    (OUT/'DIAGNOSTICS.md').write_text('\n'.join(L)+'\n')



@torch.no_grad()
def calibrated_benefit(name,batches,transformer_ce,device,ticks=(4,8,16),grid=None):
    """E2 control for confidence changes: the per-tick benefit after fitting one softmax temperature per tick.

    A tick that only becomes less confident gains on hard tokens and loses on easy ones, which looks
    like a difficulty-dependent benefit. Fitting a temperature per tick removes that effect."""
    grid=torch.linspace(0.6,1.8,121,device=device) if grid is None else grid
    model,_=load(*CTMS[name],device)
    ce={t:[] for t in ticks};entropy={t:[] for t in ticks}  # per token: cross-entropy at every grid temperature, and entropy at 1
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for x,y in batches:
            out=model(x,return_all_logits=True)['all_logits']
            for t in ticks:
                logits=out[t-1][0].float();lp=logits.log_softmax(-1);entropy[t].append(-(lp.exp()*lp).sum(-1));del lp
                ce[t].append(torch.stack([F.cross_entropy(logits/tau,y[0],reduction='none') for tau in grid],1));del logits
            del out
    del model;torch.cuda.empty_cache()
    result={'ticks':list(ticks),'temperature':{},'entropy':{},'raw_ce':{},'calibrated_ce':{}};per_token={};one=int(torch.argmin((grid-1).abs()))
    for t in ticks:
        table=torch.cat(ce[t]);best=int(table.mean(0).argmin())
        result['temperature'][t]=float(grid[best]);result['entropy'][t]=float(torch.cat(entropy[t]).mean())
        result['raw_ce'][t]=float(table[:,one].mean());result['calibrated_ce'][t]=float(table[:,best].mean());per_token[t]=table[:,best].cpu().numpy()
    delta=per_token[ticks[0]]-per_token[ticks[-1]]
    result['delta_calibrated']=float(delta.mean());result['by_transformer_loss']=binned(delta,transformer_ce,None)
    return result


if __name__=='__main__':main()
