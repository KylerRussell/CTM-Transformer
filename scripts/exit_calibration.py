"""Adaptive compute and calibration on the 400M-token checkpoints: can the CTM Transformer earn its place at equal loss? (2026-10-31)

The CTM Transformer ties Dans on loss at higher training cost. Two training-free ways it could still be better:

1. Adaptive compute. Loss is flat after about 8 ticks, so the question is which model reaches its loss with
   fewer mean ticks when each token stops thinking on its own. The first test (scripts/adaptive_compute.py,
   2026-10-18) used only the CTM's certainty threshold and compared losses at a mean of 2-4 ticks; every model
   was about 0.22 worse than at fixed ticks, the CTM Transformer no better than Dans. This extends it with:
   - a second exit rule, convergence: a token stops at the first tick t >= 2 whose readout distribution
     changed little, KL(p_t || p_{t-1}) < eps (the usual exit for recurrent-depth models);
   - the primary measure: the mean exit tick needed to come within 0.01 of the model's own 16-tick loss.
2. Calibration. The CTM paper reports well-calibrated certainty on ImageNet, computed by "averaging
   probabilities up to each tick" (arXiv 2505.05522, Figure 5). Measured here as the expected calibration
   error (ECE, 15 equal-width bins of the top-1 probability against top-1 correctness) per token, for two
   readouts: the tick-t distribution p_t and the running average (p_1 + ... + p_t) / t, as in the paper.

Data: 128 held-out FineWeb-Edu windows of 1,024 tokens (131,072 tokens), offset 40,000 (the final
evaluation's windows). Every readout of a recurrent model is computed at every tick 1..32 in one pass
(the coda and output head applied to each tick's state, as in training).

Models (two seeds each; T20 one): T8 and T20 (no ticks; calibration only), Dans, CTsq, CTnlm, CTso, CTfull,
CTsqS, CTsqH and CTsqLD (the CTM's per-tick loss with detached readouts, the objective the CTM paper's
calibration comes with).

Caveat, as before: every tick runs for every position, since later positions attend to earlier positions'
states; the exit tick measures how much thinking each token's prediction needs, not wall-clock savings.

Rules, fixed before any run. Comparisons are two-seed means against Dans; a model differs from Dans only if
the mean difference passes the threshold and both its seeds lie on the same side of both Dans seeds.
- Item 1, primary: ticks-to-loss = the smallest mean exit tick at which an exit rule's loss is within 0.01
  of the model's own fixed 16-tick loss (linear interpolation along the rule's sweep), taking the better of
  the two rules per seed. A model uses ticks more efficiently than Dans if it needs at least 0.5 fewer mean
  ticks. Reported with the same quantity for fixed ticks (the smallest T within 0.01).
- Item 1, secondary: loss at a mean of 3 and of 4 exit ticks (the lower envelope of both rules); better if
  at least 0.02 lower.
- Item 2: ECE at 16 ticks, for both readouts (tick 16 and the average of ticks 1-16), and for T8 / T20 at
  their single readout. A model is better calibrated than Dans on a readout if its ECE is at least 0.005
  lower. The CTM paper's claim is supported here if a CTM Transformer variant is better calibrated than
  Dans on the tick-averaged readout. Reported: ECE at ticks 1, 2, 4, 8 and 32 for both readouts.
- Outcome: if no variant passes a rule in item 1 or item 2, the CTM Transformer has no training-free
  advantage over Dans at equal loss on these measures.

Usage: python -m scripts.exit_calibration --gpu N --runs NAME... | --summarize
"""
import argparse,json,math,os
from pathlib import Path

OUT=Path('research/results/exit_calibration')
GROUPS={'T8':['T8_lr0.002','T8_lr0.002_s2'],'T20':['T20_lr0.002'],'Dans':['Dans_rand_lr0.002','Dans_rand_lr0.002_s2'],
        **{v:[f'{v}_lr0.002',f'{v}_lr0.002_s2'] for v in ('CTsq','CTnlm','CTso','CTfull','CTsqS','CTsqH','CTsqLD')}}
WINDOWS,OFFSET,T_MAX,BINS=128,40000,32,15
TAUS=[round(0.025*i,3) for i in range(41)]
EPSILONS=[10**(-4+0.25*i) for i in range(17)]  # 1e-4 .. 1
REPORT_TICKS=(1,2,4,8,16,32)


def ece_parts(conf,correct):
    """Per-bin (count, sum of confidence, sum of correct) for 15 equal-width bins."""
    import torch
    b=(conf*BINS).long().clamp(max=BINS-1)
    return torch.stack([torch.bincount(b,minlength=BINS).double(),torch.bincount(b,weights=conf.double(),minlength=BINS),
                        torch.bincount(b,weights=correct.double(),minlength=BINS)])


def ece(parts):
    n,c,a=parts;total=float(n.sum())
    return float(((c-a).abs()).sum()/total)


def analyse(name,device):
    import torch
    from ctm_transformer.pretrain import TokenWindows
    from scripts.code_math_probe import load,source
    out=OUT/f'{name}.json'
    if out.exists():return
    model,recurrent=load(source(name),device)
    valid=TokenWindows([Path('research/data/pretrain/fineweb_edu_32k/validation_000.bin')],1024,0)
    T=T_MAX if recurrent else 1;logV=math.log(32768)
    ce,cert,kl=[],[],[];fixed_parts=torch.zeros((T,3,BINS),dtype=torch.float64);avg_parts=torch.zeros((T,3,BINS),dtype=torch.float64);avg_ce=torch.zeros(T,dtype=torch.float64)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(OFFSET,OFFSET+WINDOWS):
            x,y=valid.batch([i]);x,y=x.to(device),y.to(device).reshape(-1)
            logits=model(x,max_thought_steps=T_MAX,return_all_logits=True)['all_logits'] if recurrent else [model(x)['logits']]
            w_ce,w_cert,w_kl=[],[],[];prev=None;running=None
            for t,l in enumerate(logits):
                logp=l.float().reshape(-1,l.shape[-1]).log_softmax(-1);p=logp.exp()
                w_ce.append(-logp.gather(-1,y[:,None]).squeeze(-1));w_cert.append(1+(p*logp).sum(-1)/logV)
                w_kl.append((p*(logp-prev)).sum(-1) if prev is not None else torch.full_like(w_ce[-1],math.inf))
                conf,pred=p.max(-1);fixed_parts[t]+=ece_parts(conf,(pred==y).float()).cpu()
                running=p if running is None else running+p;avg=running/(t+1)
                conf,pred=avg.max(-1);avg_parts[t]+=ece_parts(conf,(pred==y).float()).cpu()
                avg_ce[t]+=float(-avg.gather(-1,y[:,None]).clamp_min(1e-30).log().sum());prev=logp
            ce.append(torch.stack(w_ce).cpu());cert.append(torch.stack(w_cert).cpu());kl.append(torch.stack(w_kl).cpu())
    ce=torch.cat(ce,1);cert=torch.cat(cert,1);kl=torch.cat(kl,1);N=ce.shape[1]  # [T, tokens]
    r={'run':name,'tokens':N,'recurrent':recurrent,'fixed_ce':{str(t+1):float(ce[t].mean()) for t in range(T)},
       'ece_fixed':{str(t+1):ece(fixed_parts[t]) for t in range(T)},'ece_averaged':{str(t+1):ece(avg_parts[t]) for t in range(T)},
       'averaged_ce':{str(t+1):float(avg_ce[t])/N for t in range(T)},
       'reliability_16':{k:parts.tolist() for k,parts in (('fixed',fixed_parts[min(15,T-1)]),('averaged',avg_parts[min(15,T-1)]))}}
    if recurrent:
        def curve(reached):
            first=torch.where(reached.any(0),reached.float().argmax(0),torch.full((N,),T-1,dtype=torch.long))
            return [float(first.float().mean()+1),float(ce.gather(0,first[None]).mean())]
        r['certainty_exit']={str(tau):curve(cert>=tau) for tau in TAUS}
        r['convergence_exit']={f'{eps:.2e}':curve(kl<eps) for eps in EPSILONS}
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=1)+'\n')
    print(name,'ce16',round(r['fixed_ce'][str(min(16,T))],4),'ece16 fixed/avg',round(r['ece_fixed'][str(min(16,T))],4),round(r['ece_averaged'][str(min(16,T))],4),flush=True)


# ---------------------------------------------------------------- summary

def ticks_to(points,target):
    """Smallest mean tick at which the (mean tick, loss) curve comes within the target, interpolated."""
    pts=sorted(points);best=None
    for (t0,l0),(t1,l1) in zip(pts,pts[1:]):
        if l0<=target:return t0
        if l1<=target:return t0+(t1-t0)*(l0-target)/(l0-l1)
    return pts[-1][0] if pts and pts[-1][1]<=target else math.inf


def envelope_at(points,tick):
    """Lower envelope of the loss at a mean exit tick, by interpolation between sweep points."""
    pts=sorted(points);best=math.inf
    for (t0,l0),(t1,l1) in zip(pts,pts[1:]):
        if t0<=tick<=t1 and t1>t0:best=min(best,l0+(l1-l0)*(tick-t0)/(t1-t0))
    return best


def measures(r):
    m={'ce16':r['fixed_ce'][str(min(16,len(r['fixed_ce'])))],'ece_fixed16':r['ece_fixed'][str(min(16,len(r['ece_fixed'])))]}
    if not r['recurrent']:m['ece_averaged16']=m['ece_fixed16'];return m
    m['ece_averaged16']=r['ece_averaged']['16'];target=m['ce16']+0.01
    cert=list(r['certainty_exit'].values());conv=list(r['convergence_exit'].values())
    m['ticks_certainty']=ticks_to(cert,target);m['ticks_convergence']=ticks_to(conv,target);m['ticks_to_loss']=min(m['ticks_certainty'],m['ticks_convergence'])
    m['ticks_fixed']=next(t for t in range(1,33) if r['fixed_ce'][str(t)]<=target)
    for k in (3,4):m[f'loss_at_{k}']=min(envelope_at(cert,k),envelope_at(conv,k))
    return m


def summarize():
    res={n:measures(json.loads((OUT/f'{n}.json').read_text())) for ns in GROUPS.values() for n in ns if (OUT/f'{n}.json').exists()}
    have={g:[res[n] for n in ns if n in res] for g,ns in GROUPS.items()};mean=lambda v:sum(v)/len(v);dans=have['Dans']
    L=['# Adaptive compute and calibration','','Protocol and rules: `scripts/exit_calibration.py` (docstring). Two-seed means (T20 one seed); per seed in brackets.','']
    keys=[('ce16','loss at 16 ticks',3),('ticks_to_loss','ticks to within 0.01 (better exit rule)',2),('ticks_certainty','  certainty exit',2),
          ('ticks_convergence','  convergence exit',2),('ticks_fixed','  fixed ticks',0),('loss_at_3','loss at a mean of 3 exit ticks',3),
          ('loss_at_4','loss at a mean of 4 exit ticks',3),('ece_fixed16','ECE, tick-16 readout',4),('ece_averaged16','ECE, average of ticks 1-16',4)]
    groups=[g for g in GROUPS if have[g]]
    L+=['| Measure | '+' | '.join(groups)+' |','|---|'+'---:|'*len(groups)]
    for k,label,dp in keys:
        row=[]
        for g in groups:
            v=[r[k] for r in have[g] if k in r]
            row.append(f'{mean(v):.{dp}f} ('+'/'.join(f'{x:.{dp}f}' for x in v)+')' if v else '')
        L.append(f'| {label} | '+' | '.join(row)+' |')
    L+=['','## Rule outcomes against Dans','']
    rules=[('ticks_to_loss',0.5,'uses ticks more efficiently'),('loss_at_3',0.02,'lower loss at a mean of 3 exit ticks'),('loss_at_4',0.02,'lower loss at a mean of 4 exit ticks'),
           ('ece_fixed16',0.005,'better calibrated (tick-16 readout)'),('ece_averaged16',0.005,'better calibrated (tick-averaged readout)')]
    supported=False;any_pass=False
    for g in groups:
        if g=='Dans' or len(have[g])<2 or len(dans)<2:continue
        notes=[]
        for k,thr,text in rules:
            if k not in have[g][0]:continue
            d=mean([r[k] for r in have[g]])-mean([q[k] for q in dans])
            if d<=-thr and all(r[k]<min(q[k] for q in dans) for r in have[g]):
                notes.append(f'{text} ({d:+.3f})');any_pass|=g not in ('T8',)
                supported|=k=='ece_averaged16' and g not in ('T8',)
            if d>=thr and all(r[k]>max(q[k] for q in dans) for r in have[g]):notes.append(f'worse: {k} ({d:+.3f})')
        L.append(f'- **{g}:** '+('; '.join(notes) if notes else 'no rule passes'))
    L+=['',f'- **CTM calibration claim (a variant better calibrated than Dans on the tick-averaged readout):** {"supported" if supported else "not supported"}',
        f'- **Outcome:** '+('a CTM Transformer variant passes at least one rule' if any_pass else 'no CTM Transformer variant has a training-free advantage over Dans on these measures')]
    L+=['','## ECE by tick (fixed readout / tick-averaged readout)','']
    for g in groups:
        rs=[json.loads((OUT/f'{n}.json').read_text()) for n in GROUPS[g] if (OUT/f'{n}.json').exists()]
        if rs and rs[0]['recurrent']:
            L.append(f'- **{g}:** '+', '.join(f'{t}: {mean([r["ece_fixed"][str(t)] for r in rs]):.4f} / {mean([r["ece_averaged"][str(t)] for r in rs]):.4f}' for t in REPORT_TICKS))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'EXIT_CALIBRATION.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='*',default=[])
    p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    for n in a.runs:analyse(n,torch.device('cuda'))


if __name__=='__main__':main()
