"""Two-hop recall at width 128: is CTsqS's binding advantage specific to one-hop recall? (2026-11-18)

scripts/recall_memory.py: at width 128 CTsqS learned one-hop key-value binding in 19 of 25 runs and Dans in 4 of 25
(both at their tuned lr 1e-3, two fresh-seed confirmations); at width 256 the gap narrowed (14/15 vs 11/15, not
significant). This asks whether the advantage holds on a second binding task at width 128.

Task: 2N distinct keys from the 64 PRI tokens, A1..AN and B1..BN, and N values (16 symbols). The context holds the 2N
pairs (Ai Bi) and (Bi vi), shuffled together; then 4 queries QUERY Ai -> vi (two lookups each). N 2-16 (train,
ID), 17-32 (OOD). Everything else is recall_memory's: models at width 128, AdamW lr 1e-3 (each cell's tuned lr on
one-hop recall), warmup 200, cosine to 10%, batch 128, the pair curriculum (`cur`: N from 2..min(16, 2 + step // 700)),
the LM tick sampler, evaluation at 16 ticks (OOD also 32, 64), query accuracy, resume checkpoints.

Screen (T8, seeds 0 and 1, `cur`, up to 40,000 updates, early stop once ID is at least 95% at two checks in a row).
Rules, fixed before any run: two-hop is learnable if T8 reaches 90% ID on both seeds; the budget is the larger of
10,000 and twice the most updates T8 needed, rounded up to a multiple of 5,000, at most 40,000. If not learnable,
two-hop cannot be tested at this width, and that is reported.

Comparison (T8, Dans, CTsqS; seeds 0-9; the screen's budget, no early stop). Rules, fixed before any run:
- Primary: CTsqS binds (ID at least 90%) more reliably than Dans, one-sided Fisher exact p < 0.05 on bound counts.
- Secondary: CTsqS beats Dans on OOD at 16 ticks (mean at least 5 points higher, higher on at least 8 of 10).
- If the primary holds, it is a CTM result only after an lr check (3e-4 and 2e-3 on seeds 0-4, as recall_memory
  phase 4) and a fresh-seed test (seeds 10-19), with rules fixed before them.
- Reported: T8 against both; OOD of bound runs; ticks (OOD 64 vs 16).

Usage: python -m scripts.twohop_memory --phase screen|compare --gpu N --runs CELL:cur:SEED... | --summarize
"""
import argparse,json,os
from pathlib import Path

import scripts.recall_memory as R
from scripts.cmm_tasks import BOS,PRI,QUERY,SYM

R.OUT=Path('research/results/twohop_memory');R.CKPT=Path('research/runs/twohop_memory')
CELLS=('T8','Dans','CTsqS');SEEDS=tuple(range(10))


def example(size,dense,rng):
    keys=rng.sample(PRI,2*size);A,B=keys[:size],keys[size:];vals=[rng.choice(SYM) for _ in range(size)]
    pairs=[(a,b) for a,b in zip(A,B)]+[(b,v) for b,v in zip(B,vals)];rng.shuffle(pairs);seq=[BOS]
    for x,y in pairs:seq+=[x,y]
    mask=[False]*len(seq)
    for q in (rng.sample(range(size),4) if size>=4 else [rng.randrange(size) for _ in range(4)]):seq+=[QUERY,A[q],vals[q]];mask+=[False,True,False]
    return seq,mask


R.example=example


def summarize():
    from scipy.stats import fisher_exact
    R.choose()
    if not (R.OUT/'CHOICE.json').exists():return
    mean=lambda v:sum(v)/len(v);o=lambda r,t='16':r['ood'].get(t,r['ood'].get('None'))
    rs={c:[json.loads(f.read_text()) if (f:=R.OUT/f'compare_{c}_cur_seed{s}.json').exists() else None for s in SEEDS] for c in CELLS}
    done=[c for c in CELLS if all(rs[c])]
    L=['# Two-hop recall at width 128: comparison (seeds 0-9)','Protocol and rules: `scripts/twohop_memory.py`.','',
       '| Cell | Bound | ID | OOD at 16 | OOD at 64 | OOD of bound runs |','|---|---:|---:|---:|---:|---:|']
    for c in done:
        b=[r for r in rs[c] if r['id']>=90]
        L.append(f"| {c} | {len(b)}/10 | {mean([r['id'] for r in rs[c]]):.1f} | {mean([o(r) for r in rs[c]]):.1f} | "+(f"{mean([o(r,'64') for r in rs[c]]):.1f}" if c!='T8' else '-')+
                 f" | {mean([o(r) for r in b]) if b else float('nan'):.1f} |")
    if 'Dans' in done and 'CTsqS' in done:
        k=lambda c:sum(r['id']>=90 for r in rs[c]);p=fisher_exact([[k('CTsqS'),10-k('CTsqS')],[k('Dans'),10-k('Dans')]],alternative='greater')[1]
        oa,od=[o(r) for r in rs['CTsqS']],[o(r) for r in rs['Dans']];hi=sum(x>y for x,y in zip(oa,od))
        L+=['','## Rule outcomes','',f"- **Primary, CTsqS binds more reliably than Dans:** {'yes (lr check and fresh seeds follow)' if p<0.05 else 'no'} ({k('CTsqS')}/10 vs {k('Dans')}/10, p = {p:.3f})",
            f"- Secondary, CTsqS OOD win over Dans: {'yes' if mean(oa)-mean(od)>=5 and hi>=8 else 'no'} (mean {mean(oa):.1f} vs {mean(od):.1f}, higher on {hi} of 10)"]
        if 'T8' in done:L.append(f"- Reported, T8: {k('T8')}/10 bound")
    (R.OUT/'COMPARE.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('screen','compare'),default='screen');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    for x in a.runs:c,r,s=x.split(':');R.train(c,r,int(s),a.phase,device)


if __name__=='__main__':main()
