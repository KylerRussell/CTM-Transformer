"""Probes of the synchronization binding result (2026-11-27, after external review, concern 1).

At width 128, CTsqS learns key-value binding in 15 of 20 confirmatory runs against Dans's 2 of 20, and the phase-7
controls (scripts/recall_memory.py) attribute the gain to second-order synchronization. Bound runs of every model
generalize about equally, so synchronization changes learning, not the quality of the solution. This study asks how:
does the bound solution use the synchronization query, and does synchronization make retrieval attention appear before
the behaviour does?

Runs (phase `probe` of scripts/recall_memory.py): CTsqS and Dans retrained on the confirmatory seeds 5-24 with the
phase-2 recipe and budget at lr 1e-3, exactly as in phases 3 and 5 (same seeds, data, initialization and tick sequence;
only weight snapshots are added): weights after 0, 250, ..., 3,000 updates, every 1,000 updates to 9,000, and final
(10,000). The phase-7 controls' final weights (DansS, CTsq1S, DansT4) are reused.

Probe set: 256 examples of the ID split (N uniform 2-16, own generator), 4 queries each. At each answer position (the
queried key, whose target is its value), for every tick, thinking layer and head: the effective attention weight on the
matching value's position in the context, minus its mean weight on the other values' context positions (the value
selectivity). With the attention sink the effective weight is (1 - g) A + g A_sink on the real positions, g the layer's
per-head sink gate. A run's retrieval score is its largest mean value selectivity over ticks 1-16 (1-4 for DansT4),
thinking layers and heads; the (tick, layer, head) attaining it is its retrieval head. The same measure on the matching
key's position is reported.

Rules, fixed before any run:
- Reproduction (reported): per seed, whether the retrained run binds (ID query accuracy at least 90%) as the original
  run did. The analyses use each retrained run's own outcome.
- P1, use: in bound CTsqS runs, ID query accuracy at 16 ticks (scripts/recall_memory.accuracy, 512 examples) with the
  synchronization query removed at evaluation. The bound solution depends on the synchronization query if the mean
  drop is at least 20 points and the drop is at least 10 points in at least 80% of bound runs; it does not need it if
  the mean drop is below 5 points; otherwise it uses it in part.
- P2, precursor: at update 1,000 (when bound CTsqS runs were still at chance, 20-25% on the ID check), CTsqS's retrieval
  score is higher than Dans's over the 20 seeds each (one-sided Mann-Whitney U, p < 0.05). If this holds,
  synchronization makes retrieval attention appear before the behaviour.
- Reported: retrieval scores at every snapshot, for bound and unbound runs of both cells; within CTsqS, the AUROC of the
  update-1,000 score for later binding; final retrieval scores of bound and unbound runs per cell, controls included;
  for CTsqS at its retrieval head, the matching value's logit advantage over the other values split into the part from
  the ordinary query and the part from the synchronization query, at update 1,000 and at the final update; the
  retrieval score with the synchronization query removed.

Usage: python -m scripts.recall_probes --queue --gpu N | --analyse --gpu N | --summarize
"""
import argparse,json,os,random,time
from pathlib import Path

CELLS=('CTsqS','Dans');SEEDS=tuple(range(5,25));BUDGET=10000;PRECURSOR=1000;N_PROBE=256;MICRO=64
SNAPSHOTS=sorted(set(range(0,3001,250))|set(range(4000,10000,1000))|{BUDGET})
RM=Path('research/results/recall_memory');CKPT=Path('research/runs/recall_memory');PROBE=CKPT/'probe'
OUT=Path('research/results/recall_probes');CONTROLS=('DansS','CTsq1S','DansT4')


def run_name(cell,seed):return f'probe_{cell}_cur_seed{seed}'


def queue(gpu):
    import torch
    from scripts.recall_memory import train
    PROBE.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        for cell in CELLS:
            claim=PROBE/f'{run_name(cell,seed)}.claimed'
            try:os.close(os.open(claim,os.O_CREAT|os.O_EXCL))
            except FileExistsError:continue
            print('run',cell,seed,'gpu',gpu,time.ctime(),flush=True);train(cell,'cur',seed,'probe',torch.device('cuda'))
    print('queue empty',time.ctime(),flush=True)


def probe_set():
    """Batches of the probe examples, with each answer's position and its matching and other values' positions."""
    import torch
    from scripts.recall_memory import example
    from scripts.cmm_tasks import PAD,QUERY
    rng=random.Random('recall-probes');batches=[]
    for _ in range(N_PROBE//MICRO):
        rows=[example(rng.randint(2,16),False,rng) for _ in range(MICRO)];L=max(len(s) for s,_ in rows)-1
        x=torch.full((MICRO,L),PAD);answer=torch.zeros(MICRO,4,dtype=torch.long);value=torch.zeros(MICRO,4,dtype=torch.long)
        other=torch.zeros(MICRO,4,L,dtype=torch.bool)
        for i,(s,m) in enumerate(rows):
            x[i,:len(s)-1]=torch.tensor(s[:-1]);n=s.index(QUERY)  # the context ends at the first query
            for a,p in enumerate(p for p in range(len(m)) if m[p]):
                key=s.index(s[p]);assert key<n and key%2==1;answer[i,a]=p;value[i,a]=key+1
                for v in range(2,n,2):other[i,a,v]=v!=key+1
        batches.append((x,answer,value,other))
    return batches


class Probe:
    """Captures the effective attention of every thinking layer at each tick at the answer rows and, for a model with the
    synchronization query, the logit advantage of the matching value split into the ordinary and synchronization parts."""
    def __init__(self,model,ticks):
        self.model,self.ticks=model,ticks
        self.sync=getattr(model,'mechanisms',{}).get('sync_query',False);self.layers=len(model.core)

    def run(self,x,answer,value,other,device):
        import torch
        import torch.nn.functional as F
        from ctm_transformer import ctm_transformer as CT
        native=F.scaled_dot_product_attention;records=[];split_logits={};calls=[0];B,S=x.shape
        rows=answer.to(device);val=value.to(device);oth=other.to(device).float()
        def gather(t,offset):  # t: (B,H,L,L) -> rows (B,H,4,L)
            return t.gather(2,(rows+offset)[:,None,:,None].expand(B,t.shape[1],4,t.shape[-1]))
        def wrapped(q,k,v,*a,**kw):
            out=native(q,k,v,*a,**kw);L=q.shape[-2]
            with torch.autocast('cuda',enabled=False):
                mask=torch.ones(L,L,dtype=torch.bool,device=q.device).tril()
                p=((q.float()@k.float().transpose(-2,-1))*q.shape[-1]**-.5).masked_fill(~mask,float('-inf')).softmax(-1)
                ref=out.float();err=((p@v.float())-ref).abs().max().item()/(1+ref.abs().max().item())
                if err>0.02:raise ValueError(f'attention reconstruction error {err}')
                records.append(gather(p,0 if L==S else 1))
            return out
        layer=self.model._thinking_layer if hasattr(self.model,'_thinking_layer') else None
        def thinking(index,h,alpha,beta,*history):
            tick=calls[0]//self.layers;calls[0]+=1
            if alpha is not None:
                with torch.autocast('cuda',enabled=False):
                    block=self.model.core[index];attn=block.attn;W=h.shape[-1];H=attn.heads;n=block.attn_norm(h).float()
                    split=lambda t:t.view(B,S,H,W//H).transpose(1,2)
                    pos=attn.positions if getattr(attn,'positions',None) is not None else torch.arange(S,device=h.device)
                    qb=CT.rope(split(F.linear(n,attn.q.weight.float(),None if attn.q.bias is None else attn.q.bias.float())),pos)
                    qe=CT.rope(split(self.model.sync_queries[index]((alpha/beta).float())),pos)
                    k=CT.rope(split(F.linear(n,attn.k.weight.float(),None if attn.k.bias is None else attn.k.bias.float())),pos)
                    for part,q in (('ordinary',qb),('sync',qe)):
                        lg=(q.gather(2,rows[:,None,:,None].expand(B,H,4,W//H))@k.transpose(-2,-1))*(W//H)**-.5
                        adv=lg.gather(3,val[:,None,:,None].expand(B,H,4,1)).squeeze(-1)-(lg*oth[:,None]).sum(-1)/oth.sum(-1).clamp_min(1)[:,None]
                        split_logits.setdefault(part,{})[(tick,index)]=adv.mean((0,2)).cpu()
            return layer(index,h,alpha,beta,*history)
        F.scaled_dot_product_attention=wrapped
        if layer is not None and self.sync:self.model._thinking_layer=thinking
        try:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):self.model(x.to(device),max_thought_steps=self.ticks)
        finally:
            F.scaled_dot_product_attention=native
            if layer is not None and self.sync:del self.model._thinking_layer
        return records,split_logits


def selectivity(records,model,ticks,value,other,device):
    """(ticks, layers, heads) mean value and key selectivity from one batch's captured attention."""
    import torch
    layers=len(model.core);sink=getattr(model,'mechanisms',{}).get('attention_sink',False);per=layers*(2 if sink else 1)
    pre=len(model.prelude);assert len(records)==pre+ticks*per+len(model.coda),(len(records),pre,ticks,per)
    val=value.to(device);oth=other.to(device).float();sel={'value':[],'key':[]}
    for t in range(ticks):
        lv,lk=[],[]
        for l in range(layers):
            if sink:
                a,b=records[pre+t*per+2*l],records[pre+t*per+2*l+1];g=model.sink_gate[l].float()[None,:,None,None]
                A=(1-g)*a+g*b[...,1:]
            else:A=records[pre+t*per+l]
            def sel_at(pos,others):
                hit=A.gather(3,pos[:,None,:,None].expand(*A.shape[:3],1)).squeeze(-1)
                return (hit-(A*others[:,None]).sum(-1)/others.sum(-1).clamp_min(1)[:,None]).mean((0,2))
            lv.append(sel_at(val,oth));lk.append(sel_at(val-1,torch.roll(oth,-1,-1)))
        sel['value'].append(torch.stack(lv));sel['key'].append(torch.stack(lk))
    return {k:torch.stack(v).cpu() for k,v in sel.items()}


def build_model(cell,path,device):
    import torch
    from scripts.cmm_tasks import build
    model,_=build(cell.removesuffix('T4'),128);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return model.to(device).eval()


def score(model,ticks,batches,device,sync_off=False):
    """Mean selectivity tensors over the probe set, the retrieval score and head, and the logit split at that head."""
    import torch
    saved=getattr(model,'mechanisms',{}).get('sync_query')
    if sync_off:model.mechanisms['sync_query']=False
    try:
        probe=Probe(model,ticks);tot={'value':0,'key':0};split={}
        for x,answer,value,other in batches:
            recs,lg=probe.run(x,answer,value,other,device);s=selectivity(recs,model,ticks,value,other,device)
            for k in tot:tot[k]=tot[k]+s[k]/len(batches)
            for part,d in lg.items():
                for key,v in d.items():split.setdefault(part,{})[key]=split.get(part,{}).get(key,0)+v/len(batches)
    finally:
        if sync_off:model.mechanisms['sync_query']=saved
    v=tot['value'];i=int(v.argmax());t,l,h=i//(v.shape[1]*v.shape[2]),(i//v.shape[2])%v.shape[1],i%v.shape[2]
    out={'retrieval':float(v.max()),'head':[t+1,l,h],'key_retrieval':float(tot['key'].max()),'value_selectivity':v.tolist()}
    if split and (t,l) in split['sync']:out['logit_advantage']={p:float(split[p][(t,l)][h]) for p in ('ordinary','sync')}
    return out


def analyse(gpu):
    import torch
    from scripts.recall_memory import accuracy
    device=torch.device('cuda');batches=probe_set();(OUT/'runs').mkdir(parents=True,exist_ok=True)
    jobs=[(c,s,'probe') for s in SEEDS for c in CELLS]+[(c,s,'control') for c in CONTROLS for s in SEEDS]
    for cell,seed,kind in jobs:
        name=run_name(cell,seed) if kind=='probe' else f'control_{cell}_cur_seed{seed}';out=OUT/'runs'/f'{name}.json'
        result=RM/f'{name}.json'
        if out.exists() or not result.exists():continue
        r=json.loads(result.read_text());ticks=4 if cell.endswith('T4') else 16
        paths={st:PROBE/f'{name}.step{st:05d}.pt' for st in SNAPSHOTS} if kind=='probe' else {BUDGET:CKPT/f'{name}.final.pt'}
        rec={'cell':cell,'seed':seed,'kind':kind,'id':r['id'],'bound':r['id']>=90,'updates_to_90':r['updates_to_90'],'ticks':ticks,'snapshots':{}}
        for st,p in paths.items():
            model=build_model(cell,p,device);rec['snapshots'][str(st)]=score(model,ticks,batches,device)
            if getattr(model,'mechanisms',{}).get('sync_query'):
                if st in (PRECURSOR,BUDGET):rec['snapshots'][str(st)]['sync_off']=score(model,ticks,batches,device,sync_off=True)['retrieval']
                if st==BUDGET:
                    model.mechanisms['sync_query']=False;rec['id_sync_off']=accuracy(model,True,False,'id',ticks,512,device);model.mechanisms['sync_query']=True
        out.write_text(json.dumps(rec)+'\n');print(name,'bound' if rec['bound'] else 'unbound',{k:round(v['retrieval'],3) for k,v in rec['snapshots'].items() if k in ('1000','10000')},flush=True)


def original(cell,seed):
    for phase in ('confirm','replicate'):
        f=RM/f'{phase}_{cell}_cur_seed{seed}.json'
        if f.exists():return json.loads(f.read_text())['id']>=90


def summarize():
    from scipy.stats import mannwhitneyu
    mean=lambda v:sum(v)/len(v) if v else float('nan')
    rs={p.stem:json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*.json'))}
    probe={c:[r for r in rs.values() if r['kind']=='probe' and r['cell']==c] for c in CELLS}
    L=['# Probes of the synchronization binding result','','Protocol and rules: `scripts/recall_probes.py`.','','## Reproduction','']
    for c in CELLS:
        agree=sum(r['bound']==original(c,r['seed']) for r in probe[c])
        L.append(f"- {c}: {sum(r['bound'] for r in probe[c])}/{len(probe[c])} bound on retraining; outcome as originally in {agree} of {len(probe[c])} seeds")
    b=[r for r in probe['CTsqS'] if r['bound']]
    if b and all('id_sync_off' in r for r in b):
        drop=[r['id']-r['id_sync_off'] for r in b];share=sum(d>=10 for d in drop)/len(drop)
        verdict='depends on the synchronization query' if mean(drop)>=20 and share>=0.8 else 'does not need it' if mean(drop)<5 else 'uses it in part'
        L+=['','## P1, use','',f"- Bound CTsqS runs ({len(b)}): ID {mean([r['id'] for r in b]):.1f} intact, {mean([r['id_sync_off'] for r in b]):.1f} without the synchronization query (mean drop {mean(drop):.1f}; at least 10 points in {share:.0%})",
            f'- **Outcome:** the bound solution {verdict}']
    if all(len(probe[c])==len(SEEDS) for c in CELLS):
        a=[r['snapshots'][str(PRECURSOR)]['retrieval'] for r in probe['CTsqS']];d=[r['snapshots'][str(PRECURSOR)]['retrieval'] for r in probe['Dans']]
        p=mannwhitneyu(a,d,alternative='greater').pvalue
        L+=['','## P2, precursor (update 1,000)','',f'- Retrieval score: CTsqS {mean(a):.3f}, Dans {mean(d):.3f} (one-sided Mann-Whitney p = {p:.4f})',
            f"- **Outcome:** {'synchronization makes retrieval attention appear before the behaviour' if p<0.05 else 'no precursor at update 1,000'}"]
        pos=[r['snapshots'][str(PRECURSOR)]['retrieval'] for r in probe['CTsqS'] if r['bound']];neg=[r['snapshots'][str(PRECURSOR)]['retrieval'] for r in probe['CTsqS'] if not r['bound']]
        if pos and neg:L.append(f'- Within CTsqS, AUROC of the update-1,000 score for later binding: {sum((x>y)+0.5*(x==y) for x in pos for y in neg)/(len(pos)*len(neg)):.2f}')
    L+=['','## Retrieval score by snapshot (mean; bound / unbound runs)','','| Update | '+' | '.join(f'{c} bound | {c} unbound' for c in CELLS)+' |','|---:|'+'---:|'*(2*len(CELLS))]
    for st in SNAPSHOTS:
        cells=[]
        for c in CELLS:
            for want in (True,False):
                v=[r['snapshots'][str(st)]['retrieval'] for r in probe[c] if r['bound']==want and str(st) in r['snapshots']];cells.append(f'{mean(v):.3f} ({len(v)})' if v else '-')
        L.append(f'| {st} | '+' | '.join(cells)+' |')
    L+=['','## Final weights, every cell','','| Cell | Bound runs: retrieval | Unbound runs: retrieval | Key retrieval (bound) |','|---|---:|---:|---:|']
    for c in (*CELLS,*CONTROLS):
        v=[r for r in rs.values() if r['cell']==c];f=lambda rr,k:mean([x['snapshots'][str(BUDGET)][k] for x in rr])
        if v:L.append(f"| {c} | {f([r for r in v if r['bound']],'retrieval'):.3f} ({sum(r['bound'] for r in v)}) | {f([r for r in v if not r['bound']],'retrieval'):.3f} ({sum(not r['bound'] for r in v)}) | {f([r for r in v if r['bound']],'key_retrieval'):.3f} |")
    for c in ('CTsqS','CTsq1S'):
        for st in (PRECURSOR,BUDGET):
            v=[r['snapshots'][str(st)] for r in rs.values() if r['cell']==c and str(st) in r['snapshots'] and 'logit_advantage' in r['snapshots'][str(st)]]
            if v:L.append(f"- {c}, update {st}: logit advantage at the retrieval head, ordinary {mean([x['logit_advantage']['ordinary'] for x in v]):.2f}, synchronization {mean([x['logit_advantage']['sync'] for x in v]):.2f}; retrieval {mean([x['retrieval'] for x in v]):.3f}, without the synchronization query {mean([x['sync_off'] for x in v if 'sync_off' in x]):.3f}")
    (OUT/'PROBES.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--queue',action='store_true');g.add_argument('--analyse',action='store_true');g.add_argument('--summarize',action='store_true')
    a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    queue(a.gpu) if a.queue else analyse(a.gpu)


if __name__=='__main__':main()
