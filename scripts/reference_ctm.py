"""Reference CTM: the official Continuous Thought Machine on our tasks, and its published parity result (2026-11-23,
after external review, concern 2).

The review: no real CTM is run, so the paper cannot say whether the merge loses something the CTM has. This runs the
official model (github.com/SakanaAI/continuous-thought-machines, commit 4a6c9c3, cloned to ~/Documents/ctm-reference;
one change, a typo in a print statement of tasks/parity/train.py, args.model -> args.model_type) in its own venv
(~/.venvs/ctm-official: the research venv's torch 2.6 through a .pth file, plus the official code's extra packages).

Model (`models/ctm.py`, unchanged): d_model 512, d_input 256, 4 heads, 32 neurons each for action and output
synchronization (random selection: 528 pairs each), linear synapses (depth 1), deep neuron-level models (history 10,
hidden 16), 32 internal ticks, 1D rotational positional embedding (the parity setup's choices, resized): 1.52M
parameters on recall (parity 1.48M; copy 2.02M and sort 1.91M, whose outputs cover 64 and 50 positions), against
1.59-1.65M for T8, Dans and CTsqS at width 128. Input: the task's token sequence without its
answer tokens (embedded by a token table in place of the parity backbone's two-entry table), padded with PAD; the CTM
attends over all of it each tick. Output: every answer position at once (prediction reshaper [positions, classes], the
classes being the task's answer symbols: the 16 values, or the 2 bits for parity),
read from the output synchronization, as the official parity model reads all 64 positions.
Loss: the official CTM loss (utils/losses.py, parity_loss): per example, the cross-entropy averaged over its answer
positions, then the mean of that at the minimum-loss tick and at the most certain tick. Accuracy at the most certain
tick (as the official code reports); the final tick is also reported.
Training: our recall protocol for every model (scripts/recall_memory.py, recipe `cur`, 10,000 updates, the same data
per seed, AdamW betas 0.9/0.95, weight decay 0.1, warmup 200, cosine to 10%, clip 1, batch 128 as 2 micro-batches of
64), in float32 (as the official parity runs). Evaluation: the same 512 fixed ID and OOD examples as our models.

Phase R1, learning rate (recall, seeds 0-4): lr 1e-4 (the CTM paper's), 3e-4, 1e-3. The selected lr has the most bound
runs (ID query accuracy at least 90%), ties broken by mean ID accuracy, then by the lower lr.
Phase R2, test (recall, seeds 5-24, the selected lr). Reported against the existing runs on the same seeds (CTsqS 15/20,
Dans 2/20, T8 19/20), two-sided Fisher exact tests on bound counts:
- The merge loses binding the CTM has if the reference CTM binds more often than CTsqS (p < 0.05).
- The CTM binds less reliably than the CTM Transformer if CTsqS binds more often (p < 0.05). Otherwise neither.
- The CTM against T8 and Dans is reported by the same test.
Phase R3, the other CMM-grid tasks (scripts/cmm_tasks.py: copy, sort, parity; its protocol: 5,000 updates, lr from
R1, seeds 0-2). Reported against the grid's models (sequence accuracy, ID and OOD), by the grid's rule (a 3-seed mean
at least 5 points higher and higher on all 3 seeds).
Phase P, reproduction (runs after the LM revision queue: it needs more than 4 GB beside a CTM LM run's 17.75):
the official tasks/parity/scripts/train_ctm_75_25.sh unchanged (75 ticks, memory 25, d_model 1024, 200,000 updates)
on seeds 0, 1, 2. The paper reports 3 runs, 2 of them perfect. The reproduction succeeds if at least 2 of 3 seeds
reach at least 99% per-position test accuracy at the most certain tick; partly if 1 does; fails if none does.
If it fails, the reference-CTM results (R1-R3) are reported as unvalidated.

Usage: ~/.venvs/ctm-official/bin/python -m scripts.reference_ctm --phase R1|R2|R3 --gpu N --runs TASK:SEED:LR...
       | --queue --gpu N     (a worker: R1, then R2 and R3 once R1 has selected its lr)
       | --summarize        (run with PYTHONPATH=.:~/Documents/ctm-reference)
"""
import argparse,json,math,os,random,sys,time
from pathlib import Path

REF=Path.home()/'Documents/ctm-reference';sys.path.insert(1,str(REF))
OUT=Path('research/results/reference_ctm');CKPT=Path('research/runs/reference_ctm')
VOCAB=96;PAD=0;MICRO=64;BATCH=128;RESUME_EVERY=500;CHECK_EVERY=1000
POSITIONS={'recall':4,'copy':64,'sort':50,'parity':1}
CLASSES={t:(list(range(84,86)) if t=='parity' else list(range(4,20))) for t in POSITIONS}  # cmm_tasks SYM and BIT
R1_LRS=(1e-4,3e-4,1e-3);R1_SEEDS=tuple(range(5));R2_SEEDS=tuple(range(5,25));R3_SEEDS=(0,1,2)


def build(task):
    import torch.nn as nn
    from models.ctm import ContinuousThoughtMachine
    P,C=POSITIONS[task],len(CLASSES[task])
    net=ContinuousThoughtMachine(iterations=32,d_model=512,d_input=256,heads=4,n_synch_out=32,n_synch_action=32,synapse_depth=1,memory_length=10,
        deep_nlms=True,memory_hidden_dims=16,do_layernorm_nlm=False,backbone_type='parity_backbone',positional_embedding_type='custom-rotational-1d',
        out_dims=P*C,prediction_reshaper=[P,C],dropout=0.0,neuron_select_type='random',n_random_pairing_self=0)
    table=nn.Embedding(VOCAB,net.get_d_backbone());net.backbone.embedding=table
    net.backbone.forward=lambda x,e=table:e(x).transpose(1,2)  # tokens in place of the parity backbone's two-entry table
    return net


def rows_to_batch(rows,P,classes):
    """Token sequences with answer masks -> (inputs without answers, targets [n, P] with -100 padding)."""
    import torch
    xs,ys=[],[]
    for s,m in rows:
        ans=[classes.index(s[p+1]) for p in range(len(s)-1) if m[p]];keep=[t for i,t in enumerate(s) if not (i>0 and m[i-1])]
        xs.append(keep);ys.append(ans)
    L=max(len(x) for x in xs);x=torch.full((len(xs),L),PAD);y=torch.full((len(xs),P),-100)
    for i,(a,b) in enumerate(zip(xs,ys)):x[i,:len(a)]=torch.tensor(a);y[i,:len(b)]=torch.tensor(b[:P])
    return x,y


def ctm_loss(pred,cert,y):
    """The official parity_loss with answer masking: pred [B, P, V, T], y [B, P]."""
    import torch,torch.nn.functional as F
    B,P,V,T=pred.shape;mask=y.ne(-100)
    ce=F.cross_entropy(pred.permute(0,2,1,3),y[...,None].expand(B,P,T).clamp_min(0),reduction='none')  # [B, P, T]
    per=(ce*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)  # [B, T]
    i1=per.argmin(1);i2=cert[:,1].argmax(-1);b=torch.arange(B,device=y.device)
    return (per[b,i1].mean()+per[b,i2].mean())/2


def examples(task,split,n,rng):
    if task=='recall':
        from scripts.recall_memory import SPLITS,example
        lo,hi=SPLITS[split];return [example(rng.randint(lo,hi),False,rng) for _ in range(n)]
    from scripts.cmm_tasks import SPLITS,example
    lo,hi=SPLITS[task][split];return [example(task,rng.randint(lo,hi),rng) for _ in range(n)]


def evaluate(net,task,split,device,n=512):
    """Answer accuracy (%) and sequence accuracy (%) at the most certain and the final tick, on our fixed examples."""
    import torch
    P=POSITIONS[task];out={'certain':[0,0,0],'final':[0,0,0]}
    if task=='recall':  # the same draws as scripts/recall_memory.accuracy
        from scripts.recall_memory import SPLITS,example
        rng=random.Random(f'recall-memory-{split}-False');lo,hi=SPLITS[split];chunks=[]
        for _ in range(n//64):ns=[rng.randint(lo,hi) for _ in range(64)];chunks.append([example(k,False,rng) for k in ns])
    else:  # the same draws as scripts/cmm_tasks.evaluate
        rng=random.Random(f'eval-{task}-{split}');chunks=[examples(task,split,64,rng) for _ in range(n//64)]
    with torch.no_grad():
        for rows in chunks:
            x,y=rows_to_batch(rows,P,CLASSES[task]);x,y=x.to(device),y.to(device);pred,cert,_=net(x);pred=pred.reshape(len(rows),P,len(CLASSES[task]),-1)
            mask=y.ne(-100);b=torch.arange(len(rows),device=device)
            for k,idx in (('certain',cert[:,1].argmax(-1)),('final',torch.full((len(rows),),pred.shape[-1]-1,device=device))):
                guess=pred[b,:,:,idx].argmax(-1);right=(guess==y)&mask
                out[k][0]+=int(right.sum());out[k][1]+=int(mask.sum());out[k][2]+=int((right.sum(1)==mask.sum(1)).sum())
    return {k:{'answer':100*v[0]/v[1],'sequence':100*v[2]/n} for k,v in out.items()}


def train(task,seed,lr,phase,device):
    import torch
    from ctm_transformer.pretrain import lr_at,set_lr
    name=f'{phase}_{task}_lr{lr:g}_seed{seed}';out=OUT/f'{name}.json'
    if out.exists():return
    steps=10000 if task=='recall' else 5000;P=POSITIONS[task]
    torch.manual_seed(seed);net=build(task).to(device)
    rows=examples(task,'train' if task!='recall' else 'train',2,random.Random(0));x,_=rows_to_batch(rows,P,CLASSES[task]);net(x.to(device))  # initialise lazy layers
    opt=torch.optim.AdamW(net.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1)
    rng=random.Random(f'recall-memory-train-{seed}' if task=='recall' else f'train-{task}-{seed}');curve=[];first=0;start=time.time()
    resume=CKPT/f'{name}.resume.pt'
    if resume.exists():
        st=torch.load(resume,map_location=device,weights_only=False);net.load_state_dict(st['model']);opt.load_state_dict(st['opt'])
        first=st['step'];curve=st['curve'];rng.setstate(st['rng']);torch.set_rng_state(st['torch_rng'].cpu());torch.cuda.set_rng_state(st['cuda_rng'].cpu());start-=st['seconds']
    for step in range(first,steps):
        if step and step%RESUME_EVERY==0:
            CKPT.mkdir(parents=True,exist_ok=True);tmp=resume.with_suffix('.tmp')
            torch.save({'model':net.state_dict(),'opt':opt.state_dict(),'step':step,'curve':curve,'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state(),'seconds':time.time()-start},tmp);os.replace(tmp,resume)
        set_lr(opt,lr_at(step,lr,200,steps,0.1))
        if task=='recall':
            from scripts.recall_memory import example
            hi=min(16,2+step//700);rows=[example(rng.randint(2,hi),False,rng) for _ in range(BATCH)]
        else:rows=examples(task,'train',BATCH,rng)
        norm=0.0
        for k in range(0,BATCH,MICRO):
            x,y=rows_to_batch(rows[k:k+MICRO],P,CLASSES[task]);x,y=x.to(device),y.to(device);pred,cert,_=net(x)
            loss=ctm_loss(pred.reshape(len(x),P,len(CLASSES[task]),-1),cert,y)*len(x)/BATCH;loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.0));opt.step();opt.zero_grad(set_to_none=True)
        if (step+1)%CHECK_EVERY==0:
            net.eval();e=evaluate(net,task,'id',device,256);net.train()
            curve.append({'step':step+1,'loss':float(loss)*BATCH/len(x),'grad_norm':norm,'id_check':e['certain']['answer'],'id_check_final':e['final']['answer']})
            print(name,curve[-1],flush=True)
    net.eval();reached=next((c['step'] for c in curve if c['id_check']>=90),None)
    result={'task':task,'seed':seed,'lr':lr,'phase':phase,'steps':steps,'seconds':time.time()-start,'parameters':sum(p.numel() for p in net.parameters()),
            'updates_to_90':reached,'curve':curve,'id':evaluate(net,task,'id',device),'ood':evaluate(net,task,'ood',device)}
    OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n');resume.unlink(missing_ok=True)
    torch.save(net.state_dict(),CKPT/f'{name}.final.pt')
    print(name,'id',result['id'],'ood',result['ood'],f"{result['seconds']:.0f}s",flush=True)


def summarize():
    from scipy.stats import fisher_exact
    load=lambda f:json.loads(f.read_text()) if f.exists() else None;mean=lambda v:sum(v)/len(v)
    L=['# Reference CTM (official code)','Protocol and rules: `scripts/reference_ctm.py`.','','## R1, recall learning rate (seeds 0-4)','',
       '| lr | Bound | ID (most certain tick) | ID (final tick) |','|---:|---:|---:|---:|'];score={}
    for lr in R1_LRS:
        rs=[load(OUT/f'R1_recall_lr{lr:g}_seed{s}.json') for s in R1_SEEDS]
        if all(rs):
            b=sum(r['id']['certain']['answer']>=90 for r in rs);i=mean([r['id']['certain']['answer'] for r in rs]);score[lr]=(b,i,-lr)
            L.append(f"| {lr:g} | {b}/5 | {i:.1f} | {mean([r['id']['final']['answer'] for r in rs]):.1f} |")
    best=max(score,key=score.get) if len(score)==len(R1_LRS) else None
    L.append(f'\nSelected lr: {best:g}' if best else '\nSelected lr: pending')
    if best:(OUT/'R1_LR.json').write_text(json.dumps({'lr':best})+'\n')
    if best:
        rs=[load(OUT/f'R2_recall_lr{best:g}_seed{s}.json') for s in R2_SEEDS]
        if all(rs):
            k=sum(r['id']['certain']['answer']>=90 for r in rs)
            ref={'CTsqS':15,'Dans':2,'T8':19}  # seeds 5-24, research/results/recall_memory/CONTROL.md
            L+=['','## R2, recall test (seeds 5-24)','',f"Reference CTM: {k}/20 bound; ID {mean([r['id']['certain']['answer'] for r in rs]):.1f}; "
                f"OOD {mean([r['ood']['certain']['answer'] for r in rs]):.1f}",'']
            for c,v in ref.items():
                p=fisher_exact([[k,20-k],[v,20-v]])[1];L.append(f'- Reference CTM {k}/20 against {c} {v}/20: two-sided Fisher p = {p:.4f}'+(' (differs)' if p<0.05 else ''))
            p=fisher_exact([[k,20-k],[15,5]])[1]
            L.append('- **Outcome:** '+('the merge loses binding the CTM has' if k>15 and p<0.05 else 'the CTM binds less reliably than the CTM Transformer' if k<15 and p<0.05 else 'neither (no difference from CTsqS)'))
    rows3=[(t,s,load(OUT/f'R3_{t}_lr{best:g}_seed{s}.json')) for t in ('copy','sort','parity') for s in R3_SEEDS] if best else []
    if rows3 and all(r for *_,r in rows3):
        L+=['','## R3, other grid tasks (seeds 0-2, sequence accuracy, most certain tick)','','| Task | ID | OOD |','|---|---:|---:|']
        for t in ('copy','sort','parity'):
            rs=[r for tt,_,r in rows3 if tt==t];L.append(f"| {t} | {mean([r['id']['certain']['sequence'] for r in rs]):.1f} | {mean([r['ood']['certain']['sequence'] for r in rs]):.1f} |")
    (OUT/'REFERENCE_CTM.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def queue(gpu):
    import fcntl,torch
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(gpu));device=torch.device('cuda');CKPT.mkdir(parents=True,exist_ok=True)
    while True:
        with open(CKPT/'.queue.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (OUT/'R1_LR.json').exists() and all((OUT/f'R1_recall_lr{lr:g}_seed{s}.json').exists() for lr in R1_LRS for s in R1_SEEDS):summarize()
            runs=[('R1','recall',s,lr) for s in R1_SEEDS for lr in R1_LRS]
            if (OUT/'R1_LR.json').exists():
                lr=json.loads((OUT/'R1_LR.json').read_text())['lr'];runs+=[('R2','recall',s,lr) for s in R2_SEEDS]+[('R3',t,s,lr) for t in ('copy','sort','parity') for s in R3_SEEDS]
            todo=[r for r in runs if not (OUT/f'{r[0]}_{r[1]}_lr{r[3]:g}_seed{r[2]}.json').exists() and not (CKPT/f'{r[0]}_{r[1]}_lr{r[3]:g}_seed{r[2]}.claimed').exists()]
            if todo:(CKPT/'{}_{}_lr{:g}_seed{}.claimed'.format(todo[0][0],todo[0][1],todo[0][3],todo[0][2])).write_text(f'gpu {gpu}\n')
        if not todo:
            if (OUT/'R1_LR.json').exists():return summarize()
            time.sleep(600);continue  # R1 runs still finishing elsewhere
        ph,t,s,lr=todo[0];train(t,s,lr,ph,device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('R1','R2','R3'));p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--summarize',action='store_true');p.add_argument('--queue',action='store_true');a=p.parse_args()
    if a.summarize:return summarize()
    if a.queue:return queue(a.gpu)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    for r in a.runs:t,s,lr=r.split(':');train(t,int(s),float(lr),a.phase,device)


if __name__=='__main__':main()
