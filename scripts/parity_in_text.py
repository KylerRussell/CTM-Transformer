"""Stage A of research/PER_TOKEN_TICKS_DESIGN.md: parity inside text, with per-position tick budgets (2026-11-01).

On parity alone, training ticks that grow with the input (ceil(0.75 L) + U{0..8}) made tick-scaling length
extrapolation reliable for Dans (scripts/parity_ticks.py, _lent). A language model runs every position of a
sequence for the same tick count. This checks whether per-position budgets (rdt_recipe / ctm_transformer
``tick_budgets``) reproduce that result inside a language-model sequence, where most tokens are text.

Data: windows of SEQ tokens. Text comes from FineWeb-Edu (training text from train_000.bin, held-out text from
validation_000.bin), with tokens outside the TEXT_VOCAB most frequent ones (counted on the first 50M tokens of
train_000.bin) mapped to one unknown token. Each window carries one parity problem at a random position:
SEP b_1 .. b_n SEP answer, with bits and answer drawn from two reserved tokens; n is 1-40 in training.

Loss: the mean cross-entropy over text tokens (every target that is not part of the parity problem) plus the
cross-entropy of the answer, equally weighted. The answer is 1 token in SEQ, so without the second term the
parity signal would be diluted about SEQ times. The bits are random and are not trained.

Models: CTsq and Dans at width 128 (scripts/cmm_tasks.build; vocabulary TEXT_VOCAB + 4).
Training: 5,000 updates of batch BATCH, AdamW lr 1e-3, warmup 200, cosine to 10%, clip 1, as in the parity runs.

Arms (tick budgets during training; seeds 0-9):
- uniform: every position gets the step's tick count from the usual sampler (log-normal Poisson, mean 15, at
  most 48), as language models train now;
- pertoken: the parity problem's positions (the first SEP through the position predicting the answer) get
  ceil(0.75 n) + U{0..8}, a draw per example; every other position gets the step's sampler count;
- sequence (reported only, seeds 0-4): every position of an example gets ceil(0.75 n) + U{0..8}, as in _lent.
At evaluation every position runs the same fixed tick count (no labels or budgets are needed).

Evaluation: answer accuracy on 512 fixed examples per band of n (1-40, 41-60, 61-80, 81-120) at 1, 4, 16, 32,
64 and 128 ticks; held-out text loss (text tokens only, 1-40-bit problems inserted) on 256 windows at 16 ticks.

Rules, fixed before any run:
- A run extrapolates with ticks if its answer accuracy on 41-80 bits (mean of the 41-60 and 61-80 bands) at 64
  ticks is at least 80% and at least 20 points above its accuracy at 16 ticks.
- Per-position budgets work if a model's pertoken arm has at least 3 of 10 extrapolating seeds and at least 3
  more than its uniform arm, with its text loss (two-seed mean, seeds 0 and 1) within 0.02 of the uniform arm's.
- The synchronization queries matter if CTsq's pertoken count exceeds Dans's by at least 3.
- Decision (research/PER_TOKEN_TICKS_DESIGN.md): if per-position budgets work for neither model, Stage B is not run.

Usage: python -m scripts.parity_in_text --gpu N --runs CELL:SEED... | --summarize | --smoke CELL [--steps S]
"""
import argparse,json,math,os,random,time
from pathlib import Path
import numpy as np

OUT=Path('research/results/parity_in_text');CKPT=Path('research/runs/parity_in_text')
DATA=Path('research/data/pretrain/fineweb_edu_32k')
TEXT_VOCAB=4096;UNK,SEP,BIT0,BIT1=TEXT_VOCAB,TEXT_VOCAB+1,TEXT_VOCAB+2,TEXT_VOCAB+3;VOCAB=TEXT_VOCAB+4
SEQ,BATCH,STEPS,EXAMPLES=128,64,5000,512
CELLS=('CTsq_uniform','Dans_uniform','CTsq_pertoken','Dans_pertoken','CTsq_sequence','Dans_sequence');SEEDS=range(10)
BANDS=((1,40),(41,60),(61,80),(81,120));TICKS=(1,4,16,32,64,128);TEXT_WINDOWS=256


def remap():
    """Token id -> text-vocabulary id: the TEXT_VOCAB most frequent tokens of the first 50M training tokens, else UNK."""
    path=OUT/'vocabulary.npy'
    if path.exists():return np.load(path)
    counts=np.bincount(np.memmap(DATA/'train_000.bin',dtype=np.uint16,mode='r')[:50_000_000],minlength=32768)
    table=np.full(32768,UNK,dtype=np.int64);table[np.argsort(-counts,kind='stable')[:TEXT_VOCAB]]=np.arange(TEXT_VOCAB)
    OUT.mkdir(parents=True,exist_ok=True);np.save(path,table);return table


class Text:
    def __init__(self,split):
        self.tokens=np.memmap(DATA/('train_000.bin' if split=='train' else 'validation_000.bin'),dtype=np.uint16,mode='r');self.table=remap()

    def span(self,n,rng):
        start=rng.randrange(len(self.tokens)-n);return self.table[self.tokens[start:start+n]].tolist()


def example(text,n,rng):
    """(tokens [SEQ + 1], index of the first problem token, index of the answer token)."""
    bits=[rng.randrange(2) for _ in range(n)];problem=[SEP,*[BIT1 if b else BIT0 for b in bits],SEP,BIT1 if sum(bits)%2 else BIT0]
    p=rng.randint(0,SEQ+1-len(problem));words=text.span(SEQ+1-len(problem),rng)
    return words[:p]+problem+words[p:],p,p+len(problem)-1


def make_batch(text,lo,hi,n_examples,rng,arm=None,sampler_ticks=None):
    """x, y (text targets), answer targets [B, SEQ] (-100 elsewhere), tick budgets [B, SEQ] (for an arm), bits per example."""
    import torch
    x=torch.zeros((n_examples,SEQ),dtype=torch.long);y=torch.full((n_examples,SEQ),-100);ans=torch.full((n_examples,SEQ),-100)
    budgets=torch.full((n_examples,SEQ),sampler_ticks or 1);ns=[]
    for i in range(n_examples):
        n=rng.randint(lo,hi);seq,first,answer=example(text,n,rng);t=torch.tensor(seq);x[i]=t[:-1];ns.append(n)
        target=t[1:].clone();problem=slice(first,answer)  # positions first..answer-1 predict the problem's tokens
        y[i]=target;y[i,problem]=-100;ans[i,answer-1]=target[answer-1]
        if arm in ('pertoken','sequence'):
            scaled=math.ceil(0.75*n)+rng.randint(0,8)
            if arm=='sequence':budgets[i]=scaled
            else:budgets[i,first:answer]=scaled  # the first SEP through the position predicting the answer
    return x,y,ans,budgets,ns


def build(cell):
    from scripts.cmm_tasks import build as build_model
    return build_model(cell.split('_')[0],128,VOCAB)[0]


def answer_accuracy(model,text,lo,hi,ticks,device):
    import torch
    rng=random.Random(f'parity-text-band-{lo}-{hi}');hits=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(EXAMPLES//64):
            x,_,ans,_,_=make_batch(text,lo,hi,64,rng);x,ans=x.to(device),ans.to(device)
            pred=model(x,max_thought_steps=ticks)['logits'].argmax(-1);mask=ans.ne(-100)
            hits+=int(((pred==ans)&mask).sum())
    return 100*hits/EXAMPLES


def text_loss(model,device):
    import torch
    from torch.nn import functional as F
    text=Text('validation');rng=random.Random('parity-text-heldout');total=count=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(TEXT_WINDOWS//64):
            x,y,_,_,_=make_batch(text,1,40,64,rng);x,y=x.to(device),y.to(device)
            logits=model(x,max_thought_steps=16)['logits'].float()
            total+=float(F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100,reduction='sum'));count+=int(y.ne(-100).sum())
    return total/count


def train(cell,seed,device,steps=STEPS):
    import torch
    from torch.nn import functional as F
    from ctm_transformer.depth_sampling import lognormal_poisson
    from ctm_transformer.pretrain import lr_at,set_lr,step_depth
    out=OUT/f'{cell}_seed{seed}.json'
    if out.exists() and steps==STEPS:return
    arm=cell.split('_')[1];torch.manual_seed(seed);model=build(cell).to(device);sampler=lognormal_poisson(15,0.5,48)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    text=Text('train');rng=random.Random(f'train-parity-text-{seed}');start=time.time();curve=[]
    for step in range(steps):
        set_lr(opt,lr_at(step,1e-3,200,steps,0.1));T=step_depth(sampler,seed,step)
        x,y,ans,budgets,_=make_batch(text,1,40,BATCH,rng,arm,T);x,y,ans,budgets=x.to(device),y.to(device),ans.to(device),budgets.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,tick_budgets=budgets)['logits'].float()
            text_ce=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100)
            answer_ce=F.cross_entropy(logits.reshape(-1,VOCAB),ans.reshape(-1),ignore_index=-100)
            loss=text_ce+answer_ce
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        if (step+1)%250==0:curve.append({'step':step+1,'text':float(text_ce),'answer':float(answer_ce),'depth':int(budgets.max())})
    model.eval();result={'cell':cell,'seed':seed,'steps':steps,'seconds':time.time()-start,'curve':curve,
        'accuracy':{f'{lo}-{hi}':{str(t):answer_accuracy(model,Text('validation'),lo,hi,t,device) for t in TICKS} for lo,hi in BANDS},
        'text_loss_16':text_loss(model,device)}
    if steps==STEPS:
        CKPT.mkdir(parents=True,exist_ok=True);torch.save({'model':model.state_dict(),'cell':cell,'seed':seed},CKPT/f'{cell}_seed{seed}.pt')
        OUT.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=1)+'\n')
    g=result['accuracy'];print(cell,seed,f"{result['seconds']:.0f}s text {result['text_loss_16']:.3f} 1-40@16 {g['1-40']['16']:.1f}",
          '41-60 / 61-80 at 16/64:',[round(g[b][t],1) for b in ('41-60','61-80') for t in ('16','64')],flush=True)
    return result


def extrapolates(r):
    g=r['accuracy'];band=lambda t:(g['41-60'][t]+g['61-80'][t])/2
    return band('64')>=80 and band('64')-band('16')>=20,band('16'),band('64')


def summarize():
    res={(r['cell'],r['seed']):r for r in (json.loads(f.read_text()) for f in OUT.glob('*_seed*.json'))}
    L=['# Parity inside text: per-position tick budgets','','Protocol and rules: `scripts/parity_in_text.py` (docstring).','',
       '| Cell | Seeds | Extrapolating | Fitting 1-40 (>=95% at 16 ticks) | Text loss, seeds 0-1 | 41-80 at 16 / 64 ticks, per seed |','|---|---:|---:|---:|---:|---|']
    counts,texts={},{}
    for c in CELLS:
        rs=[res[(c,s)] for s in SEEDS if (c,s) in res];ex=[extrapolates(r) for r in rs];counts[c]=sum(e[0] for e in ex)
        tl=[res[(c,s)]['text_loss_16'] for s in (0,1) if (c,s) in res];texts[c]=sum(tl)/len(tl) if len(tl)==2 else None
        fit=sum(r['accuracy']['1-40']['16']>=95 for r in rs)
        L.append(f'| {c} | {len(rs)} | {counts[c]} | {fit} | {texts[c]:.3f} |' if texts[c] is not None else f'| {c} | {len(rs)} | {counts[c]} | {fit} | |')
        L[-1]+=' '+', '.join(f'{a:.0f}/{b:.0f}{"*" if e else ""}' for e,a,b in ex)+' |'
    L+=['','## Rule outcomes','']
    works={}
    for m in ('CTsq','Dans'):
        pt,un=f'{m}_pertoken',f'{m}_uniform'
        ok=counts[pt]>=3 and counts[pt]-counts[un]>=3 and texts[pt] is not None and texts[un] is not None and abs(texts[pt]-texts[un])<=0.02
        works[m]=ok;L.append(f'- **{m}:** pertoken {counts[pt]} vs uniform {counts[un]}; text loss {texts[pt]} vs {texts[un]}; per-position budgets work: {"yes" if ok else "no"}')
    L.append(f'- **Synchronization queries matter (CTsq pertoken count exceeds Dans by at least 3):** {"yes" if counts["CTsq_pertoken"]-counts["Dans_pertoken"]>=3 else "no"}')
    L.append(f'- **Decision:** '+('Stage B may run' if any(works.values()) else 'per-position budgets work for neither model; Stage B is not run'))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'PARITY_IN_TEXT.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='*',default=[])
    p.add_argument('--summarize',action='store_true');p.add_argument('--smoke');p.add_argument('--steps',type=int,default=50);a=p.parse_args()
    if a.summarize:summarize();return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch;device=torch.device('cuda')
    if a.smoke:train(a.smoke,0,device,a.steps);return
    for r in a.runs:c,s=r.split(':');train(c,int(s),device)


if __name__=='__main__':main()
