"""Write the 500M pretraining arm configs (research/configs/pretrain/*.json).

Arms (non-embedding parameters about 440-520M; untied 32k embeddings add the rest):
  transformer      d 1280, 24 layers
  rdt_heavy (A)    d 2304, prelude/core/coda 2/4/2: most parameters recur (as in Huginn)
  rdt_aware (B)    d 1280, 11/2/11: most parameters are applied once
  ctm_heavy (A)    d 1536, 12-layer backbone, 4,096 neurons and pairs: a large per-tick synapse
  ctm_aware (B)    d 1280, 24-layer backbone, 2,048 neurons and pairs: most parameters applied once
RDT trains with the validated randomized depth (log-normal Poisson, mean about 16, cap 48);
CTM-LM with 16 ticks, adapted for language modelling (a token-conditioned start state and the
mean cross-entropy over ticks 4, 8, 12 and 16 in place of CTM's loss; see CTM_ADAPTATIONS). Every arm: RoPE, sequence length 1,024,
262,144 tokens per optimizer step, bf16 autocast, AdamW (0.9, 0.95), weight decay 0.1.
Transformer and RDT weights use the width-scaled initialization std = sqrt(2 / (5 d)) (as in Huginn).
With the small-scale recipe's fixed 0.02, the d 2,304 RDT core's backward pass grows about 1.5x per
recurrence at initialization (gradient norm 21 at depth 1, 5.3 million at depth 32); with the scaled
std it is flat in depth. CTM-LM keeps the reference CTM initialization.
RDT input embeddings are initialized at std 1 (2026-10-02). With the small std, the sandwich-normed
RDT collapsed to the unigram level within about 75 updates: its prelude erased token identity, and
lowering the learning rate did not help. Unit-scale embeddings fixed it
(research/results/ctm_lm_probes/PROBES.md). The pre-norm Transformer learns normally with the small std.
"""
import argparse,json,math
from pathlib import Path

DATA='research/data/pretrain/fineweb_edu_32k/manifest.json';OUT=Path('research/configs/pretrain')
SEQ,GLOBAL_SEQS=1024,256
SAMPLER={'kind':'lognormal_poisson','mean':15,'sigma':0.5,'maximum':48}
# CTM-LM adapted for language modelling (user decisions 2026-10-02 and 2026-10-03; research/CTM_LM_DESIGN.md):
# a token-conditioned start state z_0 = z_init + W f_i (D) and the mean cross-entropy over ticks 4, 8, 12 and 16
# (sparse-tick loss) in place of CTM's loss. The faithful CTM-LM learned only token frequencies at this scale;
# its failure is reported as a result. Chosen over B + C (6.731), D + mean-tick (6.392, 0.61x the throughput)
# and D + final (6.313, untrained early ticks) at 6.415 nats in the 9.8M-token probes.
CTM_ADAPTATIONS=['token_start','sparse_tick_loss']
ARMS={
 'transformer':dict(family='transformer',model=dict(d_model=1280,n_layers=24,ffn_hidden_dim=3456),micro_batch=16,offload=False),
 'rdt_heavy':dict(family='rdt',model=dict(d_model=2304,prelude_layers=2,core_layers=4,coda_layers=2,ffn_hidden_dim=6144),micro_batch=2,offload=True,
                  depth_sampler=SAMPLER,eval_depth=16,eval_depths=[16,32],embedding_init_std=1.0),
 'rdt_aware':dict(family='rdt',model=dict(d_model=1280,prelude_layers=11,core_layers=2,coda_layers=11,ffn_hidden_dim=3456),micro_batch=8,offload=False,
                  depth_sampler=SAMPLER,eval_depth=16,eval_depths=[16,32],embedding_init_std=1.0),
 'ctm_heavy':dict(family='ctm_lm',model=dict(d_model=1536,n_layers=12,d_latent=4096,sync_sparse_pairs=4096,history_len=8,nlm_hidden_dim=32,max_thought_steps=16),
                  micro_batch=1,offload=True,eval_depth=16,ctm_adaptations=CTM_ADAPTATIONS),
 'ctm_aware':dict(family='ctm_lm',model=dict(d_model=1280,n_layers=24,d_latent=2048,sync_sparse_pairs=2048,history_len=8,nlm_hidden_dim=32,max_thought_steps=16),
                  micro_batch=2,offload=True,eval_depth=16,ctm_adaptations=CTM_ADAPTATIONS)}


def at_width(arm,d):
    """The arm with width d: same layer layout, same ratios of FFN width and CTM neurons and pairs to d (rounded to 64)."""
    m=dict(arm['model']);r=lambda key:max(64,round(m[key]/m['d_model']*d/64)*64)
    for key in ('ffn_hidden_dim','d_latent','sync_sparse_pairs'):
        if key in m:m[key]=r(key)
    m['d_model']=d;return {**arm,'model':m}


def config(name,arm,tokens,lr,world=2):
    micro=arm['micro_batch'];assert GLOBAL_SEQS%(world*micro)==0
    steps=round(tokens/(SEQ*GLOBAL_SEQS))
    run=json.loads(json.dumps({k:v for k,v in arm.items() if k not in ('micro_batch','offload')}))
    if arm['family']!='ctm_lm':run['model']['init_std']=round(math.sqrt(2/(5*arm['model']['d_model'])),6)
    run.update(name=name,seed=1234,data=DATA,run_directory=f'research/runs/pretrain/{name}',
        train=dict(seq_len=SEQ,micro_batch=micro,accumulation=GLOBAL_SEQS//(world*micro),total_steps=steps,warmup_steps=max(100,steps//20),lr=lr,lr_floor=0.1,
                   betas=[0.9,0.95],weight_decay=0.1,grad_clip=1.0,checkpointing=True,compile=True,offload_optimizer=arm['offload'],cpu_threads=16,
                   eval_interval=250,eval_windows=256,eval_micro_batch=1 if arm['family']=='ctm_lm' else 4,log_interval=10,checkpoint_interval=50,snapshot_interval=1000))
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tokens',type=float,default=1e9);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--suffix',default='');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for name,arm in ARMS.items():
        run=config(name+a.suffix,arm,a.tokens,a.lr);(OUT/f'{name}{a.suffix}.json').write_text(json.dumps(run,indent=2)+'\n')
        print(name+a.suffix,run['train']['total_steps'],'steps',run['train']['accumulation'],'accumulation')

if __name__=='__main__':main()
