"""Learning-rate scaling sweep for the 500M pretraining arms (research/LR_SCALING.md).

For every arm, a ladder of three narrower copies (same layer layout and width
ratios, `make_pretrain_configs.at_width`) is trained on 100M FineWeb-Edu tokens
with the 500M runs' global batch (262,144 tokens), schedule and optimizer, over
a lattice of learning rates lr = 1e-3 * 2^k. The grid is adaptive:

* the smallest width starts at k in {-1, 0, 1}; each larger width starts at
  the previous width's best k and its two neighbours;
* while the best k (lowest held-out loss; a diverged run counts as infinite)
  is at an edge of the grid, the grid is extended by one step past that edge
  (k stays within [-6, 4]);
* a width is resolved once its best k has evaluated neighbours on both sides;
* a run that ends at a held-out loss of 7.0 nats or more (no better than about
  the unigram model's 7.62) counts as collapsed, with infinite loss;
* rates above one that diverged or collapsed are not run (infinite loss), and
  a width whose every rate fails down to k = -6 is blocked, with no optimum.

At the smallest width the Transformer and both compute-aware arms are also
swept at 4x the tokens (400M), which measures how the optimum moves with
training length. Runs use one GPU each, two at a time, and evaluate on
validation windows disjoint from those the 500M runs report.

Last stage, the CTM-augmented RDT screen (research/CTM_RDT_SCREEN.md): once
the compute-aware RDT's middle width is resolved, the RDT and each CTM mechanism
(`ctm_transformer/ctm_rdt.py`) train at that width's best learning rate with
seeds 1234 and 1235. An arm name `rdt_aware+<mechanism>` adds the mechanism.

Arms listed in `research/results/lr_sweep/hold.json` (a JSON list) are not
launched while listed; running runs are unaffected. An entry "gpu:N" keeps
the sweep off GPU N. It is read on every cycle.

Restart-safe and idempotent: a relaunch resumes interrupted runs from their
checkpoints. A run that exits nonzero is recorded as `failure.json` and never
retried; it blocks the larger widths of its arm.

    python -m scripts.lr_sweep            # supervise (see research/launch_lr_sweep.sh)
    python -m scripts.lr_sweep --plan     # print the runs the current results call for
"""
import argparse,datetime,fcntl,json,math,os,subprocess,sys,time
from pathlib import Path
from scripts.make_pretrain_configs import ARMS,at_width,config

RUNS=Path('research/runs/lr_sweep');RESULTS=Path('research/results/lr_sweep')
AUTOSTART=Path.home()/'.config/autostart/ctm-lr-sweep.desktop'
LADDER={'transformer':[384,512,768],'rdt_aware':[384,512,768],'ctm_aware':[384,512,768],'rdt_heavy':[704,896,1408],'ctm_heavy':[448,640,896]}
MICRO={'transformer':[32,32,32],'rdt_aware':[16,16,16],'ctm_aware':[8,8,4],'rdt_heavy':[16,8,4],'ctm_heavy':[8,4,4]}
HORIZON_ARMS=('transformer','rdt_aware','ctm_aware')
PRIORITY=('ctm_heavy','rdt_heavy','ctm_aware','rdt_aware','transformer')
TOKENS=100_000_000;K_MIN,K_MAX=-6,4;EVAL_OFFSET=40_000
# A run whose final held-out loss is at least this has not learned beyond token frequencies (the unigram
# model scores 7.62 nats on these windows; every arm reaches about 5 nats or lower by 100M tokens when it trains).
# It counts as collapsed, with infinite loss, like a divergence. Added 2026-10-02 after RDT-heavy at lr 1e-3
# collapsed to the unigram level mid-run and every CTM-heavy rate ended at 7.66-7.73.
COLLAPSED=7.0
METRIC={'transformer':'final','rdt':'depth_16','ctm_lm':'most_certain_tick'}
SEED=1234;SCREEN_ARM,SCREEN_WIDTH=('rdt_aware',512);SCREEN_SEEDS=(1234,1235)
SCREEN_VARIANTS=('rdt_aware','rdt_aware+sync_query','rdt_aware+sync_readout','rdt_aware+learned_init')


def lr_of(k):return 1e-3*2.0**k
def run_name(arm,d,k,h,seed=SEED):return f'{arm}_d{d}_h{h}_k{k:+d}'+('' if seed==SEED else f'_s{seed}')


def sweep_config(arm,d,k,h,seed=SEED):
    base,*mechanisms=arm.split('+')
    a=at_width(ARMS[base],d);a={**a,'micro_batch':MICRO[base][LADDER[base].index(d)],'offload':False}
    name=run_name(arm,d,k,h,seed);run=config(name,a,TOKENS*h,lr_of(k),world=1);t=run['train'];steps=t['total_steps']
    run['run_directory']=str(RUNS/name)
    if mechanisms:run['mechanisms']=mechanisms
    if seed!=SEED:run['seed']=seed
    if run['family']=='rdt':run['eval_depths']=[run['eval_depth']]
    t.update(eval_interval=max(1,steps//4),eval_windows=128,final_eval_windows=1024,eval_offset=EVAL_OFFSET,checkpoint_interval=50,
             snapshot_interval=10**12,delete_checkpoint_on_complete=True)
    return run


def outcome(name):
    """'complete' with the held-out loss, 'diverged' or 'collapsed' (infinite loss), 'failed', or None (not finished)."""
    d=RUNS/name
    if (d/'failure.json').exists():return 'failed',None
    if (d/'diverged.json').exists():return 'diverged',math.inf
    if (d/'complete.json').exists():
        run=json.loads((d/'config.json').read_text());rows=[json.loads(r) for r in (d/'metrics.jsonl').read_text().splitlines() if r.strip()]
        loss=rows[-1]['validation'][METRIC[run['family']]]
        return ('collapsed',math.inf) if loss>=COLLAPSED else ('complete',loss)
    return None,None


def grid(arm,d,h,start):
    """(runs this width needs now, best k once resolved else None, blocked).

    Rates above the lowest rate that diverged or collapsed are not run: instability only grows with the
    learning rate, so they count as infinite loss. A width whose every rate fails is blocked (no optimum)."""
    ks=set(start)
    while True:
        states={k:outcome(run_name(arm,d,k,h)) for k in ks}
        bad=[k for k,(s,_) in states.items() if s in ('diverged','collapsed')]
        ks={k for k in ks if not bad or k<=min(bad)}
        if any(states[k][0]=='failed' for k in ks):return sorted(ks),None,True
        if any(states[k][0] is None for k in ks):return sorted(ks),None,False
        best=min(ks,key=lambda k:(states[k][1],k))
        if best==min(ks) and best>K_MIN:ks.add(best-1)
        elif best==max(ks) and best<K_MAX and not bad:ks.add(best+1)
        elif states[best][1]==math.inf:return sorted(ks),None,True
        else:return sorted(ks),best,False


def plan():
    """Every run the current results call for, as (arm, d, k, h, seed), and whether any arm is blocked."""
    want,blocked,screen_k=[],False,None
    for arm in PRIORITY:
        start=(-1,0,1)
        for i,d in enumerate(LADDER[arm]):
            ks,best,stop=grid(arm,d,1,start);want+=[(arm,d,k,1,SEED) for k in ks];blocked|=stop
            if best is None:break
            if (arm,d)==(SCREEN_ARM,SCREEN_WIDTH):screen_k=best
            if i==0 and arm in HORIZON_ARMS:
                hks,_,hstop=grid(arm,d,4,(best-1,best,best+1));want+=[(arm,d,k,4,SEED) for k in hks];blocked|=hstop
            start=(best-1,best,best+1)
    if screen_k is not None:
        want+=[r for v in SCREEN_VARIANTS for seed in SCREEN_SEEDS if (r:=(v,SCREEN_WIDTH,screen_k,1,seed)) not in want]
    return want,blocked


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(msg):print(f'{now()} {msg}',flush=True)


def gpu_free(index):
    out=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(index)],capture_output=True,text=True)
    return out.returncode==0 and int(out.stdout.strip())<1000


def alive(name):
    """The GPU of the running worker for `name` (possibly started by an earlier supervisor), else None."""
    pid_file=RUNS/name/'worker.pid'
    if not pid_file.exists():return None
    try:
        proc=Path(f'/proc/{int(pid_file.read_text())}')
        if f'{name}/config.json' not in (proc/'cmdline').read_bytes().decode(errors='ignore'):return None
        env=dict(e.split('=',1) for e in (proc/'environ').read_bytes().decode(errors='ignore').split('\0') if '=' in e)
        return int(env['CUDA_VISIBLE_DEVICES'])
    except (OSError,ValueError,KeyError):return None


def launch(spec,gpu):
    name=run_name(*spec);out=RUNS/name;out.mkdir(parents=True,exist_ok=True);cfg=out/'config.json'
    run=sweep_config(*spec)
    if cfg.exists():assert json.loads(cfg.read_text())==run,f'{cfg} differs from the current sweep definition'
    else:cfg.write_text(json.dumps(run,indent=2)+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
    with open(out/'train.log','a') as f:
        p=subprocess.Popen([sys.executable,'-u','-m','scripts.pretrain','--config',str(cfg)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    (out/'worker.pid').write_text(str(p.pid));log(f'launch {name} on GPU {gpu} (pid {p.pid})');return p


def supervise():
    RESULTS.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    lock=open(RESULTS/'.supervisor.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:log('another supervisor holds the lock');return
    log(f'supervisor start (pid {os.getpid()})');workers={};was_idle=False  # gpu -> (name, Popen or None for an adopted worker)
    for name in [run_name(*r) for r in plan()[0]]:
        gpu=alive(name)
        if gpu is not None:log(f'{name} is still running on GPU {gpu} from an earlier supervisor; waiting for it');workers[gpu]=(name,None)
    while True:
        for gpu,(name,p) in list(workers.items()):
            code=p.poll() if p else (None if alive(name) is not None else 0)
            if code is None:continue
            del workers[gpu];state,_=outcome(name)
            if state is None and code!=0:
                if code<0:log(f'{name} interrupted by signal {-code}; it resumes on the next launch')
                else:(RUNS/name/'failure.json').write_text(json.dumps({'returncode':code,'time_utc':now()})+'\n');log(f'{name} FAILED (exit {code}); see train.log')
            else:log(f'{name} finished: {state}')
        want,blocked=plan();busy={n for n,_ in workers.values()}
        held=set(json.loads((RESULTS/'hold.json').read_text())) if (RESULTS/'hold.json').exists() else set()
        pending=[r for r in want if outcome(run_name(*r))[0] is None and run_name(*r) not in busy and r[0].split('+')[0] not in held]
        for gpu in (0,1):
            if not pending:break
            if gpu in workers or f'gpu:{gpu}' in held or not gpu_free(gpu):continue
            r=pending.pop(0);workers[gpu]=(run_name(*r),launch(r,gpu))
        state={'time_utc':now(),'running':sorted(n for n,_ in workers.values()),'pending':[run_name(*r) for r in pending],
               'complete':sum(outcome(run_name(*r))[0] in ('complete','diverged','collapsed') for r in want),'blocked':blocked,'held':sorted(held)}
        tmp=RESULTS/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(RESULTS/'state.json')
        idle=not workers and not pending and bool(held)
        if idle and not was_idle:log(f'idle: only held arms remain ({sorted(held)})')
        was_idle=idle
        if not workers and not pending and not any(not h.startswith('gpu:') for h in held):
            if blocked:log('stopped: a failed run blocks part of the sweep');return
            subprocess.run([sys.executable,'-m','scripts.summarize_lr_sweep'],check=True)
            subprocess.run([sys.executable,'-m','scripts.summarize_ctm_rdt_screen'],check=True)
            AUTOSTART.unlink(missing_ok=True);log('sweep complete; autostart entry removed');return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);p.add_argument('--plan',action='store_true');a=p.parse_args()
    if a.plan:
        want,blocked=plan()
        for r in want:print(run_name(*r),outcome(run_name(*r))[0] or 'pending')
        print('blocked' if blocked else '');return
    supervise()

if __name__=='__main__':main()
