"""Restart-safe orchestration of the frozen presentation-control study.

This script contains no training, evaluation or analysis logic. It runs the
frozen worker (`scripts.run_registry_trials`), freeze, evaluation and summary
modules exactly as declared, and adds only what a host restart requires:

* preflight checks (frozen hashes, dataset validation, CUDA, exact archived
  replay on both GPUs, idle GPUs, disk space);
* cell-level recovery: completed cells are kept; a partially trained cell is
  moved aside unchanged and rerun from scratch with the same seeds;
* a replay audit comparing each rerun with its interrupted partial run;
* restart-safe freeze/evaluation phases;
* a lock so that only one supervisor runs.

Genuine worker errors (nonzero exit) are recorded as `<cell>.failure.json`
and are never retried automatically. A worker killed by a signal is recorded
as an infrastructure interruption; rerunning this script resumes the study.
"""
import argparse,datetime,fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/presentation_control_v1')
RUNS=Path('research/runs/presentation_control_v1')
INTERRUPTED_RUNS=Path('research/runs/presentation_control_v1_interrupted')
STATE=ROOT/'supervisor_state.json'
AUTOSTART=Path.home()/'.config/autostart/ctm-presentation-control.desktop'
# Evaluation groups mirror scripts/complete_presentation_control.py.
EVALUATION_GROUPS=[('cuda:0',['ctm']),('cuda:1',['transformer','recurrent_depth'])]
MIN_FREE_BYTES=30*2**30
BUSY_MIB=500


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


class Supervisor:
    def __init__(self):
        self.registry=read(ROOT/'registry.json')
        self.state=read(STATE) if STATE.exists() else {'attempts':[]}
        attempt=len(self.state['attempts'])+2  # Attempt 1 was the interrupted pre-supervisor launch.
        self.attempt={'attempt':attempt,'started_utc':now(),'pid':os.getpid(),
            'supervisor_sha256':file_hash(__file__),'events':[]}
        self.state['attempts'].append(self.attempt);self.state['phase']='starting'
        self.children=[]

    def event(self,phase,**extra):
        self.state['phase']=phase;self.attempt['events'].append({'phase':phase,'time_utc':now(),**extra})
        temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(self.state,indent=2)+'\n');temporary.replace(STATE)
        print(now(),phase,json.dumps(extra) if extra else '',flush=True)

    def module(self,name,*args,log=None):
        """Run a frozen module in the foreground; raise on any failure."""
        out=open(log,'a') if log else None
        try:subprocess.run([sys.executable,'-u','-m',name,*args],check=True,stdout=out,stderr=subprocess.STDOUT if out else None)
        finally:
            if out:out.close()

    # ---------------------------------------------------------------- preflight
    def preflight(self):
        pre=read(ROOT/'pre_run_source.json')
        for p,h in pre['files'].items():
            if file_hash(p)!=h:raise RuntimeError(f'Frozen file changed: {p}')
        assert file_hash(ROOT/'pre_run_source.zip')==pre['archive_sha256']
        for k in ('train','validation','ordered_train'):assert file_hash(self.registry[k+'_path'])==self.registry[k+'_sha256'],k
        # Containers may hide other PIDs from nvidia-smi, so check memory before creating our own CUDA context.
        used=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        busy=[line for line in used.strip().splitlines() if int(line.split(',')[1])>BUSY_MIB]
        if busy:raise RuntimeError('GPUs are already in use (index, MiB): '+'; '.join(busy))
        import torch
        from ctm_transformer.presentation_control import validate_presentation_control
        validate_presentation_control(self.registry['dataset'])
        if torch.__version__!='2.6.0+cu124':raise RuntimeError(f'Unexpected torch {torch.__version__}; use the pinned environment')
        if torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices are required')
        free=shutil.disk_usage(RUNS.parent).free
        if free<MIN_FREE_BYTES:raise RuntimeError(f'Only {free/2**30:.1f} GiB free')
        # The frozen checks include exact replays of the archived trainer; run
        # the replay on both GPUs so a changed environment cannot pass silently.
        for d in ('cuda:0','cuda:1'):
            subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_presentation_control.py']+
                (['-k','replays'] if d=='cuda:1' else []),check=True,env={**os.environ,'CTM_TEST_DEVICE':d})

    # ----------------------------------------------------------------- recovery
    def archive_interrupted(self):
        """Move partial outputs from an interrupted attempt aside, unchanged."""
        target=ROOT/f"interrupted_attempt{self.attempt['attempt']-1}";moved=[]
        # Unnumbered names come from the pre-supervisor launcher; supervisor logs are numbered per attempt.
        stale=[ROOT/n for n in ('continuation.json','continuation.log','gpu0.log','gpu1.log','evaluation_gpu0.log','evaluation_gpu1.log')]
        for p in stale:
            if p.exists():
                target.mkdir(exist_ok=True);shutil.move(str(p),target/p.name);moved.append(str(p))
        for e in self.registry['entries']:
            if e['reused']:continue
            run=Path(e['run_directory']);done=Path(e['summary_path']).exists()
            if done:
                if not ((run/'summary.json').exists() and (run/'final.pt').exists()):raise RuntimeError(f"{e['cell']}: summary without complete run")
                continue
            if run.exists() and any(run.iterdir()):
                dest=INTERRUPTED_RUNS/f"attempt{self.attempt['attempt']-1}"/run.relative_to(RUNS)
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(run),dest)
                rows=self.metric_rows(dest/'metrics.jsonl')
                record={'cell':e['cell'],'moved_from':str(run),'moved_to':str(dest),'last_complete_step':rows[-1]['step'] if rows else 0}
                self.state.setdefault('interrupted_runs',[]).append(record);moved.append(str(run))
        if moved:self.event('archived_interrupted_outputs',archive=str(target),moved=moved)

    @staticmethod
    def metric_rows(path):
        rows=[]
        if not Path(path).exists():return rows
        for line in Path(path).read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:break  # Truncated final line from a killed writer.
        return rows

    # ----------------------------------------------------------------- training
    def train(self):
        entries={e['cell']:e for e in self.registry['entries']}
        remaining={d:[c for c in cells if not Path(entries[c]['summary_path']).exists()] for d,cells in self.registry['gpu_queues'].items()}
        remaining={d:c for d,c in remaining.items() if c}
        if not remaining:return
        n=self.attempt['attempt']
        for d,cells in remaining.items():
            log=open(ROOT/f"gpu{d[-1]}.attempt{n}.log",'a')
            p=subprocess.Popen([sys.executable,'-u','-m','scripts.run_registry_trials','--registry',str(ROOT/'registry.json'),'--cells',*cells,'--device',d],
                stdout=log,stderr=subprocess.STDOUT);log.close();self.children.append((d,cells,p))
        self.event('training',workers={d:{'pid':p.pid,'cells':c} for d,c,p in self.children})
        # Wait for both queues so a problem on one GPU never orphans or stops the other.
        codes=[(d,cells,p.wait()) for d,cells,p in self.children];self.children=[]
        failures,interrupted=[],[]
        for d,cells,code in codes:
            if code==0:continue
            cell=next(c for c in cells if not Path(entries[c]['summary_path']).exists())
            if code<0:interrupted.append({'device':d,'cell':cell,'signal':-code});continue
            (ROOT/f'{cell}.failure.json').write_text(json.dumps({'cell':cell,'device':d,'exit_code':code,'time_utc':now(),
                'log':str(ROOT/f"gpu{d[-1]}.attempt{n}.log")},indent=2)+'\n')
            failures.append(cell)
        if failures:raise RuntimeError(f'Training failure recorded for {failures}; no favorable replacement permitted')
        if interrupted:
            self.event('interrupted',workers=interrupted);raise SystemExit('Worker killed by a signal; rerun to resume')
        self.event('training_complete')

    def replay_audit(self):
        """Check that each rerun reproduces the steps its interrupted predecessor completed."""
        results=[]
        for r in self.state.get('interrupted_runs',[]):
            old=self.metric_rows(Path(r['moved_to'])/'metrics.jsonl');new=self.metric_rows(Path(r['moved_from'])/'metrics.jsonl')
            strip=lambda x:{k:v for k,v in x.items() if k!='update_seconds'}
            mismatched=[a['step'] for a,b in zip(old,new) if strip(a)!=strip(b)]
            results.append({**r,'compared_steps':len(old),'mismatched_steps':mismatched[:20],'exact':not mismatched and len(new)>=len(old)})
        (ROOT/'interruption_replay.json').write_text(json.dumps({'runs':results},indent=2)+'\n')
        # A mismatch does not invalidate the from-scratch rerun, but it is a reproducibility finding to report.
        bad=[r['cell'] for r in results if not r['exact']]
        self.event('replay_audit_mismatch' if bad else 'replay_audit_passed',runs=len(results),mismatched=bad)

    # ------------------------------------------------------ freeze and evaluate
    def freeze(self):
        path=ROOT/'checkpoints.json'
        if path.exists():
            try:
                if read(path)['complete']:return
            except (json.JSONDecodeError,KeyError):pass
            shutil.move(str(path),ROOT/f"checkpoints.incomplete.attempt{self.attempt['attempt']}.json")
        self.event('auditing_and_freezing_final_checkpoints');self.module('scripts.freeze_presentation_control')

    def evaluate(self):
        frozen=read(ROOT/'checkpoints.json');processes=[]
        def done(cell):
            try:return read(ROOT/f'{cell}.evaluation.json')['complete']
            except (FileNotFoundError,json.JSONDecodeError,KeyError):return False
        for device,families in EVALUATION_GROUPS:
            todo=[]
            for f in families:
                cells=[t['cell'] for t in frozen['models'] if t['model_family']==f]
                if all(done(c) for c in cells):continue
                for c in cells:  # The frozen evaluator refuses existing outputs; redo the whole family.
                    p=ROOT/f'{c}.evaluation.json'
                    if p.exists():
                        target=ROOT/f"interrupted_attempt{self.attempt['attempt']-1}";target.mkdir(exist_ok=True);shutil.move(str(p),target/p.name)
                todo.append(f)
            if todo:
                log=open(ROOT/f"evaluation_gpu{device[-1]}.attempt{self.attempt['attempt']}.log",'a')
                processes.append(subprocess.Popen([sys.executable,'-u','-m','scripts.evaluate_presentation_control','--device',device,'--families',*todo],
                    stdout=log,stderr=subprocess.STDOUT));log.close()
        if not processes:return
        self.children=[(None,None,p) for p in processes]
        self.event('paired_validation_evaluation',worker_pids=[p.pid for p in processes])
        codes=[p.wait() for p in processes];self.children=[]
        if any(codes):raise RuntimeError(f'Paired evaluation failed with exit codes {codes}; inspect evaluation logs')

    # ---------------------------------------------------------------------- run
    def run(self):
        if list(ROOT.glob('*.failure.json')):
            raise RuntimeError('A training failure is recorded; resolve it manually before resuming')
        self.event('preflight');self.preflight()
        self.archive_interrupted();self.train();self.replay_audit()
        self.freeze();self.evaluate()
        self.event('summarizing');self.module('scripts.summarize_presentation_control')
        self.event('complete_pending_documentation_and_archive')
        AUTOSTART.unlink(missing_ok=True)

    def terminate_children(self,*_):
        for _,_,p in self.children:
            if p.poll() is None:p.terminate()
        raise SystemExit('Supervisor terminated')


def main():
    argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    lock=open(ROOT/'.supervisor.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:sys.exit('Another supervisor is running')
    if STATE.exists() and read(STATE).get('phase')=='complete_pending_documentation_and_archive':
        AUTOSTART.unlink(missing_ok=True);print('Study already complete',flush=True);return
    s=Supervisor();signal.signal(signal.SIGTERM,s.terminate_children)
    try:s.run()
    except BaseException as exc:
        s.event('stopped',error=repr(exc));raise

if __name__=='__main__':main()
