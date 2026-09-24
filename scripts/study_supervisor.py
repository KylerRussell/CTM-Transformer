"""Restart-safe orchestration shared by frozen development studies.

Contains no training, evaluation or analysis logic. A study supplies its
paths, frozen worker/freeze/evaluation/summary modules and preflight checks.
The supervisor then:

* verifies every file in the study's `pre_run_source.json` and runs preflight
  tests on both GPUs before any work;
* keeps completed cells, moves any partially trained cell aside unchanged and
  reruns it from scratch with the declared seeds, then audits that the rerun
  reproduces the interrupted updates;
* runs the freeze, evaluation and summary modules, which must be idempotent
  (skip completed outputs, write atomically);
* records a nonzero worker exit as `<cell>.failure.json` and never retries
  it, while a signal-killed worker is an interruption resumed on relaunch;
* holds a lock, logs every attempt in `supervisor_state.json` and removes the
  study's autostart entry when the study completes.
"""
import datetime,fcntl,json,os,shutil,signal,subprocess,sys
from dataclasses import dataclass,field
from pathlib import Path
from ctm_transformer.experiment import file_hash

COMPLETE='complete_pending_documentation_and_archive'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


@dataclass
class Study:
    root:Path                      # results directory with registry.json and pre_run_source.json
    runs:Path                      # common parent of all run directories
    worker_module:str              # accepts --device and --cells; writes each cell's summary_path
    freeze_module:str
    evaluate_module:str            # accepts --device and --cells; skips completed evaluations
    summarize_module:str
    autostart:Path
    preflight_tests:list=field(default_factory=list)  # [(pytest args, device)]
    min_free_bytes:int=30*2**30
    busy_mib:int=500

    @property
    def interrupted_runs(self):return self.runs.with_name(self.runs.name+'_interrupted')
    @property
    def state_path(self):return self.root/'supervisor_state.json'


class Supervisor:
    def __init__(self,study):
        self.s=study;self.registry=read(study.root/'registry.json')
        self.state=read(study.state_path) if study.state_path.exists() else {'attempts':[]}
        self.attempt={'attempt':len(self.state['attempts'])+1,'started_utc':now(),'pid':os.getpid(),
            'supervisor_sha256':file_hash(__file__),'events':[]}
        self.state['attempts'].append(self.attempt);self.state['phase']='starting';self.children=[]

    def event(self,phase,**extra):
        self.state['phase']=phase;self.attempt['events'].append({'phase':phase,'time_utc':now(),**extra})
        tmp=self.s.state_path.with_suffix('.tmp');tmp.write_text(json.dumps(self.state,indent=2)+'\n');tmp.replace(self.s.state_path)
        print(now(),phase,json.dumps(extra) if extra else '',flush=True)

    def entries(self):return {e['cell']:e for e in self.registry['entries']}

    def preflight(self):
        pre=read(self.s.root/'pre_run_source.json')
        for p,h in pre['files'].items():
            if file_hash(p)!=h:raise RuntimeError(f'Frozen file changed: {p}')
        if file_hash(self.s.root/'pre_run_source.zip')!=pre['archive_sha256']:raise RuntimeError('Pre-run archive changed')
        # Containers can hide other PIDs from nvidia-smi, so check memory before creating any CUDA context.
        used=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        busy=[x for x in used.strip().splitlines() if int(x.split(',')[1])>self.s.busy_mib]
        if busy:raise RuntimeError('GPUs are already in use (index, MiB): '+'; '.join(busy))
        free=shutil.disk_usage(self.s.runs.parent).free
        if free<self.s.min_free_bytes:raise RuntimeError(f'Only {free/2**30:.1f} GiB free')
        for args,device in self.s.preflight_tests:
            subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*args],check=True,env={**os.environ,'CTM_TEST_DEVICE':device})

    @staticmethod
    def metric_rows(path):
        rows=[]
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:break  # Truncated final line from a killed writer.
        return rows

    def archive_interrupted(self):
        moved=[]
        for e in self.registry['entries']:
            run=Path(e['run_directory'])
            if Path(e['summary_path']).exists():
                if not ((run/'summary.json').exists() and (run/'final.pt').exists()):raise RuntimeError(f"{e['cell']}: summary without complete run")
                continue
            if run.exists() and any(run.iterdir()):
                dest=self.s.interrupted_runs/f"attempt{self.attempt['attempt']-1}"/run.relative_to(self.s.runs)
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(run),dest)
                rows=self.metric_rows(dest/'metrics.jsonl')
                self.state.setdefault('interrupted_runs',[]).append({'cell':e['cell'],'moved_from':str(run),'moved_to':str(dest),
                    'last_complete_step':rows[-1]['step'] if rows else 0})
                moved.append(str(run))
        if moved:self.event('archived_interrupted_runs',moved=moved)

    def train(self):
        entries=self.entries();n=self.attempt['attempt']
        remaining={d:[c for c in cells if not Path(entries[c]['summary_path']).exists()] for d,cells in self.registry['gpu_queues'].items()}
        for d,cells in remaining.items():
            if not cells:continue
            with open(self.s.root/f"gpu{d[-1]}.attempt{n}.log",'a') as log:
                p=subprocess.Popen([sys.executable,'-u','-m',self.s.worker_module,'--device',d,'--cells',*cells],stdout=log,stderr=subprocess.STDOUT)
            self.children.append((d,cells,p))
        if not self.children:return
        self.event('training',workers={d:{'pid':p.pid,'cells':c} for d,c,p in self.children})
        # Wait for every queue so a problem on one GPU never orphans or stops the other.
        codes=[(d,cells,p.wait()) for d,cells,p in self.children];self.children=[]
        failures,interrupted=[],[]
        for d,cells,code in codes:
            if code==0:continue
            cell=next(c for c in cells if not Path(entries[c]['summary_path']).exists())
            if code<0:interrupted.append({'device':d,'cell':cell,'signal':-code});continue
            failure=self.s.root/f'{cell}.failure.json'
            if not failure.exists():  # The worker normally writes it with the traceback.
                failure.write_text(json.dumps({'cell':cell,'device':d,'exit_code':code,'time_utc':now()},indent=2)+'\n')
            failures.append(cell)
        if failures:raise RuntimeError(f'Training failure recorded for {failures}; no favorable replacement permitted')
        if interrupted:self.event('interrupted',workers=interrupted);raise SystemExit('Worker killed by a signal; relaunch to resume')
        self.event('training_complete')

    def replay_audit(self):
        results=[]
        for r in self.state.get('interrupted_runs',[]):
            old=self.metric_rows(Path(r['moved_to'])/'metrics.jsonl');new=self.metric_rows(Path(r['moved_from'])/'metrics.jsonl')
            strip=lambda x:{k:v for k,v in x.items() if k!='update_seconds'}
            bad=[a['step'] for a,b in zip(old,new) if strip(a)!=strip(b)]
            results.append({**r,'compared_steps':len(old),'mismatched_steps':bad[:20],'exact':not bad and len(new)>=len(old)})
        if not results:return
        (self.s.root/'interruption_replay.json').write_text(json.dumps({'runs':results},indent=2)+'\n')
        # A mismatch does not invalidate the from-scratch rerun; it is a reproducibility finding to report.
        bad=[r['cell'] for r in results if not r['exact']]
        self.event('replay_audit_mismatch' if bad else 'replay_audit_passed',runs=len(results),mismatched=bad)

    def evaluate(self):
        processes=[]
        for d,cells in self.registry['evaluation_queues'].items():
            with open(self.s.root/f"evaluation_gpu{d[-1]}.attempt{self.attempt['attempt']}.log",'a') as log:
                processes.append(subprocess.Popen([sys.executable,'-u','-m',self.s.evaluate_module,'--device',d,'--cells',*cells],stdout=log,stderr=subprocess.STDOUT))
        self.children=[(None,None,p) for p in processes]
        self.event('evaluation',worker_pids=[p.pid for p in processes])
        codes=[p.wait() for p in processes];self.children=[]
        if any(codes):raise RuntimeError(f'Evaluation failed with exit codes {codes}; inspect evaluation logs')

    def module(self,name):subprocess.run([sys.executable,'-u','-m',name],check=True)

    def run(self):
        if list(self.s.root.glob('*.failure.json')):raise RuntimeError('A training failure is recorded; resolve it manually')
        self.event('preflight');self.preflight()
        self.archive_interrupted();self.train();self.replay_audit()
        self.event('freezing_final_checkpoints');self.module(self.s.freeze_module)
        self.evaluate()
        self.event('summarizing');self.module(self.s.summarize_module)
        self.event(COMPLETE);self.s.autostart.unlink(missing_ok=True)

    def terminate_children(self,*_):
        for _,_,p in self.children:
            if p.poll() is None:p.terminate()
        raise SystemExit('Supervisor terminated')


def supervise(study):
    lock=open(study.root/'.supervisor.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:sys.exit('Another supervisor is running')
    if study.state_path.exists() and read(study.state_path).get('phase')==COMPLETE:
        study.autostart.unlink(missing_ok=True);print('Study already complete',flush=True);return
    s=Supervisor(study);signal.signal(signal.SIGTERM,s.terminate_children)
    try:s.run()
    except BaseException as exc:
        s.event('stopped',error=repr(exc));raise
