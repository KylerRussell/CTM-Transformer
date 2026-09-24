"""Finish the validation-only analysis when the declared GPU queues complete."""
import argparse,datetime,json,subprocess,sys,time
from pathlib import Path
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/recurrent_temporal_v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-pids',type=int,nargs=2,required=True);args=p.parse_args()
    registry=json.loads((ROOT/'registry.json').read_text());statepath=ROOT/'continuation.json';assert not statepath.exists()
    sources={p:file_hash(p) for p in ('scripts/complete_recurrent_temporal.py','scripts/analyze_recurrent_temporal.py')}
    state={'worker_pids':args.worker_pids,'source_sha256':sources,'events':[]}
    def event(phase,**extra):
        state['phase']=phase;state['events'].append({'phase':phase,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
        statepath.write_text(json.dumps(state,indent=2)+'\n');print(phase,extra,flush=True)
    try:
        event('training')
        expected=[Path(e['summary_path']) for e in registry['entries'] if not e['reused']]
        while not all(p.exists() for p in expected):
            if list(ROOT.glob('*.failure.json')):raise RuntimeError('Declared training failure; review failure record; no replacement permitted')
            if not any(Path(f'/proc/{pid}/cmdline').exists() for pid in args.worker_pids):raise RuntimeError('Workers exited before all six summaries appeared')
            time.sleep(10)
        for p,h in sources.items():assert file_hash(p)==h,p
        event('auditing_and_summarizing')
        subprocess.run([sys.executable,'-u','-m','scripts.analyze_recurrent_temporal'],check=True)
        event('complete_pending_documentation_and_archive')
    except BaseException as exc:
        event('failed',error=repr(exc));raise

if __name__=='__main__':main()
