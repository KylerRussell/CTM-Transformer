"""Finish the fixed-endpoint paired development evaluation after GPU training."""
import argparse,datetime,json,subprocess,sys,time
from pathlib import Path
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/presentation_control_v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-pids',type=int,nargs=2,required=True);a=p.parse_args()
    r=json.loads((ROOT/'registry.json').read_text());statepath=ROOT/'continuation.json';assert not statepath.exists()
    sources={p:file_hash(p) for p in ('scripts/complete_presentation_control.py','scripts/freeze_presentation_control.py',
             'scripts/evaluate_presentation_control.py','scripts/summarize_presentation_control.py')}
    state={'worker_pids':a.worker_pids,'source_sha256':sources,'events':[]}
    def event(phase,**extra):
        state['phase']=phase;state['events'].append({'phase':phase,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
        statepath.write_text(json.dumps(state,indent=2)+'\n');print(phase,extra,flush=True)
    def run(module):
        for p,h in sources.items():assert file_hash(p)==h,p
        subprocess.run([sys.executable,'-u','-m',module],check=True)
    try:
        event('training')
        paths=[Path(e['summary_path']) for e in r['entries'] if not e['reused']]
        while not all(p.exists() for p in paths):
            if list(ROOT.glob('*.failure.json')):raise RuntimeError('Training failure recorded; no favorable replacement permitted')
            if not any(Path(f'/proc/{pid}/cmdline').exists() for pid in a.worker_pids):raise RuntimeError('Workers exited before all summaries appeared')
            time.sleep(10)
        event('auditing_and_freezing_final_checkpoints');run('scripts.freeze_presentation_control')
        processes=[];handles=[]
        try:
            for device,families in [('cuda:0',['ctm']),('cuda:1',['transformer','recurrent_depth'])]:
                out=(ROOT/f"evaluation_gpu{device[-1]}.log").open('x');handles.append(out)
                processes.append(subprocess.Popen([sys.executable,'-u','-m','scripts.evaluate_presentation_control','--device',device,'--families',*families],stdout=out,stderr=subprocess.STDOUT))
            event('paired_validation_evaluation',worker_pids=[p.pid for p in processes])
            for p in processes:
                if p.wait()!=0:raise RuntimeError('Paired evaluation failed; inspect evaluation logs')
        finally:
            for p in processes:
                if p.poll() is None:p.terminate();p.wait()
            for out in handles:out.close()
        event('summarizing');run('scripts.summarize_presentation_control')
        event('complete_pending_documentation_and_archive')
    except BaseException as exc:
        event('failed',error=repr(exc));raise

if __name__=='__main__':main()
