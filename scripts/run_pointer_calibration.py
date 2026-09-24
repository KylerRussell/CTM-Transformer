"""Train declared pointer-calibration cells through the frozen shared runner."""
import argparse,json,traceback
from pathlib import Path
from ctm_transformer.experiment import file_hash
from scripts.run_registry_trials import run_trial

ROOT=Path('research/results/pointer_calibration_v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--cells',nargs='+',required=True)
    a=p.parse_args()
    r=json.loads((ROOT/'registry.json').read_text());pre=json.loads((ROOT/'pre_run_source.json').read_text())
    # Every split, including both large training files, is hash-checked; full semantic validation ran at freeze time.
    for path,h in pre['files'].items():assert file_hash(path)==h,path
    entries={e['cell']:e for e in r['entries']}
    if len(set(a.cells))!=len(a.cells) or any(c not in entries for c in a.cells):p.error('Unknown or duplicate cells')
    for cell in a.cells:
        e=entries[cell]
        try:
            # The frozen runner reads one training path per registry; each task supplies its own.
            run_trial({**r,'train_path':e['train_path']},e,a.device)
        except Exception as exc:
            (ROOT/f'{cell}.failure.json').write_text(json.dumps({'cell':cell,'device':a.device,'error':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
            raise

if __name__=='__main__':main()
