"""Validate the frozen presentation intervention, then use the shared GPU runner."""
import argparse,json,traceback
from pathlib import Path
from ctm_transformer.experiment import file_hash
from ctm_transformer.presentation_control import validate_presentation_control
from scripts.run_registry_trials import run_trial

ROOT=Path('research/results/presentation_control_v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);a=p.parse_args()
    r=json.loads((ROOT/'registry.json').read_text());pre=json.loads((ROOT/'pre_run_source.json').read_text())
    for path,h in pre['files'].items():assert file_hash(path)==h,path
    validate_presentation_control(r['dataset'])
    bycell={e['cell']:e for e in r['entries']}
    for cell in r['gpu_queues'][a.device]:
        try:run_trial(r,bycell[cell],a.device)
        except Exception as exc:
            (ROOT/f'{cell}.failure.json').write_text(json.dumps({'cell':cell,'error':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
            raise

if __name__=='__main__':main()
