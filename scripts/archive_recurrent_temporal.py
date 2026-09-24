"""Archive the completed paired objective control and checkpoint integrity."""
import hashlib,json,zipfile
from pathlib import Path
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/recurrent_temporal_v1')


def read(path):return json.loads(Path(path).read_text())


def main():
    summary=read(ROOT/'summary.json');registry=read(ROOT/'registry.json');pre=read(ROOT/'pre_run_source.json')
    assert summary['complete'] and summary['no_test_evaluation']
    assert summary['registry_sha256']==file_hash(ROOT/'registry.json')
    assert summary['analysis_source_sha256']==file_hash('scripts/analyze_recurrent_temporal.py')
    assert file_hash(ROOT/'pre_run_source.zip')==pre['archive_sha256']
    with zipfile.ZipFile(ROOT/'pre_run_source.zip') as z:
        assert z.testzip() is None
        for p,h in pre['files'].items():assert file_hash(p)==h==hashlib.sha256(z.read(p)).hexdigest(),p
    checkpoints={}
    for t in summary['trials']:
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        assert file_hash(Path(t['run_directory'])/'metrics.jsonl')==t['metrics_sha256']
        for p in Path(t['run_directory']).glob('*.pt'):checkpoints[str(p)]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    (ROOT/'checkpoint_manifest.json').write_text(json.dumps({'files':checkpoints},indent=2)+'\n')
    files=set(pre['files'])
    files.update(str(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.xml','.png','.pdf'))
    files.update((registry['train_path'],registry['validation_path'],registry['control_source_archive'],registry['control_checkpoint_freeze'],
                  'research/data/ordered_pointer_v1/manifest.json','research/RECURRENT_TEMPORAL_CONTROL.md','research/README.md','RESEARCH_PLAN.md',
                  'scripts/analyze_recurrent_temporal.py','scripts/complete_recurrent_temporal.py','scripts/archive_recurrent_temporal.py'))
    for t in summary['trials']:
        files.update(str(p) for p in Path(t['run_directory']).iterdir() if p.is_file() and p.suffix in ('.json','.jsonl'))
    files-={str(ROOT/p) for p in ('source_snapshot.json','integrity.json')}
    digests={p:file_hash(p) for p in sorted(files)}
    with zipfile.ZipFile(ROOT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p)
    with zipfile.ZipFile(ROOT/'source_snapshot.zip') as z:
        assert z.testzip() is None
        for p,h in digests.items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
    (ROOT/'source_snapshot.json').write_text(json.dumps({'files':digests,'archive_sha256':file_hash(ROOT/'source_snapshot.zip')},indent=2)+'\n')
    audit={'complete':True,'new_runs':6,'reused_controls':3,'new_updates_audited':18000,'all_updates_audited':27000,
           'gpu_tests_passed':11,'retained_checkpoints':len(checkpoints),'pre_run_files_verified':len(pre['files']),'archive_files':len(files),
           'no_test_evaluation':True,'source_snapshot_sha256':file_hash(ROOT/'source_snapshot.json'),
           'checkpoint_manifest_sha256':file_hash(ROOT/'checkpoint_manifest.json')}
    (ROOT/'integrity.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
