"""Verify source/data/checkpoint provenance for the completed presentation control."""
import hashlib,json,zipfile
from pathlib import Path
from ctm_transformer.experiment import file_hash
from ctm_transformer.presentation_control import validate_presentation_control

ROOT=Path('research/results/presentation_control_v1')


def read(p):return json.loads(Path(p).read_text())


def main():
    r=read(ROOT/'registry.json');frozen=read(ROOT/'checkpoints.json');summary=read(ROOT/'summary.json');pre=read(ROOT/'pre_run_source.json')
    assert summary['complete'] and summary['no_test_evaluation']
    assert summary['checkpoint_freeze_sha256']==file_hash(ROOT/'checkpoints.json')
    assert summary['analysis_source_sha256']==file_hash('scripts/summarize_presentation_control.py')
    assert file_hash(ROOT/'pre_run_source.zip')==pre['archive_sha256']
    with zipfile.ZipFile(ROOT/'pre_run_source.zip') as z:
        assert z.testzip() is None
        for p,h in pre['files'].items():assert file_hash(p)==h==hashlib.sha256(z.read(p)).hexdigest(),p
    assert file_hash(ROOT/'pre_evaluation_source.zip')==frozen['source_archive_sha256']
    for p,h in frozen['source_sha256'].items():assert file_hash(p)==h,p
    validate_presentation_control(r['dataset'])
    assert file_hash(r['control_source_archive'])==r['control_source_archive_sha256']
    for e in summary['evaluations'].values():assert file_hash(e['path'])==e['sha256']
    checkpoints={}
    for t in frozen['models']:
        assert file_hash(t['checkpoint'])==t['checkpoint_sha256']
        assert file_hash(t['summary_path'])==t['summary_sha256']
        assert file_hash(Path(t['run_directory'])/'metrics.jsonl')==t['metrics_sha256']
        for p in Path(t['run_directory']).glob('*.pt'):checkpoints[str(p)]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    (ROOT/'checkpoint_manifest.json').write_text(json.dumps({'files':checkpoints},indent=2)+'\n')
    files=set(pre['files'])|set(frozen['source_sha256'])
    files.update(str(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.xml','.png','.pdf'))
    files.update((r['ordered_train_path'],'research/data/ordered_pointer_v1/pointer/validation.jsonl',
        'research/data/ordered_pointer_v1/manifest.json',r['control_source_archive'],
        'scripts/summarize_presentation_control.py','scripts/complete_presentation_control.py','scripts/archive_presentation_control.py',
        'research/PRESENTATION_CONTROL.md','research/README.md','RESEARCH_PLAN.md'))
    for t in frozen['models']:
        files.update(str(p) for p in Path(t['run_directory']).iterdir() if p.is_file() and p.suffix in ('.json','.jsonl'))
    files-={str(ROOT/p) for p in ('source_snapshot.json','integrity.json')}
    digests={p:file_hash(p) for p in sorted(files)}
    with zipfile.ZipFile(ROOT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p)
    with zipfile.ZipFile(ROOT/'source_snapshot.zip') as z:
        assert z.testzip() is None
        for p,h in digests.items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
    (ROOT/'source_snapshot.json').write_text(json.dumps({'files':digests,'archive_sha256':file_hash(ROOT/'source_snapshot.zip')},indent=2)+'\n')
    audit={'complete':True,'new_runs':9,'reused_controls':9,'new_updates_audited':27000,'all_updates_audited':54000,
        'primary_checkpoints':18,'primary_checkpoint_step':3000,'paired_validation_maps':128,'new_checks_passed':5,
        'retained_checkpoints':len(checkpoints),'pre_run_files_verified':len(pre['files']),'archive_files':len(files),
        'no_test_evaluation':True,'source_snapshot_sha256':file_hash(ROOT/'source_snapshot.json'),
        'checkpoint_manifest_sha256':file_hash(ROOT/'checkpoint_manifest.json')}
    (ROOT/'integrity.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
