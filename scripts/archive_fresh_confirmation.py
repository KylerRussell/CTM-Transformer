"""Verify and archive the completed LR and seed-confirmation studies."""
import json
import zipfile
from pathlib import Path
from ctm_transformer.experiment import file_hash
from ctm_transformer.fresh_pointer import validate_fresh_suite
from scripts.confirmation_common import read

LR = Path('research/results/ctm_lr_v1')
ROOT = Path('research/results/fresh_confirmation_v1')


def main():
    frozen = read(ROOT / 'checkpoints.json')
    recipes = read(ROOT / 'recipes.json')
    summary = read(ROOT / 'summary.json')
    assert summary['complete'] and summary['checkpoint_freeze_sha256'] == file_hash(ROOT / 'checkpoints.json')
    validate_fresh_suite(frozen['fresh_dataset'])
    checked = {}
    for base in (LR, ROOT):
        pre = read(base / 'pre_run_source.json')
        assert file_hash(base / 'pre_run_source.zip') == pre['archive_sha256']
        with zipfile.ZipFile(base / 'pre_run_source.zip') as archive:
            assert archive.testzip() is None
        for path, digest in pre['files'].items():
            assert file_hash(path) == digest, path
        checked[str(base)] = len(pre['files'])
    assert file_hash(ROOT / 'pre_evaluation_source.zip') == frozen['source_archive_sha256']
    with zipfile.ZipFile(ROOT / 'pre_evaluation_source.zip') as archive:
        assert archive.testzip() is None
    for path, digest in frozen['source_sha256'].items():
        assert file_hash(path) == digest, path
    for value in summary['evaluations'].values():
        assert file_hash(value['path']) == value['sha256']
    records = read(LR / 'selection.json')['trials'] + frozen['models']
    directories = sorted({r['run_directory'] for r in records})
    checkpoints = {}
    for directory in directories:
        for path in sorted(Path(directory).glob('*.pt')):
            checkpoints[str(path)] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    for record in records:
        assert file_hash(record['checkpoint']) == record['checkpoint_sha256']
        assert file_hash(record['summary_path']) == record['summary_sha256']
        assert file_hash(Path(record['run_directory']) / 'metrics.jsonl') == record['metrics_sha256']
    (ROOT / 'checkpoint_manifest.json').write_text(json.dumps({'files': checkpoints}, indent=2) + '\n')
    files = set()
    for base in (LR, ROOT):
        files.update(str(p) for p in base.iterdir() if p.is_file() and p.suffix in ('.json', '.md', '.xml', '.png', '.pdf'))
    files.update(frozen['source_sha256'])
    files.update(recipes['selection_source_sha256'])
    files.update(str(p) for p in Path('research/configs/ctm_lr_v1').glob('*.json'))
    files.update(str(p) for p in Path('research/configs/fresh_confirmation_v1').glob('*.json'))
    for directory in directories:
        files.update(str(p) for p in Path(directory).iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl'))
    files.update(str(p) for p in Path(frozen['fresh_dataset']).rglob('*') if p.is_file())
    fresh_manifest = read(Path(frozen['fresh_dataset']) / 'manifest.json')
    files.update(fresh_manifest['prior_files'])
    files.update(str(p) for p in Path('research/data').rglob('manifest.json'))
    files.update(r['config'] for r in records)
    for name in ('registry.json', 'selection.json', 'pre_run_source.json'):
        files.add(str(Path('research/results/readout_comparison_v1') / name))
    for path in ('scripts/summarize_ctm_lr.py', 'scripts/summarize_fresh_confirmation.py',
                 'scripts/archive_fresh_confirmation.py', 'scripts/continue_fresh_confirmation.py',
                 'research/CTM_LR_CONFIRMATION.md', 'research/README.md', 'RESEARCH_PLAN.md'):
        files.add(path)
    # Avoid recursive inclusion of earlier versions of the archive itself.
    files -= {str(ROOT / name) for name in ('source_snapshot.json', 'integrity.json')}
    digests = {p: file_hash(p) for p in sorted(files)}
    with zipfile.ZipFile(ROOT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path)
    with zipfile.ZipFile(ROOT / 'source_snapshot.zip') as archive:
        assert archive.testzip() is None
        import hashlib
        for path, digest in digests.items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    (ROOT / 'source_snapshot.json').write_text(json.dumps({'files': digests, 'archive_sha256': file_hash(ROOT / 'source_snapshot.zip')}, indent=2) + '\n')
    audit = {'complete': True, 'pre_run_files_verified': checked, 'retained_checkpoints': len(checkpoints),
             'archived_files': len(digests), 'checkpoint_manifest_sha256': file_hash(ROOT / 'checkpoint_manifest.json'),
             'archive_manifest_sha256': file_hash(ROOT / 'source_snapshot.json'),
             'fresh_manifest_sha256': frozen['fresh_manifest_sha256'],
             'independent_confirmation_runs': 9, 'new_lr_runs': 4, 'new_training_updates': 39000,
             'reused_ctm_development_controls': 2, 'separate_development_references': 3,
             'training_scope': 'Four new optimizer-development runs plus nine independent-seed confirmations; each audited for 3000 updates and fixed exposure. Checkpoint files are hashed in place, not duplicated into the source archive.'}
    (ROOT / 'integrity.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
