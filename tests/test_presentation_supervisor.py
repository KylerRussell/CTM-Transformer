"""CPU checks of restart recovery in the presentation-control supervisor."""
import json
from pathlib import Path

import scripts.supervise_presentation_control as sup


def rows(steps, loss=1.0):
    return ''.join(json.dumps({'step': s, 'loss': loss + s, 'update_seconds': 0.1 * s}) + '\n' for s in steps)


def setup(tmp_path, monkeypatch):
    root, runs = tmp_path / 'results', tmp_path / 'runs'
    root.mkdir()
    for name, value in {'ROOT': root, 'RUNS': runs, 'INTERRUPTED_RUNS': tmp_path / 'interrupted_runs',
                        'STATE': root / 'supervisor_state.json', 'AUTOSTART': tmp_path / 'autostart.desktop'}.items():
        monkeypatch.setattr(sup, name, value)
    entries = []
    for cell, status in [('done', 'complete'), ('partial', 'partial'), ('pending', 'pending'), ('control', 'reused')]:
        run = runs / cell
        entry = {'cell': cell, 'reused': status == 'reused', 'run_directory': str(run),
                 'summary_path': str(root / f'{cell}.summary.json')}
        if status in ('complete', 'reused'):
            run.mkdir(parents=True)
            for f in ('summary.json', 'final.pt'):
                (run / f).write_text('{}')
            Path(entry['summary_path']).write_text('{}')
        elif status == 'partial':
            run.mkdir(parents=True)
            (run / 'metrics.jsonl').write_text(rows(range(1, 4)) + '{"step": 4, "lo')  # Killed mid-line.
        entries.append(entry)
    registry = {'entries': entries, 'gpu_queues': {'cuda:0': ['done', 'partial'], 'cuda:1': ['pending']}}
    (root / 'registry.json').write_text(json.dumps(registry))
    for name in ('continuation.json', 'gpu0.log', 'gpu1.log'):
        (root / name).write_text('stale')
    return root, runs


def test_recovery_keeps_completed_cells_and_moves_partial_outputs(tmp_path, monkeypatch):
    root, runs = setup(tmp_path, monkeypatch)
    s = sup.Supervisor()
    assert s.attempt['attempt'] == 2
    s.archive_interrupted()
    assert (runs / 'done' / 'final.pt').exists() and (runs / 'control' / 'final.pt').exists()
    assert not (runs / 'partial').exists()
    moved = tmp_path / 'interrupted_runs' / 'attempt1' / 'partial' / 'metrics.jsonl'
    assert moved.exists()
    assert s.state['interrupted_runs'] == [{'cell': 'partial', 'moved_from': str(runs / 'partial'),
                                            'moved_to': str(moved.parent), 'last_complete_step': 3}]
    assert sorted(p.name for p in (root / 'interrupted_attempt1').iterdir()) == ['continuation.json', 'gpu0.log', 'gpu1.log']
    state = json.loads((root / 'supervisor_state.json').read_text())
    assert state['phase'] == 'archived_interrupted_outputs'
    # A second attempt numbers itself after the first and moves nothing further.
    s2 = sup.Supervisor()
    assert s2.attempt['attempt'] == 3
    s2.archive_interrupted()
    assert s2.state['phase'] == 'starting'


def test_replay_audit_ignores_timing_and_flags_changed_values(tmp_path, monkeypatch):
    root, runs = setup(tmp_path, monkeypatch)
    s = sup.Supervisor()
    s.archive_interrupted()
    (runs / 'partial').mkdir()
    rerun = [json.loads(x) for x in rows(range(1, 7)).splitlines()]
    for r in rerun:
        r['update_seconds'] = 9.0
    (runs / 'partial' / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rerun))
    s.replay_audit()
    audit = json.loads((root / 'interruption_replay.json').read_text())['runs'][0]
    assert audit['exact'] and audit['compared_steps'] == 3
    rerun[1]['loss'] += 1e-9
    (runs / 'partial' / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rerun))
    s.replay_audit()
    audit = json.loads((root / 'interruption_replay.json').read_text())['runs'][0]
    assert not audit['exact'] and audit['mismatched_steps'] == [2]
    assert s.state['phase'] == 'replay_audit_mismatch'
