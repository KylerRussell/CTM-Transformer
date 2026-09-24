"""Guarded continuation of already-running CTM tuning through fresh evaluation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LR = Path('research/results/ctm_lr_v1')
CONF = Path('research/results/fresh_confirmation_v1')
SOURCES = ('scripts/advance_ctm_lr.py', 'scripts/confirmation_common.py',
           'scripts/run_registry_trials.py', 'scripts/freeze_fresh_confirmation.py',
           'scripts/evaluate_fresh_confirmation.py', 'scripts/summarize_ctm_lr.py',
           'scripts/summarize_fresh_confirmation.py', 'scripts/continue_fresh_confirmation.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-a-pids', nargs=2, type=int, required=True)
    args = parser.parse_args()
    assert not (CONF / 'registry.json').exists()
    statepath = LR / 'continuation.json'
    assert not statepath.exists()
    state = {'source_sha256': {p: digest(p) for p in SOURCES}, 'stage_a_pids': args.stage_a_pids, 'events': []}

    def event(phase, **details):
        state['phase'] = phase
        state['events'].append({'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': phase, **details})
        statepath.write_text(json.dumps(state, indent=2) + '\n')
        print(phase, details, flush=True)

    def check_sources():
        for p, sha in state['source_sha256'].items():
            assert digest(p) == sha, ('Continuation source changed', p)

    def run(module):
        check_sources()
        subprocess.run([sys.executable, '-u', '-m', module], check=True)

    def pair(commands, phase):
        check_sources()
        processes = []
        handles = []
        try:
            for index, command in enumerate(commands):
                handle = (CONF / f'{phase}_gpu{index}.log').open('x')
                handles.append(handle)
                processes.append(subprocess.Popen([sys.executable, '-u', '-m', *command], stdout=handle, stderr=subprocess.STDOUT))
            event(phase, worker_pids=[p.pid for p in processes])
            while any(p.poll() is None for p in processes):
                failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failures:
                    # Stop this phase on an execution failure; do not silently replace trials.
                    raise RuntimeError(f'{phase} worker failed: {failures}')
                time.sleep(5)
            assert all(p.returncode == 0 for p in processes), [(p.pid, p.returncode) for p in processes]
        except BaseException:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                p.wait()
            raise
        finally:
            for handle in handles:
                handle.close()

    try:
        event('waiting_for_stage_a')
        registry = json.loads((LR / 'registry.json').read_text())
        paths = [Path(e['summary_path']) for e in registry['entries'] if not e['reused']]
        while not all(p.exists() for p in paths):
            for pid in args.stage_a_pids:
                # A finished worker can leave while its peer is still running.
                if Path(f'/proc/{pid}/cmdline').exists():
                    assert b'scripts.run_registry_trials' in Path(f'/proc/{pid}/cmdline').read_bytes()
            if not any(Path(f'/proc/{pid}/cmdline').exists() for pid in args.stage_a_pids):
                raise RuntimeError('Stage A workers exited before all expected summaries appeared')
            time.sleep(10)
        event('freezing_recipes')
        run('scripts.advance_ctm_lr')
        run('scripts.summarize_ctm_lr')
        registry_args = ['--registry', str(CONF / 'registry.json')]
        pair([
            ['scripts.run_registry_trials', *registry_args, '--device', 'cuda:0', '--cells',
             'ctm_seed23', 'ctm_seed31', 'transformer_seed23', 'transformer_seed29', 'transformer_seed31'],
            ['scripts.run_registry_trials', *registry_args, '--device', 'cuda:1', '--cells',
             'ctm_seed29', 'recurrent_depth_seed23', 'recurrent_depth_seed29', 'recurrent_depth_seed31'],
        ], 'training_confirmation')
        event('freezing_checkpoints')
        run('scripts.freeze_fresh_confirmation')
        pair([
            ['scripts.evaluate_fresh_confirmation', '--device', 'cuda:0', '--cells',
             'ctm_seed23', 'ctm_seed29', 'ctm_seed31', 'ctm_seed17_reference'],
            ['scripts.evaluate_fresh_confirmation', '--device', 'cuda:1', '--cells',
             'transformer_seed23', 'transformer_seed29', 'transformer_seed31', 'transformer_seed17_reference',
             'recurrent_depth_seed23', 'recurrent_depth_seed29', 'recurrent_depth_seed31', 'recurrent_depth_seed17_reference'],
        ], 'evaluating_fresh_maps')
        event('summarizing')
        run('scripts.summarize_fresh_confirmation')
        event('complete_pending_documentation_and_archive')
    except BaseException as exc:
        event('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
