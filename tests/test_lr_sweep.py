"""The learning-rate sweep's adaptive grid and scaling fit recover a known optimum (scripts/lr_sweep.py, scripts/summarize_lr_sweep.py)."""
import json
import math

import pytest

from scripts import lr_sweep, summarize_lr_sweep
from scripts.lr_sweep import LADDER, TOKENS, plan, run_name, sweep_config


def true_k(arm, n, steps):
    """Known optimum in grid units: log2 lr* = log2(1e-3) + k, slope -0.6 in log2 N, -0.3 per doubling of steps."""
    offset = {'transformer': 1.0, 'rdt_aware': 0.8, 'ctm_aware': 0.2, 'rdt_heavy': 0.6, 'ctm_heavy': -0.4}[arm]
    return offset - 0.6 * math.log2(n / 2**24) - 0.3 * math.log2(steps / 381)


def finish(root, arm, d, k, h, fail=False):
    out = root / run_name(arm, d, k, h)
    out.mkdir(parents=True)
    run = sweep_config(arm, d, k, h)
    (out / 'config.json').write_text(json.dumps(run))
    if fail:
        (out / 'failure.json').write_text('{}')
        return
    n, steps = 100 * d * d, run['train']['total_steps']
    (out / 'run.json').write_text(json.dumps({'non_embedding_parameters': n, 'run': run}))
    if k >= 3:
        (out / 'diverged.json').write_text('{}')
        return
    metric = lr_sweep.METRIC[run['family']]
    loss = 3 + 0.02 * (k - true_k(arm, n, steps)) ** 2
    (out / 'metrics.jsonl').write_text(json.dumps({'step': steps, 'validation': {metric: loss}}) + '\n')
    (out / 'complete.json').write_text('{}')


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(lr_sweep, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(summarize_lr_sweep, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(summarize_lr_sweep, 'RESULTS', tmp_path / 'results')
    return tmp_path


def run_to_completion(root):
    launched = set()
    for _ in range(100):
        want, blocked = plan()
        pending = [r for r in want if lr_sweep.outcome(run_name(*r))[0] is None]
        if not pending:
            return launched, blocked
        for r in pending:
            finish(root / 'runs', *r)
            launched.add(r)
    raise AssertionError('the sweep did not terminate')


def test_sweep_configs_keep_the_global_batch_and_use_disjoint_validation_windows():
    for arm, widths in LADDER.items():
        for d in widths:
            run = sweep_config(arm, d, -1, 4)
            t = run['train']
            assert t['micro_batch'] * t['accumulation'] == 256 and t['seq_len'] == 1024 and not t['offload_optimizer']
            assert t['total_steps'] == round(4 * TOKENS / 262144) and t['lr'] == pytest.approx(5e-4)
            assert t['eval_offset'] >= 40_000 and t['final_eval_windows'] == 1024 and t['delete_checkpoint_on_complete']
            assert run['model']['d_model'] == d and run['run_directory'].endswith(run_name(arm, d, -1, 4))


def test_adaptive_grid_brackets_every_optimum_and_the_fit_recovers_the_scaling(sweep):
    launched, blocked = run_to_completion(sweep)
    assert not blocked
    for arm in LADDER:
        assert {d for a, d, _, h in launched if a == arm and h == 1} == set(LADDER[arm])
        assert any(a == arm and h == 4 for a, _, _, h in launched) == (arm in lr_sweep.HORIZON_ARMS)
    summarize_lr_sweep.main()
    out = json.loads((sweep / 'results' / 'lr_sweep.json').read_text())
    for w in out['widths']:  # the parabola vertex recovers the exact optimum
        assert w['k_star'] == pytest.approx(true_k(w['arm'], w['non_embedding_parameters'], w['steps']), abs=1e-9)
    for arm, fit in out['fits'].items():
        assert fit['b'] == pytest.approx(-0.6, abs=1e-9)
    for arm, c in out['horizon_exponents'].items():
        assert c == pytest.approx(-0.3, abs=1e-9)
    p = out['predictions']['rdt_heavy']
    expected = math.log2(1e-3) + true_k('rdt_heavy', summarize_lr_sweep.N_500['rdt_heavy'], p['steps'])
    assert math.log2(p['lr']) == pytest.approx(expected, abs=1e-6)
    assert out['shared_slope_fit']['b'] == pytest.approx(-0.6, abs=1e-9)


def test_vertex_interpolates_between_neighbours_and_falls_back_to_the_grid_point_next_to_a_divergence():
    assert summarize_lr_sweep.vertex({-1: 3.02, 0: 3.0, 1: 3.01}, 0) == pytest.approx(1 / 6)
    assert summarize_lr_sweep.vertex({-1: 3.02, 0: 3.0, 1: math.inf}, 0) == 0.0
    assert summarize_lr_sweep.vertex({-1: 3.5, 0: 3.0, 1: 3.0001}, 0) <= 1.0


def test_a_failed_run_blocks_its_arm_and_is_not_retried(sweep):
    finish(sweep / 'runs', 'ctm_heavy', 448, 0, 1, fail=True)
    want, blocked = plan()
    assert blocked and ('ctm_heavy', 448, 0, 1) in want
    assert not any(a == 'ctm_heavy' and d != 448 for a, d, _, _ in want)
    assert lr_sweep.outcome(run_name('ctm_heavy', 448, 0, 1))[0] == 'failed'
