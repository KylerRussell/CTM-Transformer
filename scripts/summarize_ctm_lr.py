"""Report all six CTM development cells without evaluating test maps."""
import json
from pathlib import Path
from ctm_transformer.experiment import file_hash
from scripts.confirmation_common import read

ROOT = Path('research/results/ctm_lr_v1')


def main():
    selected = read(ROOT / 'selection.json')
    assert selected['complete']
    trials = selected['trials']
    winner = selected['winner']
    lines = ['# CTM learning-rate development', '',
        'Six declared objective/LR cells at seed 17; the two LR 0.001 controls are reused unchanged from the preceding study. Four new trials completed the full 3,000-update budget. Checkpoint and readout selection use only the original validation maps. No test forwards belong to this development stage.', '',
        f"Selected recipe: **{winner['objective']}, peak LR {winner['learning_rate']:g}, {winner['readout']} readout**, update {winner['step']}, validation CE {winner['validation_ce']:.6g} and exact answer+EOS accuracy {winner['validation_generation']['exact_match']:.2%}.", '',
        '| Objective | Peak LR | Readout | Update | Validation CE | Validation accuracy | Training min | Run min | Reused |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|']
    curves = {}
    for t in trials:
        path = Path(t['run_directory']) / 'metrics.jsonl'
        assert file_hash(path) == t['metrics_sha256']
        curves[t['cell']] = [r for r in map(json.loads, path.read_text().splitlines()) if 'validation_readouts' in r]
        lines.append(f"| {t['objective']} | {t['learning_rate']:g} | {t['readout']} | {t['step']} | {t['validation_ce']:.6g} | {t['validation_generation']['exact_match']:.2%} | {t['training_seconds']/60:.1f} | {t['wall_seconds']/60:.1f} | {'yes' if t['reused'] else 'no'} |")
    new = [t for t in trials if not t['reused']]
    lines += ['', f"New training cost: {sum(t['training_seconds'] for t in new)/60:.1f} GPU minutes; total run cost including validation/checkpoints: {sum(t['wall_seconds'] for t in new)/60:.1f} GPU minutes. These sums are not elapsed wall time because two GPUs ran concurrently.", '',
        '![Validation CE by objective, LR, and readout](validation_curves.png)', '',
        'T16/H8, width 128, two CTM layers, 544,839 parameters, batch 32, BF16 autocast with FP32 parameters/moments. Each trial sees 96,000 examples, 5,184,000 input tokens and 192,000 supervised answer/EOS labels. The monotonic penalty is zero. The architecture and temporal objectives are unchanged.', '',
        'The next stage freezes this objective/LR/readout, retains the earlier baseline recipes, and trains all three families at seeds 23, 29 and 31 before fresh-map evaluation. Selection at one development seed can be seed-sensitive; the new-seed results are the primary confirmation. Tuning effort and compute remain unequal across families.', '',
        'See [frozen plan](PLAN_BEFORE_RUNS.md), `registry.json`, `selection.json`, and `pre_run_source.json` for exact choices and hashes.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    colors = {0.0003: 'tab:blue', 0.001: 'tab:orange', 0.003: 'tab:green'}
    for ax, objective in zip(axes, ('uniform', 'dynamic')):
        for t in trials:
            if t['objective'] != objective:
                continue
            rows = curves[t['cell']]
            for policy in ('final', 'confidence'):
                ax.plot([r['step'] for r in rows], [r['validation_readouts'][policy]['loss'] for r in rows],
                        color=colors[t['learning_rate']], linestyle='-' if policy == 'final' else '--',
                        label=f"{t['learning_rate']:g} / {policy}")
        ax.set(title=objective, xlabel='Optimizer updates', ylabel='Validation answer/EOS CE')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('CTM T16: six declared objective/LR cells, development seed 17')
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT / f'validation_curves.{ext}', dpi=180)
    plt.close(fig)
    print(lines[4])


if __name__ == '__main__':
    main()
