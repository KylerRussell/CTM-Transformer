"""Summarize fixed-recipe seed replication on paired, previously unused maps."""
import json
import statistics
from pathlib import Path
from ctm_transformer.experiment import file_hash
from scripts.confirmation_common import read

ROOT = Path('research/results/fresh_confirmation_v1')
FAMILIES = ('ctm', 'transformer', 'recurrent_depth')
SPLITS = ('test_id', 'test_shuffled')


def mean_sd(values):
    assert len(values) == 3
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'values': values, 'n_training_seeds': 3}


def main():
    frozen = read(ROOT / 'checkpoints.json')
    recipes = read(ROOT / 'recipes.json')
    evaluations = {}
    for record in frozen['models']:
        path = ROOT / f"{record['cell']}.evaluation.json"
        e = read(path)
        assert e['complete'] and e['checkpoint_freeze_sha256'] == file_hash(ROOT / 'checkpoints.json')
        assert e['checkpoint']['sha256'] == record['checkpoint_sha256'] == file_hash(record['checkpoint'])
        assert e['selected'] == record
        for source, digest in e['source_sha256'].items():
            assert file_hash(source) == digest, source
        for split in SPLITS:
            m = e['results'][split]['metrics']
            assert m['target_tokens'] == 1024 and len(m['predictions']) == 512
            assert len({p['id'] for p in m['predictions']}) == 512
            assert m['generation']['correct'] == sum(p['correct'] for p in m['predictions'])
            assert m['generation']['exact_match'] == m['generation']['correct'] / 512
        a, b = [e['results'][s]['metrics']['predictions'] for s in SPLITS]
        assert [(p['id'], p['answer']) for p in a] == [(p['id'], p['answer']) for p in b]
        evaluations[record['cell']] = e
    seeds = recipes['primary_seeds']
    assert seeds == [23, 29, 31]
    rows = [e for e in evaluations.values() if e['role'] == 'independent_confirmation']
    references = [e for e in evaluations.values() if e['role'] == 'development_reference']
    assert len(rows) == 9 and len(references) == 3
    aggregate = {}
    for family in FAMILIES:
        subset = sorted([e for e in rows if e['model_family'] == family], key=lambda e: e['seed'])
        assert [e['seed'] for e in subset] == seeds
        aggregate[family] = {
            split: {'accuracy': mean_sd([e['results'][split]['metrics']['generation']['exact_match'] for e in subset]),
                    'ce': mean_sd([e['results'][split]['metrics']['loss'] for e in subset])}
            for split in SPLITS}
        aggregate[family]['training_minutes'] = mean_sd([e['selected']['training_seconds']/60 for e in subset])
    contrasts = {}
    for family in ('transformer', 'recurrent_depth'):
        for split in SPLITS:
            deltas = []
            for seed in seeds:
                a, b = [evaluations[f'{f}_seed{seed}']['results'][split]['metrics']['predictions'] for f in ('ctm', family)]
                assert [(p['id'], p['answer']) for p in a] == [(p['id'], p['answer']) for p in b]
                deltas.append(sum(int(x['correct'])-int(y['correct']) for x,y in zip(a,b))/512)
            contrasts[f'ctm_minus_{family}_{split}'] = mean_sd(deltas)
    summary = {'complete': True, 'checkpoint_freeze_sha256': file_hash(ROOT / 'checkpoints.json'),
               'recipes_sha256': file_hash(ROOT / 'recipes.json'), 'primary_seeds': seeds, 'fresh_maps': 512,
               'aggregate': aggregate, 'paired_seed_accuracy_contrasts': contrasts,
               'evaluations': {cell: {'path': str(ROOT / f'{cell}.evaluation.json'),
                                      'sha256': file_hash(ROOT / f'{cell}.evaluation.json')} for cell in evaluations},
               'analysis_source_sha256': file_hash('scripts/summarize_fresh_confirmation.py'),
               'scope': 'Sample SD across three new training seeds on one shared set of 512 maps; development seed 17 excluded. No independence across paired maps/presentations or test-generator replication claimed.'}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Fresh-map confirmation across independent training seeds', '',
             'Recipes were frozen before new training. All nine new runs completed 3,000 updates, and all selected checkpoints were locked before any fresh-test forward. Evaluation uses 512 previously unused semantic maps, with paired ordered and shuffled presentations. Training uses the existing ordered maps.', '',
             '## Primary result: seeds 23, 29, 31', '',
             '| Family | Ordered accuracy, mean ± SD | Shuffled accuracy, mean ± SD | Ordered CE, mean ± SD | Shuffled CE, mean ± SD |',
             '|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        a = aggregate[family]
        accuracy = [f"{100*a[s]['accuracy']['mean']:.2f}% ± {100*a[s]['accuracy']['sample_sd']:.2f}" for s in SPLITS]
        ce = [f"{a[s]['ce']['mean']:.5g} ± {a[s]['ce']['sample_sd']:.3g}" for s in SPLITS]
        lines.append('| ' + ' | '.join([family] + accuracy + ce) + ' |')
    lines += ['', 'Accuracy requires unrestricted greedy generation of the correct answer and EOS. CE is teacher-forced and token-weighted over answer/EOS. SD is sample standard deviation across three training seeds, not a confidence interval. All seeds share the same 512 maps; 1,536 predictions per family do not constitute 1,536 independent test maps.', '',
              '## Locked recipes', '', '| Family | Development recipe | Peak LR | Fixed readout | Parameters |', '|---|---|---:|---|---:|']
    for family in FAMILIES:
        r = recipes['recipes'][family]
        lines.append(f"| {family} | {r['cell']} | {r['learning_rate']:g} | {r['readout']} | {r['parameters']['total']:,} |")
    lines += ['', 'The CTM objective/LR/readout was selected on the six-cell seed-17 development grid. Baseline recipes were carried forward from the earlier comparison. Only the checkpoint update is selected separately for each confirmation seed, using original validation CE under the fixed family readout.', '',
              '## Per-seed results', '', '| Family | Seed | Selected update | Validation CE | Validation accuracy | Ordered accuracy | Shuffled accuracy | Training min | Run min | Peak GiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        for e in sorted([r for r in rows if r['model_family'] == family], key=lambda e:e['seed']):
            t=e['selected']
            accuracy = [e['results'][s]['metrics']['generation']['exact_match'] for s in SPLITS]
            lines.append(f"| {family} | {e['seed']} | {t['step']} | {t['validation_ce']:.6g} | {t['validation_generation']['exact_match']:.2%} | {accuracy[0]:.2%} | {accuracy[1]:.2%} | {t['training_seconds']/60:.1f} | {t['wall_seconds']/60:.1f} | {t['peak_allocated_bytes']/2**30:.3f} |")
    lines += ['', '![Fresh-map accuracy for each independent seed](fresh_accuracy.png)', '',
              '## Development seed 17: separate reference', '',
              'Seed 17 informed recipe selection and is excluded from the primary means and SDs.', '',
              '| Family | Ordered accuracy | Shuffled accuracy | Ordered CE | Shuffled CE |', '|---|---:|---:|---:|---:|']
    for e in references:
        m = [e['results'][s]['metrics'] for s in SPLITS]
        lines.append(f"| {e['model_family']} | {m[0]['generation']['exact_match']:.2%} | {m[1]['generation']['exact_match']:.2%} | {m[0]['loss']:.6g} | {m[1]['loss']:.6g} |")
    lines += ['', '## Paired seed differences', '', '| CTM minus baseline | Mean accuracy difference (points) | Sample SD (points) |', '|---|---:|---:|']
    for label, delta in contrasts.items():
        lines.append(f"| {label} | {100*delta['mean']:+.2f} | {100*delta['sample_sd']:.2f} |")
    lines += ['', 'Each difference pairs the same training seed and the same evaluation maps. Three seeds are too few for a precise account of optimization variability. No significance claim or unpaired binomial interval is implied.', '',
              '## Exposure and cost', '',
              'Each run uses 96,000 example presentations, 5,184,000 input tokens, and 192,000 supervised answer/EOS labels. Parameters are approximately matched, within 3.5%. Training time covers all 3,000 updates, even when an earlier checkpoint is selected. Run time includes validation and checkpoint writing.', '',
              '| Family | Confirmation runs | Training GPU min | Total run GPU min |', '|---|---:|---:|---:|']
    for family in FAMILIES:
        subset = [e['selected'] for e in rows if e['model_family'] == family]
        lines.append(f"| {family} | 3 | {sum(t['training_seconds'] for t in subset)/60:.1f} | {sum(t['wall_seconds'] for t in subset)/60:.1f} |")
    lines += ['', 'These are summed device times, not concurrent elapsed time or FLOPs. They exclude development tuning, setup and final evaluation. No new inference latency benchmark was run. The previous latency comparison applies to its particular checkpoints/readouts.', '',
              '## Scope and remaining controls', '',
              '- The held-out maps exclude all 2,560 semantic maps in the frozen prior data inventory. Ordered and shuffled presentations share the same maps, starts and answers.',
              '- This is one-hop lookup on eight-node cycles under ordered training. Shuffled test presentation measures transfer; it is not evidence about performance after shuffled training or multistep composition.',
              '- Auxiliary objectives, compute and historical tuning effort differ. These are complete-recipe comparisons; architecture attribution still requires matched temporal supervision and compute budgets.',
              '- Dynamic aggregation and label-free confidence readout are distinct choices. Gold-aware prefix minima are diagnostic envelopes, not deployment policies. No historical CTM reproduction claim follows from these results.',
              '- Fresh-seed variability and task restrictions must be retained in any paper claim. Further tuning must belong to a new development study with a new final evaluation plan.', '',
              'See [CTM LR development](../ctm_lr_v1/RESULTS.md), [frozen protocol](../../CTM_LR_CONFIRMATION.md), `recipes.json`, `checkpoints.json`, source archives, and per-example `*.evaluation.json` files.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    colors = ('tab:blue', 'tab:orange', 'tab:green')
    for ax, split, title in zip(axes, SPLITS, ('Fresh ordered maps', 'Same maps, shuffled presentation')):
        for x, (family, color) in enumerate(zip(FAMILIES, colors)):
            a = aggregate[family][split]['accuracy']
            ax.scatter([x-.09, x, x+.09], [100*v for v in a['values']], color=color, s=40, zorder=3)
            ax.errorbar(x, 100*a['mean'], yerr=100*a['sample_sd'], fmt='_', markersize=22, color='black', capsize=5)
        ax.set(title=title, xticks=range(3), xticklabels=('CTM', 'Transformer', 'Recurrent depth'),
               ylabel='Exact answer + EOS (%)', ylim=(0, 105))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen recipes: dots = seeds 23/29/31; black = mean ± sample SD')
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT / f'fresh_accuracy.{ext}', dpi=180)
    plt.close(fig)
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
