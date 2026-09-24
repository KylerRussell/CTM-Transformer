"""Summarize the six exploratory runs and save standalone learning curves."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, required=True)
    parser.add_argument('--run_root', type=Path, required=True)
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    tasks = ['addition','pointer']; families = ['transformer','recurrent_depth','ctm']
    names = {'transformer':'Transformer', 'recurrent_depth':'Recurrent depth', 'ctm':'CTM'}
    records, curves = {}, {}
    for task in tasks:
        records[task], curves[task] = {}, {}
        for family in families:
            path = args.results_root / f'{task}_{family}_seed{args.seed}.summary.json'
            records[task][family] = json.loads(path.read_text())
            metric_path = args.run_root / f'{task}_seed{args.seed}' / family / 'metrics.jsonl'
            curves[task][family] = [json.loads(line) for line in metric_path.read_text().splitlines()]
            assert records[task][family]['training']['complete']
    majority = {}
    for task in tasks:
        training = [json.loads(line) for line in (args.dataset_root / task / 'train.jsonl').read_text().splitlines()]
        answer = Counter(row['answer'] for row in training).most_common(1)[0][0]
        majority[task] = {'training_majority_answer': answer, 'accuracy': {}}
        for path in sorted((args.dataset_root / task).glob('*.jsonl')):
            if path.stem == 'train': continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            majority[task]['accuracy'][path.stem] = sum(r['answer'] == answer for r in rows) / len(rows)
    summary = {'purpose':'single-seed development pilot; unmatched parameters and compute',
               'seed':args.seed, 'majority_baselines':majority, 'runs':records}
    (args.results_root / 'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines = ['# Algorithmic pilot results', '',
             'One initialization seed, 600 updates × 32 examples per model/task. Best validation answer/EOS cross-entropy selects the checkpoint. These exploratory presets are not parameter- or compute-matched.', '',
             '| Task | Model | Selected step | ID exact match | Harder operation | Longer input | Joint OOD | Training seconds |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in tasks:
        for family in families:
            row=records[task][family]
            def accuracy(split): return next(iter(row['metrics'][split].values()))['exact_match']
            depth_split='ood_carry' if task=='addition' else 'ood_depth'
            selected=row['checkpoint']['training_counters']['step']
            lines.append(f"| {task} | {names[family]} | {selected} | {accuracy('test_id'):.1%} | {accuracy(depth_split):.1%} | {accuracy('ood_length'):.1%} | {accuracy('ood_joint'):.1%} | {row['training']['training_seconds']:.1f} |")
    lines.extend(['', '“Harder operation” means two carries at fixed two-digit width for addition, and paths of depth 4–6 at fixed eight-node size for pointer chasing. Exact match requires an unconstrained greedy answer followed by EOS.', '',
                  'Training seconds include batching, transfer, forward/backward, and optimizer updates; exclude validation, checkpoint writes, and post-training generation. GPUs ran separate task queues concurrently.', '',
                  '## Pointer ID accuracy by requested path depth', '', '| Model | Depth 1 | Depth 2 | Depth 3 |', '|---|---:|---:|---:|'])
    for family in families:
        groups=next(iter(records['pointer'][family]['metrics']['test_id'].values()))['by_difficulty']
        values=[groups[f'nodes=8,steps={d}']['exact_match'] for d in (1,2,3)]
        lines.append('| '+names[family]+' | '+' | '.join(f'{v:.1%}' for v in values)+' |')
    lines.extend(['', '## Simple references', ''])
    for task in tasks:
        base=majority[task]
        lines.append(f"- {task}: always emit the training majority answer `{base['training_majority_answer']}` and EOS → ID accuracy {base['accuracy']['test_id']:.1%}.")
    lines.extend(['- Pointer uniform-node guessing with EOS has expected accuracy 12.5% on eight-node maps and 8.3% on twelve-node maps. Excluding the known-impossible start node raises these references to 14.3% and 9.1%, without reading any edges.', '',
                  'Every per-example prediction, termination flag, and difficulty group is retained in the adjacent `.eval.json` files. `pilot_summary.json` includes counts, losses, provenance, and checkpoint hashes.', '',
                  '![Validation learning curves](learning_curves.png)', '',
                  'Learned positions beyond the training range confound the length-OOD results. One seed and small evaluation sets do not support a superiority claim. See [task protocol](../../ALGORITHMIC_TASKS.md).'])
    (args.results_root / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors={'transformer':'#2369bd','recurrent_depth':'#d87818','ctm':'#28944a'}
    for ax,task in zip(axes,tasks):
        for family in families:
            rows=[r for r in curves[task][family] if 'validation' in r]
            ax.plot([r['examples_seen'] for r in rows], [r['validation']['loss'] for r in rows],
                    marker='o', markersize=3, color=colors[family], label=names[family])
        ax.set(title=task.replace('_',' ').title(), xlabel='Training examples seen', ylabel='Validation answer/EOS cross-entropy')
        ax.grid(alpha=0.25); ax.legend()
    fig.suptitle(f'Development pilot, seed {args.seed}; parameters and compute unmatched')
    fig.savefig(args.results_root/'learning_curves.png',dpi=180)
    fig.savefig(args.results_root/'learning_curves.pdf')
    plt.close(fig)
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    main()
