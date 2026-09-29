"""Tabulate the S3 recipe development probes (development only; not a paper endpoint)."""
import json,statistics
from collections import defaultdict
from pathlib import Path

OUT=Path('research/results/recipe_probes')


def main():
    groups=defaultdict(dict)
    for f in sorted(OUT.glob('*.json')):
        r=json.loads(f.read_text());ev=r['eval_by_ticks']
        trained='1' if r['model']=='transformer' else '16';extra='1' if r['model']=='transformer' else '32'
        groups[(r['model'],r['positions'],r['depth'],r['curriculum'],r['learning_rate'])][r['seed']]={
            'p1_16':ev[trained]['positions_1_16'],'p9_16':ev[trained]['positions_9_16'],
            'p17_24':statistics.mean(ev[extra]['by_position'][16:24]),'p1_16_T4':ev['4']['positions_1_16'] if '4' in ev else None,
            'minutes':r['training_minutes']}
    L=['| Model | Positions | Depth | Curriculum | LR | Seeds | Positions 9–16 per seed | Mean positions 1–16 | Mean positions 17–24 | Escaped |',
       '|---|---|---|---|---:|---|---|---:|---:|---:|']
    for (model,pos,depth,cur,lr),runs in sorted(groups.items()):
        seeds=sorted(runs);per_seed=', '.join(f"{runs[s]['p9_16']:.2f}" for s in seeds)
        L.append(f"| {model} | {pos} | {depth if model!='transformer' else '—'} | {cur} | {lr:g} | {', '.join(map(str,seeds))} | "
                 f"{per_seed} | {statistics.mean(runs[s]['p1_16'] for s in seeds):.3f} | "
                 f"{statistics.mean(runs[s]['p17_24'] for s in seeds):.3f} | {sum(runs[s]['p9_16']>=0.9 for s in seeds)}/{len(seeds)} |")
    (OUT/'probe_table.md').write_text('\n'.join(L)+'\n');print('\n'.join(L))

if __name__=='__main__':main()
