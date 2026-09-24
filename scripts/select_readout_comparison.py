"""Audit all declared trials and lock winners before held-out evaluation."""
import json,math
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash

ROOT=Path('research/results/readout_comparison_v1')


def freeze_selection():
    destination=ROOT/'selection.json'
    if destination.exists():raise ValueError('Selection is immutable; use the existing record')
    registry=json.loads((ROOT/'registry.json').read_text())
    pre=json.loads((ROOT/'pre_run_source.json').read_text())
    for path,digest in pre['files'].items():assert file_hash(path)==digest,('pre-run source changed',path)
    trials=[];reference=None
    for entry in registry['entries']:
        cell=entry['cell'];directory=Path('research/runs/readout_comparison_v1/seed17')/cell
        summary=json.loads((ROOT/f'{cell}.summary.json').read_text())
        run=json.loads((directory/'research_run.json').read_text())
        config=run['effective_config']
        assert summary['complete'] and summary['steps']==3000
        assert run['seed']==17 and run['data_seed']==18 and run['depth_seed']==19
        assert run['parameters']==entry['parameters']==summary['parameters']
        assert file_hash(entry['config'])==entry['config_sha256']==run['config_identity']['sha256']
        assert file_hash('scripts/run_readout_comparison.py')==summary['worker_source_sha256']
        for path,digest in run['code_sha256'].items():assert file_hash(path)==digest,(cell,path)
        if reference is None:reference=run
        assert run['data']==reference['data'] and run['tokenizer_sha256']==reference['tokenizer_sha256']
        assert run['checkpoint_selection']['readouts']==entry['selection_readouts']
        assert run['checkpoint_selection']['save_every_validation']
        expected={'tokens_seen':5184000,'supervised_tokens_seen':192000,'examples_seen':96000}
        for name,value in expected.items():assert summary[name]==value
        rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,3001))
        for row in rows:
            s=row['step'];factor=s/30 if s<=30 else .1+.45*(1+math.cos(math.pi*(s-30)/2970))
            assert math.isclose(row['lr'],config['learning_rate']*factor,rel_tol=1e-12)
            assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
            assert row['examples_seen']==s*32 and row['tokens_seen']==s*1728 and row['supervised_tokens_seen']==s*64
        validation=[r for r in rows if 'validation_readouts' in r];assert len(validation)==30
        candidates=[]
        for row in validation:
            assert (directory/f"step_{row['step']:06d}.pt").exists()
            values=row['validation_readouts']
            for index,policy in enumerate(entry['selection_readouts']):
                v=values[policy];assert v['target_tokens']==256 and len(v['predictions'])==128
                assert math.isfinite(v['loss'])
                assert v['generation']['correct']==sum(p['correct'] for p in v['predictions'])
                candidates.append((v['loss'],row['step'],index,policy))
            selected=min(entry['selection_readouts'],key=lambda p:values[p]['loss'])
            assert row['validation']['readout_policy']==selected and row['validation']['loss']==values[selected]['loss']
            assert sum(values['diagnostics']['confidence_tick_histogram'])==256
        best=min(candidates);assert summary['best_validation_loss']==best[0] and summary['best_readout_policy']==best[3]
        checkpoint=directory/'best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['step']==best[1] and saved['readout_policy']==best[3]
        assert saved['tokens_seen']==best[1]*1728
        record={'cell':cell,'model_family':entry['model_family'],'validation_ce':best[0],'step':best[1],'readout':best[3],
            'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':file_hash(checkpoint),
            'validation_generation':next(r for r in validation if r['step']==best[1])['validation_readouts'][best[3]]['generation'],
            'config_sha256':entry['config_sha256'],'summary_sha256':file_hash(ROOT/f'{cell}.summary.json'),
            'metrics_sha256':file_hash(directory/'metrics.jsonl'),'parameters':summary['parameters'],
            'training_seconds':summary['training_seconds'],'wall_seconds':summary['wall_seconds'],
            'peak_allocated_bytes':summary['peak_allocated_bytes'],'token_block_applications':summary['token_block_applications']}
        trials.append(record)
    winners={family:min([t for t in trials if t['model_family']==family],key=lambda t:t['validation_ce']) for family in ('ctm','transformer','recurrent_depth')}
    result={'complete':True,'rule':'minimum validation CE over declared checkpoints/readouts/trials; earlier step, then final readout, then registry trial order for exact ties',
        'no_test_forward_before_selection':True,'registry_sha256':file_hash(ROOT/'registry.json'),
        'pre_run_source_sha256':file_hash(ROOT/'pre_run_source.json'),
        'selection_source_sha256':file_hash('scripts/select_readout_comparison.py'),
        'audits':'All 27000 updates, exposure/LR counters, 270 validation checkpoints, declared policies, checkpoint identities, and frozen training sources verified.',
        'data':reference['data'],'tokenizer_sha256':reference['tokenizer_sha256'],
        'training_source_sha256':reference['code_sha256'],'trials':trials,'winners':winners}
    with destination.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(winners,indent=2))
    return result

if __name__=='__main__':freeze_selection()
