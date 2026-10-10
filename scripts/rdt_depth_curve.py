"""Held-out loss of trained RDT recipe runs against test-time depth (research/RDT_RECIPE.md, 2026-10-10).

Loads a run's final checkpoint and evaluates it, with the trainer's evaluate_lm, on the run's final
evaluation windows at each requested depth. Writes research/results/rdt_recipe/depth_curve_<run>.json.

Usage: python -m scripts.rdt_depth_curve --gpu N --runs NAME... [--depths 1 2 4 ...]
"""
import argparse,json,os
from pathlib import Path

RUNS=Path('research/runs/rdt_recipe');OUT=Path('research/results/rdt_recipe')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--depths',type=int,nargs='+',default=[1,2,4,8,12,16,24,32,48]);a=p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES',str(a.gpu))
    import torch
    from scripts.pretrain import build_config
    from ctm_transformer.pretrain import TokenWindows,evaluate_lm
    from ctm_transformer.rdt_recipe import recipe_factory
    device=torch.device('cuda');OUT.mkdir(parents=True,exist_ok=True)
    for name in a.runs:
        run=json.loads((RUNS/name/'config.json').read_text());t=run['train']
        data=json.loads(Path(run['data']).read_text());root=Path(run['data']).parent
        model=recipe_factory(run['rdt_recipe'])(build_config(run,data['vocab_size']))
        state=torch.load(RUNS/name/f'step_{t["total_steps"]:07d}.pt',map_location='cpu',weights_only=False)['model']
        model.load_state_dict(state);model.to(device)
        valid=TokenWindows([root/s for s in data['validation']],t['seq_len'],0)
        offset=t.get('eval_offset',0);n=t.get('final_eval_windows',t['eval_windows'])
        indices=list(range(offset,min(offset+n,len(valid))))
        with torch.no_grad():
            losses=evaluate_lm(model,valid,indices,t['eval_micro_batch'],device,'rdt',depths=tuple(a.depths))
        result={'run':name,'windows':len(indices),'loss_by_depth':{k.split('_')[1]:v for k,v in losses.items()}}
        (OUT/f'depth_curve_{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,' '.join(f'{d}:{v:.3f}' for d,v in result['loss_by_depth'].items()),flush=True)


if __name__=='__main__':main()
