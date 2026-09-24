"""Post-hoc fixed-final-update confidence readout; no checkpoint search."""
import json
from pathlib import Path
import torch
from scripts.run_objective_depth import confidence_evaluation
from ctm_transformer.experiment import file_hash


def main():
    torch.set_num_threads(4);torch.cuda.set_device('cuda:0')
    root=Path('research/results/objective_depth_v1');records={}
    for cell in ('uniform_t4','dynamic_t4','uniform_t16','dynamic_t16'):
        run=Path('research/runs/temporal_ablation_v1/seed17/uniform') if cell=='uniform_t4' else Path('research/runs/objective_depth_v1/seed17')/cell
        r=confidence_evaluation(run/'final.pt','cuda:0',root/f'{cell}.final.confidence.json')
        records[cell]={'checkpoint':r['checkpoint'],'metrics':r['metrics']}
        print(cell,r['metrics'],flush=True)
    (root/'final_confidence_comparison.json').write_text(json.dumps({
        'complete':True,'post_hoc':True,'selection':'Fixed final update 3000 for every cell; no checkpoint search',
        'motivation':'User clarified best-across-ticks policy after the original study began. Assess final checkpoints because final-tick CE need not select a good confidence-readout checkpoint.',
        'source_sha256':{'scripts/eval_final_confidence.py':file_hash('scripts/eval_final_confidence.py')},
        'results':records},indent=2)+'\n')

if __name__=='__main__':main()
