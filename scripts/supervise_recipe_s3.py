"""Restart-safe supervisor for the S3 recipe confirmation."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/recipe_s3_v1'),runs=Path('research/runs/recipe_s3_v1'),
    worker_module='scripts.run_recipe_s3',freeze_module='scripts.freeze_recipe_s3',
    evaluate_module='scripts.evaluate_recipe_s3',summarize_module='scripts.summarize_recipe_s3',
    autostart=Path.home()/'.config/autostart/ctm-recipe-s3.desktop',
    preflight_tests=[(['tests/test_recipe_s3.py','tests/test_positions.py','tests/test_curriculum.py','tests/test_depth_sampling.py','tests/test_ctm_lm.py','tests/test_group_s3.py'],'cuda:0'),
                     (['tests/test_positions.py','tests/test_curriculum.py','tests/test_depth_sampling.py','-k','reproduces or train'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
