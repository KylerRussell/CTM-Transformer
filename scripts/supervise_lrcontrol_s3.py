"""Restart-safe supervisor for the S3 learning-rate control."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/lrcontrol_s3_v1'),runs=Path('research/runs/lrcontrol_s3_v1'),
    worker_module='scripts.run_lrcontrol_s3',freeze_module='scripts.freeze_lrcontrol_s3',
    evaluate_module='scripts.evaluate_lrcontrol_s3',summarize_module='scripts.summarize_lrcontrol_s3',
    autostart=Path.home()/'.config/autostart/ctm-lrcontrol-s3.desktop',
    # Development probes may share the GPUs with this study, so the idle-GPU preflight is relaxed.
    busy_mib=80000,
    preflight_tests=[(['tests/test_lrcontrol_s3.py','tests/test_recipe_s3.py','tests/test_positions.py','tests/test_depth_sampling.py'],'cuda:0'),
                     (['tests/test_positions.py','tests/test_curriculum.py','-k','reproduces or train'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
