"""Restart-safe supervisor for the S3 reliability (escape probability) study."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/reliability_s3_v1'),runs=Path('research/runs/reliability_s3_v1'),
    worker_module='scripts.run_reliability_s3',freeze_module='scripts.freeze_reliability_s3',
    evaluate_module='scripts.evaluate_reliability_s3',summarize_module='scripts.summarize_reliability_s3',
    autostart=Path.home()/'.config/autostart/ctm-reliability-s3.desktop',
    preflight_tests=[(['tests/test_reliability_s3.py','tests/test_ctm_lm.py','tests/test_group_s3.py','tests/test_group_word.py','tests/test_sync_rdt.py','tests/test_presentation_control.py','-k','not replays or replays'],'cuda:0'),
                     (['tests/test_sync_rdt.py','tests/test_presentation_control.py','-k','reproduces or replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
