"""Restart-safe supervisor for the fresh-map pointer study."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/online_pointer_v1'),runs=Path('research/runs/online_pointer_v1'),
    worker_module='scripts.run_online_pointer',freeze_module='scripts.freeze_online_pointer',
    evaluate_module='scripts.evaluate_online_pointer',summarize_module='scripts.summarize_online_pointer',
    autostart=Path.home()/'.config/autostart/ctm-online-pointer.desktop',
    # Study checks, plus exact replay of the archived trainer on both GPUs to catch environment drift.
    preflight_tests=[(['tests/test_online_pointer.py','tests/test_presentation_control.py','-k','online or replays'],'cuda:0'),
                     (['tests/test_presentation_control.py','-k','replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
