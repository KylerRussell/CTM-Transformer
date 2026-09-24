"""Restart-safe supervisor for the Transformer pointer calibration."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/pointer_calibration_v1'),runs=Path('research/runs/pointer_calibration_v1'),
    worker_module='scripts.run_pointer_calibration',freeze_module='scripts.freeze_pointer_calibration',
    evaluate_module='scripts.evaluate_pointer_calibration',summarize_module='scripts.summarize_pointer_calibration',
    autostart=Path.home()/'.config/autostart/ctm-pointer-calibration.desktop',
    # Study checks, plus exact replay of the archived trainer on both GPUs to catch environment drift.
    preflight_tests=[(['tests/test_pointer_calibration.py','tests/test_presentation_control.py','-k','calibration or replays'],'cuda:0'),
                     (['tests/test_presentation_control.py','-k','replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
