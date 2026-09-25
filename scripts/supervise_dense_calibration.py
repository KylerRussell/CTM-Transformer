"""Restart-safe supervisor for the dense Transformer calibration (Stage D1a)."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/dense_calibration_v1'),runs=Path('research/runs/dense_calibration_v1'),
    worker_module='scripts.run_dense_calibration',freeze_module='scripts.freeze_dense_calibration',
    evaluate_module='scripts.evaluate_dense_calibration',summarize_module='scripts.summarize_dense_calibration',
    autostart=Path.home()/'.config/autostart/ctm-dense-calibration.desktop',
    # Dense checks (including exact v3-v2 runner parity), plus archived-trainer replay, on both GPUs.
    preflight_tests=[(['tests/test_dense_pointer.py','tests/test_presentation_control.py','-k','dense or replays'],'cuda:0'),
                     (['tests/test_dense_pointer.py','tests/test_presentation_control.py','-k','reproduces or replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
