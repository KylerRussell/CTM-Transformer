"""Restart-safe supervisor for the locked confirmation of second-order query features."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/sync_confirmation_v1'),runs=Path('research/runs/sync_confirmation_v1'),
    worker_module='scripts.run_sync_confirmation',freeze_module='scripts.freeze_sync_confirmation',
    evaluate_module='scripts.evaluate_sync_confirmation',summarize_module='scripts.summarize_sync_confirmation',
    autostart=Path.home()/'.config/autostart/ctm-sync-confirmation.desktop',
    # Confirmation checks, group and Sync-RDT checks (RDT parity), ablation checks (identity parity) and the archived-trainer replay.
    preflight_tests=[(['tests/test_sync_confirmation.py','tests/test_group_s3.py','tests/test_group_word.py','tests/test_sync_rdt.py','tests/test_sync_ablations.py','tests/test_presentation_control.py','-k','not replays or replays'],'cuda:0'),
                     (['tests/test_sync_rdt.py','tests/test_presentation_control.py','-k','reproduces or replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
