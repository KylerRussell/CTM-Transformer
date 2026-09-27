"""Restart-safe supervisor for Sync-RDT ablation A7."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/sync_ablation_v2'),runs=Path('research/runs/sync_ablation_v2'),
    worker_module='scripts.run_sync_ablation_v2',freeze_module='scripts.freeze_sync_ablation_v2',
    evaluate_module='scripts.evaluate_sync_ablation_v2',summarize_module='scripts.summarize_sync_ablation_v2',
    autostart=Path.home()/'.config/autostart/ctm-sync-ablation-a7.desktop',
    # Ablation checks (including exact identity parity), Sync-RDT parity and the archived-trainer replay, on both GPUs.
    preflight_tests=[(['tests/test_sync_ablation_self.py','tests/test_sync_ablations.py','tests/test_sync_rdt.py','tests/test_group_s3.py','tests/test_presentation_control.py','-k','not replays or replays'],'cuda:0'),
                     (['tests/test_sync_rdt.py','tests/test_presentation_control.py','-k','reproduces or replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
