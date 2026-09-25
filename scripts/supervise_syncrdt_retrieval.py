"""Restart-safe supervisor for the Sync-RDT retrieval sample-efficiency study."""
from pathlib import Path
from scripts.study_supervisor import Study,supervise

STUDY=Study(root=Path('research/results/syncrdt_retrieval_v1'),runs=Path('research/runs/syncrdt_retrieval_v1'),
    worker_module='scripts.run_syncrdt_retrieval',freeze_module='scripts.freeze_syncrdt_retrieval',
    evaluate_module='scripts.evaluate_syncrdt_retrieval',summarize_module='scripts.summarize_syncrdt_retrieval',
    autostart=Path.home()/'.config/autostart/ctm-syncrdt-retrieval.desktop',
    # Study checks (including exact RDT parity of the off cell), plus the archived-trainer replay, on both GPUs.
    preflight_tests=[(['tests/test_sync_rdt.py','tests/test_syncrdt_retrieval.py','tests/test_presentation_control.py','-k','not replays or replays'],'cuda:0'),
                     (['tests/test_sync_rdt.py','tests/test_presentation_control.py','-k','reproduces or replays'],'cuda:1')])

if __name__=='__main__':supervise(STUDY)
