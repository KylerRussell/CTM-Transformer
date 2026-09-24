import torch
from scripts.diagnose_tick_selection import trajectories


def test_gold_ce_envelope_does_not_guarantee_accuracy_monotonicity():
    # Correct at tick 1; higher probability for the gold token at tick 2,
    # but a distractor has become still more probable.
    probabilities=torch.tensor([[[.40,.35,.25],[.45,.50,.05]]])
    rows=trajectories(probabilities.log(),torch.tensor([0]))['curve']
    assert rows[1]['oracle_min_ce'] < rows[0]['oracle_min_ce']
    assert rows[0]['oracle_min_ce_tick_accuracy']==1
    assert rows[1]['oracle_min_ce_tick_accuracy']==0
    assert rows[1]['oracle_any_correct_accuracy']==1
    assert rows[1]['confidence_accuracy']==0


def test_confidence_ties_keep_earliest_tick_and_ignore_gold():
    logits=torch.zeros(2,4,3)
    a=trajectories(logits,torch.tensor([0,1]))
    b=trajectories(logits,torch.tensor([2,2]))
    for x,y in zip(a['curve'],b['curve']):
        assert x['confidence_tick_histogram']==y['confidence_tick_histogram']==[2,0,0,0]
