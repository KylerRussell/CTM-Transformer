"""Checks for the S3 learning-rate control wiring."""
import json
from pathlib import Path

from scripts.prepare_lrcontrol_s3 import CELLS, CONTROL, SEEDS
from scripts.run_lrcontrol_s3 import factory_for, sampler_for


def test_the_control_cell_is_fixed_depth_rdt_at_the_randomized_cells_learning_rate():
    control = json.loads((CONTROL / 'registry.json').read_text())
    (family, arg, lr, sampler, readout, ticks, _), = CELLS.values()
    rand = control['cells']['rdt_rope_rand']
    assert (family, arg, readout, ticks) == ('recurrent_depth', rand['factory_arg'], rand['readout'], rand['evaluation_thought_steps'])
    assert lr == rand['learning_rate'] == 1e-3 and sampler is None and control['cells']['rdt_rope_fixed']['depth_sampler'] is None
    assert list(SEEDS) == control['seeds']
    assert factory_for({'factory_kind': 'position', 'factory_arg': arg}).__qualname__ == 'position_factory[rdt,rope]'
    assert sampler_for({'depth_sampler': None}) is None


def test_the_paired_control_evaluations_are_recorded():
    summary = json.loads((CONTROL / 'summary.json').read_text())
    for cell in ('rdt_rope_rand', 'rdt_rope_fixed'):
        for seed in SEEDS:
            assert Path(summary['evaluations'][f'{cell}_seed{seed}']['path']).exists()
