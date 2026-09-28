"""Checks for the locked confirmation: the 6-of-7 rule and the factory mapping."""
from scripts.run_sync_confirmation import factory_for
from scripts.summarize_sync_confirmation import exceeds


def test_six_of_seven_rule_with_median_margin():
    base = [5, 5, 5, 5, 5, 5, 5]
    assert exceeds([9, 9, 9, 9, 9, 9, 4], base)['first_exceeds']            # 6 of 7 larger, median +4
    assert not exceeds([9, 9, 9, 9, 9, 4, 4], base)['first_exceeds']        # only 5 of 7 larger
    assert not exceeds([6, 6, 6, 6, 6, 6, 6], base)['first_exceeds']        # median +1 < 2
    assert exceeds([1, 1, 1, 1, 1, 1, 9], base)['second_exceeds']


def test_confirmation_factories_name_declared_cells():
    assert factory_for({'factory_kind': 'sync_ablation', 'factory_arg': 'current'}).__qualname__ == 'ablation_factory[current]'
    assert factory_for({'factory_kind': 'sync_rdt', 'factory_arg': 'sync'}).__qualname__ == 'cell_factory[sync]'
    assert factory_for({'factory_kind': None}) is None
