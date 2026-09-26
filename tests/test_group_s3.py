"""Checks for the S3 group-word study: data suite, correct-prefix metric and decision logic."""
import pytest

from ctm_transformer.group_suite import create_group_suite, load_group_split
from ctm_transformer.group_word import MIN_HELD_OUT
from scripts.run_group_s3 import factory_for
from scripts.summarize_group_s3 import correct_prefix, exceeds


def test_group_suite_is_deterministic_and_holds_out_long_words(tmp_path):
    a = create_group_suite(tmp_path / 'a', 'S3', [3, 5], 2000, 16, 160, 64, 32, 'x')
    create_group_suite(tmp_path / 'b', 'S3', [3, 5], 2000, 16, 160, 64, 32, 'x')
    for seed, splits in a['seeds'].items():
        for split, entry in splits.items():
            assert (tmp_path / 'a' / entry['path']).read_bytes() == (tmp_path / 'b' / entry['path']).read_bytes()
    train, evaluation = (load_group_split(tmp_path / 'a', 3, s) for s in ('train', 'evaluation'))
    assert train.metadata['lengths'] == list(range(1, 17)) and evaluation.metadata['lengths'] == [32]
    assert not (train.ids & evaluation.ids) and all(r['steps'] >= MIN_HELD_OUT for r in evaluation.records)
    with pytest.raises(ValueError, match='empty'):
        create_group_suite(tmp_path / 'a', 'S3', [3], 10, 4, 10, 10, 8, 'x')


def test_correct_prefix_counts_only_leading_positions():
    assert correct_prefix([1.0, 0.95, 0.9, 0.5, 1.0]) == 3
    assert correct_prefix([0.4, 1.0]) == 0 and correct_prefix([1.0] * 32) == 32


def test_exceeds_requires_seed_majority_and_median_margin():
    assert exceeds([12, 11, 10, 9, 3], [8, 8, 8, 8, 8])['first_exceeds']
    assert not exceeds([9, 9, 9, 9, 3], [8, 8, 8, 8, 8])['first_exceeds']      # median margin 1 < 2
    assert not exceeds([12, 12, 12, 3, 3], [8, 8, 8, 8, 8])['first_exceeds']   # only 3 seeds larger
    assert exceeds([4, 4, 4, 4, 4], [8, 8, 8, 8, 8])['second_exceeds']


def test_factory_mapping_names_declared_cells():
    assert factory_for({'factory_kind': None}) is None
    assert factory_for({'factory_kind': 'sync_rdt', 'factory_arg': 'sync'}).__qualname__ == 'cell_factory[sync]'
    assert factory_for({'factory_kind': 'ctm_variant', 'factory_arg': 'reference'}).__qualname__ == 'variant_factory[reference]'
