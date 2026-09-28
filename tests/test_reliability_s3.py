"""Checks for the S3 reliability study statistics and factory mapping."""
import pytest

from scripts.run_reliability_s3 import factory_for
from scripts.summarize_reliability_s3 import clopper_pearson, holm, mcnemar_exact, sign_flip_exact


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0 and abs(hi - 0.1684) < 1e-3
    lo, hi = clopper_pearson(10, 20)
    assert abs(lo - 0.2720) < 1e-3 and abs(hi - 0.7280) < 1e-3


def test_exact_paired_tests():
    assert mcnemar_exact([True] * 6 + [False] * 14, [False] * 20)['p'] == pytest.approx(2 * 0.5 ** 6)
    assert mcnemar_exact([True, False], [True, False])['p'] == 1.0
    result = sign_flip_exact([1.0] * 8)
    assert result['p'] == pytest.approx(2 / 256) and result['mean_difference'] == 1.0
    assert sign_flip_exact([1.0, -1.0, 1.0, -1.0])['p'] == 1.0


def test_holm_adjustment_is_monotone_and_capped():
    assert holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_factory_mapping():
    assert factory_for({'factory_kind': 'ctm_lm', 'factory_arg': 'tiny'}).__qualname__ == 'ctm_lm_factory[tiny]'
    assert factory_for({'factory_kind': 'sync_rdt', 'factory_arg': 'rdt'}).__qualname__ == 'cell_factory[rdt]'
