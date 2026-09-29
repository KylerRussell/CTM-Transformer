"""Checks for the S3 recipe-confirmation wiring and statistics."""
import itertools

import numpy as np
import torch

from scripts.run_recipe_s3 import factory_for, sampler_for
from scripts.summarize_recipe_s3 import sign_flip_monte_carlo
from scripts.summarize_reliability_s3 import sign_flip_exact


def test_factories_and_samplers_follow_the_registry_entry():
    for arg in ('transformer,rope', 'rdt,rope', 'ctm_lm,rope'):
        assert factory_for({'factory_kind': 'position', 'factory_arg': arg}).__qualname__ == f'position_factory[{arg}]'
    assert sampler_for({'depth_sampler': None}) is None
    sampler = sampler_for({'depth_sampler': {'kind': 'lognormal_poisson', 'mean': 15, 'sigma': 0.5, 'maximum': 48}})
    g = torch.Generator().manual_seed(3)
    draws = [sampler(g) for _ in range(2000)]
    assert 1 <= min(draws) and max(draws) <= 48 and 14.0 < sum(draws) / len(draws) < 17.5


def test_monte_carlo_sign_flip_matches_the_exact_test():
    rng = np.random.default_rng(0)
    for shift in (0.0, 0.05, 0.12):
        d = rng.normal(shift, 0.15, 14)
        exact, mc = sign_flip_exact(d), sign_flip_monte_carlo(d, draws=400_000)
        assert abs(exact['p'] - mc['p']) < 0.005 and exact['mean_difference'] == mc['mean_difference']
    assert sign_flip_monte_carlo([0.1] * 30)['p'] < 1e-5
    assert sign_flip_monte_carlo([0.0] * 30)['p'] == 1.0
