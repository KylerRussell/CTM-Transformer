"""Checks for the S3 randomized-depth study wiring."""
import torch

from scripts.run_randdepth_s3 import factory_for, sampler_for


def test_sampler_and_factories_follow_the_registry_entry():
    entry = {'depth_sampler': {'kind': 'lognormal_poisson', 'mean': 15, 'sigma': 0.5, 'maximum': 48}}
    sampler = sampler_for(entry)
    g = torch.Generator().manual_seed(3)
    draws = [sampler(g) for _ in range(2000)]
    assert max(draws) <= 48 and min(draws) >= 1 and 14.0 < sum(draws) / len(draws) < 17.5
    assert factory_for({'factory_kind': 'ctm_lm', 'factory_arg': 'tiny'}).__qualname__ == 'ctm_lm_factory[tiny]'
    assert factory_for({'factory_kind': 'sync_rdt', 'factory_arg': 'rdt'}).__qualname__ == 'cell_factory[rdt]'
