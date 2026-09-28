"""Training-time recurrence-depth samplers.

``lognormal_poisson`` follows the randomized-depth recipe of Geiping et al.
(2025, arXiv 2502.05171): τ ~ Normal(log(mean) − σ²/2, σ), depth ~ Poisson(e^τ) + 1,
here capped at ``maximum`` for memory. Its expected depth is about ``mean`` + 1
before capping. Samplers take the runner's seeded depth generator, so depth
sequences are reproducible.
"""
import math

import torch


def lognormal_poisson(mean,sigma=0.5,maximum=64):
    def sample(generator):
        tau=torch.normal(math.log(mean)-sigma**2/2,sigma,(),generator=generator)
        depth=int(torch.poisson(torch.exp(tau).reshape(1),generator=generator).item())+1
        return min(depth,maximum)
    sample.description=f'lognormal_poisson(mean={mean}, sigma={sigma}, maximum={maximum}); depth = min(Poisson(exp(tau)) + 1, maximum)'
    return sample


def fixed(depth):
    def sample(generator):return depth
    sample.description=f'fixed({depth})'
    return sample
