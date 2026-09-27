"""Ablation A7 of the Sync-RDT ``sync`` cell: self-pairs only (research/SYNC_ABLATION_A7.md).

Identical to ``sync`` except that each of the 128 synchronization pairs uses the
same channel twice (i, i), so every feature is one channel's decayed energy over
the 8-step state history and no feature couples two channels. The frozen
``sync_ablations`` module is not modified; this builds on its identity kind.
"""
import torch

from ctm_transformer.sync_ablations import SyncAblation


class SelfPairAblation(SyncAblation):
    def __init__(self,config):
        super().__init__(config,'identity')
        with torch.no_grad():self.sync.idxs_right.copy_(self.sync.idxs_left)
        self.kind='self'


def self_pair_factory():
    def factory(config):return SelfPairAblation(config)
    factory.__qualname__='self_pair_factory[self]'
    return factory
