# Learning-rate scaling sweep — protocol (2026-10-01)

## Purpose

This sweep picks the peak learning rate of each 500M pretraining arm:
- a separate rate per architecture;
- extrapolated from how the optimum moves with model size and training length.

It is hyperparameter selection for the pretraining runs, not a paper endpoint. The paper reports the procedure and the fitted rates.

A single shared rate (for example 3e-4) could favour one family. The optimum is also known to fall with model width and with training length, so neither a 50M optimum nor a short-run optimum transfers directly to 500M models trained on 1–2B tokens.

## Design

**Size ladder.** Each arm (`scripts/make_pretrain_configs.py`) is narrowed to three widths. The layer layout stays the same, and FFN width, CTM neurons and pairs keep their ratio to d (`at_width`).

| Arm | Widths d | 500M width |
|---|---|---:|
| Transformer | 384, 512, 768 | 1,280 |
| RDT compute-aware | 384, 512, 768 | 1,280 |
| CTM-LM compute-aware | 384, 512, 768 | 1,280 |
| RDT heavy | 704, 896, 1,408 | 2,304 |
| CTM-LM heavy | 448, 640, 896 | 1,536 |

The non-embedding parameter counts span about 4×, up to about a third of the 500M model.

**Everything else matches the 500M runs:**
- the global batch of 256 × 1,024 tokens;
- warmup to the peak (max(100, steps/20) steps), then cosine decay to 10% of it;
- AdamW (0.9, 0.95), weight decay 0.1, clipping 1.0;
- the width-scaled initialization;
- randomized depth (RDT) and 16 ticks with CTM's loss (CTM-LM);
- bf16, `torch.compile`, activation checkpointing.

The optimizer stays on the GPU, which is numerically the same as the offloaded optimizer (`tests/test_pretrain.py`). Each run uses one GPU, and two runs proceed at a time.

**Training length.** Each run trains on 100M tokens (381 steps). At the smallest width, the Transformer and both compute-aware arms are also swept at 400M tokens (1,526 steps), to measure how the optimum moves with training length.

**Learning-rate grid.** The grid is the lattice lr = 1e-3 · 2^k, adapted as follows:
- the smallest width starts at k ∈ {−1, 0, 1};
- each larger width (and the 400M sweep) starts at the previous best k and its neighbours;
- while the best k is at an edge, the grid extends past that edge (k ∈ [−6, 4]).

A width is resolved when its best k has evaluated neighbours on both sides. *(Added 2026-10-04:)* A rate that diverged or collapsed is not run at higher rates of the same width, nor at that rate or above for wider widths of the same arm. Such a rate counts as an infinite-loss neighbour. A run with a non-finite gradient norm stops as **diverged** and counts as infinite loss. *(Added 2026-10-02, before any width was resolved:)* A run ending at a held-out loss of 7.0 nats or more counts as **collapsed**, also with infinite loss. The unigram model scores 7.62. Without this rule, an arm whose every rate collapses to the unigram level would get an "optimum" chosen by noise.

**Metric.** The metric is the mean held-out cross-entropy at the end of training. It is computed on 1,024 FineWeb-Edu validation windows (about 1M tokens), starting at window 40,000. The 500M runs report windows 0–255, so selection never touches the reported evaluation. The readout scored depends on the family:
- Transformer: the output;
- RDT: depth 16;
- CTM-LM: the final tick. *(Changed 2026-10-02, before any CTM width was resolved, from the most certain tick. The adapted CTM-LM trains every tick with the mean cross-entropy, so the final tick is its natural readout. The faithful CTM-LM's sweep runs were set aside; see [CTM_LM_DESIGN.md](CTM_LM_DESIGN.md).)*

## Analysis (`scripts/summarize_lr_sweep.py`, fixed before any run)

1. **Optimum per width.** lr* is the vertex of the parabola in log₂ lr through the best grid point and its two neighbours, kept within one grid step of the best point. If a neighbour diverged, lr* is the best grid point.
2. **Size scaling per arm.** Fit log₂ lr* = a + b · log₂ N by least squares over the three widths, where N is non-embedding parameters.
3. **Training-length exponent c.** c is the change in log₂ lr* per doubling of steps, from the 100M and 400M sweeps at the smallest width. Each heavy arm uses the compute-aware arm of its family.
4. **Prediction.** For a 500M run of S steps: log₂ lr = a + b · log₂ N₅₀₀ + c · log₂(S / 381). The targets are:
   - Transformer at 1B and 2B tokens;
   - compute-aware arms at 2B;
   - heavy arms at 1B.

   This per-arm prediction is the rate used.
5. **Robustness.** A shared-slope fit (one b, a per-arm intercept) is also reported. If the two predictions for an arm differ by more than 2×, or the per-arm residuals exceed one grid step, that is reported. Before the 500M launch, the arm gets a short check at its predicted rate and at half of it.

`tests/test_lr_sweep.py` checks the adaptive grid, the stopping rule and the fit against a synthetic loss surface with a known optimum.

## Running

```
research/launch_lr_sweep.sh                       # start or resume, detached; installs an autostart entry
python -m scripts.lr_sweep --plan                 # runs the current results call for, and their state
cat research/results/lr_sweep/state.json          # running / pending / complete
```

- Run directories live in `research/runs/lr_sweep/`. Each run deletes its checkpoint when it completes.
- A run that exits nonzero is recorded as `failure.json` and never retried. It blocks the larger widths of its arm.
- When everything is resolved, the supervisor writes [results/lr_sweep/LR_SWEEP.md](results/lr_sweep/LR_SWEEP.md) and removes its autostart entry.
