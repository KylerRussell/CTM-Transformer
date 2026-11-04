# Training-rate parity (100M tokens)

Protocol and rules: `scripts/ctm_speed_check.py` (docstring).

| Arm | Loss at 16 ticks | Seconds per update | vs Dans |
|---|---:|---:|---:|
| Dans_sampler | 4.761 (4.747/4.775) | 24.4 | 1.00x |
| CTsq_sampler | 4.755 (4.725/4.786) | 36.8 | 1.51x |
| CTsq_fast | 4.753 (4.731/4.774) | 27.2 | 1.12x |

- **Same loss (within 0.01 of CTsq_sampler):** yes (-0.003)
- **Similar rate (at most 1.15x Dans):** yes (1.12x)
- **Outcome:** later CTM Transformer runs use compile_ctm without micro_batch_by_depth
