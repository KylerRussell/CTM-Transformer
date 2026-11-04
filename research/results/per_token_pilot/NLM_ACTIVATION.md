# Neuron-level activation: language-model pilot (100M tokens)

Protocol and rules: `scripts/nlm_activation_pilot.py` (docstring).

| Arm | 1 ticks | 2 ticks | 4 ticks | 8 ticks | 16 ticks | 32 ticks | 64 ticks | 96 ticks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dans_sampler | 4.794 | 4.764 | 4.761 | 4.761 | 4.761 | 4.761 | 4.761 | 4.761 |
| CTsq_sampler | 4.837 | 4.768 | 4.757 | 4.756 | 4.755 | 4.755 | 4.756 | 4.756 |
| CTnlmA_sampler | 4.878 | 4.821 | 4.769 | 4.764 | 4.764 | 4.764 | 4.764 | 4.764 |

- **Ticks pay:** no
- **A CTM win (at least 0.02 below Dans at 16 ticks):** no (+0.003)
- **No cost (within 0.01 of Dans):** yes
- **Decision:** CTnlmA does not go to 400M tokens
