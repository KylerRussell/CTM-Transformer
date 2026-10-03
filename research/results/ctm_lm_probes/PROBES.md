# CTM-LM input-use probes — results

Protocol: `scripts/ctm_lm_probes.py` (docstring). 300 steps × 32 sequences × 1,024 tokens (9.8M tokens), one GPU.
Held-out loss on 64 windows from offset 40000; unigram model: 7.62 nats.

| Probe | Held-out loss | Loss with random inputs | Context use | Works |
|---|---:|---:|---:|---|
| faithful | 7.797 | 7.798 | +0.001 | no |
| faithful_low_lr | 7.689 | 7.681 | -0.008 | no |
| A_unit_query | 7.394 | 8.311 | +0.917 | no |
| B_observe_token | 7.406 | 8.326 | +0.919 | no |
| transformer_reference | 6.311 | 8.123 | +1.812 | yes |
| rdt_aware_baseline | 7.591 | 7.591 | -0.000 | no |
| rdt_aware_low_lr | 7.589 | 7.589 | -0.000 | no |
| rdt_aware_unit_embedding | 5.979 | 8.746 | +2.767 | yes |
| faithful_unit_embedding | 7.727 | 7.726 | -0.001 | no |
| A_unit_query_unit_embedding | 7.736 | 7.735 | -0.000 | no |
| C_mean_tick_loss | 7.168 | 7.850 | +0.682 | no |
| A_C | 7.046 | 8.186 | +1.140 | no |
| B_C | 6.731 | 8.327 | +1.597 | yes |
| E_C | 7.115 | 8.188 | +1.073 | no |
| D_C | 6.392 | 8.229 | +1.838 | yes |
| B_final | 6.750 | 8.563 | +1.812 | yes |
| B_certainty | 6.802 | 8.725 | +1.923 | yes |
| A_B_C | 6.744 | 8.783 | +2.039 | yes |
| B_C_state | 6.839 | 8.518 | +1.679 | yes |
| D_final | 6.313 | 8.527 | +2.214 | yes |

**CTM-LM choice by the rule:** B_C.
