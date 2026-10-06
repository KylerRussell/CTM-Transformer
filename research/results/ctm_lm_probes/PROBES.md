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
| D_sparse | 6.415 | 8.226 | +1.810 | yes |
| D_B_sparse | 6.422 | 8.100 | +1.678 | yes |
| D_A_sparse | 6.433 | 8.194 | +1.761 | yes |
| D_H_sparse | 6.355 | 8.408 | +2.053 | yes |
| D_sparse_lr2.5e-4 | 6.626 | 8.043 | +1.416 | yes |
| D_sparse_lr1e-3 | 6.227 | 8.406 | +2.179 | yes |
| transformer_lr2.5e-4 | 6.572 | 8.080 | +1.508 | yes |
| transformer_lr1e-3 | 6.172 | 8.516 | +2.344 | yes |
| D_sparse_seed1235 | 6.434 | 8.164 | +1.730 | yes |
| transformer_seed1235 | 6.313 | 8.311 | +1.998 | yes |
| D_H_sparse_lr1e-3 | 6.205 | 8.277 | +2.073 | yes |
| D_H_sparse_lr2e-3 | 6.183 | 8.398 | +2.215 | yes |
| transformer_lr2e-3 | 6.079 | 8.883 | +2.804 | yes |
| D_H_sparse_seed1235 | 6.904 | 8.762 | +1.857 | yes |
| D_sparse_decay_softplus | 6.425 | 8.584 | +2.159 | yes |
| D_sparse_decay_spread | 6.539 | 9.045 | +2.507 | yes |

**CTM-LM choice by the rule:** B_C.
