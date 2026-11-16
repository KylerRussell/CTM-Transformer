# Associative recall inside text (d704, 100M tokens of continued pretraining)

Protocol and rules: `scripts/recall_in_text.py`. Accuracy (%) of the value at the "=" position; chance 10%.

| Run | ID at 4 / 8 / 16 / 32 / 64 ticks | OOD at 4 / 8 / 16 / 32 / 64 ticks | Text loss | Binds |
|---|---|---|---:|---|
| CTsqS_sampler_r1 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 99.4 / 99.6 / 99.8 / 99.8 / 99.8 | 3.616 | yes |
| CTsqS_sampler_r2 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 99.2 / 99.2 / 99.4 / 99.2 / 99.4 | 3.627 | yes |
| Dans_sampler_r1 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 97.1 / 97.5 / 97.5 / 97.5 / 97.5 | 3.607 | yes |
| Dans_sampler_r2 | 100.0 / 100.0 / 100.0 / 100.0 / 100.0 | 98.0 / 98.4 / 98.4 / 98.4 / 98.4 | 3.615 | yes |

## Rule outcomes

- **CTM win:** no; CTM cost: no (CTsqS minus Dans at 16 ticks: id +0.0, +0.0; ood +2.3, +1.0)
- Binding: CTsqS 2/2, Dans 2/2
- Ticks pay (OOD 64 vs 16, at least +10): Dans_sampler no, CTsqS_sampler no
