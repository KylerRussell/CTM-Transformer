# Reasoning tasks embedded in text (d704, 100M tokens of continued pretraining)

Protocol and rules: `scripts/reasoning_in_text.py` (docstring). Exact-match accuracy (%), mean of two replicates.

| Task-split | Dans_sampler 16 / 64 / span1 / span2 | CTsq_sampler 16 / 64 / span1 / span2 | Dans_span 16 / 64 / span1 / span2 | CTsq_span 16 / 64 / span1 / span2 |
|---|---|---|---|---|
| add_id | 58 / 58 / 58 / 58 | 99 / 99 / 99 / 99 | 97 / 97 / 97 / 97 | 99 / 99 / 100 / 99 |
| add_ood | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 |
| track_id | 44 / 44 / 44 / 44 | 43 / 44 / 44 / 44 | 44 / 44 / 44 / 44 | 45 / 43 / 46 / 45 |
| track_ood | 17 / 18 / 17 / 17 | 20 / 19 / 19 / 18 | 17 / 17 / 17 / 17 | 18 / 20 / 19 / 21 |
| parity_id | 52 / 51 / 52 / 52 | 50 / 49 / 50 / 50 | 52 / 51 / 54 / 53 | 48 / 48 / 49 / 51 |
| parity_ood | 51 / 52 / 52 / 51 | 49 / 49 / 48 / 48 | 49 / 50 / 50 / 50 | 49 / 50 / 50 / 49 |
| eval_id | 37 / 37 / 36 / 37 | 38 / 38 / 37 / 38 | 33 / 34 / 36 / 36 | 31 / 33 / 38 / 38 |
| eval_ood | 17 / 17 / 17 / 17 | 15 / 16 / 16 / 16 | 16 / 18 / 16 / 17 | 15 / 17 / 16 / 16 |

Text loss at 16 ticks: Dans_sampler 3.513, CTsq_sampler 3.518, Dans_span 3.512, CTsq_span 3.518

## Rule outcomes

- **Learned tasks (some arm at least 80% ID):** add yes, track no, parity no, eval no
- **sampler arms:** CTsq beats Dans on none; Dans beats CTsq on none; CTM win: no
- **span arms:** CTsq beats Dans on none; Dans beats CTsq on none; CTM win: no
