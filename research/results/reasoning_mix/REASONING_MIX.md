# Reasoning tasks in text, 50% problems and a size curriculum (d704, 100M tokens of continued pretraining)

Protocol and rules: `scripts/reasoning_mix.py` (docstring). Exact-match accuracy (%), mean of two replicates.

| Task-split | Dans_sampler 16 / 64 / span1 / span2 | CTsq_sampler 16 / 64 / span1 / span2 | Dans_span 16 / 64 / span1 / span2 | CTsq_span 16 / 64 / span1 / span2 |
|---|---|---|---|---|
| add_id | 98 / 98 / 98 / 98 | 100 / 100 / 100 / 100 | 59 / 59 / 59 / 59 | 90 / 85 / 93 / 89 |
| add_ood | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |
| track_id | 46 / 46 / 46 / 45 | 44 / 43 / 43 / 44 | 45 / 45 / 46 / 45 | 46 / 44 / 45 / 45 |
| track_ood | 16 / 16 / 16 / 16 | 16 / 17 / 16 / 17 | 16 / 16 / 16 / 16 | 18 / 18 / 17 / 18 |
| parity_id | 49 / 50 / 50 / 49 | 53 / 53 / 53 / 52 | 52 / 52 / 52 / 52 | 52 / 48 / 52 / 58 |
| parity_ood | 52 / 51 / 52 / 52 | 48 / 48 / 48 / 48 | 50 / 51 / 50 / 51 | 49 / 51 / 49 / 51 |
| eval_id | 46 / 46 / 46 / 45 | 42 / 43 / 42 / 42 | 45 / 44 / 46 / 46 | 39 / 38 / 42 / 42 |
| eval_ood | 14 / 14 / 14 / 14 | 17 / 16 / 16 / 16 | 14 / 15 / 15 / 15 | 17 / 17 / 16 / 16 |

Text loss at 16 ticks: Dans_sampler 3.598, CTsq_sampler 3.606, Dans_span 3.597, CTsq_span 3.606

## Rule outcomes

- **Learned tasks (some arm at least 80% ID):** add yes, track no, parity no, eval no
- **sampler arms:** CTsq beats Dans on none; Dans beats CTsq on none; CTM win: no
- **span arms:** CTsq beats Dans on none; Dans beats CTsq on none; CTM win: no

## Against reasoning_in_text (reported)

- **Dans_sampler:** text loss 3.598 (was 3.513); ID accuracy add 98 (was 58), track 46 (was 44), parity 49 (was 52), eval 46 (was 37)
- **CTsq_sampler:** text loss 3.606 (was 3.518); ID accuracy add 100 (was 99), track 44 (was 43), parity 53 (was 50), eval 42 (was 38)
- **Dans_span:** text loss 3.597 (was 3.512); ID accuracy add 59 (was 97), track 46 (was 44), parity 52 (was 54), eval 46 (was 36)
- **CTsq_span:** text loss 3.606 (was 3.518); ID accuracy add 93 (was 100), track 45 (was 46), parity 52 (was 49), eval 42 (was 38)
