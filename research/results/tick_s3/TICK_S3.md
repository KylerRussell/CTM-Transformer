# Adapted CTM-LM tick use on the S3 word problem

Protocol: `scripts/tick_s3.py` (docstring). Mean held-out accuracy over positions 1–16 of length-32 words, final readout, final checkpoint.

| Cell | Seeds | T trained | Mean accuracy 1–16 | Median correct prefix | Escapes (9–16 ≥ 0.9) | Accuracy 1–16 at T = 1 / 4 / 8 / 16 / 32 |
|---|---:|---:|---:|---:|---:|---|
| transformer (reliability study, paired) | 10 | 1 | 0.614 | 5.0 | 0 | |
| rdt (reliability study, paired) | 10 | 16 | 0.715 | 6.0 | 1 | |
| ctm_lm (reliability study, paired) | 10 | 16 | 0.579 | 4.0 | 0 | |
| adapt | 10 | 16 | 0.726 | 7.5 | 0 | 0.414 / 0.712 / 0.722 / 0.726 / 0.722 |
| adapt_t1 | 10 | 1 | 0.679 | 6.0 | 0 | 0.679 /  /  /  /  |
| adapt_cross | 10 | 16 | 0.856 | 9.5 | 4 | 0.463 / 0.764 / 0.844 / 0.856 / 0.848 |
| adapt_cross_t1 | 10 | 1 | 0.685 | 6.5 | 0 | 0.685 /  /  /  /  |
| adapt_wide | 10 | 16 | 0.716 | 7.0 | 0 | 0.430 / 0.698 / 0.712 / 0.716 / 0.713 |
| adapt_wide_t1 | 10 | 1 | 0.662 | 6.0 | 0 | 0.662 /  /  /  /  |
| cross_norm_sink | 1 | 16 | 0.992 | 17 | 1 | 0.465 / 0.784 / 0.965 / 0.992 / 0.910 |
| transformer_rope | 10 | 1 | 0.614 | 5.0 | 0 | 0.614 /  /  /  /  |
| rdt_rope | 1 | 16 | 0.973 | 15 | 1 | 0.322 / 0.542 / 0.756 / 0.973 / 0.979 |
| rdt_rope_t1 | 1 | 1 | 0.708 | 7 | 0 | 0.708 /  /  /  /  |

**Paired against the T = 1 control (rule: higher on ≥ 8 of 10 seeds and median gain ≥ 5 points):**

- **adapt_cross vs rdt_rope (same position encoding):** higher on 0/1 seeds, median gain -14.3 points: incomplete.
- **cross_norm_sink vs adapt_cross:** higher on 1/1 seeds, median gain +16.3 points: incomplete.
- **adapt vs adapt_t1:** higher on 9/10 seeds, median gain +3.2 points: ticks do not meet the rule.
- **adapt_cross vs adapt_cross_t1:** higher on 9/10 seeds, median gain +18.6 points: ticks contribute.
- **adapt_wide vs adapt_wide_t1:** higher on 8/10 seeds, median gain +7.0 points: ticks contribute.
- **rdt_rope vs rdt_rope_t1:** higher on 1/1 seeds, median gain +26.5 points: incomplete.
