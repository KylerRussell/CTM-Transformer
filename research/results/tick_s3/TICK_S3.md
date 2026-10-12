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
| cross_norm_sink | 10 | 16 | 0.882 | 13.0 | 5 | 0.465 / 0.753 / 0.860 / 0.882 / 0.861 |
| cross_norm_sink_t1 | 10 | 1 | 0.682 | 7.0 | 0 | 0.682 /  /  /  /  |
| cross_memory | 10 | 16 | 0.850 | 10.0 | 2 | 0.481 / 0.764 / 0.830 / 0.850 / 0.836 |
| cross_memory_t1 | 10 | 1 | 0.691 | 7.0 | 0 | 0.691 /  /  /  /  |
| cross_anchor | 10 | 16 | 0.923 | 15.5 | 7 | 0.508 / 0.799 / 0.905 / 0.923 / 0.910 |
| cross_anchor_t1 | 10 | 1 | 0.706 | 7.0 | 0 | 0.706 /  /  /  /  |
| cross_clamp | 10 | 16 | 0.808 | 8.5 | 3 | 0.469 / 0.720 / 0.787 / 0.808 / 0.782 |
| transformer_rope | 10 | 1 | 0.614 | 5.0 | 0 | 0.614 /  /  /  /  |
| rdt_rope | 10 | 16 | 0.821 | 9.0 | 1 | 0.272 / 0.496 / 0.693 / 0.821 / 0.824 |
| rdt_rope_t1 | 10 | 1 | 0.677 | 7.0 | 0 | 0.677 /  /  /  /  |
| dans | 10 | 16 | 0.962 | 17.0 | 9 | 0.355 / 0.501 / 0.721 / 0.962 / 0.951 |
| dans_t1 | 10 | 1 | 0.769 | 8.5 | 0 | 0.769 /  /  /  /  |
| dans_sync | 10 | 16 | 0.946 | 17.0 | 7 | 0.374 / 0.479 / 0.653 / 0.946 / 0.934 |
| dans_nlm | 10 | 16 | 0.965 | 17.5 | 9 | 0.340 / 0.492 / 0.699 / 0.965 / 0.945 |
| dans_carry | 10 | 16 | 0.968 | 18.0 | 9 | 0.345 / 0.497 / 0.703 / 0.968 / 0.950 |
| dans_all | 10 | 16 | 0.961 | 16.5 | 8 | 0.367 / 0.479 / 0.652 / 0.961 / 0.959 |

**Paired against the T = 1 control (rule: higher on ≥ 8 of 10 seeds and median gain ≥ 5 points):**

- **adapt_cross vs rdt_rope (same position encoding):** higher on 5/10 seeds, median gain -3.0 points: does not exceed.
- **cross_norm_sink vs adapt_cross:** higher on 6/10 seeds, median gain +1.8 points: does not improve.
- **cross_memory vs adapt_cross:** higher on 4/10 seeds, median gain -3.4 points: does not improve.
- **cross_anchor vs adapt_cross:** higher on 7/10 seeds, median gain +5.1 points: does not improve.
- **cross_clamp vs adapt_cross:** higher on 4/10 seeds, median gain -3.8 points: does not improve.
- **dans vs rdt_rope:** higher on 9/10 seeds, median gain +12.7 points: improves.
- **dans_sync vs dans:** higher on 3/10 seeds, median gain -0.1 points: does not improve.
- **dans_nlm vs dans:** higher on 5/10 seeds, median gain +0.0 points: does not improve.
- **dans_carry vs dans:** higher on 6/10 seeds, median gain +0.0 points: does not improve.
- **dans_all vs dans:** higher on 2/10 seeds, median gain -0.1 points: does not improve.
- *Exploratory:* **dans vs cross_anchor:** higher on 8/10 seeds, median gain +0.3 points.
- *Exploratory:* **dans_sync vs cross_anchor:** higher on 8/10 seeds, median gain +1.5 points.
- *Exploratory:* **dans_nlm vs cross_anchor:** higher on 8/10 seeds, median gain +0.3 points.
- *Exploratory:* **dans_carry vs cross_anchor:** higher on 8/10 seeds, median gain +0.3 points.
- *Exploratory:* **dans_all vs cross_anchor:** higher on 7/10 seeds, median gain +1.7 points.
- **adapt vs adapt_t1:** higher on 9/10 seeds, median gain +3.2 points: ticks do not meet the rule.
- **adapt_cross vs adapt_cross_t1:** higher on 9/10 seeds, median gain +18.6 points: ticks contribute.
- **adapt_wide vs adapt_wide_t1:** higher on 8/10 seeds, median gain +7.0 points: ticks contribute.
- **rdt_rope vs rdt_rope_t1:** higher on 9/10 seeds, median gain +17.5 points: ticks contribute.
- **cross_norm_sink vs cross_norm_sink_t1:** higher on 10/10 seeds, median gain +20.5 points: ticks contribute.
- **cross_memory vs cross_memory_t1:** higher on 9/10 seeds, median gain +15.9 points: ticks contribute.
- **cross_anchor vs cross_anchor_t1:** higher on 9/10 seeds, median gain +28.4 points: ticks contribute.
- **dans vs dans_t1:** higher on 9/10 seeds, median gain +23.0 points: ticks contribute.
