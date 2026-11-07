# Addition reliability (d704, 100M tokens of continued pretraining)

Protocol and rules: `scripts/addition_reliability.py` (docstring). Learned: ID addition at 16 ticks at least 90%. Spike: gradient norm above 2.0 after warmup.

| Arm | Replicate | Start | Addition ID at 4 / 16 ticks | Learned | Max gradient norm after warmup | Spike updates | Tracking / parity / expressions ID |
|---|---:|---:|---|---|---:|---:|---|
| Dans_sampler | 1 (not judged) | s1 | 95 / 98 | yes | 3.10 | 1 | 46 / 49 / 43 |
| Dans_sampler | 2 (not judged) | s2 | 17 / 18 | no | 6.26 | 2 | 42 / 54 / 30 |
| Dans_sampler | 3 | s1 | 90 / 96 | yes | 0.55 | 0 | 43 / 51 / 40 |
| Dans_sampler | 4 | s2 | 56 / 64 | no | 3.26 | 1 | 43 / 57 / 33 |
| Dans_sampler | 5 | s1 | 95 / 99 | yes | 1.29 | 0 | 41 / 55 / 37 |
| Dans_sampler | 6 | s2 | 76 / 83 | no | 0.46 | 0 | 42 / 53 / 35 |
| Dans_sampler | 7 | s1 | 96 / 99 | yes | 3.56 | 1 | 46 / 53 / 41 |
| Dans_sampler | 8 | s2 | 88 / 95 | yes | 271.42 | 2 | 43 / 52 / 34 |
| Dans_sampler | 9 | s1 | 97 / 99 | yes | 1.08 | 0 | 44 / 50 / 36 |
| Dans_sampler | 10 | s2 | 10 / 9 | no | 0.43 | 0 | 43 / 55 / 27 |
| CTsq_sampler | 1 (not judged) | s1 | 96 / 98 | yes | 0.44 | 0 | 45 / 53 / 39 |
| CTsq_sampler | 2 (not judged) | s2 | 97 / 99 | yes | 0.65 | 0 | 42 / 47 / 38 |
| CTsq_sampler | 3 | s1 | 93 / 97 | yes | 1.36 | 0 | 43 / 54 / 38 |
| CTsq_sampler | 4 | s2 | 98 / 100 | yes | 0.42 | 0 | 44 / 53 / 42 |
| CTsq_sampler | 5 | s1 | 97 / 99 | yes | 0.79 | 0 | 44 / 55 / 36 |
| CTsq_sampler | 6 | s2 | 99 / 100 | yes | 1.11 | 0 | 45 / 50 / 36 |
| CTsq_sampler | 7 | s1 | 55 / 88 | no | 0.65 | 0 | 43 / 51 / 34 |
| CTsq_sampler | 8 | s2 | 99 / 100 | yes | 0.41 | 0 | 42 / 48 / 38 |
| CTsq_sampler | 9 | s1 | 99 / 100 | yes | 5177.66 | 1 | 46 / 53 / 40 |
| CTsq_sampler | 10 | s2 | 95 / 95 | yes | 0.44 | 0 | 42 / 52 / 39 |

## Rule outcomes (replicates 3-10)

- **Learned:** Dans 5/8, CTsq 7/8 (one-sided Fisher p: CTsq more 0.285, Dans more 0.962)
- **Reliability:** no difference by the rule
- **Spiked:** Dans 3/8, CTsq 1/8; stability: no difference by the rule
- **Failed runs that spiked:** Dans 1 of 3 failures, CTsq 0 of 1
- **Pooled with replicates 1-2 (reported):** Dans 6/10, CTsq 9/10
- **By starting checkpoint (learned, fresh):** Dans_sampler s1 4/4, Dans_sampler s2 1/4, CTsq_sampler s1 3/4, CTsq_sampler s2 4/4
