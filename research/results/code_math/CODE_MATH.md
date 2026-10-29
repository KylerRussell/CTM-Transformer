# Stage 0A / 0B: code and math

Protocol and rules: `scripts/code_math_probe.py` (docstring). Loss in nats per token; two-seed means (T20 one seed).

## 0A: zero-shot (400M-token checkpoints)

| Measure | T8 | T20 | Dans | CTsq | CTnlm | CTso | CTsqH |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 3.677 | 3.587 | 3.609 | 3.614 | 3.595 | 3.608 | 3.604 |
| code | 2.605 | 2.554 | 2.563 | 2.591 | 2.588 | 2.597 | 2.574 |
| gsm8k | 3.457 | 3.384 | 3.382 | 3.405 | 3.382 | 3.372 | 3.452 |
| mbpp | 3.756 | 3.704 | 3.691 | 3.717 | 3.719 | 3.751 | 3.659 |
| humaneval | 2.564 | 2.571 | 2.460 | 2.523 | 2.506 | 2.513 | 2.495 |
| fineweb | 3.630 | 3.562 | 3.575 | 3.583 | 3.572 | 3.577 | 3.584 |

Rule outcomes:

- **T8 vs Dans:** math worse (+0.068); code worse (+0.042); gsm8k worse (+0.074); mbpp worse (+0.065); humaneval worse, specific disadvantage (+0.103); fineweb worse (+0.054)
- **CTsq vs Dans:** humaneval worse, specific disadvantage (+0.063)
- **CTnlm vs Dans:** code worse (+0.025); mbpp specific disadvantage (+0.028); humaneval worse, specific disadvantage (+0.045)
- **CTnlm vs CTsq:** no measure differs
- **CTso vs Dans:** code worse (+0.035); humaneval worse, specific disadvantage (+0.053)
- **CTso vs CTsq:** gsm8k specific advantage (-0.033)
- **CTsqH vs Dans:** gsm8k worse (+0.070); mbpp specific advantage (-0.031); humaneval worse (+0.035)
- **CTsqH vs CTsq:** mbpp better, specific advantage (-0.058); humaneval better (-0.028)

## 0B: after 100M tokens of math + code

| Measure | T8 | T20 | Dans | CTsq | CTnlm | CTso | CTsqH |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 2.517 | 2.431 | 2.460 | 2.467 | 2.456 | 2.460 | 2.470 |
| code | 1.228 | 1.175 | 1.194 | 1.199 | 1.193 | 1.196 | 1.200 |
| gsm8k | 3.072 | 2.946 | 3.024 | 3.014 | 3.028 | 3.016 | 2.991 |
| mbpp | 1.439 | 1.362 | 1.400 | 1.407 | 1.397 | 1.405 | 1.427 |
| humaneval | 0.989 | 0.935 | 0.956 | 0.958 | 0.956 | 0.959 | 0.964 |
| fineweb | 4.028 | 3.931 | 3.958 | 3.968 | 3.952 | 3.958 | 3.967 |

Rule outcomes:

- **T8 vs Dans:** math worse (+0.057); code worse, specific advantage (+0.034); gsm8k worse (+0.048); mbpp worse, specific advantage (+0.039); humaneval worse, specific advantage (+0.033); fineweb worse (+0.071)
- **CTsq vs Dans:** no measure differs
- **CTnlm vs Dans:** no measure differs
- **CTnlm vs CTsq:** gsm8k specific disadvantage (+0.014)
- **CTso vs Dans:** no measure differs
- **CTso vs CTsq:** no measure differs
- **CTsqH vs Dans:** mbpp worse (+0.027)
- **CTsqH vs CTsq:** no measure differs

## Forgetting and gains (0B minus 0A)

| Model | fineweb | math | code |
|---|---:|---:|---:|
| T8 | +0.399 | -1.160 | -1.377 |
| T20 | +0.369 | -1.156 | -1.379 |
| Dans | +0.383 | -1.149 | -1.369 |
| CTsq | +0.385 | -1.147 | -1.393 |
| CTnlm | +0.379 | -1.139 | -1.395 |
| CTso | +0.381 | -1.148 | -1.402 |
| CTsqH | +0.383 | -1.135 | -1.374 |

## Loss by ticks after 0B (64 windows; reported only)

- **Dans:** math 2.937 / 2.620 / 2.605 / 2.604 / 2.604 / 2.604; code 1.542 / 1.247 / 1.237 / 1.237 / 1.237 / 1.237 (ticks 1/4/8/16/32/64)
- **CTsq:** math 2.968 / 2.629 / 2.613 / 2.612 / 2.612 / 2.612; code 1.566 / 1.256 / 1.245 / 1.245 / 1.245 / 1.245 (ticks 1/4/8/16/32/64)
- **CTnlm:** math 2.949 / 2.616 / 2.599 / 2.599 / 2.599 / 2.599; code 1.561 / 1.249 / 1.237 / 1.237 / 1.237 / 1.237 (ticks 1/4/8/16/32/64)
- **CTso:** math 2.926 / 2.618 / 2.604 / 2.603 / 2.603 / 2.603; code 1.538 / 1.255 / 1.245 / 1.245 / 1.245 / 1.245 (ticks 1/4/8/16/32/64)
- **CTsqH:** math 2.965 / 2.636 / 2.621 / 2.620 / 2.621 / 2.621; code 1.544 / 1.257 / 1.246 / 1.246 / 1.246 / 1.246 (ticks 1/4/8/16/32/64)

## Selection of the ladder arm (0B measures + 0C task-splits against CTsq)

- **CTnlm:** wins none; losses none; qualifies: no
- **CTso:** wins none; losses none; qualifies: no
- **CTsqH:** wins none; losses none; qualifies: no
- **Ladder arm:** CTsq stays
