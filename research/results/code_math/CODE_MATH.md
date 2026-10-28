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

