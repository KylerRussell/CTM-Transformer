# Lifelong learning: task stream (seeds 0-4)

Protocol and rules: `scripts/lifelong.py`.

| Cell | Forgetting | Final accuracy | Relearning (updates) |
|---|---:|---:|---:|
| T8 | 98.4 | 24.9 | 250.0 |
| Dans | 98.8 | 24.9 | 250.0 |
| CTsqS | 98.9 | 24.8 | 262.5 |
| CTnlmA | 98.7 | 25.0 | 275.0 |

## Rule outcomes

- CTsqS beats Dans: none
- CTnlmA beats Dans: none
- Dans beats CTsqS: none
- Dans beats CTnlmA: none
- Dans beats T8: none
- T8 beats Dans: none
- T8 beats CTsqS: none
- CTsqS beats T8: none
- T8 beats CTnlmA: none
- CTnlmA beats T8: none

## Accuracy at each phase end (seed means; rows = phase, columns = copy, reverse, sort, increment)

- **T8:** 98/0/0/0 | 0/99/0/0 | 0/0/97/0 | 1/0/0/99 | 100/0/0/0 | 0/100/0/0 | 0/0/98/0 | 0/0/0/99
- **Dans:** 99/0/0/0 | 0/99/1/0 | 1/0/98/0 | 0/0/0/99 | 100/0/0/0 | 0/99/0/0 | 0/0/97/0 | 0/0/0/99
- **CTsqS:** 99/0/0/0 | 0/100/1/0 | 0/0/97/0 | 0/0/0/98 | 100/0/0/0 | 0/100/1/0 | 0/0/99/0 | 0/0/0/99
- **CTnlmA:** 99/0/0/0 | 0/99/1/0 | 1/0/98/0 | 0/0/0/98 | 100/0/0/0 | 0/100/1/0 | 1/0/98/0 | 1/0/0/100
