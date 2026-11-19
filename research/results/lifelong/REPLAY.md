# Lifelong learning: task stream (replay 0.01, seeds 1-5)

Protocol and rules: `scripts/lifelong.py`.

| Cell | Forgetting | Final accuracy | Relearning (updates) |
|---|---:|---:|---:|
| T8 | 39.5 | 80.2 | 250.0 |
| Dans | 45.3 | 76.0 | 250.0 |
| CTsqS | 49.7 | 79.5 | 250.0 |
| CTnlmA | 50.3 | 73.9 | 250.0 |

## Rule outcomes

- CTsqS beats Dans: none
- CTnlmA beats Dans: none
- Dans beats CTsqS: none
- Dans beats CTnlmA: none
- Dans beats T8: none
- T8 beats Dans: none
- T8 beats CTsqS: forgetting
- CTsqS beats T8: none
- T8 beats CTnlmA: forgetting, final
- CTnlmA beats T8: none

## Accuracy at each phase end (seed means; rows = phase, columns = copy, reverse, sort, increment)

- **T8:** 98/0/0/0 | 60/98/0/0 | 55/69/95/0 | 82/38/10/99 | 99/69/23/92 | 81/100/22/69 | 90/92/99/85 | 98/89/33/100
- **Dans:** 99/0/0/0 | 76/99/0/0 | 74/68/98/0 | 87/23/10/99 | 100/40/22/83 | 90/100/28/77 | 93/84/99/81 | 97/70/38/100
- **CTsqS:** 98/0/0/0 | 74/100/0/0 | 69/59/98/0 | 51/20/8/99 | 100/49/19/77 | 84/100/12/58 | 94/84/99/74 | 97/77/44/100
- **CTnlmA:** 98/0/0/0 | 69/99/1/0 | 61/74/98/0 | 57/32/5/99 | 99/47/16/70 | 78/100/24/54 | 84/87/98/62 | 93/71/32/100
