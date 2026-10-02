# Evaluation of `asl-transformer-s2` on `val`

- samples: 13981 · classes: 250 · signers: 3
- **top-1 0.6812** · top-5 0.8753 · macro-F1 0.6698
- per-signer top-1: 0.6820 ± 0.0412
- CPU latency (1 thread): preprocess 0.33 ms, model p50 7.95 ms / p95 10.28 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 36257 | 0.6356 | 4896.0000 |
| 27610 | 0.6964 | 4275.0000 |
|  2044 | 0.7141 | 4810.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7399 | 3268.0000 |
| 0-10%  | 0.7566 | 1323.0000 |
| 10-25% | 0.7214 | 1996.0000 |
| 25-50% | 0.7105 | 3064.0000 |
| >50%   | 0.5746 | 4330.0000 |

## 10 worst classes
| label      |   acc |
|:-----------|------:|
| give       | 0.062 |
| after      | 0.107 |
| child      | 0.111 |
| there      | 0.143 |
| hide       | 0.149 |
| snack      | 0.167 |
| look       | 0.175 |
| empty      | 0.193 |
| helicopter | 0.203 |
| tooth      | 0.218 |

## 20 most confused pairs (rate = share of the true class)
| true    | pred        |   count |   rate |
|:--------|:------------|--------:|-------:|
| give    | gift        |      43 |  0.672 |
| wake    | awake       |      33 |  0.579 |
| find    | frenchfries |      31 |  0.508 |
| mouth   | lips        |      30 |  0.536 |
| nap     | sleep       |      28 |  0.483 |
| carrot  | yes         |      28 |  0.459 |
| listen  | hear        |      28 |  0.475 |
| kiss    | home        |      27 |  0.450 |
| goose   | duck        |      25 |  0.397 |
| pen     | pencil      |      25 |  0.424 |
| tooth   | glasswindow |      23 |  0.418 |
| think   | penny       |      21 |  0.389 |
| say     | chin        |      21 |  0.382 |
| goose   | bird        |      21 |  0.333 |
| awake   | wake        |      19 |  0.322 |
| ear     | hear        |      18 |  0.310 |
| why     | sick        |      18 |  0.346 |
| smile   | cheek       |      18 |  0.300 |
| look    | face        |      18 |  0.316 |
| bedroom | bed         |      18 |  0.286 |
