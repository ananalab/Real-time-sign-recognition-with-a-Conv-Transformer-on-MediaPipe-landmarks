# Evaluation of `asl-gru-s1` on `val`

- samples: 13981 · classes: 250 · signers: 3
- **top-1 0.6535** · top-5 0.8710 · macro-F1 0.6432
- per-signer top-1: 0.6546 ± 0.0561
- CPU latency (1 thread): preprocess 0.36 ms, model p50 5.77 ms / p95 7.80 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 36257 | 0.5917 | 4896.0000 |
| 27610 | 0.6725 | 4275.0000 |
|  2044 | 0.6996 | 4810.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7200 | 3268.0000 |
| 0-10%  | 0.7173 | 1323.0000 |
| 10-25% | 0.6864 | 1996.0000 |
| 25-50% | 0.6795 | 3064.0000 |
| >50%   | 0.5503 | 4330.0000 |

## 10 worst classes
| label      |   acc |
|:-----------|------:|
| give       | 0.047 |
| there      | 0.089 |
| look       | 0.123 |
| after      | 0.125 |
| child      | 0.130 |
| empty      | 0.140 |
| helicopter | 0.153 |
| snack      | 0.167 |
| go         | 0.200 |
| nap        | 0.207 |

## 20 most confused pairs (rate = share of the true class)
| true    | pred        |   count |   rate |
|:--------|:------------|--------:|-------:|
| give    | gift        |      45 |  0.703 |
| awake   | wake        |      34 |  0.576 |
| mouth   | lips        |      33 |  0.589 |
| listen  | hear        |      32 |  0.542 |
| kiss    | home        |      30 |  0.500 |
| empty   | touch       |      25 |  0.439 |
| goose   | duck        |      25 |  0.397 |
| pen     | pencil      |      24 |  0.407 |
| nap     | sleep       |      23 |  0.397 |
| tooth   | glasswindow |      22 |  0.400 |
| goose   | bird        |      22 |  0.349 |
| say     | chin        |      22 |  0.400 |
| smile   | cheek       |      21 |  0.350 |
| ear     | hear        |      20 |  0.345 |
| wake    | awake       |      20 |  0.351 |
| find    | frenchfries |      19 |  0.311 |
| quiet   | bad         |      18 |  0.360 |
| bedroom | bed         |      18 |  0.286 |
| look    | face        |      18 |  0.316 |
| stairs  | jump        |      18 |  0.316 |
