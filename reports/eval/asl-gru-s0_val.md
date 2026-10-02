# Evaluation of `asl-gru-s0` on `val`

- samples: 13981 · classes: 250 · signers: 3
- **top-1 0.6570** · top-5 0.8755 · macro-F1 0.6473
- per-signer top-1: 0.6580 ± 0.0579
- CPU latency (1 thread): preprocess 0.31 ms, model p50 13.98 ms / p95 47.89 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 36257 | 0.5935 | 4896.0000 |
| 27610 | 0.6749 | 4275.0000 |
|  2044 | 0.7056 | 4810.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7218 | 3268.0000 |
| 0-10%  | 0.7218 | 1323.0000 |
| 10-25% | 0.6869 | 1996.0000 |
| 25-50% | 0.6926 | 3064.0000 |
| >50%   | 0.5492 | 4330.0000 |

## 10 worst classes
| label      |   acc |
|:-----------|------:|
| give       | 0.078 |
| after      | 0.143 |
| there      | 0.161 |
| ride       | 0.167 |
| goose      | 0.190 |
| go         | 0.200 |
| snack      | 0.204 |
| empty      | 0.211 |
| helicopter | 0.220 |
| child      | 0.222 |

## 20 most confused pairs (rate = share of the true class)
| true     | pred        |   count |   rate |
|:---------|:------------|--------:|-------:|
| give     | gift        |      41 |  0.641 |
| listen   | hear        |      34 |  0.576 |
| wake     | awake       |      32 |  0.561 |
| goose    | duck        |      28 |  0.444 |
| mouth    | lips        |      27 |  0.482 |
| kiss     | home        |      27 |  0.450 |
| nap      | sleep       |      26 |  0.448 |
| goose    | bird        |      21 |  0.333 |
| pen      | pencil      |      21 |  0.356 |
| smile    | cheek       |      21 |  0.350 |
| awake    | wake        |      21 |  0.356 |
| tooth    | glasswindow |      21 |  0.382 |
| find     | frenchfries |      20 |  0.328 |
| scissors | cut         |      19 |  0.322 |
| say      | chin        |      19 |  0.345 |
| look     | face        |      18 |  0.316 |
| icecream | orange      |      18 |  0.305 |
| why      | sick        |      18 |  0.346 |
| bedroom  | bed         |      18 |  0.286 |
| yes      | shoe        |      18 |  0.316 |
