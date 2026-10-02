# Evaluation of `asl-gru-s2` on `val`

- samples: 13981 · classes: 250 · signers: 3
- **top-1 0.6635** · top-5 0.8715 · macro-F1 0.6531
- per-signer top-1: 0.6641 ± 0.0508
- CPU latency (1 thread): preprocess 0.27 ms, model p50 3.35 ms / p95 3.57 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 36257 | 0.6101 | 4896.0000 |
| 27610 | 0.6711 | 4275.0000 |
|  2044 | 0.7110 | 4810.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7246 | 3268.0000 |
| 0-10%  | 0.7218 | 1323.0000 |
| 10-25% | 0.7079 | 1996.0000 |
| 25-50% | 0.6978 | 3064.0000 |
| >50%   | 0.5547 | 4330.0000 |

## 10 worst classes
| label      |   acc |
|:-----------|------:|
| give       | 0.062 |
| after      | 0.089 |
| child      | 0.093 |
| look       | 0.158 |
| there      | 0.161 |
| go         | 0.167 |
| snack      | 0.167 |
| helicopter | 0.186 |
| quiet      | 0.220 |
| ride       | 0.222 |

## 20 most confused pairs (rate = share of the true class)
| true     | pred        |   count |   rate |
|:---------|:------------|--------:|-------:|
| give     | gift        |      43 |  0.672 |
| mouth    | lips        |      33 |  0.589 |
| awake    | wake        |      32 |  0.542 |
| pen      | pencil      |      30 |  0.508 |
| listen   | hear        |      27 |  0.458 |
| say      | chin        |      25 |  0.455 |
| kiss     | home        |      25 |  0.417 |
| wake     | awake       |      23 |  0.404 |
| scissors | cut         |      23 |  0.390 |
| quiet    | bad         |      22 |  0.440 |
| nap      | sleep       |      21 |  0.362 |
| goose    | bird        |      21 |  0.333 |
| goose    | duck        |      20 |  0.317 |
| smile    | cheek       |      18 |  0.300 |
| tooth    | glasswindow |      18 |  0.327 |
| ear      | hear        |      18 |  0.310 |
| look     | face        |      18 |  0.316 |
| yes      | shoe        |      17 |  0.298 |
| why      | sick        |      17 |  0.327 |
| look     | see         |      17 |  0.298 |
