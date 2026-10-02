# Evaluation of `asl-transformer-s1` on `val`

- samples: 13981 · classes: 250 · signers: 3
- **top-1 0.6823** · top-5 0.8770 · macro-F1 0.6711
- per-signer top-1: 0.6833 ± 0.0496
- CPU latency (1 thread): preprocess 0.27 ms, model p50 7.27 ms / p95 7.74 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 36257 | 0.6272 | 4896.0000 |
| 27610 | 0.7011 | 4275.0000 |
|  2044 | 0.7216 | 4810.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7368 | 3268.0000 |
| 0-10%  | 0.7551 | 1323.0000 |
| 10-25% | 0.7174 | 1996.0000 |
| 25-50% | 0.7151 | 3064.0000 |
| >50%   | 0.5794 | 4330.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| give    | 0.062 |
| child   | 0.130 |
| after   | 0.161 |
| there   | 0.161 |
| snack   | 0.167 |
| go      | 0.183 |
| beside  | 0.190 |
| empty   | 0.193 |
| ride    | 0.204 |
| quiet   | 0.220 |

## 20 most confused pairs (rate = share of the true class)
| true    | pred        |   count |   rate |
|:--------|:------------|--------:|-------:|
| give    | gift        |      44 |  0.688 |
| mouth   | lips        |      39 |  0.696 |
| listen  | hear        |      33 |  0.559 |
| awake   | wake        |      28 |  0.475 |
| kiss    | home        |      28 |  0.467 |
| goose   | duck        |      26 |  0.413 |
| say     | chin        |      26 |  0.473 |
| wake    | awake       |      26 |  0.456 |
| nap     | sleep       |      25 |  0.431 |
| pen     | pencil      |      24 |  0.407 |
| why     | sick        |      21 |  0.404 |
| ear     | hear        |      21 |  0.362 |
| goose   | bird        |      21 |  0.333 |
| find    | frenchfries |      20 |  0.328 |
| quiet   | bad         |      19 |  0.380 |
| bedroom | bed         |      18 |  0.286 |
| carrot  | yes         |      18 |  0.295 |
| tooth   | glasswindow |      18 |  0.327 |
| think   | penny       |      18 |  0.333 |
| sleepy  | sleep       |      18 |  0.305 |
