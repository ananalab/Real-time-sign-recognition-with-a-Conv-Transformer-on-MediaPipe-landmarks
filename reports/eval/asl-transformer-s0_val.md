# Evaluation of `asl-transformer-s0` on `val`

- samples: 13981 · classes: 250 · signers: 3
- **top-1 0.6829** · top-5 0.8757 · macro-F1 0.6715
- per-signer top-1: 0.6834 ± 0.0445
- CPU latency (1 thread): preprocess 0.29 ms, model p50 7.45 ms / p95 8.22 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 36257 | 0.6356 | 4896.0000 |
| 27610 | 0.6910 | 4275.0000 |
|  2044 | 0.7237 | 4810.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7411 | 3268.0000 |
| 0-10%  | 0.7483 | 1323.0000 |
| 10-25% | 0.7219 | 1996.0000 |
| 25-50% | 0.7167 | 3064.0000 |
| >50%   | 0.5769 | 4330.0000 |

## 10 worst classes
| label      |   acc |
|:-----------|------:|
| give       | 0.078 |
| after      | 0.089 |
| there      | 0.125 |
| snack      | 0.130 |
| nap        | 0.172 |
| quiet      | 0.200 |
| ride       | 0.241 |
| mouth      | 0.250 |
| go         | 0.250 |
| helicopter | 0.254 |

## 20 most confused pairs (rate = share of the true class)
| true     | pred        |   count |   rate |
|:---------|:------------|--------:|-------:|
| give     | gift        |      44 |  0.688 |
| wake     | awake       |      36 |  0.632 |
| listen   | hear        |      34 |  0.576 |
| mouth    | lips        |      31 |  0.554 |
| pen      | pencil      |      29 |  0.492 |
| nap      | sleep       |      26 |  0.448 |
| kiss     | home        |      23 |  0.383 |
| say      | chin        |      22 |  0.400 |
| find     | frenchfries |      21 |  0.344 |
| carrot   | yes         |      21 |  0.344 |
| goose    | bird        |      21 |  0.333 |
| tooth    | glasswindow |      21 |  0.382 |
| think    | penny       |      20 |  0.370 |
| ear      | hear        |      19 |  0.328 |
| sleepy   | sleep       |      19 |  0.322 |
| bedroom  | bed         |      19 |  0.302 |
| scissors | cut         |      19 |  0.322 |
| why      | sick        |      19 |  0.365 |
| open     | book        |      18 |  0.419 |
| quiet    | bad         |      18 |  0.360 |
