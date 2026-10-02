# Evaluation of `asl-gru-s2` on `test`

- samples: 18412 · classes: 250 · signers: 4
- **top-1 0.6242** · top-5 0.8381 · macro-F1 0.6166
- per-signer top-1: 0.6238 ± 0.0342
- CPU latency (1 thread): preprocess 0.56 ms, model p50 8.33 ms / p95 11.79 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 49445 | 0.6008 | 4968.0000 |
| 32319 | 0.6066 | 4753.0000 |
| 25571 | 0.6132 | 3865.0000 |
| 55372 | 0.6745 | 4826.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.6864 | 3202.0000 |
| 0-10%  | 0.7411 | 1738.0000 |
| 10-25% | 0.7065 | 2882.0000 |
| 25-50% | 0.6377 | 5009.0000 |
| >50%   | 0.4976 | 5581.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| there   | 0.040 |
| cloud   | 0.062 |
| give    | 0.182 |
| go      | 0.194 |
| fast    | 0.200 |
| lamp    | 0.211 |
| snack   | 0.216 |
| finger  | 0.227 |
| think   | 0.237 |
| close   | 0.242 |

## 20 most confused pairs (rate = share of the true class)
| true        | pred         |   count |   rate |
|:------------|:-------------|--------:|-------:|
| wake        | awake        |      41 |  0.526 |
| say         | chin         |      35 |  0.493 |
| pen         | pencil       |      31 |  0.388 |
| awake       | wake         |      29 |  0.392 |
| think       | penny        |      29 |  0.382 |
| ear         | hear         |      27 |  0.342 |
| listen      | hear         |      27 |  0.284 |
| lips        | mouth        |      24 |  0.320 |
| look        | see          |      24 |  0.264 |
| mouth       | lips         |      23 |  0.311 |
| stay        | same         |      21 |  0.288 |
| pretend     | mouse        |      21 |  0.273 |
| glasswindow | tooth        |      19 |  0.237 |
| find        | feet         |      19 |  0.241 |
| lamp        | shhh         |      19 |  0.268 |
| uncle       | refrigerator |      19 |  0.241 |
| vacuum      | hide         |      18 |  0.273 |
| outside     | go           |      18 |  0.225 |
| there       | hesheit      |      17 |  0.227 |
| table       | minemy       |      17 |  0.266 |
