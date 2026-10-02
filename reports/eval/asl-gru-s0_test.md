# Evaluation of `asl-gru-s0` on `test`

- samples: 18412 · classes: 250 · signers: 4
- **top-1 0.6318** · top-5 0.8401 · macro-F1 0.6252
- per-signer top-1: 0.6317 ± 0.0399
- CPU latency (1 thread): preprocess 0.43 ms, model p50 7.68 ms / p95 12.05 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 49445 | 0.5992 | 4968.0000 |
| 32319 | 0.6118 | 4753.0000 |
| 25571 | 0.6264 | 3865.0000 |
| 55372 | 0.6892 | 4826.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.6993 | 3202.0000 |
| 0-10%  | 0.7417 | 1738.0000 |
| 10-25% | 0.7085 | 2882.0000 |
| 25-50% | 0.6532 | 5009.0000 |
| >50%   | 0.4999 | 5581.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| there   | 0.013 |
| finger  | 0.173 |
| go      | 0.181 |
| give    | 0.182 |
| cloud   | 0.188 |
| snack   | 0.189 |
| milk    | 0.214 |
| close   | 0.226 |
| lamp    | 0.239 |
| wake    | 0.256 |

## 20 most confused pairs (rate = share of the true class)
| true        | pred         |   count |   rate |
|:------------|:-------------|--------:|-------:|
| wake        | awake        |      46 |  0.590 |
| ear         | hear         |      36 |  0.456 |
| listen      | hear         |      30 |  0.316 |
| lips        | mouth        |      28 |  0.373 |
| stay        | same         |      28 |  0.384 |
| look        | see          |      27 |  0.297 |
| pen         | pencil       |      27 |  0.338 |
| penny       | think        |      24 |  0.308 |
| uncle       | refrigerator |      20 |  0.253 |
| think       | penny        |      20 |  0.263 |
| pretend     | mouse        |      19 |  0.247 |
| awake       | wake         |      19 |  0.257 |
| there       | boat         |      18 |  0.240 |
| glasswindow | tooth        |      18 |  0.225 |
| say         | chin         |      18 |  0.254 |
| table       | minemy       |      18 |  0.281 |
| pencil      | pen          |      17 |  0.243 |
| goose       | duck         |      17 |  0.243 |
| dirty       | pig          |      16 |  0.232 |
| cat         | kitty        |      16 |  0.213 |
