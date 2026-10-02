# Evaluation of `asl-transformer-s2` on `test`

- samples: 18412 · classes: 250 · signers: 4
- **top-1 0.6465** · top-5 0.8361 · macro-F1 0.6404
- per-signer top-1: 0.6460 ± 0.0326
- CPU latency (1 thread): preprocess 0.26 ms, model p50 7.66 ms / p95 10.60 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 49445 | 0.6176 | 4968.0000 |
| 25571 | 0.6323 | 3865.0000 |
| 32319 | 0.6415 | 4753.0000 |
| 55372 | 0.6927 | 4826.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.6958 | 3202.0000 |
| 0-10%  | 0.7681 | 1738.0000 |
| 10-25% | 0.7387 | 2882.0000 |
| 25-50% | 0.6680 | 5009.0000 |
| >50%   | 0.5135 | 5581.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| there   | 0.053 |
| finger  | 0.093 |
| give    | 0.182 |
| close   | 0.194 |
| go      | 0.236 |
| sticky  | 0.239 |
| pen     | 0.250 |
| beside  | 0.262 |
| cloud   | 0.266 |
| lamp    | 0.268 |

## 20 most confused pairs (rate = share of the true class)
| true    | pred   |   count |   rate |
|:--------|:-------|--------:|-------:|
| awake   | wake   |      37 |  0.500 |
| pen     | pencil |      36 |  0.450 |
| wake    | awake  |      30 |  0.385 |
| think   | penny  |      26 |  0.342 |
| listen  | hear   |      26 |  0.274 |
| say     | chin   |      26 |  0.366 |
| cat     | kitty  |      25 |  0.333 |
| lips    | mouth  |      24 |  0.320 |
| ear     | hear   |      23 |  0.291 |
| look    | see    |      23 |  0.253 |
| hear    | listen |      22 |  0.250 |
| mom     | farm   |      21 |  0.266 |
| there   | boat   |      20 |  0.267 |
| doll    | mouse  |      19 |  0.253 |
| pretend | mouse  |      19 |  0.247 |
| pencil  | pen    |      19 |  0.271 |
| minemy  | please |      18 |  0.234 |
| mouth   | lips   |      18 |  0.243 |
| duck    | goose  |      18 |  0.222 |
| table   | minemy |      17 |  0.266 |
