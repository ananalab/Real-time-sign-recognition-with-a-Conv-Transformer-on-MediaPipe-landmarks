# Evaluation of `asl-transformer-s1` on `test`

- samples: 18412 · classes: 250 · signers: 4
- **top-1 0.6449** · top-5 0.8422 · macro-F1 0.6378
- per-signer top-1: 0.6444 ± 0.0349
- CPU latency (1 thread): preprocess 0.32 ms, model p50 7.79 ms / p95 10.38 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 49445 | 0.6127 | 4968.0000 |
| 25571 | 0.6305 | 3865.0000 |
| 32319 | 0.6406 | 4753.0000 |
| 55372 | 0.6937 | 4826.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.6964 | 3202.0000 |
| 0-10%  | 0.7589 | 1738.0000 |
| 10-25% | 0.7328 | 2882.0000 |
| 25-50% | 0.6654 | 5009.0000 |
| >50%   | 0.5160 | 5581.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| there   | 0.053 |
| finger  | 0.093 |
| go      | 0.181 |
| beside  | 0.213 |
| snack   | 0.216 |
| give    | 0.258 |
| sticky  | 0.269 |
| close   | 0.274 |
| cloud   | 0.281 |
| wake    | 0.282 |

## 20 most confused pairs (rate = share of the true class)
| true    | pred    |   count |   rate |
|:--------|:--------|--------:|-------:|
| wake    | awake   |      45 |  0.577 |
| ear     | hear    |      37 |  0.468 |
| think   | penny   |      30 |  0.395 |
| pen     | pencil  |      29 |  0.362 |
| hello   | pretend |      27 |  0.351 |
| listen  | hear    |      25 |  0.263 |
| lips    | mouth   |      24 |  0.320 |
| have    | animal  |      24 |  0.338 |
| say     | chin    |      23 |  0.324 |
| doll    | mouse   |      21 |  0.280 |
| milk    | fast    |      20 |  0.286 |
| hear    | listen  |      20 |  0.227 |
| awake   | wake    |      20 |  0.270 |
| pretend | mouse   |      20 |  0.260 |
| look    | see     |      19 |  0.209 |
| lamp    | shhh    |      18 |  0.254 |
| mouth   | lips    |      17 |  0.230 |
| there   | hesheit |      17 |  0.227 |
| stay    | same    |      17 |  0.233 |
| table   | minemy  |      17 |  0.266 |
