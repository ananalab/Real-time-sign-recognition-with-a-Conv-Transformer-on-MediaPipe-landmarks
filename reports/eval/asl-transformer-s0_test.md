# Evaluation of `asl-transformer-s0` on `test`

- samples: 18412 · classes: 250 · signers: 4
- **top-1 0.6531** · top-5 0.8414 · macro-F1 0.6458
- per-signer top-1: 0.6526 ± 0.0310
- CPU latency (1 thread): preprocess 0.27 ms, model p50 8.40 ms / p95 11.95 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 49445 | 0.6278 | 4968.0000 |
| 25571 | 0.6404 | 3865.0000 |
| 32319 | 0.6444 | 4753.0000 |
| 55372 | 0.6979 | 4826.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.6983 | 3202.0000 |
| 0-10%  | 0.7675 | 1738.0000 |
| 10-25% | 0.7398 | 2882.0000 |
| 25-50% | 0.6712 | 5009.0000 |
| >50%   | 0.5306 | 5581.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| there   | 0.027 |
| finger  | 0.120 |
| go      | 0.181 |
| cloud   | 0.188 |
| beside  | 0.230 |
| think   | 0.237 |
| close   | 0.242 |
| snack   | 0.243 |
| give    | 0.258 |
| into    | 0.300 |

## 20 most confused pairs (rate = share of the true class)
| true     | pred   |   count |   rate |
|:---------|:-------|--------:|-------:|
| wake     | awake  |      40 |  0.513 |
| pen      | pencil |      36 |  0.450 |
| think    | penny  |      33 |  0.434 |
| awake    | wake   |      31 |  0.419 |
| duck     | goose  |      26 |  0.321 |
| look     | see    |      24 |  0.264 |
| say      | chin   |      23 |  0.324 |
| scissors | cut    |      22 |  0.328 |
| have     | animal |      22 |  0.310 |
| listen   | hear   |      21 |  0.221 |
| lips     | mouth  |      21 |  0.280 |
| there    | boat   |      20 |  0.267 |
| pretend  | mouse  |      19 |  0.247 |
| doll     | mouse  |      18 |  0.240 |
| hear     | listen |      18 |  0.205 |
| lamp     | shhh   |      17 |  0.239 |
| table    | minemy |      17 |  0.266 |
| dirty    | pig    |      17 |  0.246 |
| give     | gift   |      16 |  0.242 |
| eye      | cheek  |      15 |  0.205 |
