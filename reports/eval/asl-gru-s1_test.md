# Evaluation of `asl-gru-s1` on `test`

- samples: 18412 · classes: 250 · signers: 4
- **top-1 0.6338** · top-5 0.8431 · macro-F1 0.6270
- per-signer top-1: 0.6335 ± 0.0424
- CPU latency (1 thread): preprocess 0.32 ms, model p50 5.87 ms / p95 11.46 ms

## Per signer
|     g |    acc |         n |
|------:|-------:|----------:|
| 49445 | 0.6043 | 4968.0000 |
| 32319 | 0.6089 | 4753.0000 |
| 25571 | 0.6254 | 3865.0000 |
| 55372 | 0.6956 | 4826.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.6936 | 3202.0000 |
| 0-10%  | 0.7532 | 1738.0000 |
| 10-25% | 0.7200 | 2882.0000 |
| 25-50% | 0.6462 | 5009.0000 |
| >50%   | 0.5067 | 5581.0000 |

## 10 worst classes
| label   |   acc |
|:--------|------:|
| there   | 0.013 |
| cloud   | 0.062 |
| go      | 0.167 |
| give    | 0.167 |
| finger  | 0.173 |
| close   | 0.242 |
| lamp    | 0.254 |
| sticky  | 0.269 |
| milk    | 0.271 |
| tongue  | 0.271 |

## 20 most confused pairs (rate = share of the true class)
| true        | pred      |   count |   rate |
|:------------|:----------|--------:|-------:|
| ear         | hear      |      35 |  0.443 |
| pen         | pencil    |      33 |  0.412 |
| wake        | awake     |      33 |  0.423 |
| listen      | hear      |      28 |  0.295 |
| look        | see       |      27 |  0.297 |
| lips        | mouth     |      26 |  0.347 |
| awake       | wake      |      23 |  0.311 |
| penny       | think     |      23 |  0.295 |
| cloud       | alligator |      22 |  0.344 |
| think       | penny     |      21 |  0.276 |
| stay        | same      |      21 |  0.288 |
| tongue      | duck      |      20 |  0.286 |
| pretend     | mouse     |      20 |  0.260 |
| there       | boat      |      20 |  0.267 |
| lamp        | shhh      |      20 |  0.282 |
| glasswindow | tooth     |      19 |  0.237 |
| fish        | beside    |      19 |  0.257 |
| doll        | mouse     |      17 |  0.227 |
| table       | minemy    |      17 |  0.266 |
| there       | hesheit   |      16 |  0.213 |
