# Evaluation of `lsfb-scratch-nall-s0` on `val`

- samples: 4423 · classes: 100 · signers: 9
- **top-1 0.7723** · top-5 0.9317 · macro-F1 0.7031
- per-signer top-1: 0.7267 ± 0.0754
- CPU latency (1 thread): preprocess 0.68 ms, model p50 13.02 ms / p95 13.99 ms

## Per signer
| g    |    acc |         n |
|:-----|-------:|----------:|
| S074 | 0.5940 |  133.0000 |
| S015 | 0.6160 |  125.0000 |
| S085 | 0.7222 |  180.0000 |
| S023 | 0.7338 |  278.0000 |
| S083 | 0.7453 |  534.0000 |
| S039 | 0.7522 |  343.0000 |
| S071 | 0.7600 |   25.0000 |
| S040 | 0.7996 | 2330.0000 |
| S084 | 0.8168 |  475.0000 |

## Accuracy vs share of frames without any hand
| bin    |    acc |         n |
|:-------|-------:|----------:|
| 0      | 0.7752 | 4052.0000 |
| 0-10%  | 0.7685 |  216.0000 |
| 10-25% | 0.7258 |  124.0000 |
| 25-50% | 0.6207 |   29.0000 |
| >50%   | 0.5000 |    2.0000 |

## 10 worst classes
| label         |   acc |
|:--------------|------:|
| JUSQUE        | 0.125 |
| PAS           | 0.176 |
| CA-VEUT-DIRE  | 0.184 |
| ATTENDRE.STOP | 0.263 |
| ENVIRON       | 0.312 |
| POUR          | 0.333 |
| APPELER       | 0.360 |
| BEAUCOUP      | 0.364 |
| CONNAITRE     | 0.368 |
| FINIR         | 0.368 |

## 20 most confused pairs (rate = share of the true class)
| true         | pred       |   count |   rate |
|:-------------|:-----------|--------:|-------:|
| CA-VEUT-DIRE | DIRE       |      34 |  0.694 |
| OUI          | JUSTE.F    |      20 |  0.055 |
| MAIS         | FALLOIR    |      19 |  0.181 |
| POUR         | ENTENDANT  |      13 |  0.255 |
| FINIR        | C-EST-TOUT |      12 |  0.316 |
| JUSTE.F      | OUI        |      11 |  0.200 |
| 1            | TOUJOURS   |       9 |  0.167 |
| C-EST-TOUT   | FINIR      |       8 |  0.308 |
| CONNAITRE    | SAVOIR     |       8 |  0.421 |
| POUR         | PENSER     |       8 |  0.157 |
| MOT          | PERSONNE   |       7 |  0.212 |
| QUOI         | AVOIR      |       7 |  0.050 |
| APRES        | LAISSER    |       7 |  0.194 |
| SOURD        | FALLOIR    |       6 |  0.039 |
| OUI          | APRES      |       6 |  0.017 |
| ENVIRON      | LS         |       5 |  0.312 |
| MAIS         | PERSONNE   |       5 |  0.048 |
| PLUS         | AUSSI      |       5 |  0.208 |
| OUI          | BEAUCOUP.F |       5 |  0.014 |
| MAIS         | NON        |       5 |  0.048 |
