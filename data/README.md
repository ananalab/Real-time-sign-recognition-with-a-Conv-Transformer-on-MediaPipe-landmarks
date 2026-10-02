# Data sources

Raw data are never committed (`data/raw/`, `data/processed/` are git-ignored).
All sources are converted to the packed unified format described in `src/signrec/store.py`
(89 landmarks: 2×21 hands, 7 upper-body pose points, 40 lip-contour points; `x, y, z`; NaN = missing).

| Source | Language | Version / date | Licence | Conversion |
|---|---|---|---|---|
| Google *Isolated Sign Language Recognition* (`asl-signs`) | ASL | Kaggle competition, 2023; downloaded 2026-10 | Competition rules (no redistribution) | `scripts/build_kaggle_kernel.py --push` (runs on Kaggle), then `kaggle kernels output` |
| LSFB-ISOL v2 | LSFB (French Belgian SL) | lsfb.info.unamur.be, `lsfb_v2/isol`; downloaded 2026-10-01 | CC BY 4.0, citation required (see below) | `scripts/convert_lsfb.py` (streams `poses_raw`, keeps only the unified subset) |

## asl-signs
- 94,477 sequences, 250 signs, 21 participants, MediaPipe Holistic (legacy solution) landmarks, 543 per frame.
- Access requires accepting the competition rules on kaggle.com with your own account.

## LSFB-ISOL v2
- 120,739 isolated-sign instances cut from spontaneous conversations, 4,657 glosses, 99 signers, 50 fps.
- Poses extracted with MediaPipe (Tasks numbering: 33 pose / 21 per hand / 478 face points).
- We use **raw** poses (`poses_raw`: no interpolation nor smoothing) so that missing detections look
  like those of the webcam demo.
- Vocabulary: 100 most frequent glosses performed by ≥ 10 signers (55,599 instances, ≥ 217 per gloss).
- Official `train`/`test` splits are signer-independent (verified: no signer in both).

### Required citations
- Fink, J., Frénay, B., Meurant, L., Cleve, A. (2021). *LSFB-CONT and LSFB-ISOL: Two New Datasets for
  Vision-Based Sign Language Recognition*. IJCNN 2021.
- Meurant, L. (2015). *Corpus LSFB. Un corpus informatisé en libre accès de vidéos et d'annotations de
  la langue des signes de Belgique francophone (LSFB)*. Université de Namur.
