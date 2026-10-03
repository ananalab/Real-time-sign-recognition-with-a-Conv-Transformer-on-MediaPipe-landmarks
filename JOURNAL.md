# Research log

Decisions, rationale and measured results, in chronological order.

## Scope and environment

**Decisions**
- Scope: Google ISLR (ASL, 250 signs) end to end, then LSFB-ISOL (French Belgian Sign Language)
  and cross-lingual transfer ASL -> LSFB.
- Python 3.12 with `uv`. `mediapipe` 1.0.1 ships a `py3-none` wheel, but 3.12 is the safest version
  for the torch / onnxruntime / lightgbm stack.
- The raw Kaggle data (> 40 GB) do not fit on the development laptop (≈ 16 GB free): the conversion
  to the compact unified format runs inside a Kaggle notebook and only its output is downloaded.
- LSFB poses are streamed from the UNamur server and reduced to the unified subset on the fly
  (≈ 10 KB per clip instead of ≈ 70 KB).
- Hips are excluded from the landmark subset: they are usually out of frame (y > 1) in webcam and
  LSFB recordings. Final subset: 2 × 21 hand points, 7 upper-body points, 40 lip points = 89.

## LSFB-ISOL inspection

- 120,739 instances, 4,657 glosses, 99 signers, 50 fps; long-tailed gloss distribution.
- LSFB v2 poses follow the MediaPipe *Tasks* numbering (33 pose / 478 face points), i.e. the same
  extractor as the browser demo, whereas the Kaggle landmarks come from the legacy Holistic solution.
- The official `train`/`test` splits are signer-independent (no signer in both).
- Vocabulary: 100 most frequent glosses performed by ≥ 10 signers -> 55,599 instances (≥ 217 per gloss).
- We use the *raw* poses (no interpolation or smoothing): missing hands appear as NaN, like in the
  Kaggle data and in live webcam use.
- Clips shorter than 4 frames (mostly empty raw pose files) are dropped: 0.56 % of the selected clips
  (an early estimate of 3.1 % came from a non-random 10k sample).
- Final LSFB data: 55,286 clips; signer split 26,114 train (50 signers) / 4,423 val (9) /
  24,749 test (40, official test signers). Training examples per gloss: 72 to 1,492 (median 185).

## Baselines (ASL, signer-independent)

| Model | Val top-1 | Test top-1 | Test top-5 | Test macro-F1 |
|---|---|---|---|---|
| Chance | 0.4 % | 0.4 % | 2.0 % |, |
| Stat. features + logistic regression | 47.9 % | 42.1 % | 69.4 % | 41.9 % |
| Stat. features + LightGBM | 45.2 % | 33.8 % | 61.5 % | 33.4 % |

- First LightGBM run reached only 18.8 % (val): early stopping on the multiclass log-loss fired
  far too early. Stopping on the classification error fixed it (best iteration 293).
- The val->test gap is larger for LightGBM (−11 pts) than for the linear model (−6 pts):
  trees overfit signer-specific statistics.

## Two methodological fixes found during development

- **Augmentation placement.** Image-space translation and isotropic scaling are exactly cancelled
  by the shoulder normalisation (our invariance tests prove it), and a uniform speed change is
  undone by resampling to 64 frames. Spatial augmentation now runs after normalisation (body frame)
  and the time warp is non-uniform.
- **Aspect ratio.** MediaPipe divides x by the width and y by the height. LSFB videos are 720×576
  PAL with a 16:9 display aspect, so vertical distances are inflated by 1.78. Check: median
  nose-above-shoulders / shoulder span = 0.399 (ASL) vs 0.765 (LSFB raw) -> 0.430 after × 576/1024;
  a 4:3 assumption would give 0.574. Per-source `y_scale`: ASL 1.0, LSFB 0.5625, webcam H/W.
- Profiling: data pipeline 3.2 -> 1.8 ms/sample (training) after vectorising the resampling;
  epoch time on a Kaggle T4 was CPU-bound (≈65 s for both models).

## Baselines (LSFB-ISOL, 100 glosses, official test signers)

| Model | Val top-1 | Test top-1 | Test top-5 | Test macro-F1 |
|---|---|---|---|---|
| Chance | 1.0 % | 1.0 % | 5.0 % |, |
| Stat. features + logistic regression | 57.0 % | 49.0 % | 75.3 % | 40.9 % |
| Stat. features + LightGBM | 54.8 % | 47.1 % | 72.6 % | 37.3 % |

- Top-1 is flattered by the long tail (the 10 most frequent glosses cover 32.3 % of the test clips,
  AUSSI alone 6.2 %); macro-F1 is ~8 pts lower than top-1, unlike on the balanced ASL data.

## Latency, FP32 vs dynamic INT8 (idle M1, one thread, untrained weights)

| Model | Native FP32 | Native INT8 | Browser (wasm) FP32 | Browser INT8 | Size FP32 / INT8 |
|---|---|---|---|---|---|
| Conv-Transformer | 7.7 ms | 16.3 ms | 35.5 ms | 32.7 ms | 8.2 / 2.2 MB |
| BiGRU | 5.7 ms | 5.4 ms | 7.7 ms | 7.8 ms | 5.5 / 4.6 MB |

- Dynamic INT8 is 2.1× *slower* natively on the M1 for the Transformer (quantise/dequantise
  overhead around many small matmuls), but similar in the browser and 3.7× smaller to download.
  Decision: ship INT8 in the browser for download size; final numbers via scripts/bench_latency.py.
- The GRU op itself is not quantised by onnxruntime dynamic quantisation (size barely changes).

## Main ASL models (Kaggle T4, 50 epochs, 3 seeds, signer-independent)

| Model | Params | Val top-1 | Test top-1 | Test top-5 | Test macro-F1 | Std across test signers |
|---|---|---|---|---|---|---|
| BiGRU | 1.43 M | 65.8 ± 0.5 | 63.0 ± 0.5 | 84.0 ± 0.3 | 62.3 ± 0.6 | 3.9 |
| Conv-Transformer | 2.16 M | 68.2 ± 0.1 | 64.8 ± 0.4 | 84.0 ± 0.3 | 64.1 ± 0.4 | 3.3 |

- +22.7 pts over the best baseline (logistic regression, 42.1 % test).
- The Conv-Transformer beats the GRU on every one of the 4 test signers.
- Best checkpoints at epochs 39–46 of 50 and the validation curve is still rising -> a 150-epoch
  run (`experiments/asl_long.yaml`) measures the headroom; comparisons stay at 50 epochs.
- Accuracy falls to ≈52 % when no hand is detected in more than half of the frames (30 % of
  test clips): landmark detection quality is the main limiting factor.
- Most confused pairs are near-synonyms often signed alike: wake/awake, pen/pencil, duck/goose,
  look/see, a vocabulary property, not only a model error.
- Evaluation crashed on Kaggle (evaluate.py imported onnxruntime through export.py); fixed by
  moving `load_checkpoint` to `signrec/checkpoint.py`; evaluations were run locally.

## INT8 quantisation broke the Conv-Transformer (found with trained weights)

- Default dynamic INT8: validation top-1 **34.2 %** (2,000 clips: 40.7 %) vs 67.9 % in FP32.
  Restricting quantisation to constant-weight matmuls or per-channel weights did not help.
- Cause: rare outliers in the *hand-local* features (max 190, p99.9 = 6.2; 0.04 % of values > 10),
  produced when a detected hand is tiny (division by the wrist->middle-MCP length). A per-tensor
  dynamic scale on the first layer then wipes out the resolution of all other inputs.
- Fix: keep the input projection in FP32 -> INT8 top-1 68.4 % vs 67.9 % FP32, 2.5 MB vs 8.3 MB.
- Future work: clip the hand-size denominator in preprocessing (would require retraining).

## Ablations and leakage (Conv-Transformer, seed 0, validation signers)

Reference: 68.3 % val top-1. One factor changed at a time.

| Variant | Val top-1 | Δ | Note |
|---|---|---|---|
| pad + mask instead of resampling | 69.5 | +1.2 | |
| no lips | 68.8 | +0.5 | |
| T' = 32 | 68.4 | +0.2 | |
| hands only (no body, no lips) | 68.3 | 0.0 | |
| T' = 96 | 67.5 | −0.8 | |
| no augmentation | 67.0 | −1.3 | |
| no velocities | 66.6 | −1.7 | |
| with depth z | 65.5 | −2.8 | early-stopped at epoch 31 -> re-run without early stopping |
| no canonical hand | 64.5 | −3.8 | |
| no hand-local features | 63.8 | −4.5 | early-stopped at epoch 21 -> re-run without early stopping |

- Early stopping on a plateau while the learning rate is still high penalises a run by ~3 pts
  (it misses the end of the cosine schedule). Both affected ablations re-run for 50 full epochs.
- Seed-to-seed std of the reference is 0.1 pt (val), so ±0.5 pt differences are borderline.
- Lips and upper body bring nothing measurable on ASL; the canonical dominant hand and the
  hand-local shape features matter most.
- **Leakage**: with a random sample-level split, test top-1 = 81.7 % vs 64.8 % on unseen signers
  (+16.9 pts of optimism).

## Transfer study, long schedule, corrected ablations

**LSFB (validation top-1, 3 seeds)**

| Examples per sign | 5 | 10 | 25 | 50 | 100 | all |
|---|---|---|---|---|---|---|
| From scratch | 25.7 | 36.6 | 51.6 | 60.0 | 66.5 | 77.0 |
| ASL-pretrained, fine-tuned | 27.1 | 36.9 | 50.8 | 57.4 | 63.0 | 74.9 |

- Test (40 signers, all data): scratch 68.3 ± 0.2, fine-tuned 66.2 ± 0.2, frozen-then-tuned 66.6.
  **ASL pre-training does not help at any size** (the +1.4 at 5 examples is within one s.d.).
- Linear probe on the frozen ASL encoder: 38.3 % (val) < logistic regression on statistics (57.0 %).
- Weekly Kaggle GPU quota (30 h) exhausted; remaining checks run locally on the M1 (MPS).

**ASL**
- 150-epoch schedule: 67.5 % val vs 68.3 % at 50 epochs -> no headroom from longer training.
- Re-run without early stopping: no hand-local −0.8 (was −4.4), depth −1.1 (was −2.7).
  Four other ablations stopped at epochs 41–46 and are flagged † in the paper.
- Seed ensembles (free): ASL 67.4 % test (single 64.8), LSFB 70.9 % (single 68.3).
- Demo now ships the from-scratch LSFB model (best on validation); browser parity 40/40.

## Final latency (idle M1, one thread) and transfer checks

| Model | Native FP32 / INT8 | Browser FP32 / INT8 (p50) | Size FP32 / INT8 |
|---|---|---|---|
| Conv-Transformer | 7.8 / 16.5 ms | 16.3 / 33.4 ms | 8.3 / 2.5 MB |
| BiGRU | 5.7 / 5.5 ms | 7.9 / 8.1 ms | 5.5 / 4.8 MB |

- Earlier measurements (≈11 / 34 ms) were inflated by training jobs running in parallel.
- Low learning rate (2e-4) fine-tuning, 25 examples/sign: 46.3 % val (standard ft 50.6, scratch 51.5).
- Linear probe on the frozen ASL encoder, all data: 38.3 % val.

## Analyses added after review (no retraining)

- `scripts/stats_tests.py`: paired bootstrap of accuracy differences (hierarchical over signers,
  or over clips within signers) and exact McNemar tests on seed ensembles.
  - ASL, Conv-Transformer vs BiGRU: +1.8 pts, 95 % interval [0.9, 2.9] with signers resampled,
    McNemar p < 0.001. The ranking does not depend on the four test signers.
  - LSFB, ASL-pretrained vs scratch (all data): −2.1 pts [−2.8, −1.4] (fine-tuned) and −1.7 pts
    [−2.3, −1.2] (frozen 5 epochs): pre-training is slightly but significantly harmful.
- `scripts/analyse_lsfb_gap.py`: LSFB validation accuracy (77.2 %) exceeds test accuracy (68.3 %)
  mainly because 3 of the 9 validation signers contribute 75 % of its clips; per signer, 72.7 ± 2.5 %
  (validation) vs 67.9 ± 2.0 % (test). Re-weighting the test classes to the validation distribution
  gives 70.2 %. Test signers whose dialogue partner is a training signer: 71.8 % vs 64.7 %.
- Known limitations made explicit in the report: velocities are computed after resampling to 64
  frames (scale depends on clip length and frame rate), all transfer runs start from ASL seed 0, and
  the linear probe still updated batch-normalisation statistics of the frozen encoder.
- `tests/conftest.py` loads onnxruntime before torch: in the other order the test process aborted
  intermittently at exit on macOS ("recursive_mutex lock failed").
- The LaTeX sources are versioned in `paper/`; `make paper` rebuilds `report.pdf`, and the macro
  file is rewritten from scratch so that no stale number survives.
