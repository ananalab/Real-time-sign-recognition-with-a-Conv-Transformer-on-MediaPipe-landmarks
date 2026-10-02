# Model card

Format inspired by Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).
Models of the signrec project. All figures below come from the evaluation scripts (see `reports/`
and `report.pdf`).

## Model details

| | ASL model | LSFB model |
|---|---|---|
| Task | Isolated sign recognition (one sign per clip, closed vocabulary) | same |
| Vocabulary | 250 ASL signs (Google ISLR) | 100 LSFB glosses (most frequent in LSFB-ISOL, ≥ 10 signers) |
| Input | MediaPipe Holistic landmarks: 2 × 21 hand, 7 upper-body, 40 lip points per frame | same |
| Architecture | Conv-Transformer (depthwise temporal convolutions + self-attention), 2.2 M parameters | same, trained from scratch (ASL initialisation did not help) |
| Output | Probability over the vocabulary (top-5 shown in the demo) | same |
| Formats | PyTorch checkpoint, ONNX FP32 (shipped), ONNX dynamic INT8 | same |

## Intended use

- Demonstrations and research on pose-based isolated sign recognition.
- Educational use: practising the signs of a fixed vocabulary with immediate feedback.
- Runs locally in a web browser; video is never sent to a server.

## Out-of-scope uses

- **Not a translator.** Sign languages have their own grammar, use of space and facial
  expressions; this model recognises single signs from a fixed list and ignores all of that.
- Not suitable for any situation where an error has consequences (medical, legal, administrative,
  emergency, education assessment). Professional interpreters are irreplaceable.
- The LSFB model is **not** an LSF (French Sign Language) model: LSFB is a distinct language.
- Not intended for identifying or tracking people.

## Training data

- Google *Isolated Sign Language Recognition* (Kaggle, 2023): 94,477 clips, 21 participants,
  smartphone recordings, landmarks from the legacy MediaPipe Holistic solution. Competition rules
  apply; the data are not redistributed.
- LSFB-ISOL v2 (Fink et al., IJCNN 2021; LSFB corpus, Meurant 2015): isolated signs cut from
  spontaneous conversations, CC BY 4.0. Raw (unsmoothed) landmarks.

## Evaluation

- Signer-independent: test signers never appear in training or validation; the test sets were used
  once, for the final models.
- Metrics: top-1 / top-5 accuracy, macro-F1, accuracy per signer. Results in `report.pdf` and
  `reports/eval/`.

## Known limitations and biases

- **Few signers.** 21 ASL participants; per-signer accuracy varies markedly, so performance on a
  new user can be much lower than the average.
- **Recording conditions.** Trained on smartphone (ASL) and studio (LSFB) framings. Webcam
  distance, lighting, occlusions and fast motion degrade landmark detection, and missing hands
  strongly reduce accuracy.
- **Extractor shift.** ASL training landmarks come from an older MediaPipe pipeline than the one
  used in the browser; normalisation and augmentation reduce but do not remove this shift.
- **Non-manual features.** Only the lips are used from the face; signs distinguished by eyebrows,
  gaze or head movement are not captured.
- **Dictionary-form vs. conversational signs.** ASL signs were produced in isolation; LSFB signs
  come from conversations and are co-articulated, short and variable.
- **Vocabulary.** Any sign outside the vocabulary is mapped to its closest vocabulary sign; low
  confidence is shown but there is no explicit rejection class.

## Ethical considerations

- Landmarks of the face and body can still identify a person; the demo processes them locally
  and stores nothing.
- Sign language technologies should be built with and for Deaf communities. This project is an
  academic study and has not been evaluated by Deaf signers.

## Headline numbers (test signers never seen in training)

| | ASL model | LSFB model |
|---|---|---|
| Top-1 / top-5 | 64.8 % / 84.0 % | 68.3 % / 87.7 % |
| Macro-F1 | 64.1 % | 63.9 % |
| Accuracy range across test signers | 61.9–69.5 % (4 signers) | see `reports/eval/` (40 signers) |
| Accuracy when no hand is detected in > 50 % of frames | 52.0 % |, |
| Browser latency per sign (1 CPU thread, Apple M1) | 16 ms | 16 ms |
