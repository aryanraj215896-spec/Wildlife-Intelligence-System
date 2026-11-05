# Jaguar, Leopard, and Cheetah manual inspection

This review package was generated from the existing ResNet18 checkpoint and the existing frozen test split. It is for manual triage only; the observations below do not change any labels, dataset files, manifest rows, split assignments, or model artifacts.

## Evaluation source

- Model version: `resnet18-20261104T223635Z`
- Checkpoint: `models/runs/resnet18-20261104T223628Z/best-resnet18.pt`
- Manifest: `data/manifests/dataset_manifest-20261105-current.csv`
- Manifest SHA-256: `c8e633e6b0056aabc3436c3114a2c7737fe9b53c436cc5d568103bd1b60aa7a6`
- Evaluation report: `reports/metrics/refreshed-20261105/resnet18-test-metrics.json`
- Scores are maximum softmax scores, not calibrated probabilities.

The confusion matrix rows are actual labels and columns are predicted labels:

| Actual | Correct | Misclassified as |
|---|---:|---|
| Leopard | 168/202 | Cheetah 29, Tiger 2, Jaguar 2, Gray wolf 1 |
| Cheetah | 189/206 | Leopard 11, Jaguar 4, Lion 1, Gray wolf 1 |
| Jaguar | 128/135 | Leopard 6, Cheetah 1 |

The per-image export contains 58 available target-class test errors: 34 Leopard, 17 Cheetah, and 7 Jaguar. Another 15 available examples from other actual classes were predicted as one of these target classes (8 Leopard and 7 Cheetah); those are separated in `other_classes_predicted_as_targets.csv` and its contact sheet.

## Visual triage observations

`preliminary_visual_review_flags.csv` contains 14 initial review suggestions with full paths and blank fields for your decision and notes. These are visual screening cues, not asserted corrections:

- `Cheetah/VI8FA70VE4OW.jpg` and `Leopard/N3FBMBCJ2YP4.jpg` look snow-leopard-like. Confirm the exact species; “leopard” is not necessarily the same target as the supported Leopard class.
- Several images labeled Cheetah have conspicuous rosette-like coats and were predicted Leopard/Jaguar. The flags point to example files; check source labels before changing any.
- `Leopard/66ZHS34H7D6M.jpg` is a heavily occluded silhouette with little usable species detail.
- `Leopard/FLF4F14MW9JX.jpg` contains multiple animals, making a single-image class label ambiguous.
- `Cheetah/W5O4O7MQYF02.jpg` is an extreme close crop; `Cheetah/HNZVTD9SF7I9.jpg` is obstructed and low-confidence.

The three `misclassified_*.jpg` sheets show the target-class errors. Filenames are printed on each tile and full paths, actual labels, predicted labels, scores, and editable review fields are in `target_class_misclassifications.csv`.

## Duplicate screening

- No exact SHA-256 duplicate pairs were found among the scanned target-class files (`exact_duplicate_pairs.csv`).
- A 64-bit difference-hash screen at Hamming distance <= 3 produced 13 candidate pairs (`near_duplicate_candidates_dhash_le3.csv`). The paired-image contact sheet shows that several are visually very similar; manually verify whether each is the same photograph, a crop/resize, or merely similar subject matter.
- Seven candidate pairs cross split boundaries, including six train/test pairs and one train/validation pair. If confirmed as the same underlying photograph, these indicate possible evaluation leakage. The CSV includes both paths, labels, splits, distance, and blank review fields.
- Difference-hash matches are heuristic, not proof of duplicates. No image was moved or excluded.

## Missing files / changing local snapshot

The manifest lists 900 Jaguar records. During a read-only inventory, 15 Jaguar paths were absent from disk: 12 train, 2 test, and 1 validation. During final export verification, `Cheetah/4BZBG0RFB916.jpg` (test split) was also no longer present, although it had been read for inference and the contact sheet earlier. The known absent paths are in `manifest_images_missing_on_disk.csv`; `target_dataset_file_integrity.csv` contains per-path presence and manifest/current hash checks. I did not remove or move any dataset files. The filesystem changed during this audit, so re-check availability before relying on any source path.

The saved evaluation report is for 2,471 test examples. At the start of per-image inference, one Jaguar test image was already missing, so 2,470 examples could be rerun and checked against the report; additional target images became unavailable by final verification. No prediction or score is inferred for a missing image. The Cheetah contact sheet retains a thumbnail of the now-missing `4BZBG0RFB916.jpg`, but its source path no longer resolves. Check current file availability before relying on a path or interpreting per-image exports.

## Files

- `target_class_misclassifications.csv` — actual Jaguar/Leopard/Cheetah errors with path, predicted label, score, and manual-review fields.
- `misclassified_jaguar.jpg`, `misclassified_leopard.jpg`, `misclassified_cheetah.jpg` — contact sheets for those errors.
- `other_classes_predicted_as_targets.csv` and `other_classes_predicted_as_targets.jpg` — other actual labels predicted as a target class.
- `preliminary_visual_review_flags.csv` — initial visual triage suggestions; not corrections.
- `near_duplicate_candidates_dhash_le3.csv` and `near_duplicate_candidate_images.jpg` — heuristic candidate pairs for manual verification.
- `exact_duplicate_pairs.csv` — exact-hash pairs (empty beyond the header).
- `target_dataset_file_integrity.csv` and `manifest_images_missing_on_disk.csv` — current presence/hash audit outputs.
- `summary.json` — model, manifest, counts, and scan notes.
