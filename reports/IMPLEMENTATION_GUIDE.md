# Wildlife Intelligence System — Implementation Guide

The project is a local Streamlit demonstration for eight-class animal image classification, user-confirmed feedback, and automatic guarded retraining. It is not a public or authenticated service.

## Application

`app.py` launches `wildlife_ai.streamlit_app`. The interface is one page with an image uploader, image preview, ResNet18 prediction, alternative labels, uncalibrated score, and Correct/Incorrect controls. Corrections use the internal labels in `config/settings.py`; `Gray wolf` is displayed as `Gray Wolf`. “Other / none of these” is recorded without a supported training label.

`feedback/service.py` stores the prediction, confirmed label, model version, score, timestamp, and original upload SHA-256 in SQLite. Retention and future-training consent are distinct. Declined retention preserves metadata only; training consent cannot be used without image-retention consent. Retained files are sanitized and stored locally.

There is no user authentication, reviewer, curation, taxonomy, or model-management interface. Feedback labels are user-supplied, not independently verified. Keep the application local to the project operator's computer.

## Automatic training

`automatic_learning.py` triggers after five eligible, previously unused feedback examples have accumulated. Unsupported labels, missing/withdrawn consent, missing or invalid retained images, duplicates, and overlaps with the frozen dataset are excluded. The code reports eligibility counts and keeps attempt membership in SQLite to prevent repeatedly using the same samples.

Each update creates a new full manifest snapshot using `data/manifests/dataset_manifest.csv` as its base. Existing base rows—including validation and test—are preserved; feedback is added to train only. The alternate `dataset_manifest-20261105-current.csv` is intentionally not used. Consent, path containment, hashes, and split safety are revalidated immediately before training.

Training warm-starts from the current selected ResNet18 and uses a weighted sampler to mix feedback with original training replay. Candidate evaluation uses validation only. The candidate must remain within 1 percentage point of the current validation accuracy and macro-F1 and within 2 percentage points of each class's validation recall. The frozen test split is not used for repeated model selection. A passing candidate is selected by an atomic SQLite pointer update; a rejected or failed attempt leaves the prior checkpoint selected. Checkpoints, manifests, evaluation reports, and batch outcomes are preserved under new local paths.

## Selected checkpoint and evidence

The initial model is `resnet18-20261105T013137Z`, using `models/runs/resnet18-20261105T013130Z/best-resnet18.pt`. The application validates the checkpoint against its registry record and the validation report in `reports/model_information/`. The report is tied to the hash of the frozen `dataset_manifest.csv`, has the expected class order and sample count, and reproduces:

| Split | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 2,462 | 96.39% | 95.64% |
| Frozen test | 2,460 | 94.88% | 93.75% |

The frozen test result is historical evidence for the initial checkpoint, not a repeatedly consulted promotion metric. Model scores are uncalibrated; out-of-distribution recognition is not implemented.

## Persistence and local execution

SQLite migration 5 adds `application_model_selection`, `automatic_training_batches`, and `automatic_training_batch_feedback`. It preserves existing feedback, registry, taxonomy, and training records. Existing migration 4 remains responsible for adding and backfilling the corrected label and upload hash.

Run the app and tests with the existing virtual environment, as shown in [`../README.md`](../README.md). If code and model/data files are in different directories, set `WILDLIFE_AI_ROOT` to the directory containing the local `data/`, `models/`, and database.

The app and tests do not start a full dataset training job. Retraining only occurs after the configured eligible-feedback threshold is reached.
