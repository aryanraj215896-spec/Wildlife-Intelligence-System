# Wildlife & Animal Intelligence System — Implementation Guide

**Document date:** 2026-11-05
**Project specification:** [`../PROJECT_SPEC.md`](../PROJECT_SPEC.md)
**Roadmap and phase record:** [`../plan.md`](../plan.md)
**Experiment summary:** [`FINAL_REPORT.md`](FINAL_REPORT.md)
**Project state:** Academic research prototype

## 1. Purpose and status

This document records what is implemented, how to run it, what has been measured, and what remains unverified. It is an implementation guide and handoff record, not a declaration of production readiness.

The application is local-first. The software includes dataset inspection, training and evaluation, a versioned taxonomy importer/resolver, local inference, feedback and review, curation safeguards, Grad-CAM for ResNet18, and an auditable model registry. The current model checkpoints are registry candidates. The app offers a separate **Local development/demo (candidate)** mode for explicitly experimental inference; no candidate is production-approved or stored as active. Production inference stays unavailable until the outstanding evidence and policy gates are resolved.

No neural model, accuracy measurement, or user-entered class name substitutes for source rights, taxonomic verification, OOD validation, or governance approval. Source-page statements and the user's class-to-source assignments are documented as such, not presented as per-image proof.

## 2. Implemented capabilities

### 2.1 Supported class configuration

The configured output classes are centralized in `src/wildlife_ai/config/settings.py`:

1. Domestic Cat (`cat/`)
2. Domestic Dog (`dog/`)
3. Lion (`lion/`)
4. Tiger (`tiger/`)
5. Leopard (`leopard/`)
6. Gray wolf (`wolf/`)
7. Cheetah (`cheetah/`)
8. Jaguar (`jaguar/`)

Directory matching is case-insensitive, and the folder aliases `Cat` and `Dog` resolve to the canonical labels Domestic Cat and Domestic Dog. These model labels are not external taxonomy identifiers; a reviewed taxonomy import and explicit crosswalk are still required to claim those links.

### 2.2 Dataset preparation

`src/wildlife_ai/data/preparation.py` and `scripts/prepare_dataset.py` implement local inspection and manifest creation:

- Enumerates configured class folders without downloading or modifying original files.
- Accepts supported JPEG and PNG images; excludes non-image files from valid examples and records rejected/corrupt image records.
- Checks image decodability and dimensions, records file paths, class, hashes and available metadata.
- Supports optional metadata CSV fields including source, source image ID, licence, attribution, collection date, canonical taxon ID, review status, quality notes, and group ID.
- Detects byte-identical duplicates using SHA-256 and keeps duplicate or provided observation groups together in splitting.
- Produces deterministic 70/15/15 train/validation/test assignment from seed 42 by default.
- Preserves all eight target classes in availability reporting, including classes with no images.
- Writes CSV manifest output and reports its hash and class/split/rejection counts.

The scan does not implement perceptual/near-duplicate detection. Identical bytes and supplied `group_id` values are the available grouping evidence. Folder names and page-level dataset counts are not independently validated scientific labels.

The raw files, generated manifests, checkpoints and local database are intentionally ignored by Git. They remain local workspace artifacts and are not included in source-control changes.

### 2.3 Dataset loader and model architectures

- `src/wildlife_ai/data/loading.py` provides manifest-backed PyTorch datasets, preprocessing transforms, and configurable data loaders; paths are constrained under the configured raw-data root.
- `src/wildlife_ai/models/architectures.py` defines:
  - `ScratchCNN`, a compact four-stage convolutional network with batch normalization, ReLU, pooling, adaptive average pooling, dropout and an eight-output linear head.
  - Torchvision ResNet18 with an eight-output replacement head; optionally initialized from Torchvision's default ImageNet-1K weights, with a frozen-backbone stage and optional fine-tuning.
- `src/wildlife_ai/models/checkpoint.py` loads checkpoint state using restricted PyTorch loading and returns model metadata needed to reproduce inference preprocessing and class order.
- Training and model artifacts are local. No hosted inference API is used.

### 2.4 Training and evaluation

`src/wildlife_ai/training/engine.py` and `scripts/train.py` implement:

- Fixed-seed training with configurable model, epoch counts, batch size, image size, workers, device and learning rates.
- `auto` device selection (CUDA when available, otherwise CPU); explicitly requesting unavailable CUDA fails rather than silently pretending to use it.
- CUDA mixed precision, mini-batches, pinned memory/non-blocking transfers where appropriate, and runtime checks that model and batch devices match.
- Validation-based best checkpoint selection; atomic best/last checkpoint writes and epoch-history CSV.
- Run metadata including class ordering, manifest SHA-256, seed/configuration, preprocessing, library/platform/device details and pretrained-weight provenance.
- Consent-registry validation when a curated feedback manifest is used for training.

`src/wildlife_ai/evaluation/metrics.py` and `scripts/evaluate.py` calculate overall and per-class classification measures, confusion data, inference timing and validation-based max-softmax abstention coverage curves. Reports and SVG visualizations are written locally. The evaluation code does not select a final abstention threshold or establish OOD performance without separate OOD sets.

`scripts/benchmark_training.py` supports local throughput, step and memory benchmarking to inform batch-size choices; batch sizes are configurable rather than treated as universal hardware optima.

### 2.5 Taxonomy import and name resolution

- `src/wildlife_ai/taxonomy/importer.py` imports a caller-provided CSV only when required fields and source namespace, release, licence and attribution are present.
- Each import is associated with its source release and file SHA-256. Taxa are release-scoped; source taxon IDs are not fabricated.
- `src/wildlife_ai/taxonomy/normalization.py` provides deterministic text normalization.
- `src/wildlife_ai/taxonomy/resolver.py` returns exact/fuzzy candidates and unresolved states; a fuzzy match is never silently accepted and the app requires explicit user confirmation.
- SQLite schema supports taxonomy names, synonyms, parent taxa, internal concepts, and reviewed crosswalks across releases.

The source tree contains no reviewed production taxonomy snapshot. Therefore taxonomy lookup will remain empty or unresolved until an authorized snapshot is obtained and imported. Test fixtures are synthetic test inputs, not production taxonomy content.

### 2.6 Feedback, review, and curation

`src/wildlife_ai/feedback/service.py` and `src/wildlife_ai/curation/eligibility.py` provide:

- Feedback responses: yes, no, unsure, and unsupported.
- Model-version linkage and storage of a score explicitly typed as a softmax score, not a calibrated probability.
- Separate consent flags/timestamps for retaining an uploaded image and using it for future training; training-use consent requires retention consent.
- Optional image retention only after consent. Retained image bytes are decoded/sanitized, stored under a generated filename outside SQLite, and linked by hashes.
- Pending/verified/rejected/needs-review review states and a review history with reviewer reference and notes. Reviewer names are attribution only; this local application does not authenticate reviewers.
- Independent review gates for label, image quality, and source/licence. Eligibility requires a verified supported label, selected canonical taxon, required consents, retained image, source/licence and rights approval, quality approval, and no known duplicate.
- Curated manifest export as a new snapshot without overwriting an existing destination. The frozen base manifest and test split remain preserved; feedback examples are placed in train/validation only.
- Hash-based rejection of overlap with the frozen test or existing supported dataset, and consent revalidation at training time.
- Retained-image withdrawal and training-use withdrawal; withdrawal updates consent state and reports affected manifest records so model lineage can be reviewed.

The service cannot decide what evidence is sufficient for a real label/rights review. It does not provide authenticated roles, a formal appeal/escalation SOP, automatic verification, or automatic retraining.

### 2.7 Inference and experimental explainability

- `src/wildlife_ai/inference.py` applies checkpoint-specific image transforms and returns the predicted supported class and top alternatives.
- Scores are reported as uncalibrated softmax scores. The UI explains that they are not probabilities.
- `src/wildlife_ai/utils/images.py` performs safe local image decoding and sanitization.
- `src/wildlife_ai/explainability/gradcam.py` implements experimental Grad-CAM for ResNet18 inputs. It does not make causal claims or validate a class. ScratchCNN Grad-CAM is unsupported.
- There is no validated OOD detector or approved abstention threshold. Inputs outside the trained distribution can still receive a supported-class output in demo inference.

### 2.8 Application pages and model modes

`app.py` provides startup guidance when launched under a Python interpreter lacking required ML packages. Run the app using the project virtual-environment interpreter.

The Streamlit application in `src/wildlife_ai/streamlit_app.py` includes:

- **Classify:** Image upload, safe preview, local inference, top outputs, explicitly uncalibrated score, device selection and optional experimental ResNet18 Grad-CAM.
- **Taxonomy:** Source/release status and name search with candidate confirmation. Without an imported reviewed snapshot, no production catalogue is available.
- **Review:** Feedback review and curation eligibility/export workflow. The reviewer field is not authentication.
- **Models:** Local checkpoint candidate registration, test-report matching, candidate list, promotion policy, promotion/rejection and rollback controls.
- **Shared presentation:** Wildlife-themed typography and colors, consistent page headings/sidebar navigation, responsive spacing, readable light-theme labels and controls, and coherent upload, status, card, button, and table styling.

The Classify page separates two modes:

| Mode | Model source | Effect on production registry | Feedback |
|---|---|---|---|
| Production (active model) | Only the explicitly active registry model | Reads the active pointer; does not change it | Available under the consent and review controls |
| Local development/demo (candidate) | A registered candidate selected in the UI; this mode is selected by default if there is no active model | Does not promote, activate, or modify lifecycle state | Disabled, to avoid mixing experimental outputs with reviewed project feedback |

Both modes validate that the artifact is inside the project directory and its SHA-256 equals the registry record before loading. Candidate-demo outputs are clearly marked experimental. The demo mode is intended for local academic demonstration and is not authorization for public deployment or rights clearance.

### 2.9 Model registry and lifecycle

`src/wildlife_ai/registry/service.py` implements:

- Candidate registration from a project-contained checkpoint, restricted checkpoint validation, and artifact SHA-256.
- Optional matching frozen-test report validation for split, model version, architecture, manifest hash, class order and artifact size.
- Candidate enumeration for the local demo picker; enumeration is read-only and does not change lifecycle state.
- Explicit policy recording and candidate promotion checks for minimum overall/per-class quality, regression limits, inference time and artifact size.
- Explicit rejection and rollback with lifecycle/event history and artifact/consent checks.
- A separate singleton active-model pointer. Registering or selecting a candidate for demo does not set this pointer.

No approved numerical promotion criteria were supplied. No model is production-active, and no post-hoc policy was created from observed test performance. In the local ignored SQLite database, three versions are candidates: the refreshed `scratch_cnn-20261104T222831Z` and `resnet18-20261104T223635Z`, plus the earlier `resnet18-20261104T192318Z`. Their checkpoint files and recorded hashes were verified.

### 2.10 SQLite persistence

`src/wildlife_ai/storage/database.py` creates/upgrades the local SQLite schema and enables foreign-key enforcement. The schema covers:

- Taxonomy imports, taxa, names, concepts and crosswalks.
- Training runs, model versions, model output mappings, active pointer, promotion policies and registry events.
- Dataset manifests and per-item metadata/consent/split records.
- Feedback records, optional retained-image metadata and review history.

The database and generated artifacts are local and Git-ignored. Backup/recovery practices and authenticated multi-user access are not implemented as production operations.

## 3. Dataset and source record

### 3.1 Observed current files and manifest

Current refreshed manifest:

- Path: `data/manifests/dataset_manifest-20261105-current.csv`
- SHA-256: `c8e633e6b0056aabc3436c3114a2c7737fe9b53c436cc5d568103bd1b60aa7a6`
- Valid images: 16,478
- Split sizes: 11,534 train, 2,473 validation, 2,471 test
- Corrupt supported-format images: 0
- Exact duplicate groups: 0

| Class | Valid | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Domestic Cat | 2,371 | 1,660 | 356 | 355 |
| Domestic Dog | 4,978 | 3,484 | 747 | 747 |
| Lion | 2,007 | 1,405 | 301 | 301 |
| Tiger | 2,156 | 1,509 | 324 | 323 |
| Leopard | 1,345 | 941 | 202 | 202 |
| Gray wolf | 1,346 | 942 | 202 | 202 |
| Cheetah | 1,375 | 963 | 206 | 206 |
| Jaguar | 900 | 630 | 135 | 135 |
| **Total** | **16,478** | **11,534** | **2,473** | **2,471** |

The earlier manifest had 15,190 valid examples and 58 Gray wolf files. The 58 previously recorded Wolf paths remain with matching hashes. A refreshed scan found 1,346 images in the Wolf folder, including 1,288 absent from that earlier manifest. The earlier manifest and split were preserved; results from the earlier and refreshed experiments must not be mixed.

Counts describe the files scanned in this worktree. No files were downloaded or modified during the documented refresh. Exact-hash grouping does not detect visually similar images when different encodings, crops, or resolutions are used.

### 3.2 Source mapping and evidence limits

The user-reported mapping is Domestic Cat/Dog → Oxford-IIIT Pets; Jaguar → Kaggle `iluvchicken/cheetah-jaguar-and-tiger` (`jaguar_train_resized`); other classes → specified images.cv cards. Source-page findings and count discrepancies are in [`FINAL_REPORT.md`](FINAL_REPORT.md).

Important limitations:

- Source mapping is based on the user's statement; there is no verified record mapping each local file to a source archive entry.
- The raw per-image manifest has no confirmed source IDs or licence metadata.
- Oxford-IIIT's page states CC BY-SA 4.0 for the dataset and notes the image copyright remains with original owners.
- Kaggle's Jaguar page lists the licence as unknown.
- The cited wolf cards differ in licence statements; red/white wolf cards say academic research/original author rights and submitter licence not stated.
- Several images.cv page counts do not match the local class counts; page counts are not treated as source-file evidence or disjoint subsets.
- Folder labels and source labels have not undergone independent taxonomic/biological review.

The data is not asserted to be rights-cleared for redistribution or public deployment.

## 4. Training experiments and results

The current comparison used the same refreshed manifest, seed 42, validation-based checkpoint selection and 2,471-image test split. Test labels were not used for model selection, training or threshold tuning.

| Model | Training | Best validation accuracy | Test accuracy | Macro precision | Macro recall | Macro F1 | Forward-only throughput | Checkpoint size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ScratchCNN | 10 epochs, batch 32, 463.9 s | 71.37% | 68.35% | 67.86% | 68.63% | 67.72% | 792.5 images/s | 1,584,801 bytes |
| ResNet18 | 5 frozen-head + 5 fine-tuning epochs, batch 16, 421.4 s | 95.92% | 94.94% | 93.51% | 93.44% | 93.44% | 695.9 images/s | 44,808,267 bytes |

Both runs used the NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM, CUDA mixed precision, 224-pixel inputs and seed 42. Peak allocated VRAM: ScratchCNN 690,946,560 bytes; ResNet18 400,811,520 bytes. The throughput measurement is synchronized model-forward-only timing and excludes data loading.

ResNet18 per-class test accuracy indicators:

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Domestic Cat | 98.29% | 96.90% | 97.59% | 355 |
| Domestic Dog | 97.74% | 98.53% | 98.13% | 747 |
| Lion | 96.92% | 94.02% | 95.45% | 301 |
| Tiger | 97.82% | 97.21% | 97.52% | 323 |
| Leopard | 87.05% | 83.17% | 85.06% | 202 |
| Gray wolf | 91.09% | 91.09% | 91.09% | 202 |
| Cheetah | 83.63% | 91.75% | 87.50% | 206 |
| Jaguar | 95.52% | 94.81% | 95.17% | 135 |

These are measurements on this supplied split only, not a guarantee of performance on other sources or conditions. Softmax scores are uncalibrated. No OOD validation or false-acceptance rate is claimed.

## 5. Reproducing setup and workflows

Run commands from the repository root in PowerShell. Use `.venv`'s Python to avoid the previously observed global-Python `ModuleNotFoundError: No module named 'torch'`.

### 5.1 Environment and app

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The CUDA requirements use the official PyTorch CUDA 13.0 package index. For CPU-only environments, install `requirements-cpu.txt` instead of `requirements.txt`; do not mix both in one environment. Installing dependencies is not required for an already provisioned environment.

### 5.2 Prepare data

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --seed 42
```

To attach reviewed metadata from a CSV:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --seed 42 --metadata-csv data\manifests\source_metadata.csv
```

The optional CSV should identify files using `relative_path`. Do not fill missing source, licence, attribution, taxon, review or group fields with guesses.

### 5.3 Train and evaluate

```powershell
.\.venv\Scripts\python.exe scripts\train.py --model scratch_cnn --manifest data\manifests\dataset_manifest-20261105-current.csv --epochs 10 --seed 42 --device cuda --batch-size 32
.\.venv\Scripts\python.exe scripts\train.py --model resnet18 --manifest data\manifests\dataset_manifest-20261105-current.csv --epochs 5 --fine-tune-epochs 5 --seed 42 --device cuda --batch-size 16
```

For a reproducible test evaluation, point `scripts/evaluate.py` to a checkpoint, the same manifest, the raw data directory, and `--split test`. Do not use the test split for threshold selection.

### 5.4 Taxonomy import

Only import a reviewed source file and its actual release/licence/attribution:

```powershell
.\.venv\Scripts\python.exe scripts\import_taxonomy.py --csv data\taxonomy\reviewed-snapshot.csv --source-namespace "Catalogue of Life" --release "REVIEWED_RELEASE" --license "REVIEWED_LICENCE" --attribution "REVIEWED_ATTRIBUTION"
```

The placeholder strings above must be replaced with actual reviewed metadata. No snapshot is bundled.

The Taxonomy page can browse and confirm names only after an approved snapshot has been imported. It intentionally does not populate a fabricated catalogue or silently convert fuzzy suggestions into a selected taxon.

### 5.5 Local candidate demo

1. Start the app using the command in section 5.1.
2. Open **Classify**. If no production model is active, local demo mode is selected automatically; otherwise select **Local development/demo (candidate)**.
3. Select **ResNet18** or **ScratchCNN** from the registered candidate picker. The model must first be registered from the **Models** view if it is not already registered.
4. Upload a JPEG/PNG and press the visible **Classify image** button.
5. Treat outputs as experimental. The UI explicitly marks this mode and does not expose feedback submission there.

This does not change a candidate's lifecycle or create an active production model. Selecting **Production (active model)** requires a model that has passed the separate approved promotion workflow.

## 6. Verification performed

Latest full verification associated with the implementation and frontend audit:

- `.\.venv\Scripts\python.exe -m pytest -q` — **37 passed**.
- `.\.venv\Scripts\python.exe -m compileall -q src tests` — passed.
- `git diff --check` — passed.
- Streamlit `AppTest` exercised candidate selection, image upload, local inference, prediction/score/alternatives output, and experimental labeling; demo feedback controls were absent.
- An isolated Streamlit `AppTest` workflow imported a synthetic taxonomy fixture, searched and confirmed a name, registered a checkpoint through the Models page, selected ResNet18 and classified an uploaded image in demo mode, rendered experimental Grad-CAM, exercised the Models-page promotion form under a synthetic test policy, submitted consented feedback, approved label/rights/quality review, and exported an eligible curation manifest. The test verified the temporary active pointer and curation output; all fixture data was confined to a temporary project root and the live project registry was not promoted or modified.
- Registry unit test confirmed candidate listing leaves lifecycle state `candidate` and does not create an active-model pointer.
- The refreshed worktree app returned HTTP 200 on its Streamlit health endpoint; all four pages were navigated in the live browser without runtime exceptions. Empty live taxonomy/review state was displayed accurately.
- CUDA training and evaluation were run on the RTX 3050 for both current-manifest models.

The test suite includes dataset preparation/group leakage, corrupted/non-image handling, folder aliases and label mapping, manifest loading/path safety, model output/checkpoint loading, metrics/abstention curve behavior, taxonomy import/normalization/resolution and migrations, feedback consent/review/curation/withdrawal, inference and Grad-CAM, and candidate promotion/hash/path/rollback rules.

## 7. Repository organization

| Path | Role |
|---|---|
| `app.py` | Streamlit entry point and dependency guidance |
| `src/wildlife_ai/config/` | Class labels and project paths |
| `src/wildlife_ai/data/` | Image inspection, manifest splitting, transforms and loaders |
| `src/wildlife_ai/models/` | Architectures and restricted checkpoint loading |
| `src/wildlife_ai/training/` | Reproducible local training and run metadata |
| `src/wildlife_ai/evaluation/` | Test metrics and local visual reports |
| `src/wildlife_ai/taxonomy/` | CSV import, normalization, name resolution |
| `src/wildlife_ai/feedback/` | Feedback, consent, review and withdrawal services |
| `src/wildlife_ai/curation/` | Eligibility checks and curated manifest export |
| `src/wildlife_ai/registry/` | Candidate lifecycle, promotion, rejection and rollback |
| `src/wildlife_ai/explainability/` | Experimental ResNet18 Grad-CAM |
| `src/wildlife_ai/storage/` | SQLite schema and migrations |
| `scripts/` | Dataset, training, evaluation, taxonomy-import and benchmark CLIs |
| `tests/unit/` | Unit and workflow-level test coverage |
| `data/raw/` | Local source image folders; Git-ignored |
| `data/manifests/` | Generated manifests; Git-ignored |
| `models/runs/` | Checkpoints and training history; Git-ignored |
| `reports/metrics/` | Evaluation data and figures |
| `wildlife_ai.db` | Local registry/feedback/taxonomy state; Git-ignored |

## 8. Outstanding evidence and policy gates

These are genuine project limitations, not missing code that can safely be replaced by assumptions:

1. **Per-image provenance and rights:** Match every local image to a source record and preserve its actual licence/attribution. Resolve the unknown Kaggle Jaguar licence and the rights/terms of the cited wolf sources before claiming redistribution or public-use rights.
2. **Taxonomy:** Select and review a specific taxonomy source/release/licence, import its source data, and approve explicit mappings for model classes. The catalogue is not populated with fabricated taxa.
3. **Label review:** Independently inspect/verify the local source labels, including whether red/white wolf examples are appropriate Gray wolf labels.
4. **OOD evaluation:** Obtain distinct source-reviewed OOD development and final holdout data, document grouping/leakage controls, select the method/threshold from development data, and reserve final OOD data for final assessment only.
5. **Promotion criteria:** Agree on numerical acceptance/regression criteria before a future candidate evaluation. Do not select thresholds post hoc from already observed test results.
6. **Consent and review SOP:** Approve exact user-facing consent wording, retention period, rights/quality/label evidence requirements, disagreement handling and withdrawal/descendant-retirement process.
7. **Reviewer security:** The local reviewer identity is self-reported and unauthenticated. The app is for a trusted local workspace; it is not a secure multi-user review service.
8. **Near duplicates and groups:** Provide source/observation grouping metadata or approve a perceptual near-duplicate method to catch related images beyond identical file bytes.
9. **Operations and evidence:** Define backup/recovery and artifact retention practice; persistent demo screenshots have not been committed.

Until these gates are addressed, retain the academic-prototype label, use candidate inference only in explicit local demo mode, and do not claim production readiness, cleared data, calibrated confidence, OOD recognition, or taxonomy validity.
