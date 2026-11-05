# Wildlife & Animal Intelligence System

This is a local, Windows-compatible Python project for wildlife image classification and taxonomy-assisted feedback. `PROJECT_SPEC.md` is the authoritative specification, and `plan.md` defines the implementation milestones.

See [`reports/IMPLEMENTATION_GUIDE.md`](reports/IMPLEMENTATION_GUIDE.md) for the detailed implementation inventory, workflows, reproducibility instructions, verification and outstanding gates. [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) contains the architecture diagram, current dataset counts, refreshed training/test results, demonstration observations and concise project summary.

## Project status

This repository is finalized as an **academic research prototype**. Implementation and local model experiments are complete, but there is no production-approved active model. The explicit local development/demo mode can run a registered candidate; production inference remains unavailable until data rights/provenance, taxonomy, out-of-distribution evaluation, consent/review policy, and model-promotion gates are resolved. Dataset source pages and user-reported mappings are not treated as proof of the provenance or reuse rights of individual local images. Do not use the reported scores as evidence of production readiness.

## Environment setup

Create and activate a virtual environment only for a new environment. Do not recreate an existing environment to change the installation type.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### CUDA-enabled installation (recommended when supported)

The verified environment uses PyTorch 2.14.1+cu130 and Torchvision 0.29.1+cu130. `requirements.txt` pins these builds and uses the official PyTorch CUDA 13.0 package index:

```powershell
python -m pip install -r requirements.txt
```

The configured index is `https://download.pytorch.org/whl/cu130`.

### CPU-only fallback

For machines without a supported CUDA setup, install the CPU builds instead:

```powershell
python -m pip install -r requirements-cpu.txt
```

This uses the official PyTorch CPU package index. The CPU requirements are a separate alternative to `requirements.txt`; do not install both into the same environment.

## Verify PyTorch and GPU

Run these commands from the activated environment:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

When CUDA is available, verify a tensor operation on the GPU:

```powershell
python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x = torch.tensor([1.0, 2.0], device=device); print('Device:', device); print('Tensor result:', (x * 2).tolist())"
```

On a CPU-only installation, expect `CUDA available: False`, `PyTorch CUDA: None`, and `GPU: CPU`; the tensor operation should still succeed on CPU.

## Tests

```powershell
python -m pytest -q
```

## Dataset inspection and preparation

After selecting and reviewing licensed sources, place copies under the configured raw-data class folders:

```text
data/raw/
  Cat/       # Domestic Cat
  Dog/       # Domestic Dog
  Cheetah/
  Jaguar/
  Leopard/
  Lion/
  Tiger/
  Wolf/      # Gray wolf; leave absent until data is available
```

Folder names are matched case-insensitively. The eight target classes remain in the report; classes with no valid images are marked `awaiting_data`. Only JPEG and PNG files are inspected: other files are excluded, and unreadable or corrupt JPEG/PNG files are recorded as rejected.

Inspect the files and write a reproducible manifest with the default seed:

```powershell
python scripts\prepare_dataset.py --seed 42
```

Optional per-image provenance and observation groups can be supplied in a CSV with a `relative_path` column plus fields such as `source`, `source_image_id`, `license`, `attribution`, `collection_date`, `canonical_taxon_id`, `review_status`, `quality_notes`, and `group_id`:

```powershell
python scripts\prepare_dataset.py --seed 42 --metadata-csv data\manifests\source_metadata.csv
```

The script writes `data/manifests/dataset_manifest.csv` and prints class availability and valid/rejected/split counts, rejection reasons, and the manifest SHA-256. It records valid and rejected image files, content hashes, image dimensions, class labels, provenance fields, and splits. Missing source, licence, attribution, and canonical taxonomy metadata remain blank until reviewed; this generated manifest alone does not make images eligible for training. The script does not download, delete, or redistribute images. Near-duplicate image detection is not implemented yet; exact byte-identical duplicates and rows sharing a supplied `group_id` are kept together to prevent split leakage.

Import only a reviewed, versioned taxonomy CSV. The importer requires source metadata and the documented columns; it does not fetch or fabricate a snapshot:

```powershell
python scripts\import_taxonomy.py --csv data\taxonomy\reviewed-snapshot.csv --source-namespace "Catalogue of Life" --release "REVIEWED_RELEASE" --license "REVIEWED_LICENCE" --attribution "REVIEWED_ATTRIBUTION"
```

## Model training

Activate the project environment and use the generated manifest. The default automatic device selection uses CUDA when available and otherwise falls back to CPU. The defaults use two persistent data-loader workers, pinned memory and non-blocking CUDA copies, batch size 32 for the scratch CNN or 16 for ResNet18, 224-pixel inputs, and mixed precision on CUDA. These are starting settings, not claims of optimality.

```powershell
python scripts\train.py --model scratch_cnn --epochs 10
python scripts\train.py --model resnet18 --epochs 5 --fine-tune-epochs 5
```

Training writes best/last checkpoints and an epoch-history CSV under `models/runs/`. The ResNet18 run uses Torchvision's default ImageNet-1K pretrained weights, downloaded by Torchvision if they are not already cached. Checkpoints include the manifest hash, split seed, class order, preprocessing, configuration, library/device details, validation history, and weight provenance. Test images are not read by the training command.

To reproduce the refreshed 2026-11-05 experiment from the current generated manifest:

```powershell
python scripts\train.py --model scratch_cnn --manifest data\manifests\dataset_manifest-20261105-current.csv --epochs 10 --seed 42 --device cuda --batch-size 32
python scripts\train.py --model resnet18 --manifest data\manifests\dataset_manifest-20261105-current.csv --epochs 5 --fine-tune-epochs 5 --seed 42 --device cuda --batch-size 16
python scripts\evaluate.py --checkpoint models\runs\scratch_cnn-20261104T222824Z\best-scratch_cnn.pt --manifest data\manifests\dataset_manifest-20261105-current.csv --raw-dir data\raw --output-dir reports\metrics\refreshed-20261105 --split test --batch-size 32 --device cuda
python scripts\evaluate.py --checkpoint models\runs\resnet18-20261104T223628Z\best-resnet18.pt --manifest data\manifests\dataset_manifest-20261105-current.csv --raw-dir data\raw --output-dir reports\metrics\refreshed-20261105 --split test --batch-size 16 --device cuda
```

The measured current-manifest test results are ScratchCNN accuracy 68.35% / macro F1 67.72% and ResNet18 accuracy 94.94% / macro F1 93.44%, each on the same 2,471-image test split. Gray wolf has 202 test examples in this refreshed split. See `plan.md` for per-class metrics, measured inference throughput, checkpoint sizes, GPU training times, and blockers. The previous checkpoint/results used an older manifest and must not be mixed with these results.

Benchmark loader throughput and real model steps before choosing a batch size. The benchmark compares loader workers 0 and 2, then tests batch sizes 16, 32, and 64 (including CUDA transfer, forward/backward, mixed precision, peak VRAM, and system RAM):

```powershell
python scripts\benchmark_training.py --model scratch_cnn
```

Use the best stable measured throughput rather than maximizing batch size or GPU utilization. A 64-sample batch can fit while delivering lower throughput; reduce batch size or worker count if available system RAM or VRAM becomes constrained. `train.py` reports verified model/batch device placement, per-epoch sample throughput, and RAM/GPU memory telemetry.

## Local application

Start the Streamlit application from the project root using the project environment's interpreter. This avoids launching a globally installed Streamlit with a different Python that does not contain PyTorch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If PyTorch is not installed in `.venv`, install the project's CUDA-enabled requirements into that existing environment (do not recreate it), then launch with the same interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For CPU-only environments, use `requirements-cpu.txt` instead. Do not mix the CUDA and CPU requirements in one environment.

The app provides **Classify**, **Taxonomy**, **Review**, and **Models** views. The Classify view separates **Production (active model)** from **Local development/demo (candidate)**. Demo mode requires a registered candidate, verifies its project-contained path and artifact hash, and runs it locally without changing registry state or the production active-model pointer. It is visibly marked experimental and disables feedback submission to avoid mixing candidate-demo output with reviewed feedback. An active model is still required for production inference. A reviewed taxonomy snapshot must be imported before name searches can return candidates; the current source tree does not include that external snapshot.

The Models view validates and registers a local best checkpoint as a candidate. It can attach a matching `*-test-metrics.json` report when present. Promotion requires a separately approved policy with explicit test and regression limits; no numeric policy is preconfigured. The app does not set an active model automatically. Reviewer references are self-declared and do not provide authentication.

Feedback can be submitted without retaining the image. Retention and future training use are separate consents. Retained images are sanitized, stored outside SQLite under `data/feedback/images/`, and use generated filenames. Reviewers must separately verify the label, image quality, and source/licence before the item is eligible for a curation export. Withdrawal removes the retained copy and invalidates its curated training consent; training checks the consent state again before using a registered curated manifest.

Curated manifests are versioned full snapshots of the frozen supported manifest plus reviewed additions. Original split assignments are preserved; feedback additions are assigned to train or validation only and never to the frozen test split. The existing loader accepts the generated manifest paths. A single added item cannot form both train and validation sets, so a useful training run needs enough reviewed, eligible feedback. Exact duplicates and frozen-test overlap are rejected; perceptual near-duplicate detection is not implemented. To explicitly train from a curated snapshot:

```powershell
python scripts\train.py --model resnet18 --manifest data\manifests\curated-feedback-<id>.csv
```

Training rechecks the consent registry; if the database is missing, the curated manifest is unregistered, or consent has been withdrawn, training is refused.

The ResNet18 view offers an experimental Grad-CAM visualization for its processed input. It is not a causal explanation, does not verify the predicted class, and is not supported for the scratch CNN. The interface does not claim OOD recognition: no approved abstention threshold or separately sourced OOD development/final set is available.

## Verified development environment

The Phase 1 environment check reported Python 3.14.8, PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, CUDA 13.0, and an NVIDIA GeForce RTX 3050 6GB Laptop GPU. These are environment observations, not a guarantee that every supported machine has CUDA.
