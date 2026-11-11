# Wildlife & Animal Intelligence System

A locally runnable academic demonstration that classifies an uploaded animal image, collects a user's confirmation or correction, and automatically trains an evaluated model candidate from accumulated consented feedback.

## Run locally

Use the existing project environment and start Streamlit from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If dependencies still need to be installed in that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The application needs the local ResNet18 checkpoint, `data/manifests/dataset_manifest.csv`, and its referenced dataset files. If application code and data live in different directories, point the app to the directory containing `data/`, `models/`, and `wildlife_ai.db`:

```powershell
$env:WILDLIFE_AI_ROOT = 'D:\path\to\Wildlife-Intelligence-System'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app initializes or migrates its local SQLite database on startup. It does not need a production-active model or network service.

## Use the application

The single page supports JPG, JPEG, and PNG uploads up to 10 MiB. It previews the image and predicts one of eight configured classes: Domestic Cat, Domestic Dog, Cheetah, Jaguar, Leopard, Lion, Tiger, or Gray Wolf. The result includes alternative matches, the selected model version, and an explicitly **uncalibrated** score.

Choose **Correct** to confirm the predicted class, or **Incorrect** to select the confirmed class. **Other / none of these** records an unsupported response without creating a training label. Image-retention consent and future-training consent are separate. Without retention consent, only feedback metadata and the original upload hash are saved. Without training consent, a retained image is not eligible for training.

Feedback is stored locally. The app has no user accounts or reviewer workflow; treat it as a local operator/teacher demonstration, not a public service. Confirmed labels are user-supplied and are not independently verified. A sanitized image is kept only with retention consent.

## Automatic learning safeguards

After **five new, unique, eligible examples** have both consents and a supported confirmed label, the app automatically:

1. Revalidates consent, image integrity, class label, duplicate hashes, and the frozen dataset files.
2. Writes a new manifest that leaves the frozen train/validation/test rows unchanged and adds feedback images to **train only**.
3. Warm-starts ResNet18 from the currently selected checkpoint and balances feedback examples with replay from the original training data.
4. Evaluates the candidate on the unchanged validation split. The frozen test split is never used for training or repeated promotion decisions.
5. Selects the candidate only if validation accuracy and macro-F1 each remain within 1 percentage point of the current model, and every class's validation recall remains within 2 percentage points.

An accepted update changes the app's selected-model pointer only after evaluation passes. A rejected or failed update leaves the previous selected checkpoint intact; generated checkpoints and batch outcomes remain available locally. Retraining is triggered by eligible feedback, not by ordinary app reruns or unsupported/no-consent submissions.

The base dataset is `data/manifests/dataset_manifest.csv`. The alternate `dataset_manifest-20261105-current.csv` has unreconciled missing paths and is not used. Original dataset files and prior checkpoints are not overwritten or deleted. Training runs and automatic manifests are written to new, unique local paths.

## Selected model

The default model is the verified ResNet18 checkpoint `resnet18-20261105T013137Z` (`models/runs/resnet18-20261105T013130Z/best-resnet18.pt`). The checkpoint and recorded validation report were matched by model version, artifact hash, manifest hash, class order, and evaluation sample count; the validation report was reproduced against the frozen manifest.

| Evaluation split | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 2,462 | 96.39% | 95.64% |
| Frozen test | 2,460 | 94.88% | 93.75% |

These are results on the supplied frozen dataset only, not a guarantee of performance on new sources. Scores are not calibrated probabilities; the model has no out-of-distribution detector.

## Tests

Run the existing suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Further implementation and safety details are in [`reports/IMPLEMENTATION_GUIDE.md`](reports/IMPLEMENTATION_GUIDE.md).
