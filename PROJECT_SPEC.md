# Wildlife & Animal Intelligence System

**Document type:** Product and technical specification  
**Status:** Draft for architecture review  
**Intended use:** Authoritative project brief for the development team and GitHub Copilot  
**Initial implementation target:** Local, Windows-compatible college minor project  
**Primary language:** Python

---

## 0. Instructions for GitHub Copilot and contributors

Treat this document as the authoritative description of the intended system. Read the relevant sections before proposing or implementing a feature.

### Development rules

1. Do not implement the entire project in one operation. Work in small, reviewable milestones.
2. Before implementation, inspect the existing repository and report which requirements are already implemented, missing, or contradictory.
3. Do not silently remove, weaken, or reinterpret a requirement. Explain any proposed change and its trade-offs first.
4. Do not invent training results, accuracy values, dataset sizes, citations, or completed features. Report measured results only.
5. Do not use external AI inference APIs or hosted animal-recognition services. Image inference and model training must run locally.
6. Pretrained model weights may be used for transfer learning, provided the project documents their source and does its own training/fine-tuning and evaluation.
7. Keep taxonomy data, model-supported classes, user feedback, and curated training data separate. They represent different things and must not be conflated.
8. Never add unverified user-submitted labels to the training dataset automatically.
9. Never retrain or replace the active model automatically after one user submission.
10. Keep the code modular, typed where useful, documented, testable, and compatible with Windows. Use configuration rather than scattering constants across modules.
11. Do not add dependencies without explaining why they are needed. Pin or constrain versions after confirming compatibility in the actual environment.
12. Add or update tests with each meaningful implementation. Run the relevant tests and report the real output.
13. Avoid destructive actions. Do not delete datasets, checkpoints, user feedback, or existing project files without explicit approval.
14. When uncertain about a taxonomy match, ask for confirmation or mark it unresolved. Do not silently guess.
15. Do not represent a model's estimated class score as a calibrated probability unless calibration has been measured and documented.

### Required response format for each development milestone

Before editing files, state:
- What the milestone will implement.
- Which requirements it addresses.
- Which files it expects to create or modify.
- Any assumptions or decisions that need approval.

After editing files, state:
- What changed.
- Commands to run.
- Tests executed and their actual results.
- Known limitations or unfinished work.
- The next logical milestone.

If the repository's current state conflicts with this specification, stop and explain the conflict before making broad changes.

---

# 1. Project overview

## 1.1 Project name

**Wildlife & Animal Intelligence System**

## 1.2 Project concept

Build a local machine-learning application that analyses an uploaded animal photograph, predicts one of the animal categories supported by its currently deployed image-classification model, communicates uncertainty, and lets a user submit corrections.

The application also maintains a separate animal taxonomy reference catalogue. This catalogue contains animal names and taxonomic information even when the image model has never been trained to recognise those animals.

User feedback is stored separately, reviewed, and may later be converted into curated training examples. In a controlled training cycle, the team can expand the model's supported categories and evaluate a new model version before deployment.

The intended system therefore combines three distinct capabilities:

1. **Visual classification:** Learn visual patterns from labelled images and predict supported animal categories.
2. **Taxonomic name resolution:** Resolve common names, scientific names, synonyms, and likely spelling errors to a canonical taxonomy record.
3. **Human-guided dataset improvement:** Collect, review, and curate feedback to improve future training datasets and model versions.

## 1.3 Problem statement

Basic image classifiers often force every input into one of their known output classes. If a classifier has been trained on domestic cats and dogs, lions, tigers, leopards, gray wolves, cheetahs, and jaguars, an uploaded bear image may still be labelled as one of those eight classes.

This project aims to make that limitation visible and provide a controlled feedback process. The system should be able to abstain when an image is not confidently recognised, let a user identify an unsupported animal through a separate taxonomy catalogue, and preserve the correction for later review.

The system must not claim that a name lookup teaches the neural network what the animal looks like. The taxonomy database knows names and relationships; the ML model learns visual patterns from images.

## 1.4 Primary objectives

- Build and evaluate an image-classification model using a prepared animal-image dataset.
- Compare a CNN trained from scratch with a transfer-learning model.
- Provide an interface for image upload and local inference.
- Display the predicted category, a meaningful score, and alternative predictions.
- Provide a cautious unknown/unsupported-input path rather than always forcing a known class.
- Maintain a taxonomy catalogue independent of model classes.
- Resolve user-entered animal names such as `hipo`, `hippo`, and `Hippopotamus` to a canonical taxonomy record, with confirmation when appropriate.
- Collect user confirmations and corrections without automatically trusting them.
- Introduce a human verification and data-curation workflow.
- Support controlled dataset expansion, retraining, model comparison, versioning, and rollback.
- Produce reproducible experiments and a documented evaluation suitable for a college minor-project submission.

## 1.5 Non-objectives for the initial release

The initial release is not intended to:

- Identify every animal species in the world.
- Reliably infer an animal's health, behaviour, mood, danger, or intent.
- Perform wildlife population estimation or conservation monitoring.
- Track animals across video frames.
- Detect and classify multiple animals in one image.
- Identify individual animals or recognise a particular pet.
- Provide veterinary or ecological advice.
- Automatically accept every user correction as ground truth.
- Automatically learn a new species from a single uploaded photograph.
- Continuously retrain itself in production.
- Depend on paid APIs or cloud inference services.

These may be considered future extensions, but they are not part of the minimum viable product (MVP).

---

# 2. Users and intended operating environment

## 2.1 Intended users

- The student development team.
- A project evaluator or administrator who reviews submitted corrections.
- A person testing the application by uploading an animal image and optionally submitting feedback.

## 2.2 Initial deployment model

The MVP should run locally on a Windows development machine. Model training and image inference should use the local machine's CPU or supported NVIDIA CUDA GPU.

The first version may use a local SQLite database and local image folders. It does not require public multi-user hosting, account registration, or a remote backend.

Design modules so that a future web deployment is possible, but do not add distributed-system complexity to the MVP without a concrete need.

## 2.3 Hardware target

The primary development machine has an NVIDIA laptop GPU with 6 GB of VRAM. The project must therefore:

- Make batch size configurable.
- Support CPU fallback.
- Avoid loading the full dataset into GPU memory.
- Use data loaders and mini-batches.
- Permit mixed-precision training where supported and tested.
- Save checkpoints safely and often enough to recover from interrupted training.
- Allow image size and worker count to be configured.

Do not assume CUDA is installed correctly merely because an NVIDIA GPU exists. The setup process must test `torch.cuda.is_available()` and report the detected device.

---

# 3. Scope and classification categories

## 3.1 Initial model classes

The first image classifier should support these eight user-facing categories:

1. Domestic Cat
2. Domestic Dog
3. Lion
4. Tiger
5. Leopard
6. Gray wolf
7. Cheetah
8. Jaguar

The label-to-taxonomy mapping must be stored in configuration or model metadata, not duplicated as hard-coded assumptions throughout the code.

Candidate scientific-name mappings are:

| Display label | Candidate scientific taxon | Notes |
|---|---|---|
| Domestic Cat | *Felis catus* | A domestic species; breed-level prediction is out of scope. |
| Domestic Dog | *Canis lupus familiaris* or the accepted equivalent in the chosen taxonomy source | Taxonomic backbones may represent domestic dogs differently. Use the selected backbone's accepted record. |
| Lion | *Panthera leo* | Species-level category. |
| Tiger | *Panthera tigris* | Species-level category. |
| Leopard | *Panthera pardus* | Species-level category. |
| Gray wolf | *Canis lupus* | Use the intended gray-wolf record from the selected taxonomy backbone. |
| Cheetah | *Acinonyx jubatus* | Species-level category. |
| Jaguar | *Panthera onca* | Species-level category. |

Do not invent or hard-code external taxonomy identifiers. Resolve the canonical identifiers from the chosen, versioned taxonomy source and save the source namespace and identifier together.

## 3.2 Category boundaries

- Cat and dog images may include different breeds, ages, coat colours, poses, and backgrounds, but the output remains the broad domestic-cat or domestic-dog category.
- The wildlife classes should represent the intended species, not merely visually similar animals.
- Images with multiple animals, no animal, severe obstruction, or an animal outside the eight supported classes should be treated as difficult or unsupported inputs, not silently assigned a reliable label.
- The MVP is a single-primary-animal image classifier, not an object detector.

## 3.3 Taxonomy entries are not model classes

The taxonomy database must contain more animals than the image model supports.

Examples of possible taxonomy entries that may exist without being model classes:

- Hippopotamus
- Brown bear
- Polar bear
- Giraffe
- Rhinoceros
- Zebra
- Elephant
- Fox
- Deer
- Crocodile

These are illustrative entries. The actual catalogue comes from the imported taxonomy source and curated additions.

A taxonomy entry existing in the database does **not** mean the model can recognise it. The interface must distinguish:

- **Known in taxonomy:** a corresponding animal record exists.
- **Has curated images:** reviewed images are available for that taxon.
- **Supported by model:** the active model has an output class mapped to that taxon.

Do not turn `Other`, `Unknown`, or `Unsupported` into a normal animal training class. These are workflow outcomes, not coherent biological categories.

---

# 4. User-facing application workflow

## 4.1 Main workflow

1. User opens the application.
2. User uploads an image in an accepted format.
3. The application validates and previews the image.
4. The active model runs local inference.
5. The application displays either a supported prediction or an uncertainty/unsupported result.
6. The application offers `Correct` and `Incorrect` feedback.
7. For `Correct`, the predicted class is recorded as the user's confirmed label.
8. For `Incorrect`, the user selects the actual animal from the model's eight supported classes.
9. The correction is stored with a pending verification status. Image-retention consent and future-training consent are requested separately; training consent requires retention consent.
10. A reviewer can inspect the feedback and either verify, reject, or request more information.
11. A verified, eligible correction can be included in a curated training dataset. It is not automatically used for training.

## 4.2 Prediction screen

The prediction screen should include:

- Uploaded image preview.
- Predicted display label.
- A clearly named model score or estimated confidence, with explanation that it is not certainty.
- Top three model outputs where meaningful.
- Model version identifier.
- Feedback controls: `Correct` and `Incorrect`. An incorrect response exposes a dropdown with exactly `Domestic Cat`, `Domestic Dog`, `Cheetah`, `Jaguar`, `Leopard`, `Lion`, `Tiger`, and `Gray Wolf`.
- A link or button to view Grad-CAM if supported by the selected model.

Do not show a fabricated percentage. If a score is displayed as a percentage, label it as a model score unless calibration has been tested. A softmax score is not automatically a calibrated probability.

## 4.3 Correct-prediction path

When the user selects `Correct`:

- Store the response as user feedback, including the model version and prediction details.
- Store the predicted class as the directly corrected/confirmed class.
- Do not automatically treat the response as verified scientific ground truth.
- Do not automatically add the image to the training dataset.
- If retaining the image for review or future use, require the relevant consent and record it.

## 4.4 Incorrect-prediction path

When the user selects `Incorrect`:

- Show a dropdown containing exactly the model's eight supported display classes: `Domestic Cat`, `Domestic Dog`, `Cheetah`, `Jaguar`, `Leopard`, `Lion`, `Tiger`, and `Gray Wolf`.
- Store the selected class directly as the corrected label, separately from the original prediction.
- Do not require an external taxonomy database, taxonomy identifier, or scientific-name lookup for this correction.
- Keep any taxonomy search as a separate reference workflow; it does not determine the classifier's corrected class.

The display option `Gray Wolf` maps to the existing internal class label `Gray wolf` so current checkpoints and manifests remain compatible.

## 4.5 Unknown or unsupported prediction path

When the model abstains or the input is considered unsupported:

- State that the model could not confidently identify the animal among its supported categories.
- Do not claim that the system has recognised a new species.
- Offer taxonomy search so the user can identify the animal by name.
- Make it clear that identifying the name does not add visual recognition capability to the model.

## 4.6 Image retention and consent

Inference should work without storing the uploaded image permanently.

If an image is to be retained with feedback for review or possible dataset use:

- Explain the purpose before submission.
- Obtain explicit consent.
- Record the consent state and timestamp.
- Provide a way for the project administrator to remove a retained image and its associated feedback record where feasible.
- Avoid collecting unnecessary personal information or image metadata.

For the local MVP, do not build an elaborate account system. Use a local session identifier or reviewer-entered contributor reference only if needed. Do not claim that one browser session represents an independently verified human.

---

# 5. Animal taxonomy reference system

## 5.1 Purpose

The taxonomy system resolves animal names and provides taxonomic context independently of the ML model. It is a reference catalogue, not a visual classifier.

## 5.2 Data source strategy

Use an established taxonomy source such as Catalogue of Life as the initial candidate source. Before importing data, inspect:

- Available download format and subset options.
- Data licence and attribution requirements.
- Release/version information.
- Whether common names and synonyms are included in the selected dataset.
- Taxonomic coverage and known limitations.

Do not assume any taxonomy source is perfectly complete or permanently current. Store source name, release/version, import date, and licence/attribution details.

For the MVP, prefer a downloaded offline snapshot or selected subset so that name resolution works locally. Do not require an internet connection for every name lookup.

Potential reference: <https://www.catalogueoflife.org/data/download>

The taxonomy-provider layer should be replaceable. Keep source-specific import logic separate from the application's internal taxonomy representation.

## 5.3 Internal taxonomy record

Each record should support fields equivalent to:

- Internal primary key.
- Source namespace/provider.
- Source taxon identifier.
- Accepted scientific name.
- Accepted common name, where available.
- Other common names.
- Synonyms.
- Taxonomic rank.
- Parent taxon identifier, where available.
- Taxonomic status (accepted, synonym, uncertain, or source-specific equivalent).
- Source release/version.
- Attribution/licence metadata.

Use a compound identity such as `(source_namespace, source_taxon_id)`. Do not assume a numeric identifier from one provider is globally unique across all providers.

## 5.4 Name and alias handling

The resolver should support:

1. Exact accepted common-name match.
2. Exact scientific-name match.
3. Exact synonym match.
4. Exact curated-alias match.
5. Normalised matching that ignores case, repeated whitespace, and harmless punctuation differences.
6. Fuzzy matching for likely misspellings.

Normalisation should be deterministic. It may include Unicode normalisation, case folding, trimming, punctuation handling, and whitespace collapsing. Preserve the original user input separately for audit purposes.

Use a local fuzzy-matching library such as RapidFuzz if appropriate. Do not use an external language-model API for name resolution.

## 5.5 Example: resolving `hipo`

Expected user journey:

- User enters `hipo`.
- The resolver normalises the string and searches exact names and aliases.
- No exact result may exist.
- Fuzzy matching ranks likely candidates.
- The interface proposes `Hippopotamus` with its scientific name and taxonomic rank.
- The user confirms the intended record.
- The feedback stores the original input `hipo` and the canonical taxonomy record identifier.

The resolver must not blindly convert every similar spelling into Hippopotamus. For ambiguous or low-scoring matches, display several candidates or ask the user to refine the search.

A typo such as `hipo` must not automatically become an official synonym in the taxonomy database. It may be recorded as a user-entered query or proposed alias. Only a reviewed alias should be added to the curated alias table.

## 5.6 Ambiguous common names

Some common names refer to multiple taxa, broad groups, or different animals in different regions. The resolver must show enough context to distinguish candidates, such as:

- Accepted common name.
- Scientific name.
- Taxonomic rank.
- Parent group.

Example: `bear` may refer to bears generally or to a particular bear species. Do not silently choose a particular species when the user intended a broad group.

## 5.7 Unresolved names

If no suitable record is found:

- Keep the original string.
- Store it as an unresolved name suggestion or feedback item.
- Mark it for manual review.
- Do not automatically create an accepted taxon.
- Do not use the unresolved string as a model class.

## 5.8 Taxonomy update process

A taxonomy snapshot update must be deliberate and versioned. The application must be able to report which taxonomy source and snapshot were used to resolve a name. An update must not silently alter the identity of records already stored in feedback or model metadata.

---

# 6. Dataset acquisition and management

## 6.1 Dataset sources

Candidate sources include:

- Oxford-IIIT Pet Dataset for domestic cats and dogs.
- iNaturalist datasets or other appropriately licensed wildlife image datasets for selected wild-animal categories.
- Additional openly licensed sources where required to fill gaps.

Potential references:

- <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- <https://research.google/pubs/the-inaturalist-species-classification-and-detection-dataset/>

These are candidate sources, not blanket permission to redistribute every image. Review each source's licence, terms, attribution rules, and permitted use before downloading or including images in a submission or public repository.

Do not scrape random image search results without checking provenance and usage rights.

## 6.2 Initial data target

For the first experiments, target approximately 500–1,000 usable images per class if a suitable licensed dataset is available. This is a planning target, not a guarantee of performance and not a hard requirement if the dataset is smaller.

The team must report the actual count per class and explain any imbalance. A smaller clean, diverse dataset is preferable to a larger dataset with incorrect labels, duplicates, or unclear rights.

## 6.3 Raw and processed data separation

Keep downloaded originals separate from processed splits.

Suggested layout:

```text
data/
  raw/
    cat/
    dog/
    lion/
    tiger/
    leopard/
    wolf/
    cheetah/
    jaguar/
  processed/
    train/
    validation/
    test/
  manifests/
  feedback_quarantine/
```

Do not commit image datasets to Git. Store acquisition notes, manifests, scripts, and small synthetic test fixtures in the repository as appropriate.

## 6.4 Dataset manifest and provenance

For every usable image, maintain a manifest with fields equivalent to:

- Image identifier.
- Relative file path.
- Label and canonical taxon identifier.
- Original source/dataset.
- Original source image identifier or URL where permitted.
- Licence/attribution reference.
- Collection/import date.
- Dataset split.
- Duplicate group or hash.
- Review status.
- Notes on quality or uncertain labels.

Do not store a source URL if the source's terms prohibit it; document the permitted provenance instead.

## 6.5 Data validation

The preparation pipeline should:

- Validate supported image formats.
- Detect unreadable or corrupted files.
- Detect exact duplicates using hashes.
- Where feasible, detect likely near-duplicates using perceptual hashes or image embeddings.
- Verify that each image has one valid label for the current single-label task.
- Flag extreme blur, severe corruption, and unusable image dimensions.
- Report counts before and after filtering.
- Preserve a record of rejected files and reasons rather than silently deleting source data.

## 6.6 Train, validation, and test splits

Initial target split:

- Training: 70%.
- Validation: 15%.
- Test: 15%.

Use a fixed, documented random seed and stratify by class where possible. Prevent identical or near-identical images from crossing splits. If images originate from the same video, observation, photographer sequence, or burst, group them where possible to reduce leakage.

The test set must remain untouched during model selection and tuning. Feedback images that later enter training must never be allowed to leak into the frozen test set.

For future model versions, maintain a stable evaluation set or a carefully versioned evaluation protocol. Document any change to the test set and why it was necessary.

## 6.7 Image preprocessing

Initial baseline configuration:

- Resize/crop to 224 × 224 pixels unless an experiment justifies a different size.
- Convert to RGB.
- Convert to tensors.
- Apply the normalisation expected by the selected model's pretrained weights for transfer learning.
- Apply augmentation to training images only.

Possible training augmentations:

- Random horizontal flip where biologically and semantically safe.
- Small rotations.
- Random resized crop with controlled scale.
- Moderate brightness/contrast changes.

Do not apply validation or test augmentations that alter evaluation examples unpredictably. Use deterministic preprocessing for validation and testing.

---

# 7. Machine-learning architecture

## 7.1 Framework

Use Python with PyTorch and Torchvision. Keep training, inference, and evaluation code separate from the Streamlit UI.

## 7.2 Baseline model: CNN from scratch

Implement a compact convolutional neural network trained from random initialisation on the prepared eight-class dataset.

The model should include a reasonable sequence of convolution, activation, pooling, and classification layers. Dropout or other regularisation may be used where justified.

The goal is not to maximise performance at any cost. It provides a baseline for comparison with transfer learning and demonstrates the training process directly.

Record the architecture and hyperparameters used.

## 7.3 Transfer-learning model: ResNet18

Implement ResNet18 using Torchvision pretrained ImageNet weights, subject to the weights' licence/terms and documented provenance.

Initial strategy:

1. Load the specified pretrained weights.
2. Replace the final classification layer with an eight-output layer.
3. Freeze the backbone initially and train the new classifier.
4. Optionally unfreeze selected or all backbone layers for fine-tuning.
5. Use validation performance to guide model selection.
6. Save the best validation checkpoint and its full metadata.

The project must perform its own training/fine-tuning and evaluation. Simply calling an external ready-made classifier is not acceptable.

## 7.4 Model comparison

Evaluate the scratch CNN and ResNet18 on the same untouched test set and report:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Per-class precision, recall, and F1-score.
- Confusion matrix.
- Model file size.
- Inference time under a documented measurement setup.
- Training time and hardware configuration.

Do not use test-set results to select hyperparameters. Use the validation set for development decisions.

## 7.5 Training configuration

Hyperparameters must be configurable, including:

- Model architecture.
- Input dimensions.
- Batch size.
- Number of epochs.
- Learning rate.
- Optimiser.
- Weight decay.
- Scheduler.
- Number of data-loader workers.
- Random seed.
- Device selection.
- Mixed-precision setting.
- Checkpoint directory.

Use sensible defaults but do not claim they are optimal before experiments.

## 7.6 Reproducibility

Record enough information to reproduce a training run:

- Dataset manifest version/hash.
- Dataset split seed.
- Model architecture.
- Pretrained-weight identifier, if applicable.
- Training configuration.
- Python and key library versions.
- Device and CUDA details where relevant.
- Git commit hash where available.
- Start/end time and training history.

Perfect determinism may not be available on every GPU operation. Document limitations rather than promising identical results across all machines.

## 7.7 Checkpoints

A checkpoint must contain or reference:

- Model state dictionary.
- Architecture identifier.
- Class names and class-to-index mapping.
- Canonical taxonomy mapping for each output class.
- Input size and preprocessing configuration.
- Normalisation parameters.
- Model version.
- Training configuration.
- Dataset manifest identifier/hash.
- Validation metrics.
- Pretrained-weight provenance, if applicable.

Do not load arbitrary untrusted checkpoint files. Use trusted local files and follow PyTorch's safe-loading guidance for the installed version.

---

# 8. Uncertainty and unknown/unsupported inputs

## 8.1 Problem

A standard closed-set classifier returns one of its known classes for almost any input. It may therefore label a bear as a dog or a hippopotamus as a pig-like known category even though the animal is outside the model's supported categories.

## 8.2 MVP abstention behaviour

Implement an initial rejection/abstention mechanism using validation data. A simple confidence threshold may be used as a baseline, but its limitations must be measured and explained.

The system should be able to return a status such as:

- `supported_prediction` — the input passes the current acceptance rule.
- `uncertain_prediction` — the model's output is too uncertain under the configured rule.
- `unsupported_or_out_of_distribution` — evidence suggests the input may not belong to supported classes.
- `invalid_input` — the file cannot be processed or fails image validation.

Do not claim that a confidence threshold alone provides reliable open-set recognition. Softmax scores can be high for unfamiliar inputs.

## 8.3 Evaluation of abstention

Create a separate set of out-of-scope examples, such as bear, hippopotamus, giraffe, elephant, and non-animal images, where licences permit.

Measure both:

- The model's ability to correctly classify supported test images.
- Its ability to abstain on unsupported inputs.

Report false acceptance and false rejection behaviour. Do not tune the rejection method on the final evaluation set.

## 8.4 Future experiments

After the baseline, investigate methods such as:

- Entropy-based uncertainty.
- Energy-based scores.
- Embedding-distance or nearest-neighbour approaches.
- Calibration methods.
- Dedicated open-set recognition methods.

Only implement more advanced approaches when their assumptions, evaluation requirements, and additional complexity are understood.

---

# 9. Human-in-the-loop feedback system

## 9.1 Purpose

The feedback system collects possible corrections from users. It does not make every user submission trustworthy by default.

## 9.2 Feedback types

Store at least:

- User confirmed the displayed prediction (`Correct`).
- User rejected the displayed prediction and selected one of the model's eight classes (`Incorrect`).

Keep positive confirmations separate from verified labels. A user selecting `Correct` is useful feedback, but it is not automatically an expert-verified ground-truth annotation.

## 9.3 Feedback record

A feedback record should contain fields equivalent to:

- Feedback identifier.
- Image reference, if retention consent was given.
- Original prediction label.
- Corrected class label selected directly from the model's eight supported classes (equal to the predicted class for a positive confirmation).
- Original model score and score type.
- Model version used for prediction.
- User response (`yes`/`no` or equivalent).
- Verification status.
- Submission timestamp.
- Consent state and timestamp.
- Reviewer decision, reviewer reference, and review timestamp where applicable.
- Duplicate or related-feedback reference where detected.

Store the image separately from its database metadata. Do not store raw image bytes in ordinary feedback rows unless a clear technical reason exists.

## 9.4 Verification states

Use these states:

| State | Meaning | Eligible for training? |
|---|---|---|
| `pending` | Submitted but not reviewed | No |
| `corroborated` | Multiple independent submissions agree, but no final review has occurred | No, not by itself |
| `verified` | Accepted by an authorised reviewer using the project's review procedure | Eligible after data-quality checks |
| `rejected` | Label or image was rejected | No |
| `needs_review` | Ambiguous, incomplete, or requires more evidence | No |

Corroboration is supporting evidence, not proof. Repeated feedback may come from duplicated images, copied answers, or users repeating the same mistake.

## 9.5 Review procedure

The local MVP should include a simple administrator/reviewer workflow to:

- View the image, if it was retained with consent.
- View the original model output and model version.
- View the user's directly selected corrected class.
- Inspect source or supporting notes where available.
- Accept the correction, reject it, or mark it as needing further review.
- Record reviewer action and timestamp.

For a student project, the team may act as reviewers. The application must still preserve the distinction between user-submitted and reviewer-verified labels.

## 9.6 Data credibility

Do not invent a numerical credibility score without a defensible methodology. Use explicit states and provenance fields first. If a future data-quality score is introduced, document its inputs and ensure it is not confused with the model's prediction confidence.

## 9.7 Anti-duplication and abuse controls

- Detect exact duplicate images.
- Where feasible, flag near-duplicates.
- Do not count repeated submissions of the same image as independent confirmations.
- Apply basic input validation and submission limits if the application becomes multi-user.
- Preserve a review trail for status changes.

---

# 10. Curated dataset expansion and retraining

## 10.1 Separation of stages

The following stages must remain separate:

1. User submission.
2. Feedback storage.
3. Human verification.
4. Image and label quality checks.
5. Curated dataset inclusion.
6. Training/validation assignment.
7. Model training.
8. Model evaluation.
9. Model promotion or rejection.

A record being `verified` makes it eligible for curation; it does not bypass duplicate detection, image quality checks, licensing/consent checks, or split rules.

## 10.2 Adding an unsupported species

Example: users upload bear images while the model only supports the original eight categories.

Expected process:

1. The model abstains or gives an uncertain result.
2. The user selects the appropriate bear taxon through taxonomy search.
3. The correction is stored as pending.
4. A reviewer checks the label and image.
5. Accepted images enter the curated feedback pool.
6. The team gathers enough diverse, correctly labelled examples for a new category.
7. The class is added to a proposed training configuration and class-to-taxon mapping.
8. A new model is trained and evaluated.
9. The model is promoted only if it meets the project's evaluation criteria.

One image or one user correction is never enough to establish reliable support for a new species.

## 10.3 Training data eligibility

Before a supported-class feedback image can enter a training run, verify:

- It has explicit retention/training consent where required.
- Its directly selected corrected class is one of the model's eight supported classes; taxonomy lookup is not required.
- Its label has been verified.
- It is not corrupted or unusable.
- It is not an exact or near duplicate of another selected example, unless deliberately retained for a documented reason.
- Its source and use rights are documented.
- It does not violate the frozen evaluation split.

## 10.4 Model retraining policy

- Retraining is initiated explicitly by a developer or administrator.
- It runs as a controlled experiment, not automatically after every submission.
- Every run records its configuration and dataset manifest.
- The previous model remains available.
- The new model is evaluated on the agreed evaluation protocol.
- The new model is not promoted solely because its training accuracy improved.
- The system must support rollback to the previous model version.

## 10.5 New-category promotion criteria

The team must define category-specific minimum data and performance criteria before expanding the model. Do not hard-code an arbitrary universal number of images as proof that a class is ready.

Consider:

- Number of verified examples.
- Diversity of individuals, poses, lighting, backgrounds, and image sources.
- Label quality and taxonomic consistency.
- Class balance.
- Per-class validation and test performance.
- Confusion with visually similar classes.
- Out-of-distribution performance.

If evidence is insufficient, keep the animal in the taxonomy catalogue but mark it as unsupported by the active model.

---

# 11. Explainable AI

## 11.1 Grad-CAM

Implement Grad-CAM for the selected CNN architecture where the architecture and target layer support it. For ResNet18, select and document an appropriate final convolutional layer.

The application should allow the user to view:

- Original image.
- Heatmap.
- Heatmap overlay.

## 11.2 Explanation limitations

A Grad-CAM heatmap is a visual diagnostic of model activation/gradient influence. It is not proof that the model reasoned correctly, nor a causal explanation of its decision.

Do not use the heatmap as a substitute for proper evaluation. Document examples where the model attends to background, watermarks, cages, or other spurious features.

---

# 12. Application interface and inference

## 12.1 UI framework

Use Streamlit for the MVP unless the team documents a strong reason to choose another framework.

## 12.2 Main pages or views

Suggested MVP views:

1. **Classify image:** upload, preview, predict, show result, and submit feedback.
2. **Explore taxonomy:** search animal names and inspect scientific/common names and taxonomic rank.
3. **Feedback review:** local administrator view for pending submissions.
4. **Model information:** active model version, supported categories, and evaluation summary.

Keep the user interface clear and usable. Visual styling must not take priority over a functioning ML pipeline.

## 12.3 Inference requirements

- Load a saved local checkpoint.
- Load the exact class mapping and preprocessing configuration from checkpoint metadata or an associated versioned file.
- Use the same normalisation expected during training.
- Set the model to evaluation mode.
- Disable gradient calculation for ordinary prediction.
- Handle invalid, oversized, corrupt, or unsupported files gracefully.
- Do not retrain the model when the app starts or when a user uploads an image.
- Do not call external inference APIs.

## 12.4 Image upload validation

At minimum:

- Allow common image types such as JPEG and PNG; optionally support WebP after testing.
- Check file size and image dimensions against configurable limits.
- Decode the image safely.
- Convert to RGB consistently.
- Handle corrupt or malformed files without crashing.
- Avoid trusting a filename extension as proof of actual file format.
- Avoid retaining unnecessary EXIF metadata.

## 12.5 Application errors

Use clear user-facing error messages and useful developer logs. Do not expose stack traces, local filesystem paths, or sensitive internal information to ordinary users.

---

# 13. Storage and database design

## 13.1 MVP storage choices

Use SQLite for structured local records unless the architecture review identifies a concrete limitation. Store image files in controlled local directories and reference them from the database.

Keep persistence code behind repository/service modules so the storage layer can be changed later.

## 13.2 Logical entities

The database design should support at least these logical entities:

- `taxa`: canonical taxonomy records.
- `taxon_names`: accepted names, common names, synonyms, and aliases.
- `taxonomy_imports`: source and snapshot metadata.
- `model_versions`: registered model versions and their metadata.
- `model_classes`: output labels mapped to canonical taxa for each model version.
- `feedback`: user confirmation/correction records.
- `feedback_images`: retained image references and consent metadata.
- `reviews`: reviewer decisions and audit history.
- `dataset_items`: curated images, labels, provenance, and split assignment.
- `training_runs`: configuration, manifest references, metrics, and output checkpoints.

The implementation may combine or split physical tables where appropriate, but it must preserve these logical distinctions.

## 13.3 Data integrity

- Use primary keys and appropriate foreign keys.
- Use unique constraints for provider namespace plus provider taxon ID.
- Keep raw user input separately from resolved taxonomy identifiers.
- Avoid storing an animal's name as the only identity key.
- Preserve model-version mappings so past predictions remain interpretable after model updates.
- Record status transitions and review actions.

---

# 14. Suggested source-code structure

The final structure may evolve after architecture review, but it should preserve separation of responsibilities.

```text
Wildlife-Intelligence-System/
├── .github/
│   ├── copilot-instructions.md
│   └── prompts/
├── app/
│   ├── streamlit_app.py
│   └── pages/
├── configs/
│   ├── data.yaml
│   ├── training.yaml
│   └── taxonomy.yaml
├── data/
│   ├── raw/                 # ignored by Git
│   ├── processed/           # ignored by Git
│   ├── taxonomy/            # imported data; licence-aware and normally ignored
│   ├── feedback/             # retained images; ignored by Git
│   └── manifests/
├── models/
│   ├── checkpoints/          # ignored by Git
│   └── registry/             # small metadata may be version-controlled
├── reports/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── import_taxonomy.py
│   ├── train.py
│   ├── evaluate.py
│   └── register_model.py
├── src/
│   └── wildlife_ai/
│       ├── config/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── taxonomy/
│       ├── feedback/
│       ├── curation/
│       ├── explainability/
│       ├── storage/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── .gitignore
├── README.md
├── PROJECT_SPEC.md
└── requirements.txt
```

Do not create every file as an empty placeholder merely to make the folder tree look complete. Add modules when their milestone begins.

## 14.1 Git exclusions

At minimum, exclude:

- `.venv/` and other virtual environments.
- `__pycache__/` and compiled Python files.
- Raw and processed datasets unless a small, legally redistributable sample is intentionally included.
- User-submitted feedback images.
- Large model checkpoints and generated weights.
- Local databases containing user feedback.
- Logs, caches, temporary files, and local secrets.

Provide an `.env.example` only if environment variables are actually needed. Never commit credentials or private data.

---

# 15. Testing and quality assurance

## 15.1 Unit tests

Test at least:

- Image loading and invalid-file handling.
- Image preprocessing and expected tensor dimensions.
- Dataset label mapping.
- Train/validation/test split reproducibility.
- Duplicate detection.
- Model output dimensions.
- Checkpoint save/load and metadata consistency.
- Taxonomy exact-name lookup.
- Scientific-name and synonym lookup.
- Name normalisation.
- Fuzzy matching and ambiguous candidate handling.
- Unresolved-name handling.
- Feedback state transitions.
- Consent checks before image retention.
- Curated-dataset eligibility rules.

## 15.2 Integration tests

Test complete paths such as:

1. Load a sample image and run local inference.
2. Submit a correction and confirm it is stored as pending.
3. Resolve `hipo` to a Hippopotamus candidate and require user confirmation.
4. Verify a feedback item and confirm it becomes eligible for curation only after all other checks pass.
5. Confirm that pending or rejected records cannot enter a training manifest.
6. Load a model checkpoint and confirm that class mappings and preprocessing match.
7. Run the application without a CUDA device using CPU fallback.

## 15.3 ML evaluation tests

- Confirm that train, validation, and test manifests do not overlap by image identifier or detected duplicate group.
- Confirm that the test set is not used by the training loop.
- Confirm that class-to-index mapping is stable and stored with the checkpoint.
- Confirm that evaluation metrics are computed from actual model predictions.
- Confirm that unsupported-input evaluation is separate from ordinary eight-class test accuracy.

## 15.4 Manual testing

Test at least:

- Clear image of each supported animal.
- Unusual pose or lighting for supported animals.
- Bear and hippopotamus images as unsupported examples; test clear images of all eight supported classes.
- Non-animal images.
- Blurry or partially obstructed images.
- Multiple-animal images.
- Corrupted image files.
- Incorrect user correction.
- Ambiguous animal name.
- Misspelled animal name.

Record observed behaviour. Do not choose only successful examples for the final demonstration.

---

# 16. Evaluation methodology and project reporting

## 16.1 Required results

The final report should include:

- Dataset sources and licences/attributions.
- Actual class counts and split counts.
- Preprocessing and augmentation methods.
- Architecture descriptions.
- Training configurations.
- Training/validation loss and metric curves.
- Confusion matrices.
- Accuracy, macro precision, macro recall, macro F1, and per-class metrics.
- Model size and inference-time methodology.
- Comparison of scratch CNN and ResNet18.
- Examples of correct, incorrect, and uncertain predictions.
- Unsupported-input evaluation.
- Limitations and likely sources of bias.
- Human-feedback and verification workflow demonstration.

## 16.2 Research questions

The team should select a small number of research questions and answer them with experiments. Suggested questions:

1. Does transfer learning with ResNet18 outperform a compact CNN trained from scratch on the same dataset and split?
2. How do data augmentation and fine-tuning affect per-class generalisation?
3. Which categories are most frequently confused, and what visual or data-related factors may explain the confusion?
4. How does the chosen abstention method trade off correct supported predictions against false acceptance of unsupported images?
5. Does a curated set of verified feedback images improve a later model version without degrading performance on the frozen evaluation set?

Do not promise a positive result for any research question. The value is in conducting and reporting the experiment honestly.

## 16.3 Limitations to discuss

- Dataset imbalance and sampling bias.
- Differences between curated photographs and real-world uploads.
- Background leakage and spurious correlations.
- Similar-looking species.
- Limited number of initial classes.
- Taxonomy source incompleteness and naming differences.
- Subjectivity or error in user-submitted labels.
- Limitations of confidence scores and open-set recognition.
- Grad-CAM's interpretability limitations.
- Hardware and training-time constraints.

---

# 17. Security, privacy, and responsible use

- Process uploads locally in the MVP.
- Do not send user images to external inference APIs.
- Do not retain images without clear consent for retention and potential dataset use.
- Validate file type, size, dimensions, and decoding.
- Generate safe internal filenames rather than trusting uploaded filenames.
- Prevent path traversal when saving files.
- Keep user images, local databases, and checkpoints out of Git.
- Avoid unnecessary collection of personal data.
- Record consent and review actions where applicable.
- Explain that predictions are machine estimates and may be wrong.
- Do not present the tool as a substitute for a wildlife expert, veterinarian, or conservation authority.
- Respect dataset and pretrained-weight licences and attribution requirements.

---

# 18. Development roadmap and milestone gates

Implementation must proceed in the following order. Later work may be adjusted after architecture review, but the team should not skip the milestone gates.

## Milestone 0 — Requirements and architecture

Deliverables:
- Reviewed project architecture.
- Agreed MVP scope.
- Initial taxonomy-source decision.
- Initial dataset-source and licence plan.
- Database design proposal.
- Updated repository structure.

Exit criteria:
- The team understands the separation between taxonomy, model classes, feedback, and curated training data.
- Major assumptions and risks are documented.

## Milestone 1 — Development environment and repository

Deliverables:
- Git repository initialised.
- Python virtual environment.
- PyTorch installed.
- CUDA/CPU device check.
- Dependency file.
- Basic tests and linting/formatting decisions.

Exit criteria:
- A reproducible environment setup is documented.
- A small PyTorch tensor operation runs on the selected device.

## Milestone 2 — Dataset inspection and preparation

Deliverables:
- Acquired and licence-reviewed initial data.
- Dataset inspection report.
- Cleaned manifest.
- Duplicate/corruption report.
- Reproducible train/validation/test splits.

Exit criteria:
- Class mappings and split counts are known.
- No known duplicate leakage exists across splits.

## Milestone 3 — Baseline CNN

Deliverables:
- Scratch CNN.
- Training and validation loop.
- Checkpoint saving.
- Training graphs.

Exit criteria:
- The model trains end-to-end and can run inference from a saved checkpoint.

## Milestone 4 — ResNet18 transfer learning

Deliverables:
- ResNet18 training pipeline.
- Frozen-backbone stage.
- Optional fine-tuning stage.
- Saved checkpoint metadata.

Exit criteria:
- The model trains and reproduces inference from its checkpoint.

## Milestone 5 — Evaluation and comparison

Deliverables:
- Test metrics.
- Confusion matrices.
- Per-class report.
- Model comparison.
- Documented limitations.

Exit criteria:
- Results are generated from the untouched test set and are reproducible from recorded artefacts.

## Milestone 6 — Taxonomy catalogue and name resolver

Deliverables:
- Versioned taxonomy import.
- Internal taxonomy schema.
- Exact and synonym lookup.
- Fuzzy candidate suggestions.
- Ambiguous and unresolved-name handling.

Exit criteria:
- `hipo` can produce a Hippopotamus suggestion, but the user must confirm it.
- The selected canonical taxon is stored separately from the original typed string.
- A taxonomy entry does not automatically become a model class.

## Milestone 7 — Feedback and verification

Deliverables:
- Feedback storage.
- Correct/Incorrect workflow with direct correction selection from the eight model classes.
- Candidate-demo feedback linked to the predicting model version without changing its lifecycle state.
- Pending/corroborated/verified/rejected/needs-review states.
- Reviewer interface or workflow.
- Separate image-retention and future-training consent checks.
- Curated verified corrections assigned only to train or validation.

Exit criteria:
- Unverified feedback cannot enter a training manifest.
- Every reviewed item has an auditable state and provenance.

## Milestone 8 — Inference application

Deliverables:
- Streamlit upload interface.
- Local checkpoint loading.
- Prediction and abstention presentation.
- Top-three outputs where applicable.
- Feedback integration.

Exit criteria:
- The app runs locally and does not retrain on startup or call external inference APIs.

## Milestone 9 — Grad-CAM and usability

Deliverables:
- Grad-CAM visualisation for the supported architecture.
- Error handling.
- Manual usability tests.

Exit criteria:
- Heatmaps can be generated and their limitations are documented.

## Milestone 10 — Controlled data expansion and model versioning

Deliverables:
- Curated feedback export/manifest process.
- New training-run process.
- Model registry/version metadata.
- Model comparison and promotion decision.
- Rollback procedure.

Exit criteria:
- A candidate model can be trained without modifying the active model.
- Promotion is deliberate and evidence-based.

## Milestone 11 — Final documentation and demonstration

Deliverables:
- README setup instructions.
- Final project report.
- Architecture diagram.
- Dataset and licence notes.
- Experimental results.
- Screenshots and demonstration cases.
- Known limitations and future scope.

Exit criteria:
- Another student can follow the documented instructions to set up the project and reproduce the principal workflow.

---

# 19. MVP acceptance criteria

The MVP is considered functionally complete only when all applicable criteria below are satisfied.

### Data and model

- [ ] The eight initial animal categories are represented by documented, correctly mapped labels.
- [ ] The data pipeline validates images and records rejected files.
- [ ] Train, validation, and test splits are reproducible and checked for leakage.
- [ ] A scratch CNN trains and saves a usable checkpoint.
- [ ] ResNet18 transfer learning trains and saves a usable checkpoint.
- [ ] Both models are evaluated on the same untouched test set.
- [ ] Metrics and graphs are generated from real predictions.

### Taxonomy

- [ ] The taxonomy catalogue can be imported and versioned.
- [ ] The catalogue exists independently of model-supported classes.
- [ ] Exact common names, scientific names, and synonyms can be resolved.
- [ ] Fuzzy matching suggests likely matches without silently accepting ambiguous candidates.
- [ ] `hipo` can suggest Hippopotamus and requires user confirmation.
- [ ] An unresolved name is stored for review without being invented as an accepted taxon.

### Feedback

- [ ] The application asks for feedback after prediction.
- [ ] An incorrect prediction can be corrected by selecting directly from the eight model classes, without a taxonomy import or lookup.
- [ ] Original prediction, corrected class, model version, score, submission time, and verification status are retained as separate feedback data.
- [ ] Unverified corrections remain outside the training dataset.
- [ ] A reviewer can verify, reject, or request further review.
- [ ] Image retention for feedback is consent-based.
- [ ] Verified eligible corrections enter only train or validation, never the frozen test split.
- [ ] Candidate-demo feedback is linked to its candidate model version and does not change model lifecycle state.

### Application

- [ ] Images can be uploaded and validated.
- [ ] Inference uses a locally saved model.
- [ ] The application displays a result or an uncertainty/unsupported status.
- [ ] The application displays top alternatives where relevant.
- [ ] The application does not retrain on individual uploads.
- [ ] The application does not call an external inference API.

### Reproducibility and documentation

- [ ] Model metadata includes class mapping and preprocessing configuration.
- [ ] Training runs record configuration and dataset manifest references.
- [ ] The project has a working setup guide.
- [ ] Results, limitations, and data-source acknowledgements are documented.

---

# 20. Future scope

Possible extensions after the MVP:

- Expand to more species after collecting sufficient verified and diverse images.
- Add an animal detector for photographs containing multiple animals.
- Add bird classification as a dedicated, carefully scoped model or model family.
- Explore hierarchical classification using taxonomic relationships.
- Add image-embedding search for visually similar examples.
- Improve open-set recognition and confidence calibration.
- Add additional languages and regional common names.
- Add reviewer roles and multi-user authentication for a hosted deployment.
- Add audio or video analysis as a separate research component.
- Provide a dataset quality dashboard and annotation disagreement analysis.

Future scope must not be represented as completed MVP functionality.

---

# 21. Architecture summary

The intended logical flow is:

```text
                    IMAGE UPLOAD
                         |
                         v
                INPUT VALIDATION
                         |
                         v
                 ACTIVE ML MODEL
                         |
              +----------+----------+
              |                     |
              v                     v
       SUPPORTED OUTPUT       ABSTAIN / UNCERTAIN
              |                     |
              +----------+----------+
                         |
                         v
                 DISPLAY RESULT
                         |
                         v
                  USER FEEDBACK
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       CONFIRM        CORRECT         UNSURE
          |              |
          |              v
          |       TAXONOMY SEARCH
          |              |
          |              v
          |       NAME RESOLUTION
          |              |
          |              v
          |       USER CONFIRMATION
          |              |
          +--------------+
                         |
                         v
                  FEEDBACK STORE
                         |
                         v
                  HUMAN REVIEW
                         |
                +--------+--------+
                |                 |
                v                 v
             VERIFIED          REJECTED
                |
                v
          QUALITY / RIGHTS CHECKS
                |
                v
          CURATED DATASET POOL
                |
                v
         CONTROLLED TRAINING RUN
                |
                v
          CANDIDATE MODEL VERSION
                |
                v
           MODEL EVALUATION
                |
          +-----+------+
          |            |
          v            v
       PROMOTE       REJECT
          |
          v
      ACTIVE MODEL
```

The taxonomy catalogue is a separate reference service for animal-name lookup and unsupported-animal research. It does not determine direct corrections among the model's eight classes, perform image recognition, or automatically add output classes to the model.

---

# 22. Decisions that must be confirmed during architecture review

The following are deliberate open decisions. Copilot should analyse and recommend options rather than silently deciding them.

1. **Taxonomy source:** Choose the initial provider/snapshot after reviewing coverage, format, licence, and available common names/synonyms.
2. **Database:** Confirm SQLite for the local MVP and define a migration strategy.
3. **Dataset composition:** Confirm actual sources and counts for lion, tiger, leopard, gray wolf, cheetah, and jaguar; do not assume the chosen sources provide balanced data.
4. **Dog taxonomy record:** Confirm the accepted dog record used by the selected taxonomy backbone.
5. **Abstention method:** Start with a measurable baseline and decide whether a more advanced open-set method is feasible.
6. **Review authority:** Define who can mark a correction as verified in the college-project setting.
7. **Image retention:** Decide the exact consent wording and local retention policy.
8. **New-class criteria:** Define evidence-based minimum data and performance gates before adding a new output class.
9. **Model promotion:** Define acceptable regression limits and required evaluation checks before replacing the active model.
10. **Team responsibilities:** Assign dataset, ML, taxonomy/feedback, application, and documentation work among team members.

Until these decisions are reviewed, treat them as open questions rather than implementation facts.

---

# 23. Initial instruction to GitHub Copilot

After placing this file at the repository root, open Copilot Chat in **Plan mode** and use the following prompt:

```text
Read PROJECT_SPEC.md completely. Treat it as the authoritative project
specification.

Do not implement code or create files yet.

Analyse the specification and produce:
1. A proposed high-level architecture.
2. A module-by-module responsibility breakdown.
3. A database schema proposal and entity relationships.
4. A proposed taxonomy import and name-resolution design.
5. A proposed dataset manifest and split strategy.
6. A proposed model-training and evaluation pipeline.
7. A proposed feedback verification and curation workflow.
8. A proposed model-versioning and promotion workflow.
9. A phased implementation plan with dependencies.
10. A list of contradictions, risks, missing requirements, and decisions
    that need human approval.

Pay particular attention to keeping these separate:
- Taxonomy records.
- Model-supported output classes.
- User feedback.
- Verified/curated training examples.
- Model versions.

Do not assume that a taxonomy entry is visually recognisable by the model.
Do not assume user feedback is ground truth.
Do not propose automatic retraining after each correction.
Do not fabricate dataset availability, model performance, or test results.

Recommend the simplest architecture that satisfies the MVP without
unnecessary infrastructure. Wait for human review before implementation.
```

**End of specification.**
