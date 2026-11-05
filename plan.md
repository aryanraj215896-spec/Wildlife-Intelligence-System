# Wildlife & Animal Intelligence System — Implementation Status

## Status and scope

The approved local Windows MVP architecture has been implemented and exercised. This file records the existing roadmap and its execution status; it does not restart architecture review. The project is finalized as a research prototype, not a rights-cleared or production-ready classifier. Software implementation and local experiments are complete, while data-rights/provenance, taxonomy, OOD, consent/review, and promotion evidence remain unresolved and intentionally block active inference.

Confirmed decisions: Catalogue of Life is the initial taxonomy-source candidate, subject to source/release review; consent for retaining feedback images is separate from consent for potential training use; designated project reviewers may verify feedback; use cautious abstention and claim OOD only if separately validated; use grouped, stratified 70/15/15 initial supported-class splits with a frozen test set plus separate OOD development and final OOD evaluation sets; and require affected model artifacts to be retired or retrained if a contributing image is withdrawn.

The central invariant is that five different concepts remain distinct:

1. **Taxonomy records** describe accepted taxa and their names/relationships.
2. **Model-supported output classes** are the explicit outputs of one particular model version, each mapped to a taxon.
3. **User feedback** records what a user said about a particular prediction; it is unverified.
4. **Verified/curated examples** are retained, reviewed and quality-checked data selected for a dataset manifest.
5. **Model versions** are immutable trained artifacts and metadata; promotion changes which version is active but does not rewrite previous predictions or feedback.

A taxonomy entry, a selected feedback taxon, or a verified label does not itself make a taxon visually recognizable by the active model.

## Feature classification: MVP, advanced milestones, future scope

The architecture covers all proposed capabilities, but implementation is staged. This classification sets the first-release boundary without removing advanced requirements from the system design.

This table classifies every proposed feature; the phase table below assigns delivery order and dependencies without changing these scope tiers.

| Area | MVP — essential first release | Advanced milestones — after MVP gates | Future scope — separate follow-on work |
|---|---|---|---|
| Runtime and interface | Local Windows-compatible Python app; Streamlit classify, taxonomy, feedback/review, and model-information views; local CPU fallback and tested optional CUDA | Usability refinements and richer diagnostics | Hosted or distributed deployment; multi-user service operation |
| Image handling | JPEG/PNG (WebP only after testing), safe decode, size/dimension checks, RGB preprocessing, safe generated filenames, no unconsented retention, useful errors | Expanded format support and more extensive usability testing | Video ingestion and multi-frame analysis |
| Taxonomy | One reviewed Catalogue of Life snapshot; internal immutable release-scoped records; exact common/scientific/synonym lookup; deterministic normalization; fuzzy suggestions with user confirmation; unresolved-name handling; model mappings kept separate | Reviewed cross-release reconciliation tooling and taxonomy update workflow; richer ambiguity/context support | Multiple taxonomy providers, additional languages/regional names, hierarchical taxonomy-based classification |
| Supported classes | Explicit eight initial output classes (Domestic Cat, Domestic Dog, Lion, Tiger, Leopard, Gray wolf, Cheetah, Jaguar) with model-version-specific taxon mappings; no `Other`/`Unknown` training class | Add a category only after approved data-diversity and performance gates | Broad species coverage, dedicated bird model, other model families |
| Data and provenance | Source/licence review for all eight classes; raw/processed separation, eight-class manifest, image validation, exact/near-duplicate checks where feasible, grouped reproducible 70/15/15 supported-class split, frozen supported test set | Expand curated data; more complete duplicate/source lineage and automated withdrawal-impact reports | Large-scale data curation or population-monitoring datasets |
| Models and training | Eight-output compact scratch CNN and ResNet18 transfer learning; configurable local training, reproducibility metadata, safe checkpoints, validation selection, measured metrics | Controlled new-category training, additional fine-tuning experiments, richer registry/promotion tooling | Additional architectures and research model families |
| Inference and uncertainty | Local checkpoint loading; exact class/preprocessing metadata; top alternatives; cautious baseline abstention; separate OOD development and final evaluation data; no final-set tuning | Evaluate calibration, energy/embedding-distance or dedicated open-set methods; promote only if validated | Advanced open-set recognition and calibrated probabilities if measured; image-embedding similarity search |
| Feedback and review | Yes/no/not-sure/unsupported feedback; pending/verified/rejected/needs-review; local project-team review interface; separate retention/training consent; no automatic verification or retraining | Disagreement review aids and more complete reviewer/curation audit | Corroboration only with authenticated independent reviewers; authenticated roles/multi-user authorization |
| Curation and versioning | Eligibility checks before manifest inclusion; immutable run/model metadata; explicit candidate promotion/rejection; documented rollback; withdrawal retires affected artifacts | Full source-to-model lineage UI, automated withdrawal impact tracing, expanded promotion criteria/reporting | Automated retraining orchestration or enterprise registry; dataset-quality dashboard and annotation-disagreement analysis |
| Explainability and modalities | Not required for MVP acceptance; architecture seam retained | Grad-CAM for supported CNN/ResNet layer, with limitations documented | More advanced explainability research; audio/video analysis |
| Recognition scope | Single-primary-animal classifier; no claim to identify every species or individual | New supported species only after gates | Object detection/multiple animals; dedicated bird model; hierarchical classification; broader species coverage |
| Languages and deployment | Local, single-project deployment; configured labels and taxonomy names | Improve local reviewer workflow without implying authentication | Additional languages/regional names; hosted reviewer roles and authenticated multi-user deployment |

MVP review is a local project-team interface with no authenticated multi-user access or secure authorization: access to the project machine, running app, database, or files can expose or alter review data. A manually entered reviewer name is an attribution note only, not authentication.

Advanced and future items remain in scope for the overall architecture, but are not represented as delivered MVP functionality. None permits automatic acceptance of user labels or automatic retraining after a correction.

## 1. Proposed high-level architecture

Use a small layered Python application rather than a distributed service:

```text
Streamlit UI
    |
Application services (classify, search taxa, submit feedback, review, curate)
    |
Domain rules and typed records
    +-- inference / training / evaluation / explainability
    +-- taxonomy import / name resolution
    +-- feedback verification / dataset curation
    |
Repository interfaces
    +-- SQLite repositories and migrations
    +-- local image and checkpoint storage
    +-- versioned configuration and manifests
```

- **UI:** Streamlit pages for image classification, taxonomy exploration, feedback review, and model information. UI code calls services and does not own model, taxonomy, or persistence rules.
- **Application/domain:** Typed request/result objects and explicit state transitions. Enforce consent, reviewer actions, model-class lookup, and curation eligibility here, not only in widgets. A user correction never triggers training; only a separate, explicit developer/administrator action can start a controlled training run after review and curation.
- **ML:** Separate data preparation, model definitions, training, evaluation, checkpoint loading, local inference, abstention evaluation, and Grad-CAM. Training is a command-driven workflow, never an upload/startup side effect.
- **Taxonomy:** Provider-specific import adapters feed a stable internal taxonomy schema. Name resolution returns ranked candidates and requires user confirmation; it never creates model classes.
- **Persistence:** SQLite for local structured records; controlled local directories for retained images, manifests, logs, and checkpoints. Hide storage behind repository/service boundaries to leave future migration possible without adding server infrastructure now.
- **Configuration:** Central configuration for paths, image limits, class mappings, preprocessing, training, and resolver thresholds. Versioned model metadata is authoritative for inference preprocessing and class mapping.

Suggested implementation modules (create only when their milestones begin):

| Module | Responsibility |
|---|---|
| `app/streamlit_app.py`, `app/pages/` | Views, upload controls, result display, feedback/reviewer interactions |
| `src/wildlife_ai/config/` | Typed loading/validation of app, data, taxonomy, and training configuration |
| `src/wildlife_ai/domain/` | Shared types, enumerations, and explicit transition/eligibility rules |
| `src/wildlife_ai/storage/` | SQLite connection, schema migrations, repository implementations, transactions |
| `src/wildlife_ai/data/` | Image validation, provenance, hashes, duplicate grouping, manifests, split generation, loaders |
| `src/wildlife_ai/models/` | Scratch CNN and ResNet18 construction; architecture identifiers |
| `src/wildlife_ai/training/` | Reproducible training loop, checkpointing, run metadata and history |
| `src/wildlife_ai/evaluation/` | Classification metrics, confusion matrix, OOD/abstention evaluation and reports |
| `src/wildlife_ai/inference/` | Safe local checkpoint selection/loading, preprocessing, prediction and abstention result |
| `src/wildlife_ai/taxonomy/` | Import adapters, normalized names, exact/fuzzy search and candidate ranking |
| `src/wildlife_ai/feedback/` | Feedback submission, consent-aware image handling, reviewer decisions/audit |
| `src/wildlife_ai/curation/` | Eligibility checks and immutable training-manifest export |
| `src/wildlife_ai/explainability/` | Architecture-specific Grad-CAM and overlay generation |
| `scripts/` | Small command-line entry points that orchestrate package services |
| `tests/unit/`, `tests/integration/`, `tests/fixtures/` | Rules, persistence, pipeline invariants and small synthetic fixtures |

## 2. Database schema proposal and relationships

Use internal primary keys plus stable provider identity; foreign keys should be enabled in SQLite. Use explicit migrations. Store image files outside ordinary feedback rows.

| Logical table | Core fields and constraints |
|---|---|
| `taxonomy_imports` | `id`, provider/source namespace, release/version, import timestamp, licence/attribution, source file/checksum, import status |
| `taxon_concepts` | Internal stable identity for a reconciled taxonomic concept; not a provider ID or display name; lifecycle/audit metadata |
| `taxa` | Immutable provider record for one release: internal `taxon_version_id`, `taxonomy_import_id`, provider namespace/ID, names, rank/status, nullable release-scoped parent and nullable reconciled `taxon_concept_id`; unique `(taxonomy_import_id, source_namespace, source_taxon_id)` |
| `taxon_names` | `id`, release-scoped `taxon_version_id`, display name, normalized name, name kind (scientific/common/synonym/curated alias), language/region, source/import reference, alias review state; indexes on normalized name |
| `taxon_crosswalks` | Explicit relation between old/new release-scoped taxon records, relation type (same concept, rename, split, merge, synonym/accepted change, uncertain), reviewer decision, source evidence, timestamp and review state |
| `model_versions` | immutable version ID, architecture, lifecycle state, artifact path/hash, metadata path, training-run reference, creation time, metrics/evaluation reference |
| `model_classes` | `model_version_id`, output index, display label, exact release-scoped `taxon_version_id`; unique `(model_version_id, output_index)` and unique label within a version |
| `feedback` | ID, submission time, user response, original prediction label and exact predicted `taxon_version_id`/score/score type, model-version reference, original typed correction, nullable resolved `taxon_version_id` and resolver taxonomy snapshot, resolver method/candidate score, verification state |
| `feedback_images` | ID, feedback reference, safe relative file path, content hash/duplicate group, retention/training consent states and timestamps, deletion state |
| `reviews` | append-only review event: feedback reference, prior/new state, decision, reviewer reference (self-declared in MVP, not authenticated), notes/evidence, timestamp |
| `dataset_manifests` | immutable manifest ID/hash, creation time, source/split policy version, seed, purpose/evaluation-set version |
| `dataset_items` | manifest reference, source feedback or acquired-image reference, path/hash/duplicate group, release-scoped canonical `taxon_version_id`, label-review and rights/consent checks, split, provenance |
| `training_runs` | run ID, config, code/library/device metadata, dataset manifest hash, seed, clean base checkpoint/reference and parent lineage, start/end/status, metrics, output artifact references |

Relationships and invariants:

- Every import/release record is immutable. The same provider identifier may appear in multiple releases, so uniqueness is scoped to the import/release. `taxon_concepts` gives this application a stable internal identity only after reconciliation; `taxon_version_id` preserves the exact source record/version. Cross-release continuity is explicit in `taxon_crosswalks`; a changed, split, merged, absent or uncertain record is never silently overwritten or redirected.
- A model version has its own output-class mapping. A model class points to a taxonomy record, but taxonomy records do not imply any `model_classes` row.
- Feedback references the exact model version and taxonomy snapshot, storing the original output as a historical snapshot; it separately stores user response, free-text input, and optional release-scoped resolved taxon. Model mappings likewise reference a specific release-scoped record. Taxonomy updates never rewrite historical feedback or model mappings.
- Retained image metadata is separate from feedback and requires explicit consent. Feedback may exist without a retained image.
- Review events preserve history rather than overwriting who made a decision. `verified` is a review state, not automatic dataset inclusion.
- Dataset items are selected into immutable manifests only after consent/rights, accepted label, quality, duplicate and split checks pass. Dataset membership is not represented by feedback state alone.
- Training runs consume manifest IDs/hashes and produce candidate model versions. Preserve source-image-to-manifest-to-run-to-checkpoint lineage, including parent/base checkpoint. Promotion changes an explicit active-version pointer/registry state; it does not mutate prior versions or their class mappings.

SQLite implementation should define deletion behavior carefully: user-requested image deletion must remove the retained file and update references/status consistently. Confirmed withdrawal direction: remove the image and any identified duplicate group from future manifests and require every affected model artifact/descendant to be retired or replaced. Any replacement model intended to remove the withdrawn image's learned influence must be retrained from a clean base checkpoint using a newly generated, updated eligible dataset manifest that excludes the withdrawn image/group and passes all consent, rights, label, quality and duplicate checks. The clean base must not be the previously fine-tuned model or any descendant that learned from the withdrawn image; use random initialization or a documented pretrained base whose provenance shows it is unaffected. Never claim that deleting the file alone removes learned influence. Preserve only a minimal withdrawal/impact audit record, not the image. If lineage cannot establish a clean artifact, retire it pending a clean retrain.

## 3. Taxonomy import and name-resolution design

1. **Source review:** Review the Catalogue of Life candidate's coverage, download formats, release IDs, names/synonyms, licence and attribution before choosing a specific snapshot. Do not import until that review is complete.
2. **Staged immutable import:** Preserve source snapshot metadata and checksum; parse provider files into staging/validated records; validate required identifiers and relationships; report rejected rows; apply each release transactionally as immutable `taxa`/`taxon_names` records. Keep import code provider-specific and internal entities provider-neutral.
3. **Cross-release reconciliation:** For each new release, compare records against prior snapshots using provider identifiers as candidate evidence (not unquestionable identity), accepted/scientific names, synonyms, rank, parentage and source change metadata. When a provider identifier or accepted name changes, create an explicit candidate crosswalk to the prior release record and require review before mapping it to the same stable `taxon_concept_id`. A reviewer-approved unchanged/renamed/accepted-name change may map the new release record to that same internal concept. A split or merge creates distinct internal concepts as appropriate and explicit one-to-many/many-to-one crosswalk relations; it does not collapse historical records. If evidence is ambiguous, leave it unreconciled/uncertain and require review. Never auto-remap old feedback or model mappings.
4. **Identity and names:** Store provider namespace and provider taxon ID together, scoped to a release. Store accepted scientific/common names, source synonyms and reviewed local aliases as separate name rows with provenance and normalized search form. Never promote a user's typo directly to an official synonym.
5. **Deterministic normalization:** Preserve raw query; normalize Unicode, case, whitespace and harmless punctuation deterministically. Keep normalization implementation/version available for reproducibility.
6. **Resolution order:** Exact normalized accepted common/scientific name; exact source synonym; exact reviewed alias; then fuzzy ranked candidates. Return candidate context (scientific name, rank, parent, accepted name) and method/score. RapidFuzz is a plausible local dependency if evaluation shows it is useful; no external inference or language-model service.
7. **Confirmation and ambiguity:** Fuzzy output is a suggestion, not a resolution. `hipo` may rank Hippopotamus, but user confirmation is required. Ambiguous names such as `bear` return multiple appropriately ranked taxa or a broad taxon when the source provides one; do not silently select a species. Low-score/no-match queries remain unresolved and reviewable.
8. **Historical references:** Feedback, model mappings and dataset manifests point to exact immutable `taxon_version_id` records and record the taxonomy release. Internal `taxon_concept_id` supports reviewed continuity queries only; it never replaces the historical foreign key. Future release imports add records and reviewed crosswalks, and cannot rewrite the taxon reference attached to an earlier prediction, correction, curated item or model version.
9. **Separate support state:** Model support is determined only by the active model version's `model_classes`, not by taxonomy presence, a fuzzy match, feedback, or curation.

## 4. Dataset manifest and split strategy

- Keep acquired originals, processed data, retained feedback images, and manifests separate. Ignore large/private image files and local databases in Git.
- Each manifest item records stable image ID, relative path, canonical taxon, source and source image reference where permitted, licence/attribution, acquisition/import date, review state, SHA-256, perceptual/duplicate group, split, and quality notes. Store permitted provenance rather than a prohibited URL.
- Validate decode/format/dimensions and single-label suitability. Record rejection reasons; do not silently delete originals. Detect exact duplicates and, when feasible, near-duplicates. Keep groups from the same observation, sequence, photographer burst, or source together.
- Initial supported-class target covers all eight configured classes and is 70/15/15 train/validation/test, stratified by class and grouped by duplicate/observation group, using a fixed recorded seed. It is a target, not a claim about available data.
- Keep the label-to-taxonomy mapping for all eight classes in configuration or model metadata; resolve canonical provider identifiers only after selecting and reviewing the versioned taxonomy source.
- Keep supported-class test data, OOD development data, and final OOD evaluation data as separate, versioned manifests. Use validation plus OOD development data to choose an abstention rule. Seal the final OOD evaluation set from all threshold/method tuning; run it only after the method is frozen and report results only. If those results lead to a method/model change, the old final set is consumed and a new untouched final OOD set is required before the next final evaluation.
- Freeze supported test membership for model selection and tuning; do not use supported test metrics to adjust thresholds or hyperparameters. Use training/validation and OOD development sets for development and acceptance-threshold fitting.
- Enforce no overlap of exact IDs, hashes, or duplicate groups across splits and between feedback data and frozen test data. Report counts, imbalances, rejected items, group/split rules, and manifest hash.
- For subsequent data additions, version manifests and preserve the original frozen evaluation protocol. If an evaluation set must change, document why and report results separately rather than silently replacing it.
- Augment training only. Use deterministic evaluation transforms; document the pretrained normalization and the exact transform config with each checkpoint.

## 5. Model-training and evaluation pipeline

1. Inspect licensed data and generate validated immutable manifests/splits.
2. Seed Python/NumPy/PyTorch where applicable; record limitations of deterministic GPU operations, versions, device/CUDA details, Git revision, data hash and config.
3. Build data loaders with configurable workers/batch size and CPU fallback; never load the whole dataset onto GPU. Test actual CUDA availability. Make mixed precision opt-in/configurable and verify it on the target hardware.
4. Train compact scratch CNN as baseline, with validation monitoring and safe periodic/best-validation checkpoints.
5. Train ResNet18 using documented Torchvision weights: replace final head, train head with frozen backbone, optionally fine-tune selected/all layers using validation decisions. Record pretrained-weight identifier and provenance/licence.
6. Save a versioned checkpoint bundle: state dict, architecture, output labels/indexes and taxon IDs, input size, normalization/preprocessing, training config, manifest hash, validation results, weight provenance and model version. Load only trusted local artifacts; use safe loading supported by the installed PyTorch version.
7. Select configurations only on validation data. Evaluate both final candidates against the same untouched supported test set. Report accuracy, macro precision/recall/F1, per-class results, confusion matrix, file size, training time and measured inference time with setup.
8. Separately use supported validation examples and an OOD development set for abstention-method/threshold design. Keep the final OOD evaluation set separate and untouched; freeze the method and all thresholds before running it, and never tune against its results. Report false rejection and false acceptance on the final evaluation along with supported accuracy. If the result informs a method/model change, retire that set as a final holdout and obtain a new untouched set for the next final evaluation. Treat threshold output as an operational abstention estimate, not reliable proof of OOD recognition. Distinguish `uncertain` from `unsupported_or_out_of_distribution` only if evidence supports that distinction; otherwise present a cautious abstention without claiming the cause.
9. Generate plots/reports from real predictions and preserve the test protocol and run metadata. Never invent results.

## 6. Feedback verification and curation workflow

1. Inference returns prediction/abstention with model version, score type and alternatives. Feedback options are yes/no/not sure/unsupported. A confirmation remains user feedback, not verified ground truth.
2. For a correction, retain raw typed name and resolution trace; show taxonomy candidates and require explicit user selection/confirmation. Allow unresolved/cancel.
3. Store the feedback record as `pending`, including model prediction and explicit consent state. Inference must work without retention; require consent before storing the image, and keep it outside the database. A submission cannot launch training or alter the active model.
4. **Simple MVP review:** the application provides a local project-team review interface. A reviewer inspects the consented retained image, original output/version, proposed taxon, provenance/evidence, and duplicate flags; then records `verified`, `rejected`, or `needs_review` with a timestamp and optional self-declared reviewer reference. The initial application provides no secure multi-user authentication or authorization. A manually entered name is an attribution note only, not authentication or proof that the action came from an authorized person. Anyone with access to the trusted project machine/app/database/files may be able to access or change review data; designated reviewer status is enforced only by the team's offline procedure.
5. A verified correction becomes **eligible for curation review only**. Curation separately checks retention/training consent, permitted rights/provenance, canonical taxon, image validity/quality, duplicate/near-duplicate status, and frozen test leakage. Reject/hold items explicitly with reasons.
6. A curation export creates a versioned manifest item and split assignment. For a new category, gather diverse verified data and meet pre-agreed per-class evidence/performance gates before adding it to a proposed model mapping. Until then the taxon remains searchable but unsupported.

MVP state progression is `pending` -> `verified`, `rejected`, or `needs_review`; corroboration is not part of the essential workflow. Only an explicit verified reviewer decision establishes verification. Repeated submissions, even if labels agree, are not independent verification. Corroboration may be considered only as a future extension with authenticated, independent reviewers and duplicate/related-submission checks; it remains supporting evidence, never automatic ground truth or training eligibility. Verification does not directly create a dataset item.

## 7. Model versioning and promotion workflow

- Assign each training run and produced model a unique immutable version. Keep artifacts, metadata, mappings, evaluation reports, and clean-base/parent lineage; never overwrite the currently active checkpoint.
- Register a new run as a candidate only after checkpoint metadata and artifact integrity are validated. The candidate's supported outputs are its explicit `model_classes`.
- Compare candidate to active model on the agreed frozen test/evaluation protocol and separately inspect validation, per-class, confusion, abstention/OOD results, model size and inference constraints.
- Promote only through an explicit project-team developer/reviewer decision under the local team's offline procedure after passing pre-agreed minimum criteria and regression limits. Record decision, self-declared approver reference, timestamp, evaluation IDs and rationale; this record is not authenticated by the MVP.
- Switch a small active-version registry/config pointer atomically. Inference reads one selected immutable version and its paired preprocessing/class map. Past feedback continues to reference its original model version.
- Rejection leaves active version unchanged. Rollback re-points to a previous registered version only if that version remains eligible (including withdrawal/lineage checks), records the action, and does not destroy candidate history.
- Do not define arbitrary numeric new-class requirements until the team has reviewed dataset feasibility and desired evidence. Promotion criteria and rollback verification are approval items.

## 8. Phased implementation plan and dependencies

Dependencies below are logical prerequisites; source review and repository inventory can run in parallel during architecture review. Phases 1–8 establish the MVP core; phase 10 includes the minimum immutable model metadata and explicit candidate/active selection needed by the MVP, with advanced lineage automation and broader expansion retained as later work. Phase 9 and the advanced items above remain in the architecture without being required to claim the MVP is complete.

| Phase | Tier | Deliverable / exit gate | Depends on |
|---|---|---|---|
| 0. Requirements and architecture review | Prerequisite | Confirm scope, taxonomy provider strategy, data/licence plan, logical schema, review/consent policy, class-expansion and promotion gates | None |
| 1. Repository/environment baseline | MVP | Inspect current repository before edits; document implemented/missing/conflicting requirements; Python environment, device check, dependencies, tests and quality conventions | Phase 0 |
| 2. Data acquisition and preparation | MVP | Source/licence-reviewed data for the eight configured classes, actual per-class count report, validator, provenance manifest, duplicate report, reproducible grouped splits | Phase 0, 1 |
| 3. Scratch CNN | MVP | Model, train/validation loop, checkpoint metadata and inference smoke test | Phase 2 |
| 4. ResNet18 transfer learning | MVP | Frozen-head stage, optional fine-tuning, provenance and checkpoint | Phase 2 |
| 5. Evaluation and abstention baseline | MVP | Same frozen supported test protocol for both models; separate OOD development/final evaluation manifests; threshold frozen before final OOD evaluation | Phases 3, 4 |
| 6. Taxonomy catalogue/resolver | MVP | Versioned immutable import, internal identity and reviewed cross-release reconciliation, exact/synonym/fuzzy lookup, confirmation and unresolved cases | Phase 0, 1 |
| 7. Feedback/review/curation | MVP | Consent-aware feedback, simple pending/verified/rejected/needs-review workflow, local project-team review, review history, eligibility checks and curated manifest export; no authentication/corroboration claim | Phases 1, 2, 6 |
| 8. Inference application | MVP | Streamlit upload, local checkpoint inference, abstention/result UX, taxonomy correction and feedback | Phases 5, 6, 7 |
| 9. Grad-CAM/usability | Advanced milestone | Supported architecture heatmap, limitations, error handling and extended manual workflow cases | Phase 8 |
| 10. Controlled expansion/versioning | MVP foundation + advanced extension | MVP: immutable model metadata, explicit candidate promotion/rejection and documented rollback. Advanced: lineage impact tooling, new-category expansion and richer registry workflows | Phases 5, 7, 8 |
| 11. Final documentation/demo | MVP deliverable | Reproducible setup, sources/licences, actual experiments, limitations and demonstration | Phases 0–10 |

Do not create all proposed modules as empty placeholders. Add tested, coherent vertical slices at each phase.

### Execution status (updated 2026-11-05)

- **Phase 2:** Inspected the user-supplied `data/raw/` files and generated the Git-ignored manifest `data/manifests/dataset_manifest.csv` with seed 42. The manifest contains 15,190 valid JPEG/PNG images across all eight configured classes; four non-image files were excluded and no supported-format image was rejected. The per-class counts and split counts are recorded in the source-feasibility audit above and in the manifest. Exact byte-hash duplicate groups: 0. No dataset provenance, source licence, or canonical external taxon IDs were supplied; these remain blockers for rights-cleared public distribution and taxonomy-backed claims.
- **Phase 3:** Implemented `ScratchCNN`, manifest-backed PyTorch datasets/loaders, training/validation, mixed precision, atomic best/last checkpoints, and run metadata. Trained 10 epochs with seed 42 on CUDA using the NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,441,926,656 bytes VRAM), batch size 32, 224-pixel inputs, and mixed precision. Best validation accuracy was 0.6956 at epoch 7; training duration was 1,137 seconds. The frozen test evaluation later measured 69.18% accuracy and 0.5577 macro F1. Checkpoints and history are under ignored `models/runs/scratch_cnn-20261104T190023Z/`.
- **Phase 4:** Implemented pretrained Torchvision ResNet18, frozen-head training and optional full-backbone fine-tuning, safe atomic checkpoints, and run provenance. The official `ResNet18_Weights.DEFAULT` (ImageNet-1K V1) artifact was downloaded from the Torchvision URL and loaded with its hash check. Trained 5 frozen-head plus 5 fine-tuning epochs on the RTX 3050, batch size 16, 224-pixel input, mixed precision, seed 42. Best validation accuracy was 0.9592. On the same frozen 2,278-image test split, accuracy was 0.9579, macro precision 0.9525, macro recall 0.9382, macro F1 0.9445. Measured CUDA model-forward-only throughput was 739.8 images/second at batch size 16 (first batch warmed up, CUDA synchronized; data loading excluded); checkpoint size was 44,808,075 bytes. These are measurements on this supplied split, not source/licence validation or a guarantee of generalization. Checkpoints and history are under ignored `models/runs/resnet18-20261104T192314Z/`.
- **Phase 5:** Implemented same-frozen-test metrics, per-class results, confusion CSV/SVG, training-history SVG, inference timing, and validation-only max-softmax abstention coverage curves. Best ResNet18 test accuracy was 0.9579 versus 0.6918 for scratch CNN; model scores are not calibrated probabilities. No threshold was selected because acceptance criteria are not defined. **Blocked:** no separately versioned, source-reviewed OOD development/final datasets were supplied, so false acceptance/rejection on OOD data and the final OOD gate cannot be evaluated. Test results are reported but not used to tune thresholds.
- **Phase 6:** Implemented SQLite taxonomy tables, schema versioning, a release/provenance-aware CSV importer, deterministic normalization, and exact/fuzzy candidate resolution that always requires user confirmation. Synthetic fixtures test resolver behavior only. **Blocked:** no reviewed taxonomy snapshot, source release/licence, or provider identity mapping was supplied, so the project catalogue remains empty and no real taxonomy data was invented.
- **Phase 7:** Implemented consent-aware feedback storage with safe generated filenames, JPEG/PNG validation and sanitization, separate retention/training consent timestamps, prediction/model linkage, duplicate linkage, review history, reviewer attribution, rights and image-quality review gates, image/training-consent withdrawal, and curated-manifest export. Exports preserve the existing frozen supported manifest and split membership, add reviewed items to train/validation only, reject exact/frozen-test hash overlap, and recheck consent before training. The local reviewer remains unauthenticated by design; exact consent wording/retention period, review SOP, data provenance, and near-duplicate detection are unresolved. **Complete as an implementation slice; operational-policy blockers remain.**
- **Phase 8:** Implemented Streamlit classify/taxonomy/review/model views, local checkpoint inference, top outputs, explicitly uncalibrated softmax scores, optional confirmed taxonomy correction, consent-gated feedback, and curation controls. When no production model is active, Classify opens directly in local development/demo mode, offers registered ResNet18 and ScratchCNN candidates, and keeps the image uploader and Classify button visible without a promotion step. Candidate artifact paths and hashes are checked; invalid candidates are identified in the UI. Predictions are marked experimental, the production active-model pointer is untouched, and feedback submission is disabled in demo mode. Production mode still requires an explicitly promoted active model. The frontend now has a responsive wildlife-themed visual system and explicit light-theme contrast for navigation, widget labels, candidate selectors, upload controls, cards, buttons, tables, and status messages. Streamlit `AppTest` exercised candidate registration, ResNet18 selection, upload, inference, prediction/score/alternatives, experimental Grad-CAM, and the isolated end-to-end taxonomy-confirmation → inference → consented feedback → review → eligible curation export flow. The real local taxonomy/review database is empty, so those pages accurately show empty-state guidance until reviewed data is supplied.
- **Phase 9:** Implemented and tested ResNet18 Grad-CAM for the processed model input, with unsupported-architecture and invalid-input handling. The visualization is described as exploratory, not causal evidence. ScratchCNN Grad-CAM remains unsupported.
- **Phase 10:** Implemented immutable candidate registration with restricted checkpoint loading and hash checks, matching frozen-test report validation, explicit candidate rejection, policy-gated promotion with per-class regression limits, lifecycle history, and rollback with artifact/consent checks. Streamlit `AppTest` exercised candidate registration and the Models-page promotion form using only an isolated synthetic project and test policy; it verified the active pointer there. **Real-project promotion remains blocked** until numerical acceptance/regression criteria are established and approved before candidate evaluation. No active model is selected in the current local database. New-category expansion remains beyond the current MVP foundation.
- **Phase 11:** `README.md`, `reports/FINAL_REPORT.md`, and the detailed `reports/IMPLEMENTATION_GUIDE.md` document reproducible setup, component implementation, application workflows, dataset/source audit, measured model results, validation and limitations. The refreshed worktree app was started on `127.0.0.1:8502`; all four pages were navigated in the live browser, and Streamlit `AppTest` exercised the end-to-end workflows. The latest isolated AppTest also completed candidate registration, ResNet18 demo inference and Grad-CAM, then tested the Models-page promotion flow without touching the live registry. **Software roadmap complete as an academic research prototype; production/specification gates remain open:** reconcile the user's class-to-source mapping against per-image/local archive provenance and licences (including the unknown Kaggle Jaguar licence and images.cv red/white wolf terms), import a reviewed taxonomy snapshot, provide OOD development/final assets, confirm consent/review policies, and approve promotion criteria before any future candidate evaluation. The explicit local candidate-demo path is implemented; it does not promote or activate candidates. Final full suite: 37 tests passed.
- **Dataset refresh and current model results (2026-11-05):** A later inspection found 1,346 valid Gray wolf images in `data/raw/wolf/`, of which 1,288 were absent from the earlier training manifest. The earlier 58 image paths all still exist with matching SHA-256 hashes. The original manifest and its frozen test split were preserved. A new complete manifest was generated at `data/manifests/dataset_manifest-20261105-current.csv` (16,478 valid images; 0 corrupt and 0 exact duplicate groups; SHA-256 `c8e633e6b0056aabc3436c3114a2c7737fe9b53c436cc5d568103bd1b60aa7a6`). Its split sizes are 11,534 train, 2,473 validation and 2,471 test. Both models were retrained using this manifest on CUDA with seed 42 and evaluated on the same held-out test split without tuning on test results:
  - ScratchCNN: 10 epochs, batch size 32, 463.9 seconds, best validation accuracy 0.7137; test accuracy 0.6835, macro precision 0.6786, macro recall 0.6863, macro F1 0.6772; forward-only throughput 792.5 images/second; checkpoint 1,584,801 bytes. Gray wolf test recall was 0.6733 (202 examples).
  - ResNet18: 5 frozen-head + 5 fine-tuning epochs, batch size 16, 421.4 seconds, best validation accuracy 0.9592; test accuracy 0.9494, macro precision 0.9351, macro recall 0.9344, macro F1 0.9344; forward-only throughput 695.9 images/second; checkpoint 44,808,267 bytes. Gray wolf test recall was 0.9109 (202 examples).
  - Checkpoints and histories are under `models/runs/scratch_cnn-20261104T222824Z/` and `models/runs/resnet18-20261104T223628Z/`; test reports and confusion figures are under `reports/metrics/refreshed-20261105/` and `reports/metrics/figures/`. A real Wolf test-image inference smoke check on CUDA predicted Gray wolf; this is one example, not a population-level claim.
  - Both updated checkpoints are registered as local `candidate` model versions with their verified artifact hashes and matching test-evaluation reports. No model is active: promotion remains blocked pending policy criteria approved before candidate evaluation. Data provenance/licensing remain unverified; numeric metrics do not establish deployment readiness or generalization to other sources.

### Application functionality audit (2026-11-05)

| Area | Status | Evidence and boundary |
|---|---|---|
| Classify | Functional for local demo inference | Registered ResNet18 candidate selection, image upload, inference, top prediction, uncalibrated score, alternatives, and experimental labeling passed Streamlit `AppTest`; live browser shows the candidate picker and upload/classify controls. Demo mode cannot activate a model or submit feedback. |
| Taxonomy | Integrated; data-blocked in the real project DB | Live page, search, and explicit confirmation are wired; isolated app test imported a synthetic fixture and exercised confirmation. No reviewed real snapshot is installed, so the real page correctly reports zero imported taxa. |
| Review and feedback | Functional; empty in the real project DB | The isolated app test submitted consented feedback and saved a review. The live Review page renders the queue and reports zero items. Reviewer identity remains unauthenticated/self-declared. |
| Curation | Functional with eligible reviewed data; unavailable in the live DB | The isolated end-to-end test exported a manifest after consent, label, rights, and quality checks. The live DB has no eligible examples, so export is correctly unavailable. |
| Models and registry | Integrated and browsable | Live Models page reports three candidates, zero active models, and no policy; current checkpoints are present and hash-valid. Isolated AppTest exercises checkpoint registration and the Models-page promotion form using a synthetic policy; registry tests cover rejection and rollback services. Candidate inference remains separate from production activation. No live project promotion or rollback action was taken. |
| SQLite and services | Functional for tested local workflows | AppTest verified taxonomy, inference, feedback, review, and manifest records in an isolated temporary database. The live project's real DB was inspected read-only and left unchanged. |

Frontend audit also caught and corrected low-contrast default-theme text and controls that were not visible against the light research palette. Remaining limitations are missing source/licence provenance, reviewed taxonomy/OOD data, approved consent/review and promotion policies, near-duplicate grouping, and authenticated reviewer access.

## 9. Contradictions, risks, missing requirements, and approvals

### Specification tensions to resolve

- It asks for distinct `uncertain_prediction` and `unsupported_or_out_of_distribution` outcomes while also noting that a simple softmax threshold is not reliable open-set recognition. A threshold can justify abstention, not confidently diagnose the reason; UI semantics should be cautious unless a separately validated method is adopted.
- The target split is 70/15/15 while later versions should keep a stable evaluation set. Treat this as the initial supported-class split; preserve it thereafter and version any protocol change. Keep OOD development and final OOD evaluation sets separately versioned; final OOD data must never be used to tune thresholds or methods.
- It calls for feedback `corroborated` status, but the local MVP cannot authenticate independent reviewers. Keep corroboration out of the MVP; consider it later only with authenticated independent reviewers and duplicate/related-submission checks.
- It permits image withdrawal while artifacts may contain learned influence from that image. Remove withdrawn items from future manifests, retire affected model descendants, and if retraining to remove influence, start from a clean base rather than any contaminated fine-tuned checkpoint.
- It requests a candidate model's evaluation and evidence-based promotion but does not define numeric acceptance/regression criteria. They must be set before promotion, not improvised after viewing candidate results.

### Risks and missing requirements

- The Catalogue of Life candidate is selected as the direction, but no release or cross-release identity policy has been validated against actual source data. Provider identifiers may change or taxonomic concepts may split/merge; release-scoped immutable records and explicit reviewed crosswalks are required.
- Source feasibility audit (2026-11-04): Oxford-IIIT Pets is a candidate for domestic cats and dogs; its official page describes 37 breed categories with roughly 200 images per breed and a CC BY-SA 4.0 licence, while noting that image copyrights remain with their original owners. The iNaturalist 2021 `train_mini` annotation archive was checksum-verified (MD5 `395a35be3651d86dc3b0d365b8ea5f92`) without downloading image files. Its metadata contains 50 examples each for *Acinonyx jubatus*, *Canis lupus*, *Felis catus*, *Panthera leo*, *Panthera onca*, and *Panthera pardus*; it has no exact category for *Canis lupus familiaris* or *Panthera tigris*. These mini-set counts are below the 500–1,000 per-class planning target. The iNaturalist terms restrict the dataset to non-commercial educational/research use and prohibit redistributing images; per-image metadata includes mixed licences, including NonCommercial and NoDerivatives variants. Sources: [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [iNaturalist 2021 competition dataset and terms](https://github.com/visipedia/inat_comp/tree/master/2021).
- Open Images V6 was audited using only its official class-description CSV; no image or annotation archive was downloaded. Its class-description metadata includes broad labels for Cat, Dog, Tiger, Leopard, Lion, Cheetah, Jaguar (Animal), and Wolf, but no exact Gray wolf class, so several labels may not establish the required species-level ground truth. Its documentation provides human-verified image-level labels and per-image source, author, and licence metadata. Per-class counts and the suitability of those broad labels remain unverified: the official human-verified training-label CSV is 2,474,671,021 bytes, so it was not downloaded or processed. Source: [Open Images V6/V7 download documentation](https://storage.googleapis.com/openimages/web/download.html).
- The official LILA WCS Camera Traps species-count CSV (14,954 bytes; metadata only) lists 1,677 *Canis familiaris*, 822 *Canis lupus*, 9,435 *Panthera onca*, 470 *P. pardus*, 429 *P. leo*, 321 *P. tigris*, and 54 *Acinonyx jubatus* labeled images; it has no *Felis catus* entry. The dataset page describes about 1.4M images across 675 species, about 50% empty images, sequence-level label caveats, and the Community Data License Agreement (CDLA) Permissive 1.0. This is a potential source for seven classes, but four wild classes fall below the 500–1,000 per-class planning target, and *Canis familiaris* still needs an approved taxonomy mapping. No animal images were downloaded. Sources: [WCS Camera Traps](https://lila.science/datasets/wcscameratraps) and [official species-count CSV](https://storage.googleapis.com/public-datasets-lila/wcs/wcs_specieslist.csv).
- The official LILA Snapshot Serengeti species-count CSV (894 bytes; metadata only) lists 6,806 cheetah, 650 leopard, and 22,166 lion-labeled images (female, male, and cub categories combined). Its page notes that labels are reliable at sequence level rather than necessarily for each frame, and the dataset is under CDLA Permissive 1.0. It is a supplementary candidate for three African classes, not a source for the other five target classes. No animal images were downloaded. Sources: [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti) and [official species-count CSV](https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2.1.species_list.csv).
- Oxford-IIIT Pets remains a candidate for domestic cats and dogs, but its official page gives approximately 200 images per breed, not verified per-species totals, and states that image copyrights remain with the original owners despite the dataset's CC BY-SA 4.0 availability. The source research above predates the later user-supplied local data and does not establish that local data's provenance or licence.
- Follow-up actual dataset inspection (2026-11-05): the user supplied local files in `data/raw/`, and later supplied the source mapping: Domestic Cat/Dog from Oxford-IIIT, Jaguar from Kaggle `iluvchicken/cheetah-jaguar-and-tiger` (`jaguar_train_resized`), and the other classes from images.cv pages (wolf, red wolf, white wolf, leopard, cheetah, tiger, lion). Those source pages and their stated counts/licences are summarized in `reports/FINAL_REPORT.md`. The manifest scan found 2,371 Domestic Cat, 4,978 Domestic Dog, 1,375 Cheetah, 900 Jaguar, 1,345 Leopard, 2,007 Lion, 2,156 Tiger, and initially 58 Gray wolf images (15,190 valid total), with zero corrupt supported-format images and four non-image files excluded. The refreshed folder contained 1,346 Gray wolf images, giving 16,478 total. Exact duplicate groups found: 0. The source mapping is user-reported and page-level; raw images have no per-file provenance metadata, some linked licences are unknown/restrictive, and page totals do not match several local counts. Do not claim file-level provenance, label verification, or rights clearance without reconciling local files against source archives and rights terms.
- Actual licensed images, especially balanced wild-animal classes and unsupported OOD examples, are unknown. The image-count target is explicitly aspirational.
- The accepted domestic-dog taxon and gray-wolf mapping depend on the selected backbone and need verified provider IDs.
- No review SOP specifies evidence for a verified taxon label or how disagreement/escalation works. Designated project reviewers are the intended authority, but MVP reviewer identity is not authenticated.
- Exact consent wording and retention duration remain unspecified; retention consent and potential training-use consent will be separate.
- Local reviewer access is a trusted-machine/workspace boundary only: anyone with access to the running local app, its database, or files may be able to view/change review data. A manually entered reviewer name is an attribution note, not authentication, authorization, or non-repudiation. Do not treat hidden pages or UI-only passwords as multi-user security.
- Name-resolution fuzzy thresholds, ambiguous-result policy and evaluation examples need empirical tests; typo suggestions must never be silently accepted.
- Per-class criteria for adding a category and quantitative model-promotion/regression limits are open. Avoid arbitrary universal image counts or fabricated thresholds.
- SQLite migration/versioning, backup/recovery, deletion consistency, and a way to identify all model descendants affected by withdrawal need acceptance criteria even for a local MVP.
- Dataset source licensing may constrain storage, derivative use, or public distribution; source-by-source review is required.
- A stable public-facing model version identifier and artifact retention policy are not specified.
- The implementation now covers the executable portions of Phases 1–11. The unresolved gates above prevent claims of rights-cleared data, populated taxonomy, OOD performance, or an active/promoted production model. No further training or promotion is performed without the relevant input and policy.

### Remaining decisions requiring human approval

Confirmed choices are recorded above; the remaining decisions are:

1. **Taxonomy snapshot and crosswalk procedure:** Which Catalogue of Life release/licence is usable, how often to import updates, and what evidence/authority is required to approve cross-release `same/split/merge/uncertain` relationships. Historical records remain immutable regardless.
2. **Consent and retention policy:** Exact user-facing wording, retention duration, deletion procedure, and how to identify/retire all affected model descendants. Separate retention and training-use consent is confirmed.
3. **Review SOP:** What evidence is sufficient for a designated project reviewer to mark a label verified, how to resolve disagreements, and whether project constraints require actual authentication (which is outside the proposed local MVP).
4. **Dataset feasibility:** Which licensed datasets and actual per-class counts are available for Domestic Cat, Domestic Dog, Lion, Tiger, Leopard, Gray wolf, Cheetah, and Jaguar, and whether separate OOD development and final evaluation sets can be built without duplicate/source leakage.
5. **Abstention evaluation protocol:** Which baseline score/rule to evaluate and what validation/OOD-development criteria justify user-facing uncertainty; final OOD evaluation remains strictly holdout-only.
6. **Class-expansion and promotion gates:** Evidence-based per-class data/diversity requirements and numerical model regression/promotion limits, set before evaluating candidates.
7. **Local access boundary:** Confirm that trusted-machine access and unauthenticated reviewer attribution are acceptable for the MVP; otherwise approve adding authentication and its scope.
8. **Operational policies:** SQLite backup/recovery, model artifact retention, withdrawal audit details, and team responsibilities.

The local layered architecture, SQLite with migrations, Catalogue of Life as candidate, separate retention/training consent, designated project reviewers, cautious abstention, grouped stratified 70/15/15 supported-class split, separate OOD development/final sets, and withdrawal-driven artifact retirement/retraining are confirmed design directions. No decision relaxes the prohibition on automatic retraining after user feedback.

## Assumptions deliberately not made

- No taxonomy data source, image source, image count, licence clearance, accepted taxon identifier, model result, accuracy, or working implementation is asserted.
- No taxonomy presence or feedback label is treated as model support or ground truth.
- No feedback is automatically verified, curated, added to training, or used to trigger retraining. Corrections never trigger automatic training; training requires a separate explicit action.
- No model score is described as calibrated probability without a measured calibration procedure.
- No out-of-scope result is called a recognized new species.
