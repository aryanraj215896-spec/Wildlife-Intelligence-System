from __future__ import annotations

import csv
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
import torch
from streamlit.testing.v1 import AppTest

from wildlife_ai.config.settings import SUPPORTED_CLASSES, ProjectSettings
from wildlife_ai.models.architectures import build_model
from wildlife_ai.registry.service import (
    create_promotion_policy,
    register_candidate,
)
from wildlife_ai.storage.database import connect_database
from wildlife_ai.taxonomy.importer import import_taxonomy_csv
from wildlife_ai.training.engine import _save_checkpoint


def _image_bytes(color: tuple[int, int, int] = (125, 80, 35)) -> bytes:
    image = Image.new("RGB", (48, 36), color)
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def _prepare_application_project(
    root: Path,
    *,
    register_model: bool = True,
    activate_candidate: bool = True,
) -> tuple[str, bytes]:
    settings = ProjectSettings.from_repo_root(root)
    settings.ensure_directories()
    base_manifest = settings.manifests_dir / "dataset_manifest.csv"
    base_fields = [
        "image_id",
        "relative_path",
        "label",
        "sha256",
        "duplicate_group",
        "group_id",
        "image_status",
        "rejection_reason",
        "width",
        "height",
        "split",
        "source",
        "source_image_id",
        "license",
        "attribution",
        "collection_date",
        "canonical_taxon_id",
        "review_status",
        "quality_notes",
        "feedback_id",
    ]
    with base_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=base_fields)
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "test-heldout",
                "relative_path": "cat/test-heldout.jpg",
                "label": "Domestic Cat",
                "sha256": "f" * 64,
                "image_status": "valid",
                "split": "test",
                "review_status": "verified",
            }
        )
    manifest_hash = hashlib.sha256(base_manifest.read_bytes()).hexdigest()

    taxonomy_csv = settings.taxonomy_dir / "synthetic-test-taxonomy.csv"
    taxonomy_fields = [
        "source_taxon_id",
        "accepted_scientific_name",
        "accepted_common_name",
        "rank",
        "taxonomic_status",
    ]
    with taxonomy_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=taxonomy_fields)
        writer.writeheader()
        for index, label in enumerate(SUPPORTED_CLASSES):
            writer.writerow(
                {
                    "source_taxon_id": f"synthetic-{index}",
                    "accepted_scientific_name": f"Synthetic species {index}",
                    "accepted_common_name": label,
                    "rank": "species",
                    "taxonomic_status": "accepted",
                }
            )
    import_taxonomy_csv(
        taxonomy_csv,
        settings.database_path,
        source_namespace="synthetic-test-only",
        release="fixture-1",
        license="test fixture only",
        attribution="test fixture only",
    )
    connection = connect_database(settings.database_path)
    try:
        taxon_ids = {
            row["accepted_common_name"]: row["taxon_version_id"]
            for row in connection.execute(
                "SELECT taxon_version_id, accepted_common_name FROM taxa"
            )
        }
    finally:
        connection.close()

    architecture = "resnet18"
    model = build_model(architecture)
    with torch.no_grad():
        model.fc.bias[0] = 12.0
    version = "streamlit-integration-candidate"
    checkpoint = settings.root_dir / "models" / "runs" / version / "best-resnet18.pt"
    metadata = {
        "model_version": version,
        "architecture": architecture,
        "class_names": list(SUPPORTED_CLASSES),
        "class_taxonomy_ids": taxon_ids,
        "input_size": [224, 224],
        "normalization": "imagenet",
        "dataset_manifest_sha256": manifest_hash,
    }
    _save_checkpoint(checkpoint, model, metadata)

    evaluation_path = (
        settings.root_dir
        / "reports"
        / "metrics"
        / "resnet18-test-metrics.json"
    )
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation = {
        "model_version": version,
        "architecture": architecture,
        "split": "test",
        "dataset_manifest_sha256": manifest_hash,
        "metrics": {"accuracy": 1.0, "macro_f1": 1.0},
        "per_class": [
            {"label": label, "recall": 1.0, "support": 1}
            for label in SUPPORTED_CLASSES
        ],
        "inference": {"images_per_second": 10.0},
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "confusion_matrix": {"labels": list(SUPPORTED_CLASSES)},
    }
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    candidate = None
    if register_model:
        candidate = register_candidate(
            settings.root_dir,
            settings.database_path,
            checkpoint,
            evaluation_path=evaluation_path,
        )
    policy_id = create_promotion_policy(
        settings.database_path,
        {
            "minimum_test_accuracy": 0.9,
            "minimum_test_macro_f1": 0.9,
            "minimum_per_class_recall": 0.9,
            "maximum_accuracy_drop": 0.1,
            "maximum_macro_f1_drop": 0.1,
            "maximum_per_class_recall_drop": 0.1,
            "maximum_inference_ms_per_image": 200.0,
            "maximum_checkpoint_size_bytes": 100_000_000,
        },
        approved_by="integration test fixture",
        approval_reference="synthetic test only",
    )
    if activate_candidate and candidate is not None:
        promote_candidate(
            settings.root_dir,
            settings.database_path,
            str(candidate["model_version_id"]),
            policy_id=policy_id,
            actor_reference="integration test fixture",
            rationale="Enable production-page UI test in isolated temporary project only",
        )
    return version, _image_bytes()


def _select_workspace_page(app: AppTest, page: str) -> None:
    next(item for item in app.radio if item.label == "Workspace").set_value(page).run(
        timeout=45
    )
    assert not app.exception, [item.message for item in app.exception]


def test_streamlit_pages_and_feedback_review_curation_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    version, upload = _prepare_application_project(
        tmp_path,
        register_model=False,
        activate_candidate=False,
    )
    monkeypatch.setenv("WILDLIFE_AI_ROOT", str(tmp_path))
    project_root = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(project_root / "app.py"))
    app.session_state["inference-mode"] = "Production (active model)"
    app.run(timeout=45)
    assert not app.exception, [item.message for item in app.exception]
    assert next(item for item in app.radio if item.label == "Inference mode").value == (
        "Local development/demo (candidate)"
    )
    assert app.file_uploader
    assert any(item.label == "Classify image" for item in app.button)

    _select_workspace_page(app, "Models")
    assert any(item.label == "Local checkpoint" for item in app.selectbox)
    next(
        item
        for item in app.button
        if item.label == "Validate and register candidate"
    ).click().run(timeout=45)
    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM model_versions WHERE lifecycle_state = 'candidate'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM promotion_policies"
        ).fetchone()[0] == 1
    finally:
        connection.close()
    _select_workspace_page(app, "Classify")
    assert any(item.label == "Candidate model" for item in app.selectbox)
    app.file_uploader[0].upload("wildlife-demo.jpg", upload, "image/jpeg").run(
        timeout=30
    )
    next(item for item in app.button if item.label == "Classify image").click().run(
        timeout=90
    )
    assert not app.exception, [item.message for item in app.exception]
    assert any(item.value == "Domestic Cat" for item in app.subheader)
    assert any("EXPERIMENTAL LOCAL DEMO" in item.value for item in app.warning)
    assert not any(item.label in {"Yes", "No", "Not sure"} for item in app.button)
    next(
        item for item in app.button if item.label == "View experimental Grad-CAM"
    ).click().run(timeout=90)
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]
    assert any(
        "Experimental Grad-CAM on the processed model input" in item.value
        for item in app.caption
    )

    _select_workspace_page(app, "Models")
    next(
        item
        for item in app.text_input
        if item.label == "Decision-maker reference"
    ).set_value("integration reviewer")
    next(
        item
        for item in app.text_area
        if item.label == "Promotion rationale"
    ).set_value("Synthetic fixture only; exercise the registry UI")
    next(
        item
        for item in app.button
        if item.label == "Evaluate criteria and promote"
    ).click().run(timeout=45)
    assert not app.error, [item.value for item in app.error]
    assert any(
        version in item.value and "active model" in item.value.lower()
        for item in app.success
    )
    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        active_version = connection.execute(
            "SELECT model_version_id FROM active_model WHERE singleton = 1"
        ).fetchone()["model_version_id"]
    finally:
        connection.close()
    assert active_version == version
    _select_workspace_page(app, "Classify")
    app.run(timeout=45)
    next(
        item
        for item in app.radio
        if item.label == "Inference mode"
    ).set_value("Production (active model)").run(timeout=45)

    _select_workspace_page(app, "Taxonomy")
    assert app.metric[0].value == "8"
    next(item for item in app.text_input if item.label == "Animal name").set_value(
        "Domestic Cat"
    ).run(timeout=45)
    assert any("Exact and fuzzy candidates are suggestions" in item.value for item in app.caption)
    next(item for item in app.checkbox if item.label == "I confirm this is the animal I intended").set_value(
        True
    ).run(timeout=45)
    assert any("Confirmed selection" in item.value for item in app.success)

    _select_workspace_page(app, "Classify")
    app.file_uploader[0].upload("wildlife.jpg", upload, "image/jpeg").run(timeout=30)
    next(item for item in app.button if item.label == "Classify image").click().run(
        timeout=90
    )
    assert not app.exception, [item.message for item in app.exception]
    assert any(item.value == "Domestic Cat" for item in app.subheader)
    assert any(f"`{version}`" in item.value for item in app.markdown)

    next(item for item in app.button if item.label == "No").click().run(timeout=45)
    next(item for item in app.text_input if item.label == "Animal name").set_value(
        "Domestic Cat"
    ).run(timeout=45)
    next(item for item in app.checkbox if item.label == "I confirm this is the animal I intended").set_value(
        True
    ).run(timeout=45)
    next(
        item
        for item in app.checkbox
        if item.label == "I consent to retain this sanitized image for project review."
    ).set_value(True).run(timeout=45)
    next(
        item
        for item in app.checkbox
        if item.label == "I separately consent to use the retained image in a future training dataset."
    ).set_value(True).run(timeout=45)
    next(
        item
        for item in app.text_input
        if item.label == "Image source or permitted source reference"
    ).set_value("synthetic integration fixture").run(timeout=45)
    next(item for item in app.text_input if item.label == "Image licence or rights basis").set_value(
        "test-only fixture"
    ).run(timeout=45)
    next(item for item in app.text_input if item.label == "Required attribution, if any").set_value(
        "test-only"
    ).run(timeout=45)
    next(item for item in app.button if item.label == "Submit feedback").click().run(
        timeout=45
    )
    assert any("Feedback saved as pending review" in item.value for item in app.success)

    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        feedback_id = connection.execute(
            "SELECT feedback_id FROM feedback"
        ).fetchone()["feedback_id"]
    finally:
        connection.close()

    _select_workspace_page(app, "Review")
    assert app.selectbox
    assert any("Pending" in item.label for item in app.metric)
    next(
        item
        for item in app.text_input
        if item.label == "Reviewer reference (self-declared)"
    ).set_value("integration reviewer")
    next(
        item
        for item in app.checkbox
        if item.label == "I reviewed image quality and suitability"
    ).set_value(True)
    next(
        item
        for item in app.checkbox
        if item.label == "I reviewed the source and licence and confirm they permit this use"
    ).set_value(True)
    next(item for item in app.button if item.label == "Save review decision").click().run(
        timeout=45
    )
    assert not app.exception, [item.message for item in app.exception]
    eligible = next(item for item in app.multiselect if item.label == "Eligible feedback")
    eligible.set_value([feedback_id]).run(timeout=45)
    next(
        item
        for item in app.button
        if item.label == "Export selected curation manifest"
    ).click().run(timeout=45)
    assert any("Created 1 item manifest" in item.value for item in app.success)
    assert app.get("download_button")

    _select_workspace_page(app, "Models")
    assert any(version in item.value for item in app.success)
    assert any(item.label == "Local checkpoint" for item in app.selectbox)
    _select_workspace_page(app, "Classify")
    assert not app.exception

    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        reviewed = connection.execute(
            "SELECT verification_status, retention_consent, training_use_consent FROM feedback WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
        dataset_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_manifests WHERE purpose = 'curated_feedback'"
        ).fetchone()[0]
        active = connection.execute(
            "SELECT model_version_id FROM active_model WHERE singleton = 1"
        ).fetchone()["model_version_id"]
    finally:
        connection.close()
    assert tuple(reviewed) == ("verified", 1, 1)
    assert dataset_count == 1
    assert active == version
