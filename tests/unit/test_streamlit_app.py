from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from wildlife_ai.automatic_learning import INITIAL_CHECKPOINT, INITIAL_MODEL_VERSION
from wildlife_ai.config.settings import SUPPORTED_CLASSES, ProjectSettings
from wildlife_ai.storage.database import connect_database
from wildlife_ai.training.engine import _save_checkpoint


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (48, 36), color)
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def _prepare_application_project(root: Path) -> tuple[str, bytes]:
    settings = ProjectSettings.from_repo_root(root)
    settings.ensure_directories()
    manifest_path = settings.manifests_dir / "dataset_manifest.csv"
    fields = [
        "image_id",
        "relative_path",
        "label",
        "sha256",
        "duplicate_group",
        "image_status",
        "split",
        "source",
        "license",
        "attribution",
        "feedback_id",
    ]
    base_rows = []
    for split, color in (
        ("train", (21, 33, 45)),
        ("validation", (55, 67, 79)),
        ("test", (89, 101, 113)),
    ):
        image_bytes = _image_bytes(color)
        relative = f"Cat/{split}.jpg"
        image_path = settings.raw_data_dir / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        base_rows.append(
            {
                "image_id": f"base-{split}",
                "relative_path": relative,
                "label": SUPPORTED_CLASSES[0],
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "duplicate_group": hashlib.sha256(image_bytes).hexdigest(),
                "image_status": "valid",
                "split": split,
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(base_rows)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    from wildlife_ai.models.architectures import build_model

    network = build_model("resnet18", pretrained=False)
    with torch.no_grad():
        network.fc.bias[0] = 12.0
    version = INITIAL_MODEL_VERSION
    checkpoint = root / INITIAL_CHECKPOINT
    metadata = {
        "model_version": version,
        "architecture": "resnet18",
        "class_names": list(SUPPORTED_CLASSES),
        "class_taxonomy_ids": {},
        "input_size": [224, 224],
        "normalization": "imagenet",
        "dataset_manifest_sha256": manifest_hash,
    }
    _save_checkpoint(checkpoint, network, metadata)
    report = {
        "model_version": version,
        "architecture": "resnet18",
        "split": "validation",
        "dataset_manifest_sha256": manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "sample_count": 1,
        "class_names": list(SUPPORTED_CLASSES),
        "metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "sample_count": 1,
        },
        "per_class": [
            {"label": label, "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
            for label in SUPPORTED_CLASSES
        ],
    }
    report_path = (
        root
        / "reports"
        / "model_information"
        / f"{INITIAL_MODEL_VERSION}-validation.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return version, _image_bytes((125, 80, 35))


def _run_app(root: Path, monkeypatch, upload: bytes) -> AppTest:
    monkeypatch.setenv("WILDLIFE_AI_ROOT", str(root))
    project_root = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(project_root / "app.py"))
    app.run(timeout=60)
    assert not app.exception, [item.message for item in app.exception]
    assert app.file_uploader, [item.value for item in app.error]
    app.file_uploader[0].upload("wildlife-demo.jpg", upload, "image/jpeg").run(
        timeout=90
    )
    assert not app.exception, [item.message for item in app.exception]
    return app


def test_single_page_classification_and_correct_feedback_without_image_retention(
    tmp_path: Path,
    monkeypatch,
) -> None:
    version, upload = _prepare_application_project(tmp_path)
    app = _run_app(tmp_path, monkeypatch, upload)

    assert not app.radio
    assert any(item.value == "Domestic Cat" for item in app.subheader)
    assert any("Uncalibrated model score" in item.value for item in app.caption)
    assert any(version in item.value for item in app.caption)
    assert {item.label for item in app.button} >= {"Correct", "Incorrect"}

    next(item for item in app.button if item.label == "Correct").click().run(timeout=45)
    assert not app.exception, [item.message for item in app.exception]
    assert any("feedback has been saved" in item.value for item in app.success)

    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        feedback = connection.execute(
            """SELECT response, predicted_label, corrected_label, model_version_id,
                      retention_consent, training_use_consent, image_sha256
               FROM feedback"""
        ).fetchone()
        image_count = connection.execute("SELECT COUNT(*) FROM feedback_images").fetchone()[0]
        selected = connection.execute(
            "SELECT model_version_id FROM application_model_selection WHERE singleton = 1"
        ).fetchone()
    finally:
        connection.close()
    assert tuple(feedback[:6]) == (
        "yes",
        "Domestic Cat",
        "Domestic Cat",
        version,
        0,
        0,
    )
    assert feedback["image_sha256"] == hashlib.sha256(upload).hexdigest()
    assert image_count == 0
    assert selected["model_version_id"] == version


def test_incorrect_feedback_maps_gray_wolf_and_separates_consents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    version, upload = _prepare_application_project(tmp_path)
    app = _run_app(tmp_path, monkeypatch, upload)
    next(item for item in app.button if item.label == "Incorrect").click().run(timeout=45)
    correction = next(
        item for item in app.selectbox if item.label == "Choose the correct animal"
    )
    assert list(correction.options) == [
        *[("Gray Wolf" if label == "Gray wolf" else label) for label in SUPPORTED_CLASSES],
        "Other / none of these",
    ]
    correction.set_value("Gray Wolf").run(timeout=45)
    next(item for item in app.checkbox if "sanitized copy" in item.label).check().run(
        timeout=45
    )
    next(item for item in app.button if item.label == "Save correction").click().run(
        timeout=45
    )
    assert not app.exception, [item.message for item in app.exception]
    assert app.success, {
        "success": [item.value for item in app.success],
        "error": [item.value for item in app.error],
        "buttons": [item.label for item in app.button],
        "mode": app.session_state.get("wildlife_feedback_mode"),
    }

    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        feedback = connection.execute(
            """SELECT response, predicted_label, corrected_label, model_version_id,
                      retention_consent, training_use_consent
               FROM feedback"""
        ).fetchone()
        retained = connection.execute(
            "SELECT COUNT(*) FROM feedback_images"
        ).fetchone()[0]
        batch_count = connection.execute(
            "SELECT COUNT(*) FROM automatic_training_batches"
        ).fetchone()[0]
    finally:
        connection.close()
    assert tuple(feedback) == ("no", "Domestic Cat", "Gray wolf", version, 1, 0)
    assert retained == 1
    assert batch_count == 0


def test_other_feedback_is_saved_but_has_no_training_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _prepare_application_project(tmp_path)
    app = _run_app(tmp_path, monkeypatch, _image_bytes((125, 80, 35)))
    next(item for item in app.button if item.label == "Incorrect").click().run(timeout=45)
    next(item for item in app.selectbox if item.label == "Choose the correct animal").set_value(
        "Other / none of these"
    ).run(timeout=45)
    next(item for item in app.button if item.label == "Save correction").click().run(
        timeout=45
    )
    assert not app.exception, [item.message for item in app.exception]

    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        feedback = connection.execute(
            "SELECT response, corrected_label, training_use_consent FROM feedback"
        ).fetchone()
    finally:
        connection.close()
    assert tuple(feedback) == ("unsupported", "", 0)


def test_consented_correction_reports_automatic_batch_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _prepare_application_project(tmp_path)
    app = _run_app(tmp_path, monkeypatch, _image_bytes((125, 80, 35)))
    next(item for item in app.button if item.label == "Incorrect").click().run(timeout=45)
    next(item for item in app.selectbox if item.label == "Choose the correct animal").set_value(
        "Gray Wolf"
    ).run(timeout=45)
    next(item for item in app.checkbox if "sanitized copy" in item.label).check().run(
        timeout=45
    )
    next(item for item in app.checkbox if "separately consent" in item.label).check().run(
        timeout=45
    )
    next(item for item in app.button if item.label == "Save correction").click().run(
        timeout=45
    )
    assert not app.exception, [item.message for item in app.exception]
    assert any("1 of 5 consented, unique examples" in item.value for item in app.info)

    connection = connect_database(tmp_path / "wildlife_ai.db")
    try:
        feedback = connection.execute(
            "SELECT retention_consent, training_use_consent FROM feedback"
        ).fetchone()
    finally:
        connection.close()
    assert tuple(feedback) == (1, 1)
