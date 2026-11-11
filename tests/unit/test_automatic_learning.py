from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image

from wildlife_ai.automatic_learning import (
    AUTOMATIC_BATCH_SIZE,
    INITIAL_MODEL_VERSION,
    automatic_update_if_ready,
    feedback_eligibility,
    get_selected_model,
    initialize_selected_model,
)
from wildlife_ai.config.settings import SUPPORTED_CLASSES, ProjectSettings
from wildlife_ai.data.loading import ManifestImageDataset
from wildlife_ai.feedback.service import submit_feedback
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.storage.database import connect_database
from wildlife_ai.training.engine import TrainingConfig, _save_checkpoint


def _image_bytes(seed: int) -> bytes:
    image = Image.new("RGB", (48, 36))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel(
                (x, y),
                ((seed * 31 + x) % 256, (seed * 47 + y) % 256, (seed * 19) % 256),
            )
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def _make_project(root: Path) -> tuple[ProjectSettings, Path, Path]:
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
        "feedback_id",
    ]
    base_rows = []
    for split, seed in (("train", 11), ("validation", 12), ("test", 13)):
        image_bytes = _image_bytes(seed)
        relative = f"Cat/{split}.jpg"
        image_path = settings.raw_data_dir / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        digest = hashlib.sha256(image_bytes).hexdigest()
        base_rows.append(
            {
                "image_id": split,
                "relative_path": relative,
                "label": SUPPORTED_CLASSES[0],
                "sha256": digest,
                "duplicate_group": digest,
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

    model = build_model("resnet18", pretrained=False)
    checkpoint_path = root / "models" / "runs" / INITIAL_MODEL_VERSION / "best-resnet18.pt"
    metadata = {
        "model_version": INITIAL_MODEL_VERSION,
        "architecture": "resnet18",
        "class_names": list(SUPPORTED_CLASSES),
        "class_taxonomy_ids": {},
        "input_size": [224, 224],
        "normalization": "imagenet",
        "dataset_manifest_sha256": manifest_hash,
    }
    _save_checkpoint(checkpoint_path, model, metadata)
    report_path = root / "reports" / f"{INITIAL_MODEL_VERSION}-validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_version": INITIAL_MODEL_VERSION,
        "architecture": "resnet18",
        "split": "validation",
        "dataset_manifest_sha256": manifest_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
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
    report_path.write_text(json.dumps(report), encoding="utf-8")
    initialize_selected_model(
        root,
        settings.database_path,
        checkpoint_path,
        report_path,
    )
    return settings, manifest_path, checkpoint_path


def _submit(
    settings: ProjectSettings,
    image: bytes,
    *,
    retain: bool = True,
    train: bool = True,
) -> str:
    return submit_feedback(
        settings.database_path,
        settings.feedback_dir,
        response="no",
        predicted_label="Domestic Cat",
        corrected_label="Gray wolf",
        model_version_id=INITIAL_MODEL_VERSION,
        model_score=0.8,
        image_bytes=image,
        retention_consent=retain,
        training_use_consent=train,
    )


def test_automatic_feedback_needs_both_consents_and_excludes_duplicates_and_test_leaks(
    tmp_path: Path,
) -> None:
    settings, manifest_path, _ = _make_project(tmp_path)
    _submit(settings, _image_bytes(21), retain=False, train=False)
    _submit(settings, _image_bytes(22), retain=True, train=False)
    duplicate_image = _image_bytes(23)
    _submit(settings, duplicate_image)
    _submit(settings, duplicate_image)

    test_image = (settings.raw_data_dir / "Cat" / "test.jpg").read_bytes()
    _submit(settings, test_image)
    eligible, reasons = feedback_eligibility(
        settings.root_dir,
        settings.database_path,
        manifest_path,
    )
    assert len(eligible) == 1, reasons
    assert reasons["image_retention_not_currently_consented"] >= 1
    assert reasons["training_use_not_currently_consented"] >= 1
    assert reasons["duplicate_feedback_image"] >= 1
    assert reasons["image_overlaps_frozen_dataset"] >= 1
    selected = get_selected_model(settings.root_dir, settings.database_path)
    assert selected["model_version_id"] == INITIAL_MODEL_VERSION


def test_batch_training_keeps_frozen_splits_and_promotes_only_passing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, manifest_path, initial_checkpoint = _make_project(tmp_path)
    for seed in range(30, 30 + AUTOMATIC_BATCH_SIZE):
        _submit(settings, _image_bytes(seed))

    training_configs: list[TrainingConfig] = []
    mode = {"regress": False}

    def fake_train(config: TrainingConfig) -> dict[str, object]:
        training_configs.append(config)
        rows = list(csv.DictReader(config.manifest_path.open(encoding="utf-8")))
        additions = [row for row in rows if row.get("feedback_id")]
        assert len(additions) == AUTOMATIC_BATCH_SIZE
        assert all(row["split"] == "train" for row in additions)
        assert sum(row["split"] == "test" for row in rows) == 1
        assert sum(row["split"] == "validation" for row in rows) == 1
        train_data = ManifestImageDataset(config.manifest_path, config.raw_root, "train")
        val_data = ManifestImageDataset(config.manifest_path, config.raw_root, "validation")
        test_data = ManifestImageDataset(config.manifest_path, config.raw_root, "test")
        assert len(train_data) == AUTOMATIC_BATCH_SIZE + 1
        assert len(val_data) == 1
        assert len(test_data) == 1
        assert sum(train_data.feedback_flags) == AUTOMATIC_BATCH_SIZE
        assert not any(val_data.feedback_flags)
        assert not any(test_data.feedback_flags)

        model, metadata = load_checkpoint(config.initial_checkpoint_path, device=torch.device("cpu"))
        metadata = dict(metadata)
        metadata["model_version"] = config.model_version
        metadata["dataset_manifest_sha256"] = hashlib.sha256(
            config.manifest_path.read_bytes()
        ).hexdigest()
        checkpoint = config.output_dir / "best-resnet18.pt"
        _save_checkpoint(checkpoint, model, metadata)
        return {"best_checkpoint": str(checkpoint)}

    def fake_evaluate(checkpoint, manifest, raw_root, output, **kwargs):
        _, metadata = load_checkpoint(checkpoint, device=torch.device("cpu"))
        metrics = {
            "accuracy": 0.8 if mode["regress"] else 1.0,
            "macro_f1": 0.8 if mode["regress"] else 1.0,
            "sample_count": 1,
        }
        report = {
            "model_version": metadata["model_version"],
            "architecture": "resnet18",
            "split": "validation",
            "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
            "class_names": list(SUPPORTED_CLASSES),
            "metrics": metrics,
            "sample_count": 1,
            "confusion_matrix": {"labels": list(SUPPORTED_CLASSES), "counts": []},
            "per_class": [
                {
                    "label": label,
                    "precision": metrics["accuracy"],
                    "recall": metrics["accuracy"],
                    "f1": metrics["macro_f1"],
                    "support": 1,
                }
                for label in SUPPORTED_CLASSES
            ],
        }
        report_path = Path(output) / "resnet18-validation-metrics.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    monkeypatch.setattr("wildlife_ai.automatic_learning.train_model", fake_train)
    monkeypatch.setattr("wildlife_ai.automatic_learning.evaluate_checkpoint", fake_evaluate)

    accepted = automatic_update_if_ready(
        settings.root_dir,
        settings.database_path,
        manifest_path,
    )
    assert accepted["status"] == "accepted"
    assert training_configs[0].initial_checkpoint_path == initial_checkpoint
    selected_after_acceptance = get_selected_model(settings.root_dir, settings.database_path)
    assert selected_after_acceptance["model_version_id"] == accepted["candidate_model_version_id"]

    for seed in range(40, 40 + AUTOMATIC_BATCH_SIZE):
        _submit(settings, _image_bytes(seed))
    mode["regress"] = True
    rejected = automatic_update_if_ready(
        settings.root_dir,
        settings.database_path,
        manifest_path,
    )
    assert rejected["status"] == "rejected"
    assert "accuracy_regressed_beyond_limit" in rejected["issues"]
    selected_after_rejection = get_selected_model(settings.root_dir, settings.database_path)
    assert selected_after_rejection["model_version_id"] == accepted["candidate_model_version_id"]

    connection = connect_database(settings.database_path)
    try:
        batches = connection.execute(
            "SELECT status FROM automatic_training_batches ORDER BY started_at"
        ).fetchall()
        test_feedback = connection.execute(
            """SELECT COUNT(*) FROM dataset_items di
               JOIN dataset_manifests dm USING (dataset_manifest_id)
               WHERE dm.purpose = 'automatic_feedback'
                 AND di.feedback_id IS NOT NULL AND di.split = 'test'"""
        ).fetchone()[0]
    finally:
        connection.close()
    assert [row["status"] for row in batches] == ["accepted", "rejected"]
    assert test_feedback == 0
