from __future__ import annotations

import json
import hashlib
import csv
import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.evaluation.metrics import evaluate_checkpoint
from wildlife_ai.inference import predict_image
from wildlife_ai.models.architectures import build_model
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.registry.service import (
    create_promotion_policy,
    list_candidate_models,
    promote_candidate,
    register_candidate,
    reject_candidate,
    rollback_model,
)
from wildlife_ai.storage.database import connect_database
from wildlife_ai.training.engine import TrainingConfig, _save_checkpoint, train_model

_MANIFEST_HASH = "b" * 64


def _write_candidate(
    project: Path,
    version: str,
    *,
    accuracy: float = 0.8,
    macro_f1: float = 0.8,
    recall: float = 0.8,
) -> tuple[Path, Path]:
    model_path = project / "models" / "runs" / version / "best-scratch_cnn.pt"
    model = build_model("scratch_cnn", pretrained=False)
    metadata = {
        "model_version": version,
        "architecture": "scratch_cnn",
        "class_names": list(SUPPORTED_CLASSES),
        "class_taxonomy_ids": {label: None for label in SUPPORTED_CLASSES},
        "input_size": [224, 224],
        "normalization": "scratch",
        "dataset_manifest_sha256": _MANIFEST_HASH,
    }
    _save_checkpoint(model_path, model, metadata)
    evaluation = {
        "model_version": version,
        "architecture": "scratch_cnn",
        "split": "test",
        "dataset_manifest_sha256": _MANIFEST_HASH,
        "metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
        "per_class": [
            {"label": label, "recall": recall, "support": 2}
            for label in SUPPORTED_CLASSES
        ],
        "inference": {"images_per_second": 20.0},
        "checkpoint_size_bytes": model_path.stat().st_size,
        "confusion_matrix": {"labels": list(SUPPORTED_CLASSES)},
    }
    evaluation_path = project / "reports" / f"{version}-test.json"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    return model_path, evaluation_path


def _policy(database_path: Path, **overrides: float | int) -> str:
    criteria: dict[str, float | int] = {
        "minimum_test_accuracy": 0.7,
        "minimum_test_macro_f1": 0.7,
        "minimum_per_class_recall": 0.7,
        "maximum_accuracy_drop": 0.1,
        "maximum_macro_f1_drop": 0.1,
        "maximum_per_class_recall_drop": 0.1,
        "maximum_inference_ms_per_image": 100.0,
        "maximum_checkpoint_size_bytes": 100_000_000,
    }
    criteria.update(overrides)
    return create_promotion_policy(
        database_path,
        criteria,
        approved_by="test reviewer",
        approval_reference="approved fixture",
    )


def test_registry_promotes_candidates_and_rolls_back_explicitly(tmp_path: Path) -> None:
    database_path = tmp_path / "wildlife_ai.db"
    first_path, first_evaluation = _write_candidate(tmp_path, "candidate-one")
    second_path, second_evaluation = _write_candidate(
        tmp_path, "candidate-two", accuracy=0.77, macro_f1=0.78, recall=0.77
    )
    first = register_candidate(
        tmp_path, database_path, first_path, evaluation_path=first_evaluation
    )
    second = register_candidate(
        tmp_path, database_path, second_path, evaluation_path=second_evaluation
    )
    assert first["lifecycle_state"] == second["lifecycle_state"] == "candidate"
    policy_id = _policy(database_path)

    promoted_first = promote_candidate(
        tmp_path,
        database_path,
        str(first["model_version_id"]),
        policy_id=policy_id,
        actor_reference="test reviewer",
        rationale="Passed the approved fixture policy",
    )
    assert promoted_first["lifecycle_state"] == "active"
    promoted_second = promote_candidate(
        tmp_path,
        database_path,
        str(second["model_version_id"]),
        policy_id=policy_id,
        actor_reference="test reviewer",
        rationale="Passed the approved fixture policy",
    )
    assert promoted_second["previous_model_version_id"] == first["model_version_id"]

    rollback = rollback_model(
        tmp_path,
        database_path,
        str(first["model_version_id"]),
        actor_reference="test reviewer",
        rationale="Restore prior registered model",
    )
    assert rollback["lifecycle_state"] == "active"
    connection = connect_database(database_path)
    try:
        active = connection.execute(
            "SELECT model_version_id FROM active_model WHERE singleton = 1"
        ).fetchone()[0]
        states = {
            row["model_version_id"]: row["lifecycle_state"]
            for row in connection.execute(
                "SELECT model_version_id, lifecycle_state FROM model_versions"
            )
        }
        event_count = connection.execute(
            "SELECT COUNT(*) FROM model_registry_events"
        ).fetchone()[0]
    finally:
        connection.close()
    assert active == first["model_version_id"]
    assert states[first["model_version_id"]] == "active"
    assert states[second["model_version_id"]] == "retired"
    assert event_count == 5


def test_promotion_requires_policy_matching_evaluation_and_rejects_failed_gates(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "wildlife_ai.db"
    checkpoint, evaluation = _write_candidate(
        tmp_path, "candidate-low", accuracy=0.6
    )
    candidate = register_candidate(
        tmp_path, database_path, checkpoint, evaluation_path=evaluation
    )
    policy_id = _policy(database_path)
    with pytest.raises(ValueError, match="minimum_test_accuracy"):
        promote_candidate(
            tmp_path,
            database_path,
            str(candidate["model_version_id"]),
            policy_id=policy_id,
            actor_reference="reviewer",
            rationale="Attempt promotion",
        )
    with pytest.raises(ValueError, match="missing"):
        create_promotion_policy(
            database_path,
            {"minimum_test_accuracy": 0.5},
            approved_by="reviewer",
            approval_reference="unapproved",
        )

    reject_candidate(
        database_path,
        str(candidate["model_version_id"]),
        actor_reference="reviewer",
        rationale="Failed minimum test accuracy",
    )
    connection = connect_database(database_path)
    try:
        state = connection.execute(
            "SELECT lifecycle_state FROM model_versions WHERE model_version_id = ?",
            (candidate["model_version_id"],),
        ).fetchone()[0]
    finally:
        connection.close()
    assert state == "rejected"


def test_registry_rejects_untrusted_paths_and_wrong_split(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "wildlife_ai.db"
    checkpoint, evaluation = _write_candidate(tmp_path, "candidate-validation")
    report = json.loads(evaluation.read_text(encoding="utf-8"))
    report["split"] = "validation"
    evaluation.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="test-split"):
        register_candidate(
            tmp_path, database_path, checkpoint, evaluation_path=evaluation
        )
    with pytest.raises(ValueError, match="inside the project directory"):
        register_candidate(
            tmp_path,
            database_path,
            Path(torch.__file__),
        )


def test_candidate_listing_does_not_promote_or_activate_model(tmp_path: Path) -> None:
    database_path = tmp_path / "wildlife_ai.db"
    checkpoint, evaluation = _write_candidate(tmp_path, "candidate-demo")
    candidate = register_candidate(
        tmp_path, database_path, checkpoint, evaluation_path=evaluation
    )

    candidates = list_candidate_models(database_path)

    assert candidates == [
        {
            "model_version_id": candidate["model_version_id"],
            "architecture": "scratch_cnn",
            "artifact_path": str(checkpoint.relative_to(tmp_path)).replace("\\", "/"),
            "artifact_sha256": candidate["artifact_sha256"],
        }
    ]
    connection = connect_database(database_path)
    try:
        active_count = connection.execute(
            "SELECT COUNT(*) FROM active_model"
        ).fetchone()[0]
        lifecycle_state = connection.execute(
            "SELECT lifecycle_state FROM model_versions WHERE model_version_id = ?",
            (candidate["model_version_id"],),
        ).fetchone()[0]
    finally:
        connection.close()
    assert active_count == 0
    assert lifecycle_state == "candidate"


def test_small_resnet_run_evaluates_and_registers_without_replacing_prior_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import wildlife_ai.training.engine as training_engine

    database_path = tmp_path / "wildlife_ai.db"
    previous_checkpoint, previous_evaluation = _write_candidate(
        tmp_path, "previous-academic-candidate"
    )
    previous = register_candidate(
        tmp_path,
        database_path,
        previous_checkpoint,
        evaluation_path=previous_evaluation,
    )
    previous_bytes = previous_checkpoint.read_bytes()

    raw_root = tmp_path / "data" / "raw"
    manifest_path = tmp_path / "data" / "manifests" / "tiny.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "validation", "test"):
        for index, label in enumerate(SUPPORTED_CLASSES):
            image_bytes = io.BytesIO()
            Image.new("RGB", (40, 40), (index * 23, len(split) * 13, 70)).save(
                image_bytes, format="JPEG"
            )
            relative_path = f"fixture-{index}/{split}.jpg"
            image_path = raw_root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes.getvalue())
            rows.append(
                {
                    "image_status": "valid",
                    "split": split,
                    "sha256": hashlib.sha256(image_bytes.getvalue()).hexdigest(),
                    "relative_path": relative_path,
                    "label": label,
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    actual_build_model = build_model

    def build_without_download(model_name, *, pretrained=False, freeze_backbone=False):
        return actual_build_model(
            model_name,
            pretrained=False,
            freeze_backbone=freeze_backbone,
        )

    monkeypatch.setattr(training_engine, "build_model", build_without_download)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training = train_model(
            TrainingConfig(
                model_name="resnet18",
                manifest_path=manifest_path,
                raw_root=raw_root,
                output_dir=tmp_path / "models" / "runs" / "tiny-feedback-run",
                epochs=1,
                fine_tune_epochs=0,
                batch_size=8,
                num_workers=0,
                device="cpu",
                mixed_precision=False,
                image_size=32,
            )
        )
    finally:
        torch.set_num_threads(previous_threads)

    checkpoint = Path(str(training["best_checkpoint"]))
    evaluation = evaluate_checkpoint(
        checkpoint,
        manifest_path,
        raw_root,
        tmp_path / "reports" / "tiny-run",
        split="test",
        batch_size=8,
        num_workers=0,
        device_name="cpu",
    )
    candidate = register_candidate(
        tmp_path,
        database_path,
        checkpoint,
        evaluation_path=str(evaluation["report_path"]),
    )
    loaded_model, metadata = load_checkpoint(checkpoint, device=torch.device("cpu"))
    sample_image = (raw_root / "fixture-0" / "test.jpg").read_bytes()
    prediction = predict_image(
        loaded_model,
        metadata,
        sample_image,
        device=torch.device("cpu"),
    )

    connection = connect_database(database_path)
    try:
        states = {
            row["model_version_id"]: row["lifecycle_state"]
            for row in connection.execute(
                "SELECT model_version_id, lifecycle_state FROM model_versions"
            )
        }
        new_metadata = json.loads(
            connection.execute(
                "SELECT metadata_json FROM model_versions WHERE model_version_id = ?",
                (candidate["model_version_id"],),
            ).fetchone()["metadata_json"]
        )
        active_count = connection.execute("SELECT COUNT(*) FROM active_model").fetchone()[0]
    finally:
        connection.close()

    assert previous_checkpoint.read_bytes() == previous_bytes
    assert checkpoint.is_file() and checkpoint != previous_checkpoint
    assert states[str(previous["model_version_id"])] == "candidate"
    assert states[str(candidate["model_version_id"])] == "candidate"
    assert active_count == 0
    assert new_metadata["test_evaluation"]["metrics"]["sample_count"] == 8
    assert prediction["model_version"] == metadata["model_version"]
    assert prediction["predicted_label"] in SUPPORTED_CLASSES


def test_promotion_rejects_checkpoint_hash_changes(tmp_path: Path) -> None:
    database_path = tmp_path / "wildlife_ai.db"
    checkpoint, evaluation = _write_candidate(tmp_path, "candidate-mutated")
    candidate = register_candidate(
        tmp_path, database_path, checkpoint, evaluation_path=evaluation
    )
    policy_id = _policy(database_path)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        promote_candidate(
            tmp_path,
            database_path,
            str(candidate["model_version_id"]),
            policy_id=policy_id,
            actor_reference="reviewer",
            rationale="Attempt promotion",
        )
