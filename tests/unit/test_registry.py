from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.models.architectures import build_model
from wildlife_ai.registry.service import (
    create_promotion_policy,
    list_candidate_models,
    promote_candidate,
    register_candidate,
    reject_candidate,
    rollback_model,
)
from wildlife_ai.storage.database import connect_database
from wildlife_ai.training.engine import _save_checkpoint

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
