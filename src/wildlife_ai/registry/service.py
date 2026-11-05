from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

import torch

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.storage.database import connect_database

_POLICY_RATES = {
    "minimum_test_accuracy",
    "minimum_test_macro_f1",
    "minimum_per_class_recall",
    "maximum_accuracy_drop",
    "maximum_macro_f1_drop",
    "maximum_per_class_recall_drop",
}
_POLICY_LIMITS = {"maximum_inference_ms_per_image", "maximum_checkpoint_size_bytes"}
_POLICY_FIELDS = _POLICY_RATES | _POLICY_LIMITS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact_path(project_root: str | Path, artifact: str | Path) -> Path:
    root = Path(project_root).resolve()
    path = Path(artifact)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Model artifacts must be inside the project directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Model artifact not found: {resolved}")
    return resolved


def _read_test_evaluation(
    evaluation_path: str | Path | None,
    *,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if evaluation_path is None:
        return None
    path = Path(evaluation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Test evaluation report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("split") != "test":
        raise ValueError("Model registration requires a test-split evaluation report")
    if report.get("architecture") != checkpoint_metadata.get("architecture"):
        raise ValueError("Evaluation architecture does not match the checkpoint")
    if report.get("model_version") != checkpoint_metadata.get("model_version"):
        raise ValueError("Evaluation model version does not match the checkpoint")
    if report.get("dataset_manifest_sha256") != checkpoint_metadata.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("Evaluation and checkpoint use different dataset manifests")
    if report.get("confusion_matrix", {}).get("labels") != list(SUPPORTED_CLASSES):
        raise ValueError("Evaluation class order does not match configured classes")
    return report


def register_candidate(
    project_root: str | Path,
    database_path: str | Path,
    checkpoint_path: str | Path,
    *,
    evaluation_path: str | Path | None = None,
) -> dict[str, object]:
    artifact = _safe_artifact_path(project_root, checkpoint_path)
    model, metadata = load_checkpoint(artifact, device=torch.device("cpu"))
    del model
    digest = _sha256(artifact)
    evaluation = _read_test_evaluation(
        evaluation_path,
        checkpoint_metadata=metadata,
    )
    if evaluation is not None and evaluation.get("checkpoint_size_bytes") != artifact.stat().st_size:
        raise ValueError("Evaluation report checkpoint size does not match the candidate artifact")
    metadata = dict(metadata)
    if evaluation is not None:
        metadata["test_evaluation"] = evaluation
    model_version_id = str(metadata.get("model_version") or f"model-{digest[:16]}")
    relative_artifact = artifact.relative_to(Path(project_root).resolve()).as_posix()
    connection = connect_database(database_path)
    try:
        with connection:
            existing = connection.execute(
                """SELECT model_version_id, artifact_sha256
                   FROM model_versions WHERE model_version_id = ?""",
                (model_version_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError(f"Model version is already registered: {model_version_id}")
            connection.execute(
                """INSERT INTO model_versions
                   (model_version_id, architecture, lifecycle_state, artifact_path,
                    artifact_sha256, metadata_json)
                   VALUES (?, ?, 'candidate', ?, ?, ?)""",
                (
                    model_version_id,
                    metadata["architecture"],
                    relative_artifact,
                    digest,
                    json.dumps(metadata),
                ),
            )
            taxonomy_ids = metadata.get("class_taxonomy_ids", {})
            connection.executemany(
                """INSERT INTO model_classes
                   (model_version_id, output_index, display_label, taxon_version_id)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        model_version_id,
                        index,
                        label,
                        taxonomy_ids.get(label),
                    )
                    for index, label in enumerate(SUPPORTED_CLASSES)
                ],
            )
            connection.execute(
                """INSERT INTO model_registry_events
                   (event_id, action, model_version_id, actor_reference, rationale)
                   VALUES (?, 'candidate_registered', ?, 'local_registration', ?)""",
                (
                    str(uuid.uuid4()),
                    model_version_id,
                    f"Validated checkpoint SHA-256 {digest}",
                ),
            )
    finally:
        connection.close()
    return {
        "model_version_id": model_version_id,
        "lifecycle_state": "candidate",
        "artifact_path": relative_artifact,
        "artifact_sha256": digest,
        "has_test_evaluation": evaluation is not None,
    }


def list_candidate_models(database_path: str | Path) -> list[dict[str, str]]:
    connection = connect_database(database_path)
    try:
        rows = connection.execute(
            """SELECT model_version_id, architecture, artifact_path, artifact_sha256
               FROM model_versions WHERE lifecycle_state = 'candidate'
               ORDER BY created_at DESC, model_version_id"""
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def create_promotion_policy(
    database_path: str | Path,
    criteria: dict[str, float | int],
    *,
    approved_by: str,
    approval_reference: str,
) -> str:
    if set(criteria) != _POLICY_FIELDS:
        missing = sorted(_POLICY_FIELDS - set(criteria))
        extra = sorted(set(criteria) - _POLICY_FIELDS)
        raise ValueError(f"Invalid promotion criteria; missing={missing}, extra={extra}")
    for key in _POLICY_RATES:
        value = float(criteria[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if not math.isfinite(float(criteria["maximum_inference_ms_per_image"])) or float(
        criteria["maximum_inference_ms_per_image"]
    ) <= 0:
        raise ValueError("maximum_inference_ms_per_image must be positive")
    maximum_size = criteria["maximum_checkpoint_size_bytes"]
    if isinstance(maximum_size, bool) or int(maximum_size) != maximum_size or int(maximum_size) <= 0:
        raise ValueError("maximum_checkpoint_size_bytes must be positive")
    if not approved_by.strip() or not approval_reference.strip():
        raise ValueError("Promotion policy requires a named approval reference")
    policy_id = str(uuid.uuid4())
    connection = connect_database(database_path)
    try:
        with connection:
            connection.execute(
                """INSERT INTO promotion_policies
                   (policy_id, approved_by, approval_reference, criteria_json)
                   VALUES (?, ?, ?, ?)""",
                (
                    policy_id,
                    approved_by.strip(),
                    approval_reference.strip(),
                    json.dumps(criteria, sort_keys=True),
                ),
            )
    finally:
        connection.close()
    return policy_id


def _evaluation_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    report = metadata.get("test_evaluation")
    if not isinstance(report, dict) or report.get("split") != "test":
        raise ValueError("Model has no matching frozen-test evaluation report")
    metrics = report.get("metrics")
    per_class = report.get("per_class")
    inference = report.get("inference")
    if not isinstance(metrics, dict) or not isinstance(per_class, list) or not isinstance(inference, dict):
        raise ValueError("Test evaluation report is missing required metrics")
    if report.get("confusion_matrix", {}).get("labels") != list(SUPPORTED_CLASSES):
        raise ValueError("Test evaluation uses an incompatible class order")
    if not inference.get("images_per_second") or inference["images_per_second"] <= 0:
        raise ValueError("Test evaluation is missing valid inference throughput")
    recalls = {
        str(row["label"]): float(row["recall"])
        for row in per_class
        if row.get("label") and row.get("support", 0) > 0
    }
    if set(recalls) != set(SUPPORTED_CLASSES):
        raise ValueError("Test evaluation must include supported-test examples for every class")
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_recall": recalls,
        "inference_ms_per_image": 1000 / float(inference["images_per_second"]),
        "checkpoint_size_bytes": int(report["checkpoint_size_bytes"]),
        "manifest_sha256": report["dataset_manifest_sha256"],
    }


def _ensure_model_data_consent(
    connection,
    metadata: dict[str, Any],
    project_root: str | Path,
) -> None:
    manifest_hash = metadata.get("dataset_manifest_sha256")
    if not manifest_hash:
        return
    manifest = connection.execute(
        """SELECT dataset_manifest_id FROM dataset_manifests
           WHERE manifest_sha256 = ? AND purpose = 'curated_feedback'""",
        (manifest_hash,),
    ).fetchone()
    if manifest is None:
        return
    rows = connection.execute(
        """SELECT di.consent_training, di.relative_path, f.training_use_consent,
                  f.retention_consent, i.withdrawn_at, i.relative_path AS feedback_path
           FROM dataset_items di
           LEFT JOIN feedback f ON f.feedback_id = di.feedback_id
           LEFT JOIN feedback_images i ON i.feedback_id = di.feedback_id
           WHERE di.dataset_manifest_id = ? AND di.feedback_id IS NOT NULL""",
        (manifest["dataset_manifest_id"],),
    ).fetchall()
    root = Path(project_root).resolve()
    for row in rows:
        image_path = (root / str(row["feedback_path"] or "")).resolve()
        if (
            not row["consent_training"]
            or not row["training_use_consent"]
            or not row["retention_consent"]
            or row["withdrawn_at"]
            or not image_path.is_relative_to(root)
            or not image_path.is_file()
        ):
            raise ValueError(
                "Model training lineage includes feedback with withdrawn consent or a missing image"
            )


def _evaluate_candidate(
    candidate_metadata: dict[str, Any],
    active_metadata: dict[str, Any] | None,
    criteria: dict[str, Any],
) -> list[str]:
    candidate = _evaluation_metrics(candidate_metadata)
    failures = []
    if candidate["accuracy"] < criteria["minimum_test_accuracy"]:
        failures.append("minimum_test_accuracy")
    if candidate["macro_f1"] < criteria["minimum_test_macro_f1"]:
        failures.append("minimum_test_macro_f1")
    if min(candidate["per_class_recall"].values()) < criteria["minimum_per_class_recall"]:
        failures.append("minimum_per_class_recall")
    if candidate["inference_ms_per_image"] > criteria["maximum_inference_ms_per_image"]:
        failures.append("maximum_inference_ms_per_image")
    if candidate["checkpoint_size_bytes"] > criteria["maximum_checkpoint_size_bytes"]:
        failures.append("maximum_checkpoint_size_bytes")
    if active_metadata is not None:
        active = _evaluation_metrics(active_metadata)
        if candidate["manifest_sha256"] != active["manifest_sha256"]:
            raise ValueError("Candidate and active model were not evaluated on the same frozen manifest")
        if active["accuracy"] - candidate["accuracy"] > criteria["maximum_accuracy_drop"]:
            failures.append("maximum_accuracy_drop")
        if active["macro_f1"] - candidate["macro_f1"] > criteria["maximum_macro_f1_drop"]:
            failures.append("maximum_macro_f1_drop")
        if any(
            active["per_class_recall"][label] - candidate["per_class_recall"][label]
            > criteria["maximum_per_class_recall_drop"]
            for label in SUPPORTED_CLASSES
        ):
            failures.append("maximum_per_class_recall_drop")
    return failures


def promote_candidate(
    project_root: str | Path,
    database_path: str | Path,
    model_version_id: str,
    *,
    policy_id: str,
    actor_reference: str,
    rationale: str,
) -> dict[str, object]:
    if not actor_reference.strip() or not rationale.strip():
        raise ValueError("Promotion requires an actor reference and rationale")
    connection = connect_database(database_path)
    try:
        candidate_row = connection.execute(
            "SELECT * FROM model_versions WHERE model_version_id = ?",
            (model_version_id,),
        ).fetchone()
        if candidate_row is None:
            raise KeyError(f"Model version does not exist: {model_version_id}")
        if candidate_row["lifecycle_state"] != "candidate":
            raise ValueError("Only a candidate model can be promoted")
        policy_row = connection.execute(
            "SELECT * FROM promotion_policies WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        if policy_row is None:
            raise KeyError(f"Promotion policy does not exist: {policy_id}")
        candidate_metadata = json.loads(candidate_row["metadata_json"])
        artifact = _safe_artifact_path(project_root, candidate_row["artifact_path"])
        if _sha256(artifact) != candidate_row["artifact_sha256"]:
            raise ValueError("Candidate checkpoint hash differs from its registry record")
        active_row = connection.execute(
            """SELECT mv.* FROM active_model am JOIN model_versions mv
               ON mv.model_version_id = am.model_version_id WHERE am.singleton = 1"""
        ).fetchone()
        active_metadata = json.loads(active_row["metadata_json"]) if active_row else None
        _ensure_model_data_consent(connection, candidate_metadata, project_root)
        if active_metadata is not None:
            _ensure_model_data_consent(connection, active_metadata, project_root)
        criteria = json.loads(policy_row["criteria_json"])
        failures = _evaluate_candidate(candidate_metadata, active_metadata, criteria)
        if failures:
            raise ValueError(f"Promotion criteria failed: {', '.join(failures)}")

        previous_id = active_row["model_version_id"] if active_row else None
        with connection:
            if previous_id:
                connection.execute(
                    """UPDATE model_versions SET lifecycle_state = 'retired',
                       decision_at = CURRENT_TIMESTAMP, decision_by = ?,
                       decision_rationale = ? WHERE model_version_id = ?""",
                    (actor_reference, f"Superseded by {model_version_id}: {rationale}", previous_id),
                )
            connection.execute(
                """UPDATE model_versions SET lifecycle_state = 'active',
                   decision_at = CURRENT_TIMESTAMP, decision_by = ?,
                   decision_rationale = ? WHERE model_version_id = ?""",
                (actor_reference, rationale, model_version_id),
            )
            connection.execute(
                """INSERT INTO active_model(singleton, model_version_id)
                   VALUES (1, ?)
                   ON CONFLICT(singleton) DO UPDATE SET
                   model_version_id = excluded.model_version_id,
                   updated_at = CURRENT_TIMESTAMP""",
                (model_version_id,),
            )
            connection.execute(
                """INSERT INTO model_registry_events
                   (event_id, action, model_version_id, previous_model_version_id,
                    policy_id, actor_reference, rationale)
                   VALUES (?, 'promoted', ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    model_version_id,
                    previous_id,
                    policy_id,
                    actor_reference,
                    rationale,
                ),
            )
        return {
            "model_version_id": model_version_id,
            "previous_model_version_id": previous_id,
            "lifecycle_state": "active",
            "policy_id": policy_id,
        }
    finally:
        connection.close()


def reject_candidate(
    database_path: str | Path,
    model_version_id: str,
    *,
    actor_reference: str,
    rationale: str,
) -> None:
    if not actor_reference.strip() or not rationale.strip():
        raise ValueError("Candidate rejection requires an actor reference and rationale")
    connection = connect_database(database_path)
    try:
        with connection:
            row = connection.execute(
                "SELECT lifecycle_state FROM model_versions WHERE model_version_id = ?",
                (model_version_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Model version does not exist: {model_version_id}")
            if row["lifecycle_state"] != "candidate":
                raise ValueError("Only a candidate model can be rejected")
            connection.execute(
                """UPDATE model_versions SET lifecycle_state = 'rejected',
                   decision_at = CURRENT_TIMESTAMP, decision_by = ?,
                   decision_rationale = ? WHERE model_version_id = ?""",
                (actor_reference, rationale, model_version_id),
            )
            connection.execute(
                """INSERT INTO model_registry_events
                   (event_id, action, model_version_id, actor_reference, rationale)
                   VALUES (?, 'rejected', ?, ?, ?)""",
                (str(uuid.uuid4()), model_version_id, actor_reference, rationale),
            )
    finally:
        connection.close()


def rollback_model(
    project_root: str | Path,
    database_path: str | Path,
    model_version_id: str,
    *,
    actor_reference: str,
    rationale: str,
) -> dict[str, object]:
    if not actor_reference.strip() or not rationale.strip():
        raise ValueError("Rollback requires an actor reference and rationale")
    connection = connect_database(database_path)
    try:
        target = connection.execute(
            "SELECT * FROM model_versions WHERE model_version_id = ?",
            (model_version_id,),
        ).fetchone()
        if target is None:
            raise KeyError(f"Model version does not exist: {model_version_id}")
        if target["lifecycle_state"] not in {"active", "retired"}:
            raise ValueError("Rollback target must be an active or previously retired model")
        artifact = _safe_artifact_path(project_root, target["artifact_path"])
        if _sha256(artifact) != target["artifact_sha256"]:
            raise ValueError("Rollback checkpoint hash differs from its registry record")
        metadata = json.loads(target["metadata_json"])
        _ensure_model_data_consent(connection, metadata, project_root)
        active = connection.execute(
            """SELECT model_version_id FROM active_model WHERE singleton = 1"""
        ).fetchone()
        previous_id = active["model_version_id"] if active else None
        if previous_id == model_version_id:
            raise ValueError("The selected model is already active")
        with connection:
            if previous_id and previous_id != model_version_id:
                connection.execute(
                    """UPDATE model_versions SET lifecycle_state = 'retired'
                       WHERE model_version_id = ?""",
                    (previous_id,),
                )
            connection.execute(
                """UPDATE model_versions SET lifecycle_state = 'active',
                   decision_at = CURRENT_TIMESTAMP, decision_by = ?,
                   decision_rationale = ? WHERE model_version_id = ?""",
                (actor_reference, rationale, model_version_id),
            )
            connection.execute(
                """INSERT INTO active_model(singleton, model_version_id)
                   VALUES (1, ?)
                   ON CONFLICT(singleton) DO UPDATE SET
                   model_version_id = excluded.model_version_id,
                   updated_at = CURRENT_TIMESTAMP""",
                (model_version_id,),
            )
            connection.execute(
                """INSERT INTO model_registry_events
                   (event_id, action, model_version_id, previous_model_version_id,
                    actor_reference, rationale)
                   VALUES (?, 'rolled_back', ?, ?, ?, ?)""",
                (str(uuid.uuid4()), model_version_id, previous_id, actor_reference, rationale),
            )
        return {
            "model_version_id": model_version_id,
            "previous_model_version_id": previous_id,
            "lifecycle_state": "active",
        }
    finally:
        connection.close()
