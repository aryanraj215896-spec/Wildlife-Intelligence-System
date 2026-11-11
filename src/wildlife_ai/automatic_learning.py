from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.curation.eligibility import (
    _read_base_manifest,
    _verify_base_manifest_files,
    validate_training_manifest_consent,
)
from wildlife_ai.evaluation.metrics import evaluate_checkpoint
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.registry.service import register_candidate
from wildlife_ai.storage.database import connect_database
from wildlife_ai.training.engine import TrainingConfig, train_model

AUTOMATIC_BATCH_SIZE = 5
MAX_ACCURACY_DROP = 0.01
MAX_MACRO_F1_DROP = 0.01
MAX_CLASS_RECALL_DROP = 0.02
INITIAL_MODEL_VERSION = "resnet18-20261105T013137Z"
INITIAL_CHECKPOINT = Path("models/runs/resnet18-20261105T013130Z/best-resnet18.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_selected_model(
    project_root: str | Path,
    database_path: str | Path,
    checkpoint_path: str | Path,
    validation_report_path: str | Path,
    *,
    expected_version: str = INITIAL_MODEL_VERSION,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    report_path = Path(validation_report_path).resolve()
    frozen_manifest = root / "data" / "manifests" / "dataset_manifest.csv"
    if not frozen_manifest.is_file():
        raise FileNotFoundError(f"Frozen dataset manifest not found: {frozen_manifest}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected model checkpoint not found: {checkpoint}")
    if not report_path.is_file():
        raise FileNotFoundError(f"Selected model validation report not found: {report_path}")
    model, metadata = load_checkpoint(checkpoint, device=torch.device("cpu"))
    del model
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_hash = _sha256(frozen_manifest)
    checkpoint_hash = _sha256(checkpoint)
    with frozen_manifest.open(newline="", encoding="utf-8") as stream:
        validation_count = sum(
            row.get("image_status") == "valid" and row.get("split") == "validation"
            for row in csv.DictReader(stream)
        )
    per_class = report.get("per_class")
    metric_names = {
        row.get("label") for row in per_class if isinstance(row, dict)
    } if isinstance(per_class, list) else set()
    if metadata.get("model_version") != expected_version:
        raise ValueError("Selected checkpoint has an unexpected model version")
    if metadata.get("architecture") != "resnet18" or metadata.get("class_names") != list(
        SUPPORTED_CLASSES
    ):
        raise ValueError("Selected checkpoint is not a ResNet18")
    if (
        report.get("model_version") != expected_version
        or report.get("architecture") != "resnet18"
        or report.get("split") != "validation"
        or report.get("dataset_manifest_sha256") != manifest_hash
        or report.get("checkpoint_sha256") != checkpoint_hash
        or report.get("checkpoint_size_bytes") != checkpoint.stat().st_size
        or report.get("class_names") != list(SUPPORTED_CLASSES)
        or int(report.get("sample_count", 0)) != validation_count
        or not isinstance(report.get("metrics"), dict)
        or int(report.get("metrics", {}).get("sample_count", 0)) != validation_count
        or metric_names != set(SUPPORTED_CLASSES)
    ):
        raise ValueError("Selected checkpoint does not match its verified validation report")
    if metadata.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("Selected checkpoint was not trained against the frozen manifest")

    connection = connect_database(database_path)
    try:
        selected = connection.execute(
            "SELECT model_version_id FROM application_model_selection WHERE singleton = 1"
        ).fetchone()
        registered = None if selected is not None else connection.execute(
            """SELECT model_version_id, artifact_path, artifact_sha256
               FROM model_versions WHERE model_version_id = ?""",
            (expected_version,),
        ).fetchone()
    finally:
        connection.close()

    if selected is not None:
        return get_selected_model(project_root, database_path)
    if registered is None:
        register_candidate(root, database_path, checkpoint)
    else:
        registered_path = (root / registered["artifact_path"]).resolve()
        if (
            registered_path != checkpoint
            or registered["artifact_sha256"] != checkpoint_hash
        ):
            raise ValueError("Selected checkpoint does not match its registered artifact")
    relative_report_path = str(report_path)
    connection = connect_database(database_path)
    try:
        with connection:
            connection.execute(
                """INSERT INTO application_model_selection
                   (singleton, model_version_id, validation_report_path, validation_report_json)
                   VALUES (1, ?, ?, ?)""",
                (expected_version, relative_report_path, json.dumps(report, sort_keys=True)),
            )
    finally:
        connection.close()
    return get_selected_model(project_root, database_path)


def get_selected_model(
    project_root: str | Path,
    database_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    connection = connect_database(database_path)
    try:
        row = connection.execute(
            """SELECT s.model_version_id, s.validation_report_path,
                      s.validation_report_json, m.architecture, m.artifact_path,
                      m.artifact_sha256, m.metadata_json
               FROM application_model_selection s
               JOIN model_versions m USING (model_version_id)
               WHERE s.singleton = 1"""
        ).fetchone()
        if row is None:
            raise RuntimeError("The selected model has not been initialized")
    finally:
        connection.close()
    checkpoint = (root / row["artifact_path"]).resolve()
    if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
        raise FileNotFoundError("Selected model checkpoint is missing or outside the project")
    if _sha256(checkpoint) != row["artifact_sha256"]:
        raise ValueError("Selected model checkpoint hash does not match its registry record")
    report_path = Path(row["validation_report_path"]).resolve()
    report = json.loads(row["validation_report_json"])
    metadata = json.loads(row["metadata_json"])
    if report.get("model_version") != row["model_version_id"]:
        raise ValueError("Selected model validation report does not match the model version")
    if (
        metadata.get("model_version") != row["model_version_id"]
        or metadata.get("architecture") != "resnet18"
        or row["architecture"] != "resnet18"
    ):
        raise ValueError("Selected model registry metadata does not match its version")
    return {
        "model_version_id": row["model_version_id"],
        "architecture": row["architecture"],
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": row["artifact_sha256"],
        "validation_report_path": report_path,
        "validation_report": report,
        "metadata": metadata,
    }


def feedback_eligibility(
    project_root: str | Path,
    database_path: str | Path,
    frozen_manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    root = Path(project_root).resolve()
    base_path = Path(frozen_manifest_path).resolve()
    fields, base_rows = _read_base_manifest(base_path)
    del fields
    _verify_base_manifest_files(root, base_rows)
    base_hashes = {
        str(row.get("sha256"))
        for row in base_rows
        if row.get("image_status") == "valid" and row.get("sha256")
    }
    connection = connect_database(database_path)
    try:
        attempted = {
            row["feedback_id"]
            for row in connection.execute(
                "SELECT feedback_id FROM automatic_training_batch_feedback"
            )
        }
        feedback_rows = connection.execute(
            """SELECT f.feedback_id, f.response, f.predicted_label, f.corrected_label,
                      f.image_sha256 AS upload_sha256, f.retention_consent,
                      f.retention_withdrawn_at, f.training_use_consent,
                      f.training_use_withdrawn_at, f.duplicate_of_feedback_id,
                      i.relative_path, i.sha256 AS retained_sha256,
                      i.upload_sha256 AS retained_upload_sha256, i.withdrawn_at
               FROM feedback f
               LEFT JOIN feedback_images i USING (feedback_id)
               ORDER BY f.rowid"""
        ).fetchall()
    finally:
        connection.close()

    eligible: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    seen_hashes: set[str] = set()
    for row in feedback_rows:
        item = dict(row)
        hashes = {
            str(value)
            for value in (
                item["upload_sha256"],
                item["retained_sha256"],
                item["retained_upload_sha256"],
            )
            if value
        }
        why = []
        if item["response"] not in {"yes", "no"}:
            why.append("no_supported_confirmation")
        if item["corrected_label"] not in SUPPORTED_CLASSES:
            why.append("no_supported_confirmed_label")
        elif item["response"] == "yes" and item["corrected_label"] != item["predicted_label"]:
            why.append("confirmed_label_mismatch")
        elif item["response"] == "no" and item["corrected_label"] == item["predicted_label"]:
            why.append("correction_does_not_change_prediction")
        if not item["retention_consent"] or item["retention_withdrawn_at"]:
            why.append("image_retention_not_currently_consented")
        if not item["training_use_consent"] or item["training_use_withdrawn_at"]:
            why.append("training_use_not_currently_consented")
        if not item["relative_path"] or item["withdrawn_at"]:
            why.append("retained_image_missing_or_withdrawn")
        else:
            image_path = (root / str(item["relative_path"])).resolve()
            feedback_root = (root / "data" / "feedback").resolve()
            if not image_path.is_relative_to(feedback_root):
                why.append("retained_image_path_unsafe")
            elif not image_path.is_file():
                why.append("retained_image_file_missing")
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    actual_hash = _sha256(image_path)
                    if actual_hash != item["retained_sha256"]:
                        why.append("retained_image_hash_mismatch")
                except (
                    OSError,
                    ValueError,
                    Image.DecompressionBombError,
                    Image.DecompressionBombWarning,
                ):
                    why.append("retained_image_invalid")
        if hashes & base_hashes:
            why.append("image_overlaps_frozen_dataset")
        if item["duplicate_of_feedback_id"] or hashes & seen_hashes:
            why.append("duplicate_feedback_image")
        if item["feedback_id"] in attempted:
            why.append("already_used_in_training_attempt")
        if why:
            for reason in set(why):
                reasons[reason] = reasons.get(reason, 0) + 1
        else:
            eligible.append(item)
        seen_hashes.update(hashes)
    return eligible, reasons


def _write_automatic_manifest(
    project_root: Path,
    database_path: Path,
    frozen_manifest_path: Path,
    feedback_items: list[dict[str, Any]],
    batch_id: str,
) -> tuple[Path, str]:
    root = project_root.resolve()
    fields, base_rows = _read_base_manifest(frozen_manifest_path)
    _verify_base_manifest_files(root, base_rows)
    base_hashes = {
        str(row.get("sha256"))
        for row in base_rows
        if row.get("image_status") == "valid" and row.get("sha256")
    }
    seen_feedback_hashes: set[str] = set()
    extra_fields = [
        "dataset_item_id",
        "feedback_id",
        "width",
        "height",
        "duplicate_group",
        "consent_training",
    ]
    output_fields = list(dict.fromkeys([*fields, *extra_fields]))
    destination = root / "data" / "manifests" / "automatic" / f"{batch_id}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite automatic manifest for batch {batch_id}")

    feedback_manifest_rows = []
    base_seen = set(base_hashes)
    for item in feedback_items:
        image_path = (root / str(item["relative_path"])).resolve()
        digest = str(item["retained_sha256"])
        if digest in base_seen or digest in seen_feedback_hashes:
            raise ValueError("Duplicate or frozen-dataset image found while building training manifest")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        feedback_manifest_rows.append({
            "dataset_item_id": str(uuid.uuid4()),
            "feedback_id": item["feedback_id"],
            "relative_path": item["relative_path"],
            "sha256": digest,
            "image_status": "valid",
            "split": "train",
            "label": item["corrected_label"],
            "width": width,
            "height": height,
            "duplicate_group": digest,
            "consent_training": 1,
        })
        seen_feedback_hashes.add(digest)
    try:
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(base_rows)
            writer.writerows(feedback_manifest_rows)
        manifest_hash = _sha256(temporary)
        temporary.replace(destination)
        manifest_id = str(uuid.uuid4())
        with connect_database(database_path) as connection:
            connection.execute(
                """INSERT INTO dataset_manifests
                   (dataset_manifest_id, manifest_sha256, purpose, split_policy_version,
                    split_seed, manifest_path)
                   VALUES (?, ?, 'automatic_feedback', 'frozen-splits-feedback-train-v1', 42, ?)""",
                (manifest_id, manifest_hash, str(destination.resolve())),
            )
            base_items = [
                (
                    str(uuid.uuid4()),
                    manifest_id,
                    row["relative_path"],
                    row["sha256"],
                    row.get("duplicate_group") or "",
                    row.get("review_status") or "unreviewed",
                    row.get("source") or "",
                    row.get("license") or "",
                    row["split"],
                )
                for row in base_rows
                if row.get("image_status") == "valid"
                and row.get("split") in {"train", "validation", "test"}
            ]
            connection.executemany(
                """INSERT INTO dataset_items
                   (dataset_item_id, dataset_manifest_id, relative_path, sha256,
                    duplicate_group, label_status, source, license, consent_training, split)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                base_items,
            )
            connection.executemany(
                """INSERT INTO dataset_items
                   (dataset_item_id, dataset_manifest_id, relative_path, sha256,
                    duplicate_group, label_status, source, license, consent_training,
                    split, feedback_id)
                   VALUES (?, ?, ?, ?, ?, 'user_confirmed', '', '', 1, 'train', ?)""",
                [
                    (
                        item["dataset_item_id"],
                        manifest_id,
                        item["relative_path"],
                        item["sha256"],
                        item["duplicate_group"],
                        item["feedback_id"],
                    )
                    for item in feedback_manifest_rows
                ],
            )
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    return destination, manifest_hash


def _compare_validation(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    for key, maximum_drop in (
        ("accuracy", MAX_ACCURACY_DROP),
        ("macro_f1", MAX_MACRO_F1_DROP),
    ):
        if key not in baseline_metrics or key not in candidate_metrics:
            issues.append(f"missing_{key}")
        elif float(candidate_metrics[key]) < float(baseline_metrics[key]) - maximum_drop:
            issues.append(f"{key}_regressed_beyond_limit")
    if int(candidate_metrics.get("sample_count", -1)) != int(
        baseline_metrics.get("sample_count", -2)
    ):
        issues.append("validation_sample_count_changed")
    baseline_recalls = {
        row.get("label"): float(row.get("recall", -1))
        for row in baseline.get("per_class", [])
    }
    candidate_recalls = {
        row.get("label"): float(row.get("recall", -1))
        for row in candidate.get("per_class", [])
    }
    if set(baseline_recalls) != set(SUPPORTED_CLASSES) or set(candidate_recalls) != set(
        SUPPORTED_CLASSES
    ):
        issues.append("per_class_validation_metrics_incomplete")
    else:
        for label in SUPPORTED_CLASSES:
            if candidate_recalls[label] < baseline_recalls[label] - MAX_CLASS_RECALL_DROP:
                issues.append(f"{label}_recall_regressed_beyond_limit")
    return not issues, issues


def automatic_update_if_ready(
    project_root: str | Path,
    database_path: str | Path,
    frozen_manifest_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    db_path = Path(database_path).resolve()
    manifest_path = Path(frozen_manifest_path).resolve()
    batch_id = str(uuid.uuid4())
    eligible, _ = feedback_eligibility(root, db_path, manifest_path)
    connection = connect_database(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        selected_row = connection.execute(
            """SELECT s.model_version_id, s.validation_report_path,
                      s.validation_report_json, m.artifact_path
               FROM application_model_selection s
               JOIN model_versions m USING (model_version_id)
               WHERE s.singleton = 1"""
        ).fetchone()
        if selected_row is None:
            raise RuntimeError("The selected model has not been initialized")
        attempted = {
            row["feedback_id"]
            for row in connection.execute(
                "SELECT feedback_id FROM automatic_training_batch_feedback"
            )
        }
        eligible = [item for item in eligible if item["feedback_id"] not in attempted]
        if len(eligible) < AUTOMATIC_BATCH_SIZE:
            connection.commit()
            connection.close()
            return {
                "status": "collecting",
                "eligible_count": len(eligible),
                "required_count": AUTOMATIC_BATCH_SIZE,
            }
        selected_items = eligible[:AUTOMATIC_BATCH_SIZE]
        connection.execute(
            "INSERT INTO automatic_training_batches (batch_id, status, message) VALUES (?, 'running', ?)",
            (batch_id, "Automatic training started after enough eligible feedback was collected."),
        )
        connection.executemany(
            """INSERT INTO automatic_training_batch_feedback (batch_id, feedback_id)
               VALUES (?, ?)""",
            [(batch_id, item["feedback_id"]) for item in selected_items],
        )
        connection.commit()
        selected_version = selected_row["model_version_id"]
        selected_checkpoint = (root / selected_row["artifact_path"]).resolve()
        baseline_report = json.loads(selected_row["validation_report_json"])
    except Exception:
        connection.rollback()
        connection.close()
        raise
    connection.close()

    output_dir = root / "models" / "runs" / f"automatic-{batch_id}"
    manifest_output: Path | None = None
    candidate_version: str | None = None
    try:
        manifest_output, _ = _write_automatic_manifest(
            root, db_path, manifest_path, selected_items, batch_id
        )
        validate_training_manifest_consent(db_path, manifest_output)
        trained = train_model(
            TrainingConfig(
                model_name="resnet18",
                manifest_path=manifest_output,
                raw_root=root / "data" / "raw",
                output_dir=output_dir,
                database_path=db_path,
                epochs=1,
                fine_tune_epochs=1,
                device="auto",
                initial_checkpoint_path=selected_checkpoint,
                feedback_fraction=0.25,
                model_version=f"resnet18-auto-{batch_id}",
            )
        )
        checkpoint = Path(str(trained["best_checkpoint"]))
        evaluation = evaluate_checkpoint(
            checkpoint,
            manifest_output,
            root / "data" / "raw",
            output_dir,
            split="validation",
            num_workers=0,
            device_name="auto",
        )
        _, candidate_metadata = load_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
        )
        if (
            evaluation.get("model_version") != candidate_metadata.get("model_version")
            or evaluation.get("architecture") != "resnet18"
            or evaluation.get("split") != "validation"
            or evaluation.get("dataset_manifest_sha256")
            != candidate_metadata.get("dataset_manifest_sha256")
            or evaluation.get("confusion_matrix", {}).get("labels")
            != list(SUPPORTED_CLASSES)
            or evaluation.get("sample_count")
            != evaluation.get("metrics", {}).get("sample_count")
        ):
            raise ValueError("Candidate validation report does not match the checkpoint")
        report_path = Path(str(evaluation["report_path"]))
        passed, issues = _compare_validation(baseline_report, evaluation)
        candidate = register_candidate(root, db_path, checkpoint)
        candidate_version = str(candidate["model_version_id"])
        status = "accepted" if passed else "rejected"
        message = (
            "The update passed validation and is now the selected model."
            if passed
            else "The update did not meet validation safeguards; the previous model remains selected."
        )
        connection = connect_database(db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            latest_selected = connection.execute(
                "SELECT model_version_id FROM application_model_selection WHERE singleton = 1"
            ).fetchone()
            if latest_selected is None or latest_selected["model_version_id"] != selected_version:
                status = "rejected"
                message = "Another model update completed first; this candidate was not selected."
                issues.append("selected_model_changed_during_training")
            if status == "accepted":
                connection.execute(
                    """UPDATE application_model_selection
                       SET model_version_id = ?, validation_report_path = ?,
                           validation_report_json = ?, updated_at = ?
                       WHERE singleton = 1""",
                    (
                        candidate_version,
                        str(report_path.resolve()),
                        json.dumps(evaluation, sort_keys=True),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            else:
                connection.execute(
                    "UPDATE model_versions SET lifecycle_state = 'rejected' WHERE model_version_id = ?",
                    (candidate_version,),
                )
            connection.execute(
                """UPDATE automatic_training_batches
                   SET status = ?, candidate_model_version_id = ?, evaluation_json = ?,
                       message = ?, completed_at = ?
                   WHERE batch_id = ?""",
                (
                    status,
                    candidate_version,
                    json.dumps({"evaluation": evaluation, "issues": issues}, sort_keys=True),
                    message,
                    datetime.now(timezone.utc).isoformat(),
                    batch_id,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return {
            "status": status,
            "message": message,
            "eligible_count": AUTOMATIC_BATCH_SIZE,
            "required_count": AUTOMATIC_BATCH_SIZE,
            "candidate_model_version_id": candidate_version,
            "issues": issues,
            "evaluation": evaluation,
        }
    except Exception as exc:
        connection = connect_database(db_path)
        try:
            with connection:
                connection.execute(
                    """UPDATE automatic_training_batches
                       SET status = 'failed', candidate_model_version_id = ?,
                           message = ?, completed_at = ?
                       WHERE batch_id = ?""",
                    (
                        candidate_version,
                        f"Automatic training failed: {type(exc).__name__}: {exc}",
                        datetime.now(timezone.utc).isoformat(),
                        batch_id,
                    ),
                )
        finally:
            connection.close()
        return {
            "status": "failed",
            "message": f"Automatic training failed: {type(exc).__name__}: {exc}",
            "eligible_count": AUTOMATIC_BATCH_SIZE,
            "required_count": AUTOMATIC_BATCH_SIZE,
            "candidate_model_version_id": candidate_version,
        }
