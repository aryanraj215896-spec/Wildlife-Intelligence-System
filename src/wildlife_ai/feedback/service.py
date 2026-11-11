from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.storage.database import connect_database
from wildlife_ai.utils.images import sanitize_image

_RESPONSES = {"yes", "no", "unsure", "unsupported"}
_REVIEW_STATES = {"verified", "rejected", "needs_review"}


def submit_feedback(
    database_path: str | Path,
    image_storage_dir: str | Path,
    *,
    response: str,
    predicted_label: str,
    corrected_label: str | None = None,
    model_version_id: str | None,
    model_score: float | None,
    original_typed_name: str = "",
    resolved_taxon_version_id: str | None = None,
    resolver_method: str = "",
    image_bytes: bytes | None = None,
    retention_consent: bool = False,
    training_use_consent: bool = False,
    source: str = "",
    license: str = "",
    attribution: str = "",
) -> str:
    if response not in _RESPONSES:
        raise ValueError(f"Unsupported feedback response: {response}")
    if predicted_label not in SUPPORTED_CLASSES:
        raise ValueError(f"Unsupported predicted label: {predicted_label}")
    if response == "yes":
        if corrected_label is None:
            corrected_label = predicted_label
        if corrected_label != predicted_label:
            raise ValueError("A correct prediction must retain the predicted label")
    elif response == "no":
        if corrected_label not in SUPPORTED_CLASSES:
            raise ValueError("An incorrect prediction requires a supported corrected label")
        if corrected_label == predicted_label:
            raise ValueError("An incorrect prediction must be corrected to a different label")
    elif corrected_label is not None and corrected_label not in SUPPORTED_CLASSES:
        raise ValueError(f"Unsupported corrected label: {corrected_label}")
    if image_bytes is None:
        raise ValueError("The uploaded image is required to preserve its SHA-256 hash")
    if model_score is not None and not 0 <= model_score <= 1:
        raise ValueError("model_score must be between 0 and 1")
    if training_use_consent and not retention_consent:
        raise ValueError("Training-use consent requires separate image-retention consent")
    if retention_consent and image_bytes is None:
        raise ValueError("An image is required when retention consent is given")
    feedback_id = str(uuid.uuid4())
    image_path: Path | None = None
    sanitized_image: bytes | None = None
    image_hash: str | None = None
    upload_hash = hashlib.sha256(image_bytes).hexdigest()
    if retention_consent:
        sanitized_image = sanitize_image(image_bytes)
        image_hash = hashlib.sha256(sanitized_image).hexdigest()
        image_path = Path(image_storage_dir).resolve() / f"{feedback_id}.png"
        try:
            relative_image_path = image_path.relative_to(
                Path(database_path).resolve().parent
            ).as_posix()
        except ValueError as exc:
            raise ValueError("Feedback image storage must be inside the project directory") from exc

    connection = connect_database(database_path)
    try:
        duplicate = None
        if image_hash is not None:
            duplicate_row = connection.execute(
                "SELECT feedback_id FROM feedback_images WHERE sha256 = ? LIMIT 1",
                (image_hash,),
            ).fetchone()
            duplicate = duplicate_row["feedback_id"] if duplicate_row else None
        consent_at = datetime.now(timezone.utc).isoformat()
        with connection:
            connection.execute(
                """INSERT INTO feedback
                   (feedback_id, response, predicted_label, corrected_label, image_sha256,
                    model_version_id, model_score,
                    original_typed_name, resolved_taxon_version_id, resolver_method,
                    duplicate_of_feedback_id, retention_consent, retention_consent_at,
                    training_use_consent, training_use_consent_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    feedback_id,
                    response,
                    predicted_label,
                    corrected_label or "",
                    upload_hash,
                    model_version_id,
                    model_score,
                    original_typed_name,
                    resolved_taxon_version_id,
                    resolver_method,
                    duplicate,
                    int(retention_consent),
                    consent_at if retention_consent else None,
                    int(training_use_consent),
                    consent_at if training_use_consent else None,
                ),
            )
            if image_path is not None and sanitized_image is not None and image_hash is not None:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                with image_path.open("xb") as stream:
                    stream.write(sanitized_image)
                connection.execute(
                    """INSERT INTO feedback_images
                       (feedback_image_id, feedback_id, relative_path, sha256, upload_sha256,
                        source, license, attribution)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        feedback_id,
                        relative_image_path,
                        image_hash,
                        upload_hash,
                        source,
                        license,
                        attribution,
                    ),
                )
    except Exception:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    return feedback_id


def list_feedback(database_path: str | Path, *, status: str | None = None) -> list[dict[str, object]]:
    connection = connect_database(database_path)
    try:
        if status is None:
            rows = connection.execute(
                """SELECT f.*, i.relative_path AS image_path, i.sha256, i.source,
                          i.license, i.attribution, i.rights_review_status,
                          i.quality_review_status
                   FROM feedback f LEFT JOIN feedback_images i ON i.feedback_id = f.feedback_id
                   ORDER BY f.submitted_at"""
            ).fetchall()
        else:
            rows = connection.execute(
                """SELECT f.*, i.relative_path AS image_path, i.sha256, i.source,
                          i.license, i.attribution, i.rights_review_status,
                          i.quality_review_status
                   FROM feedback f LEFT JOIN feedback_images i ON i.feedback_id = f.feedback_id
                   WHERE f.verification_status = ? ORDER BY f.submitted_at""",
                (status,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def review_feedback(
    database_path: str | Path,
    feedback_id: str,
    *,
    decision: str,
    corrected_label: str | None = None,
    reviewer_reference: str = "",
    notes: str = "",
    resolved_taxon_version_id: str | None = None,
    rights_approved: bool = False,
    quality_approved: bool = False,
) -> None:
    if decision not in _REVIEW_STATES:
        raise ValueError(f"Unsupported review decision: {decision}")
    connection = connect_database(database_path)
    try:
        with connection:
            row = connection.execute(
                """SELECT verification_status, response, predicted_label, corrected_label
                   FROM feedback WHERE feedback_id = ?""",
                (feedback_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Feedback item does not exist: {feedback_id}")
            previous = row["verification_status"]
            if previous not in {"pending", "needs_review", "verified"}:
                raise ValueError(f"Feedback in {previous!r} state cannot be reviewed again")
            label = corrected_label or row["corrected_label"]
            if corrected_label is not None and corrected_label not in SUPPORTED_CLASSES:
                raise ValueError(f"Unsupported corrected label: {corrected_label}")
            if decision == "verified":
                if row["response"] not in {"yes", "no"}:
                    raise ValueError("Only a correct or incorrect class response can be verified")
                if label not in SUPPORTED_CLASSES:
                    raise ValueError("A supported corrected label is required before verification")
                if row["response"] == "yes" and label != row["predicted_label"]:
                    raise ValueError("A confirmed prediction must match the predicted label")
                if row["response"] == "no" and label == row["predicted_label"]:
                    raise ValueError("An incorrect prediction must have a different corrected label")
            if rights_approved:
                image = connection.execute(
                    """SELECT 1 FROM feedback_images
                       WHERE feedback_id = ? AND withdrawn_at IS NULL""",
                    (feedback_id,),
                ).fetchone()
                if decision != "verified" or image is None:
                    raise ValueError(
                        "Rights approval requires verified feedback with a retained image"
                    )
                connection.execute(
                    """UPDATE feedback_images SET rights_review_status = 'approved'
                       WHERE feedback_id = ?""",
                    (feedback_id,),
                )
            if quality_approved:
                image = connection.execute(
                    """SELECT 1 FROM feedback_images
                       WHERE feedback_id = ? AND withdrawn_at IS NULL""",
                    (feedback_id,),
                ).fetchone()
                if decision != "verified" or image is None:
                    raise ValueError(
                        "Image-quality approval requires verified feedback with a retained image"
                    )
                connection.execute(
                    """UPDATE feedback_images SET quality_review_status = 'approved'
                       WHERE feedback_id = ?""",
                    (feedback_id,),
                )
            connection.execute(
                """UPDATE feedback SET verification_status = ?,
                   corrected_label = ?,
                   resolved_taxon_version_id = COALESCE(?, resolved_taxon_version_id)
                   WHERE feedback_id = ?""",
                (decision, label or "", resolved_taxon_version_id, feedback_id),
            )
            connection.execute(
                """INSERT INTO reviews
                   (review_id, feedback_id, previous_status, new_status, reviewer_reference, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), feedback_id, previous, decision, reviewer_reference, notes),
            )
    finally:
        connection.close()


def withdraw_retained_image(database_path: str | Path, feedback_id: str) -> dict[str, object]:
    connection = connect_database(database_path)
    try:
        with connection:
            row = connection.execute(
                """SELECT relative_path, withdrawn_at FROM feedback_images
                   WHERE feedback_id = ?""",
                (feedback_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No retained image exists for feedback: {feedback_id}")
            if row["withdrawn_at"] is not None:
                manifests = connection.execute(
                    """SELECT DISTINCT dm.dataset_manifest_id, dm.manifest_sha256
                       FROM dataset_items di
                       JOIN dataset_manifests dm USING (dataset_manifest_id)
                       WHERE di.feedback_id = ?""",
                    (feedback_id,),
                ).fetchall()
                return {"feedback_id": feedback_id, "affected_manifests": [dict(item) for item in manifests]}
            project_root = Path(database_path).resolve().parent
            image_path = (project_root / row["relative_path"]).resolve()
            if not image_path.is_relative_to(project_root):
                raise ValueError("Stored feedback image path escapes the project directory")
            manifests = connection.execute(
                """SELECT DISTINCT dm.dataset_manifest_id, dm.manifest_sha256
                   FROM dataset_items di
                   JOIN dataset_manifests dm USING (dataset_manifest_id)
                   WHERE di.feedback_id = ?""",
                (feedback_id,),
            ).fetchall()
            connection.execute(
                "UPDATE feedback_images SET withdrawn_at = ? WHERE feedback_id = ?",
                (datetime.now(timezone.utc).isoformat(), feedback_id),
            )
            connection.execute(
                """UPDATE feedback
                   SET retention_consent = 0, retention_withdrawn_at = ?,
                       training_use_consent = 0, training_use_withdrawn_at = ?
                   WHERE feedback_id = ?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    feedback_id,
                ),
            )
            connection.execute(
                "UPDATE dataset_items SET consent_training = 0 WHERE feedback_id = ?",
                (feedback_id,),
            )
        image_path.unlink(missing_ok=True)
        return {
            "feedback_id": feedback_id,
            "affected_manifests": [dict(item) for item in manifests],
        }
    finally:
        connection.close()


def withdraw_training_consent(database_path: str | Path, feedback_id: str) -> dict[str, object]:
    connection = connect_database(database_path)
    try:
        with connection:
            row = connection.execute(
                "SELECT training_use_consent FROM feedback WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Feedback item does not exist: {feedback_id}")
            manifests = connection.execute(
                """SELECT DISTINCT dm.dataset_manifest_id, dm.manifest_sha256
                   FROM dataset_items di
                   JOIN dataset_manifests dm USING (dataset_manifest_id)
                   WHERE di.feedback_id = ?""",
                (feedback_id,),
            ).fetchall()
            if row["training_use_consent"]:
                connection.execute(
                    """UPDATE feedback SET training_use_consent = 0,
                       training_use_withdrawn_at = ? WHERE feedback_id = ?""",
                    (datetime.now(timezone.utc).isoformat(), feedback_id),
                )
                connection.execute(
                    "UPDATE dataset_items SET consent_training = 0 WHERE feedback_id = ?",
                    (feedback_id,),
                )
            return {
                "feedback_id": feedback_id,
                "affected_manifests": [dict(item) for item in manifests],
            }
    finally:
        connection.close()
