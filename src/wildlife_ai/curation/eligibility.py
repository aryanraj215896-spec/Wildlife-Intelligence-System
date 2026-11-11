from __future__ import annotations

import csv
import hashlib
import random
import uuid
from pathlib import Path

from PIL import Image

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.data.preparation import DatasetImage
from wildlife_ai.storage.database import connect_database


def _protected_test_hashes(manifest_path: str | Path) -> set[str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen supported-test manifest is required: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["sha256"]
            for row in csv.DictReader(stream)
            if row.get("image_status") == "valid"
            and row.get("split") == "test"
            and row.get("sha256")
        }


def _existing_dataset_hashes(manifest_path: str | Path) -> set[str]:
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["sha256"]
            for row in csv.DictReader(stream)
            if row.get("image_status") == "valid" and row.get("sha256")
        }


def _read_base_manifest(manifest_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {
            "image_status",
            "split",
            "sha256",
            "relative_path",
            "label",
        }.issubset(reader.fieldnames):
            raise ValueError("Frozen supported manifest is missing required dataset columns")
        return list(reader.fieldnames), list(reader)


def _verify_base_manifest_files(
    project_root: Path,
    rows: list[dict[str, str]],
) -> None:
    raw_root = (project_root / "data" / "raw").resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Frozen supported dataset directory is required: {raw_root}")
    for row in rows:
        if row.get("image_status") != "valid":
            continue
        image_path = (raw_root / row["relative_path"]).resolve()
        if not image_path.is_relative_to(raw_root):
            raise ValueError(f"Frozen dataset path escapes data/raw: {row['relative_path']}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Frozen dataset image is missing: {image_path}")
        expected_hash = row.get("sha256", "")
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"Frozen dataset image hash mismatch: {row['relative_path']}")


def _eligibility_row(
    connection,
    feedback_id: str,
    project_root: Path,
) -> tuple[dict[str, object], list[str]]:
    row = connection.execute(
        """SELECT f.feedback_id, f.response, f.predicted_label, f.corrected_label,
                  f.image_sha256 AS upload_sha256, f.verification_status,
                  f.retention_consent, f.training_use_consent, f.duplicate_of_feedback_id,
                  i.relative_path, i.sha256, i.source, i.license,
                  i.attribution, i.rights_review_status, i.quality_review_status, i.withdrawn_at
           FROM feedback f LEFT JOIN feedback_images i ON i.feedback_id = f.feedback_id
           WHERE f.feedback_id = ?""",
        (feedback_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Feedback item does not exist: {feedback_id}")
    item = dict(row)
    reasons = []
    if item["verification_status"] != "verified":
        reasons.append("label_not_verified")
    if item["corrected_label"] not in SUPPORTED_CLASSES:
        reasons.append("corrected_label_not_supported")
    if item["response"] == "yes" and item["corrected_label"] != item["predicted_label"]:
        reasons.append("confirmed_label_mismatch")
    if item["response"] == "no" and item["corrected_label"] == item["predicted_label"]:
        reasons.append("correction_does_not_change_prediction")
    if not item["retention_consent"]:
        reasons.append("image_retention_not_consented")
    if not item["training_use_consent"]:
        reasons.append("training_use_not_consented")
    if not item["relative_path"] or item["withdrawn_at"]:
        reasons.append("retained_image_missing_or_withdrawn")
    else:
        image_path = (project_root / str(item["relative_path"])).resolve()
        if not image_path.is_relative_to(project_root.resolve()):
            reasons.append("retained_image_path_unsafe")
        elif not image_path.is_file():
            reasons.append("retained_image_file_missing")
    if item["quality_review_status"] != "approved":
        reasons.append("image_quality_not_reviewed")
    if item["duplicate_of_feedback_id"]:
        reasons.append("duplicate_feedback_image")
    return item, reasons


def check_training_eligibility(
    database_path: str | Path,
    feedback_id: str,
    *,
    protected_test_manifest: str | Path,
) -> dict[str, object]:
    protected_hashes = _protected_test_hashes(protected_test_manifest)
    _, base_rows = _read_base_manifest(protected_test_manifest)
    project_root = Path(database_path).resolve().parent
    _verify_base_manifest_files(project_root, base_rows)
    existing_hashes = _existing_dataset_hashes(protected_test_manifest)
    connection = connect_database(database_path)
    try:
        item, reasons = _eligibility_row(
            connection, feedback_id, project_root
        )
        item_hashes = {item.get("sha256"), item.get("upload_sha256")}
        if item_hashes & protected_hashes:
            reasons.append("image_hash_overlaps_frozen_test")
        if item_hashes & existing_hashes:
            reasons.append("image_hash_overlaps_existing_supported_dataset")
        return {"feedback_id": feedback_id, "eligible": not reasons, "reasons": reasons}
    finally:
        connection.close()


def validate_training_manifest_consent(
    database_path: str | Path,
    manifest_path: str | Path,
) -> None:
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        manifest_rows = list(reader)
        has_feedback_ids = any(row.get("feedback_id") for row in manifest_rows)
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    project_root = Path(database_path).resolve().parent
    raw_root = (project_root / "data" / "raw").resolve()
    feedback_rows = {
        row["feedback_id"]: row for row in manifest_rows if row.get("feedback_id")
    }
    for row in manifest_rows:
        if row.get("image_status") != "valid":
            continue
        relative = Path(row["relative_path"])
        if row.get("feedback_id"):
            image_path = (project_root / relative).resolve()
            allowed_root = (project_root / "data" / "feedback").resolve()
        else:
            image_path = (raw_root / relative).resolve()
            allowed_root = raw_root
        if not image_path.is_relative_to(allowed_root):
            raise ValueError(f"Training manifest image path escapes its data directory: {relative}")
        if not image_path.is_file():
            if row.get("feedback_id"):
                raise ValueError(
                    "Curated training manifest contains feedback with withdrawn or missing consent/image"
                )
            raise FileNotFoundError(f"Training manifest image is missing: {image_path}")
        expected_hash = row.get("sha256", "")
        if expected_hash and hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Training manifest image hash mismatch: {relative}")
        if row.get("feedback_id") and row.get("split") == "test":
            raise ValueError("Curated feedback must never be assigned to the frozen test split")
    connection = connect_database(database_path)
    try:
        manifest = connection.execute(
            """SELECT dataset_manifest_id FROM dataset_manifests
               WHERE manifest_sha256 = ?
                 AND purpose IN ('curated_feedback', 'automatic_feedback')""",
            (manifest_hash,),
        ).fetchone()
        if manifest is None:
            if has_feedback_ids:
                raise ValueError("Curated feedback manifest is missing from the consent registry")
            return
        manifest_record = connection.execute(
            """SELECT dataset_manifest_id, purpose FROM dataset_manifests
               WHERE manifest_sha256 = ?
                AND purpose IN ('curated_feedback', 'automatic_feedback')""",
            (manifest_hash,),
        ).fetchone()
        rows = connection.execute(
            """SELECT di.feedback_id, di.consent_training, di.relative_path,
                      di.sha256, di.split, f.training_use_consent,
                      f.retention_consent, f.verification_status, f.response,
                      f.training_use_withdrawn_at, f.retention_withdrawn_at,
                      f.corrected_label,
                      i.sha256 AS retained_sha256, i.withdrawn_at, i.relative_path AS image_path
               FROM dataset_items di
               LEFT JOIN feedback f ON f.feedback_id = di.feedback_id
               LEFT JOIN feedback_images i ON i.feedback_id = di.feedback_id
               WHERE di.dataset_manifest_id = ?""",
            (manifest_record["dataset_manifest_id"],),
        ).fetchall()
        for row in rows:
            if row["feedback_id"] is None:
                continue
            manifest_row = feedback_rows.get(str(row["feedback_id"]))
            image_path = (
                project_root / str(row["image_path"] or "")
            ).resolve()
            if manifest_row is None:
                raise ValueError("Registered curated manifest is missing a feedback row")
            if (
                not row["consent_training"]
                or not row["training_use_consent"]
                or not row["retention_consent"]
                or row["training_use_withdrawn_at"]
                or row["retention_withdrawn_at"]
                or (
                    manifest_record["purpose"] == "curated_feedback"
                    and row["verification_status"] != "verified"
                )
                or (
                    manifest_record["purpose"] == "automatic_feedback"
                    and row["response"] not in {"yes", "no"}
                )
                or row["corrected_label"] not in SUPPORTED_CLASSES
                or manifest_row.get("label") != row["corrected_label"]
                or manifest_row.get("sha256") != row["sha256"]
                or row["split"] == "test"
                or row["withdrawn_at"]
                or not image_path.is_relative_to(project_root)
                or not image_path.is_file()
                or row["retained_sha256"] != row["sha256"]
            ):
                raise ValueError(
                    "Curated training manifest contains feedback with withdrawn or missing consent/image"
                )
    finally:
        connection.close()


def export_curated_manifest(
    database_path: str | Path,
    feedback_ids: list[str],
    *,
    protected_test_manifest: str | Path,
    destination: str | Path,
    seed: int = 42,
) -> dict[str, object]:
    if not feedback_ids:
        raise ValueError("At least one feedback ID is required")
    if len(set(feedback_ids)) != len(feedback_ids):
        raise ValueError("Feedback IDs must be unique")
    protected_hashes = _protected_test_hashes(protected_test_manifest)
    existing_hashes = _existing_dataset_hashes(protected_test_manifest)
    base_fields, base_rows = _read_base_manifest(protected_test_manifest)
    project_root = Path(database_path).resolve().parent
    _verify_base_manifest_files(project_root, base_rows)
    fields = list(
        dict.fromkeys(
            [
                *base_fields,
                *DatasetImage.__dataclass_fields__,
                "dataset_item_id",
                "canonical_taxon_version_id",
                "rights_review_status",
                "quality_review_status",
            ]
        )
    )
    connection = connect_database(database_path)
    manifest_id = str(uuid.uuid4())
    output_path = Path(destination)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing curated manifest: {output_path}")
    rows = []
    try:
        items = []
        for feedback_id in feedback_ids:
            item, reasons = _eligibility_row(
                connection, feedback_id, project_root
            )
            item_hashes = {item.get("sha256"), item.get("upload_sha256")}
            if item_hashes & protected_hashes:
                reasons.append("image_hash_overlaps_frozen_test")
            if item_hashes & existing_hashes:
                reasons.append("image_hash_overlaps_existing_supported_dataset")
            if reasons:
                raise ValueError(f"Feedback {feedback_id} is ineligible: {', '.join(reasons)}")
            items.append(item)
        hashes = [str(item["sha256"]) for item in items]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Selected feedback images contain exact duplicates")

        rng = random.Random(seed)
        split_by_hash: dict[str, str] = {}
        shuffled_hashes = sorted(hashes, key=lambda digest: rng.random())
        train_count = round(len(shuffled_hashes) * 0.8235)
        if len(shuffled_hashes) > 1:
            train_count = min(len(shuffled_hashes) - 1, max(1, train_count))
        for index, digest in enumerate(shuffled_hashes):
            split_by_hash[digest] = "train" if index < train_count else "validation"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{manifest_id}.tmp")
        manifest_rows = [
            {**row, "feedback_id": row.get("feedback_id", "")}
            for row in base_rows
        ]
        try:
            with temporary_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(manifest_rows)
                for item in items:
                    split = split_by_hash[str(item["sha256"])]
                    image_path = Path(database_path).resolve().parent / str(item["relative_path"])
                    with Image.open(image_path) as image:
                        width, height = image.size
                    row = {field: "" for field in fields}
                    row.update({
                        "dataset_item_id": str(uuid.uuid4()),
                        "feedback_id": item["feedback_id"],
                        "relative_path": item["relative_path"],
                        "sha256": item["sha256"],
                        "image_id": str(uuid.uuid4()),
                        "image_status": "valid",
                        "width": width,
                        "height": height,
                        "label": item["corrected_label"],
                        "duplicate_group": item["sha256"],
                        "canonical_taxon_version_id": "",
                        "canonical_taxon_id": "",
                        "source": item["source"],
                        "license": item["license"],
                        "attribution": item["attribution"],
                        "review_status": "verified",
                        "rights_review_status": "approved",
                        "quality_review_status": "approved",
                        "split": split,
                    })
                    rows.append(row)
                    writer.writerow(row)
            manifest_hash = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
            temporary_path.replace(output_path)
            with connection:
                connection.execute(
                    """INSERT INTO dataset_manifests
                       (dataset_manifest_id, manifest_sha256, purpose, split_policy_version,
                        split_seed, manifest_path)
                       VALUES (?, ?, 'curated_feedback', 'train-validation-only-v1', ?, ?)""",
                    (manifest_id, manifest_hash, seed, str(output_path.resolve())),
                )
                base_items = []
                for row in base_rows:
                    if row.get("image_status") != "valid" or row.get("split") not in {
                        "train",
                        "validation",
                        "test",
                    }:
                        continue
                    taxon_id = row.get("canonical_taxon_id") or None
                    if taxon_id and not connection.execute(
                        "SELECT 1 FROM taxa WHERE taxon_version_id = ?",
                        (taxon_id,),
                    ).fetchone():
                        taxon_id = None
                    base_items.append(
                        (
                            str(uuid.uuid4()),
                            manifest_id,
                            row["relative_path"],
                            row["sha256"],
                            row.get("duplicate_group") or "",
                            taxon_id,
                            row.get("review_status") or "unreviewed",
                            row.get("source") or "",
                            row.get("license") or "",
                            row["split"],
                        )
                    )
                connection.executemany(
                    """INSERT INTO dataset_items
                       (dataset_item_id, dataset_manifest_id, relative_path, sha256,
                        duplicate_group, taxon_version_id, label_status, source, license,
                        consent_training, split)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    base_items,
                )
                connection.executemany(
                    """INSERT INTO dataset_items
                       (dataset_item_id, dataset_manifest_id, relative_path, sha256,
                       duplicate_group, taxon_version_id, label_status, source, license, consent_training,
                       split, feedback_id)
                       VALUES (?, ?, ?, ?, ?, ?, 'verified', ?, ?, 1, ?, ?)""",
                    [
                        (
                            row["dataset_item_id"],
                            manifest_id,
                            row["relative_path"],
                            row["sha256"],
                            row["duplicate_group"],
                            row["canonical_taxon_version_id"] or None,
                            row["source"],
                            row["license"],
                            row["split"],
                            row["feedback_id"],
                        )
                        for row in rows
                    ],
                )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise
        return {
            "dataset_manifest_id": manifest_id,
            "manifest_path": str(output_path.resolve()),
            "manifest_sha256": manifest_hash,
            "item_count": len(rows),
            "split_policy": "deterministic train/validation 70:15 ratio among additions; never assign to frozen test",
            "seed": seed,
        }
    finally:
        connection.close()
