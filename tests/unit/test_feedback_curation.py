from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from wildlife_ai.curation.eligibility import (
    check_training_eligibility,
    export_curated_manifest,
    validate_training_manifest_consent,
)
from wildlife_ai.data.loading import ManifestImageDataset
from wildlife_ai.feedback.service import (
    review_feedback,
    submit_feedback,
    withdraw_retained_image,
    withdraw_training_consent,
)
from wildlife_ai.storage.database import connect_database


def _image_bytes(color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (18, 12), color).save(output, format="JPEG")
    return output.getvalue()


def _setup_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path
    database_path = project / "wildlife_ai.db"
    protected_manifest = project / "data" / "manifests" / "supported.csv"
    protected_manifest.parent.mkdir(parents=True, exist_ok=True)
    test_image_path = project / "data" / "raw" / "Cat" / "frozen-test.jpg"
    test_image_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_test_image = _image_bytes((88, 99, 111))
    test_image_path.write_bytes(frozen_test_image)
    test_hash = hashlib.sha256(frozen_test_image).hexdigest()
    protected_manifest.write_text(
        "image_status,split,sha256,relative_path,label,image_id,duplicate_group\n"
        f"valid,test,{test_hash},Cat/frozen-test.jpg,Domestic Cat,test-image,{test_hash}\n",
        encoding="utf-8",
    )
    return database_path, protected_manifest


def _submit_verified_feedback(
    project: Path,
    database_path: Path,
    *,
    image: bytes | None = None,
) -> tuple[str, bytes]:
    image = image or _image_bytes()
    feedback_id = submit_feedback(
        database_path,
        project / "data" / "feedback" / "images",
        response="no",
        predicted_label="Domestic Dog",
        corrected_label="Domestic Cat",
        model_version_id=None,
        model_score=0.82,
        image_bytes=image,
        retention_consent=True,
        training_use_consent=True,
        source="unit test",
        license="CC0",
        attribution="Synthetic unit-test image",
    )
    review_feedback(
        database_path,
        feedback_id,
        decision="verified",
        reviewer_reference="test reviewer",
        notes="Verified synthetic fixture",
        rights_approved=True,
        quality_approved=True,
    )
    return feedback_id, image


def test_feedback_consent_controls_retention_and_validates_upload(tmp_path: Path) -> None:
    database_path, _ = _setup_project(tmp_path)
    upload = _image_bytes()
    no_consent_id = submit_feedback(
        database_path,
        tmp_path / "data" / "feedback" / "images",
        response="unsure",
        predicted_label="Lion",
        model_version_id=None,
        model_score=None,
        image_bytes=upload,
        retention_consent=False,
        training_use_consent=False,
    )
    connection = connect_database(database_path)
    try:
        record = connection.execute(
            """SELECT retention_consent, training_use_consent, image_sha256,
                      corrected_label, submitted_at
               FROM feedback WHERE feedback_id = ?""",
            (no_consent_id,),
        ).fetchone()
        assert tuple(record) == (
            0,
            0,
            hashlib.sha256(upload).hexdigest(),
            "",
            record["submitted_at"],
        )
        assert record["submitted_at"]
        assert connection.execute("SELECT COUNT(*) FROM feedback_images").fetchone()[0] == 0
    finally:
        connection.close()

    with pytest.raises(ValueError, match="requires separate image-retention consent"):
        submit_feedback(
            database_path,
            tmp_path / "data" / "feedback" / "images",
            response="yes",
            predicted_label="Domestic Cat",
            model_version_id=None,
            model_score=0.7,
            image_bytes=_image_bytes(),
            retention_consent=False,
            training_use_consent=True,
        )
    with pytest.raises(ValueError, match="corrupt or unsafe"):
        submit_feedback(
            database_path,
            tmp_path / "data" / "feedback" / "images",
            response="yes",
            predicted_label="Domestic Cat",
            model_version_id=None,
            model_score=0.7,
            image_bytes=b"not an image",
            retention_consent=True,
            training_use_consent=False,
        )


def test_correct_and_incorrect_responses_store_direct_supported_labels(tmp_path: Path) -> None:
    database_path, _ = _setup_project(tmp_path)
    image = _image_bytes()
    correct_id = submit_feedback(
        database_path,
        tmp_path / "data" / "feedback" / "images",
        response="yes",
        predicted_label="Gray wolf",
        model_version_id=None,
        model_score=0.91,
        image_bytes=image,
    )
    with pytest.raises(ValueError, match="requires a supported corrected label"):
        submit_feedback(
            database_path,
            tmp_path / "data" / "feedback" / "images",
            response="no",
            predicted_label="Lion",
            model_version_id=None,
            model_score=0.8,
            image_bytes=image,
        )
    with pytest.raises(ValueError, match="different label"):
        submit_feedback(
            database_path,
            tmp_path / "data" / "feedback" / "images",
            response="no",
            predicted_label="Lion",
            corrected_label="Lion",
            model_version_id=None,
            model_score=0.8,
            image_bytes=image,
        )
    correction_id = submit_feedback(
        database_path,
        tmp_path / "data" / "feedback" / "images",
        response="no",
        predicted_label="Lion",
        corrected_label="Gray wolf",
        model_version_id=None,
        model_score=0.8,
        image_bytes=image,
    )
    connection = connect_database(database_path)
    try:
        rows = connection.execute(
            """SELECT response, predicted_label, corrected_label, model_version_id,
                      image_sha256, verification_status
               FROM feedback WHERE feedback_id IN (?, ?) ORDER BY response""",
            (correct_id, correction_id),
        ).fetchall()
    finally:
        connection.close()
    assert tuple(rows[0]) == (
        "no",
        "Lion",
        "Gray wolf",
        None,
        hashlib.sha256(image).hexdigest(),
        "pending",
    )
    assert tuple(rows[1]) == (
        "yes",
        "Gray wolf",
        "Gray wolf",
        None,
        hashlib.sha256(image).hexdigest(),
        "pending",
    )


def test_review_rejects_unverified_labels_and_rejected_items_are_ineligible(
    tmp_path: Path,
) -> None:
    database_path, protected_manifest = _setup_project(tmp_path)
    feedback_id = submit_feedback(
        database_path,
        tmp_path / "data" / "feedback" / "images",
        response="no",
        predicted_label="Lion",
        corrected_label="Domestic Cat",
        model_version_id=None,
        model_score=0.7,
        image_bytes=_image_bytes(),
        retention_consent=True,
        training_use_consent=True,
    )
    with pytest.raises(ValueError, match="supported corrected label"):
        review_feedback(
            database_path,
            feedback_id,
            decision="verified",
            corrected_label="not an animal",
        )
    review_feedback(
        database_path,
        feedback_id,
        decision="rejected",
        notes="Correction could not be substantiated",
    )
    eligibility = check_training_eligibility(
        database_path,
        feedback_id,
        protected_test_manifest=protected_manifest,
    )
    assert not eligibility["eligible"]
    assert "label_not_verified" in eligibility["reasons"]


def test_review_and_curation_export_are_eligible_and_loader_compatible(tmp_path: Path) -> None:
    database_path, protected_manifest = _setup_project(tmp_path)
    feedback_id, _ = _submit_verified_feedback(tmp_path, database_path)
    eligibility = check_training_eligibility(
        database_path, feedback_id, protected_test_manifest=protected_manifest
    )
    assert eligibility == {"feedback_id": feedback_id, "eligible": True, "reasons": []}

    destination = tmp_path / "data" / "manifests" / "curated.csv"
    result = export_curated_manifest(
        database_path,
        [feedback_id],
        protected_test_manifest=protected_manifest,
        destination=destination,
    )
    assert result["item_count"] == 1
    with pytest.raises(ValueError, match="missing from the consent registry"):
        validate_training_manifest_consent(tmp_path / "missing.db", destination)
    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = next(row for row in rows if row["feedback_id"] == feedback_id)
    assert row["label"] == "Domestic Cat"
    assert row["split"] in {"train", "validation"}
    assert row["image_status"] == "valid"
    assert row["relative_path"].startswith("data/feedback/images/")

    dataset = ManifestImageDataset(
        destination,
        tmp_path / "data" / "raw",
        row["split"],
    )
    assert len(dataset) == 1
    image, target = dataset[0]
    assert image.size == (18, 12)
    assert target == 0
    test_dataset = ManifestImageDataset(
        destination,
        tmp_path / "data" / "raw",
        "test",
    )
    assert len(test_dataset) == 1

    connection = connect_database(database_path)
    try:
        item = connection.execute(
            """SELECT feedback_id, consent_training, split FROM dataset_items
               WHERE feedback_id = ?""",
            (feedback_id,),
        ).fetchone()
        assert tuple(item) == (feedback_id, 1, row["split"])
    finally:
        connection.close()


def test_duplicate_and_frozen_test_images_are_not_curation_eligible(tmp_path: Path) -> None:
    database_path, protected_manifest = _setup_project(tmp_path)
    shared_image = _image_bytes()
    first_id, _ = _submit_verified_feedback(
        tmp_path, database_path, image=shared_image
    )
    duplicate_id, _ = _submit_verified_feedback(
        tmp_path, database_path, image=shared_image
    )
    duplicate_result = check_training_eligibility(
        database_path, duplicate_id, protected_test_manifest=protected_manifest
    )
    assert "duplicate_feedback_image" in duplicate_result["reasons"]

    manifest_hash = hashlib.sha256(shared_image).hexdigest()
    (tmp_path / "data" / "raw" / "Cat" / "frozen-test.jpg").write_bytes(shared_image)
    protected_manifest.write_text(
        "image_status,split,sha256,relative_path,label,image_id,duplicate_group\n"
        f"valid,test,{manifest_hash},Cat/frozen-test.jpg,Domestic Cat,test-image,{manifest_hash}\n",
        encoding="utf-8",
    )
    test_overlap = check_training_eligibility(
        database_path, first_id, protected_test_manifest=protected_manifest
    )
    assert "image_hash_overlaps_frozen_test" in test_overlap["reasons"]
    with pytest.raises(ValueError, match="image_hash_overlaps_frozen_test"):
        export_curated_manifest(
            database_path,
            [first_id],
            protected_test_manifest=protected_manifest,
            destination=tmp_path / "curated.csv",
        )


def test_withdrawal_removes_image_and_invalidates_curation(tmp_path: Path) -> None:
    database_path, protected_manifest = _setup_project(tmp_path)
    feedback_id, _ = _submit_verified_feedback(tmp_path, database_path)
    destination = tmp_path / "data" / "manifests" / "curated.csv"
    export_curated_manifest(
        database_path,
        [feedback_id],
        protected_test_manifest=protected_manifest,
        destination=destination,
    )
    validate_training_manifest_consent(database_path, destination)
    connection = connect_database(database_path)
    try:
        image_path = tmp_path / connection.execute(
            "SELECT relative_path FROM feedback_images WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()[0]
    finally:
        connection.close()

    withdrawal = withdraw_retained_image(database_path, feedback_id)
    assert not image_path.exists()
    assert withdrawal["affected_manifests"]
    result = check_training_eligibility(
        database_path, feedback_id, protected_test_manifest=protected_manifest
    )
    assert not result["eligible"]
    assert "training_use_not_consented" in result["reasons"]
    assert "retained_image_missing_or_withdrawn" in result["reasons"]
    with pytest.raises(ValueError, match="withdrawn or missing consent"):
        validate_training_manifest_consent(database_path, destination)


def test_training_consent_can_be_withdrawn_without_deleting_retained_image(
    tmp_path: Path,
) -> None:
    database_path, protected_manifest = _setup_project(tmp_path)
    feedback_id, _ = _submit_verified_feedback(tmp_path, database_path)
    affected = withdraw_training_consent(database_path, feedback_id)
    assert affected["affected_manifests"] == []
    connection = connect_database(database_path)
    try:
        row = connection.execute(
            """SELECT training_use_consent, training_use_withdrawn_at
               FROM feedback WHERE feedback_id = ?""",
            (feedback_id,),
        ).fetchone()
        image_path = tmp_path / connection.execute(
            "SELECT relative_path FROM feedback_images WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert row["training_use_consent"] == 0
    assert row["training_use_withdrawn_at"]
    assert image_path.is_file()
    result = check_training_eligibility(
        database_path, feedback_id, protected_test_manifest=protected_manifest
    )
    assert "training_use_not_consented" in result["reasons"]
