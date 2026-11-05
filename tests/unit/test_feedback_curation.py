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
from wildlife_ai.taxonomy.importer import import_taxonomy_csv


def _image_bytes(color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (18, 12), color).save(output, format="JPEG")
    return output.getvalue()


def _setup_project(tmp_path: Path) -> tuple[Path, Path, str]:
    project = tmp_path
    database_path = project / "wildlife_ai.db"
    taxonomy_path = project / "taxonomy.csv"
    taxonomy_path.write_text(
        "source_taxon_id,accepted_scientific_name,accepted_common_name,rank,taxonomic_status\n"
        "cat,Felis catus,Domestic Cat,species,accepted\n",
        encoding="utf-8",
    )
    import_taxonomy_csv(
        taxonomy_path,
        database_path,
        source_namespace="test-fixture",
        release="fixture-1",
        license="fixture licence",
        attribution="Synthetic unit-test row",
    )
    connection = connect_database(database_path)
    try:
        taxon_id = connection.execute(
            "SELECT taxon_version_id FROM taxa WHERE source_taxon_id = 'cat'"
        ).fetchone()[0]
    finally:
        connection.close()

    protected_manifest = project / "data" / "manifests" / "supported.csv"
    protected_manifest.parent.mkdir(parents=True, exist_ok=True)
    test_image_path = project / "data" / "raw" / "Cat" / "frozen-test.jpg"
    test_image_path.parent.mkdir(parents=True, exist_ok=True)
    test_image_path.write_bytes(_image_bytes((88, 99, 111)))
    protected_manifest.write_text(
        "image_status,split,sha256,relative_path,label,image_id,duplicate_group\n"
        "valid,test,"
        + ("a" * 64)
        + ",Cat/frozen-test.jpg,Domestic Cat,test-image,"
        + ("a" * 64)
        + "\n",
        encoding="utf-8",
    )
    return database_path, protected_manifest, taxon_id


def _submit_verified_feedback(
    project: Path,
    database_path: Path,
    taxon_id: str,
    *,
    image: bytes | None = None,
) -> tuple[str, bytes]:
    image = image or _image_bytes()
    feedback_id = submit_feedback(
        database_path,
        project / "data" / "feedback" / "images",
        response="no",
        predicted_label="Domestic Dog",
        model_version_id=None,
        model_score=0.82,
        original_typed_name="Domestic Cat",
        resolved_taxon_version_id=taxon_id,
        resolver_method="exact",
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
    database_path, _, taxon_id = _setup_project(tmp_path)
    no_consent_id = submit_feedback(
        database_path,
        tmp_path / "data" / "feedback" / "images",
        response="unsure",
        predicted_label="Lion",
        model_version_id=None,
        model_score=None,
        resolved_taxon_version_id=taxon_id,
        image_bytes=_image_bytes(),
        retention_consent=False,
        training_use_consent=False,
    )
    connection = connect_database(database_path)
    try:
        record = connection.execute(
            "SELECT retention_consent, training_use_consent FROM feedback WHERE feedback_id = ?",
            (no_consent_id,),
        ).fetchone()
        assert tuple(record) == (0, 0)
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


def test_review_and_curation_export_are_eligible_and_loader_compatible(tmp_path: Path) -> None:
    database_path, protected_manifest, taxon_id = _setup_project(tmp_path)
    feedback_id, _ = _submit_verified_feedback(tmp_path, database_path, taxon_id)
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
    database_path, protected_manifest, taxon_id = _setup_project(tmp_path)
    shared_image = _image_bytes()
    first_id, _ = _submit_verified_feedback(
        tmp_path, database_path, taxon_id, image=shared_image
    )
    duplicate_id, _ = _submit_verified_feedback(
        tmp_path, database_path, taxon_id, image=shared_image
    )
    duplicate_result = check_training_eligibility(
        database_path, duplicate_id, protected_test_manifest=protected_manifest
    )
    assert "duplicate_feedback_image" in duplicate_result["reasons"]

    manifest_hash = hashlib.sha256(shared_image).hexdigest()
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
    database_path, protected_manifest, taxon_id = _setup_project(tmp_path)
    feedback_id, _ = _submit_verified_feedback(tmp_path, database_path, taxon_id)
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
    database_path, protected_manifest, taxon_id = _setup_project(tmp_path)
    feedback_id, _ = _submit_verified_feedback(tmp_path, database_path, taxon_id)
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
