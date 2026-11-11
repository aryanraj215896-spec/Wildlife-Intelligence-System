from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from wildlife_ai.storage.database import connect_database
from wildlife_ai.taxonomy.importer import import_taxonomy_csv
from wildlife_ai.taxonomy.normalization import normalize_name
from wildlife_ai.taxonomy.resolver import resolve_name


def _write_taxonomy_csv(path: Path) -> None:
    rows = [
        {
            "source_taxon_id": "test-hippo",
            "accepted_scientific_name": "Hippopotamus amphibius",
            "accepted_common_name": "Hippopotamus",
            "other_common_names": "Common hippopotamus",
            "rank": "species",
            "parent_source_taxon_id": "",
            "taxonomic_status": "accepted",
            "synonyms": "Hippopotamus amphibius",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_normalization_handles_case_unicode_punctuation_and_whitespace() -> None:
    assert normalize_name("  HÍPPO-potamus  ") == "híppo potamus"


def test_imported_taxonomy_resolves_exact_and_fuzzy_with_confirmation(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "source.csv"
    database_path = tmp_path / "taxonomy.db"
    _write_taxonomy_csv(csv_path)
    imported = import_taxonomy_csv(
        csv_path,
        database_path,
        source_namespace="test-fixture",
        release="fixture-1",
        license="fixture licence",
        attribution="Synthetic unit-test row",
    )
    connection = connect_database(database_path)
    try:
        exact = resolve_name(connection, "hippopotamus")
        fuzzy = resolve_name(connection, "hipo")
    finally:
        connection.close()

    assert imported["row_count"] == 1
    assert exact["method"] == "exact"
    assert exact["requires_confirmation"] is False
    assert exact["ambiguous"] is False
    assert exact["candidates"][0]["source_taxon_id"] == "test-hippo"
    assert fuzzy["method"] == "fuzzy"
    assert fuzzy["requires_confirmation"] is True
    assert fuzzy["candidates"][0]["accepted_common_name"] == "Hippopotamus"


def test_taxonomy_import_requires_release_metadata_and_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("source_taxon_id\nx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        import_taxonomy_csv(
            csv_path,
            tmp_path / "taxonomy.db",
            source_namespace="fixture",
            release="fixture-1",
            license="fixture",
            attribution="fixture",
        )


def test_database_migration_is_repeatable_and_enables_foreign_keys(tmp_path: Path) -> None:
    database_path = tmp_path / "wildlife.db"
    first = connect_database(database_path)
    first.close()
    second = connect_database(database_path)
    try:
        assert second.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert second.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 5
        tables = {
            row["name"]
            for row in second.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "application_model_selection",
            "automatic_training_batches",
            "automatic_training_batch_feedback",
        } <= tables
    finally:
        second.close()


def test_feedback_schema_upgrade_adds_phase_seven_columns(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """CREATE TABLE schema_migrations (
               version INTEGER PRIMARY KEY,
               applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
           );
           INSERT INTO schema_migrations(version) VALUES (1);
           CREATE TABLE dataset_manifests (
               dataset_manifest_id TEXT PRIMARY KEY,
               manifest_sha256 TEXT NOT NULL,
               purpose TEXT NOT NULL,
               split_policy_version TEXT NOT NULL,
               split_seed INTEGER NOT NULL
           );
           CREATE TABLE dataset_items (
               dataset_item_id TEXT PRIMARY KEY,
               dataset_manifest_id TEXT NOT NULL,
               relative_path TEXT NOT NULL,
               sha256 TEXT NOT NULL,
               split TEXT NOT NULL
           );
           CREATE TABLE feedback (
               feedback_id TEXT PRIMARY KEY,
               verification_status TEXT NOT NULL,
               retention_consent INTEGER NOT NULL DEFAULT 0,
               training_use_consent INTEGER NOT NULL DEFAULT 0,
               response TEXT NOT NULL DEFAULT 'no',
               predicted_label TEXT NOT NULL DEFAULT 'Lion'
           );
           CREATE TABLE feedback_images (
               feedback_image_id TEXT PRIMARY KEY,
               feedback_id TEXT NOT NULL,
               relative_path TEXT NOT NULL,
               sha256 TEXT NOT NULL
           );
           INSERT INTO feedback(feedback_id, verification_status)
           VALUES ('legacy-feedback', 'pending');"""
    )
    connection.close()

    upgraded = connect_database(database_path)
    try:
        assert upgraded.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0] == 5
        legacy = upgraded.execute(
            """SELECT feedback_id, verification_status, corrected_label, image_sha256
               FROM feedback WHERE feedback_id = 'legacy-feedback'"""
        ).fetchone()
        assert tuple(legacy) == ("legacy-feedback", "pending", "", "")
        assert {
            row["name"] for row in upgraded.execute("PRAGMA table_info(feedback_images)")
        } >= {
            "upload_sha256",
            "rights_review_status",
            "quality_review_status",
            "withdrawn_at",
        }
        assert {
            row["name"] for row in upgraded.execute("PRAGMA table_info(dataset_items)")
        } >= {"feedback_id"}
        tables = {
            row["name"]
            for row in upgraded.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"promotion_policies", "model_registry_events"} <= tables
    finally:
        upgraded.close()
