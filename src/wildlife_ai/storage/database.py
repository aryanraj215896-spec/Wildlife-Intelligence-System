from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE taxonomy_imports (
    taxonomy_import_id TEXT PRIMARY KEY,
    source_namespace TEXT NOT NULL,
    release TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    license TEXT NOT NULL,
    attribution TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    UNIQUE(source_namespace, release)
);
CREATE TABLE taxon_concepts (
    taxon_concept_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE taxa (
    taxon_version_id TEXT PRIMARY KEY,
    taxonomy_import_id TEXT NOT NULL REFERENCES taxonomy_imports(taxonomy_import_id),
    source_namespace TEXT NOT NULL,
    source_taxon_id TEXT NOT NULL,
    accepted_scientific_name TEXT NOT NULL,
    accepted_common_name TEXT NOT NULL DEFAULT '',
    rank TEXT NOT NULL,
    parent_taxon_version_id TEXT REFERENCES taxa(taxon_version_id),
    parent_source_taxon_id TEXT NOT NULL DEFAULT '',
    taxonomic_status TEXT NOT NULL,
    source_release TEXT NOT NULL,
    taxon_concept_id TEXT REFERENCES taxon_concepts(taxon_concept_id),
    UNIQUE(taxonomy_import_id, source_namespace, source_taxon_id)
);
CREATE TABLE taxon_names (
    taxon_name_id INTEGER PRIMARY KEY,
    taxon_version_id TEXT NOT NULL REFERENCES taxa(taxon_version_id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    name_kind TEXT NOT NULL CHECK(name_kind IN ('scientific', 'common', 'synonym', 'curated_alias')),
    language TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'source',
    UNIQUE(taxon_version_id, normalized_name, name_kind)
);
CREATE INDEX idx_taxon_names_normalized ON taxon_names(normalized_name);
CREATE TABLE taxon_crosswalks (
    taxon_crosswalk_id TEXT PRIMARY KEY,
    from_taxon_version_id TEXT NOT NULL REFERENCES taxa(taxon_version_id),
    to_taxon_version_id TEXT NOT NULL REFERENCES taxa(taxon_version_id),
    relationship TEXT NOT NULL CHECK(relationship IN ('same', 'rename', 'split', 'merge', 'uncertain')),
    review_status TEXT NOT NULL CHECK(review_status IN ('pending', 'approved', 'rejected')),
    evidence TEXT NOT NULL DEFAULT '',
    reviewer_reference TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT
);
CREATE TABLE training_runs (
    training_run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    dataset_manifest_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_path TEXT NOT NULL DEFAULT ''
);
CREATE TABLE model_versions (
    model_version_id TEXT PRIMARY KEY,
    architecture TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL CHECK(lifecycle_state IN ('candidate', 'active', 'rejected', 'retired')),
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    training_run_id TEXT REFERENCES training_runs(training_run_id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decision_at TEXT,
    decision_by TEXT NOT NULL DEFAULT '',
    decision_rationale TEXT NOT NULL DEFAULT ''
);
CREATE TABLE model_classes (
    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
    output_index INTEGER NOT NULL,
    display_label TEXT NOT NULL,
    taxon_version_id TEXT REFERENCES taxa(taxon_version_id),
    PRIMARY KEY(model_version_id, output_index),
    UNIQUE(model_version_id, display_label)
);
CREATE TABLE active_model (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE dataset_manifests (
    dataset_manifest_id TEXT PRIMARY KEY,
    manifest_sha256 TEXT NOT NULL,
    purpose TEXT NOT NULL,
    split_policy_version TEXT NOT NULL,
    split_seed INTEGER NOT NULL,
    manifest_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE dataset_items (
    dataset_item_id TEXT PRIMARY KEY,
    dataset_manifest_id TEXT NOT NULL REFERENCES dataset_manifests(dataset_manifest_id),
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    duplicate_group TEXT NOT NULL DEFAULT '',
    taxon_version_id TEXT REFERENCES taxa(taxon_version_id),
    label_status TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL DEFAULT '',
    consent_training INTEGER NOT NULL DEFAULT 0 CHECK(consent_training IN (0, 1)),
    split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'test')),
    feedback_id TEXT REFERENCES feedback(feedback_id),
    UNIQUE(dataset_manifest_id, relative_path)
);
CREATE TABLE feedback (
    feedback_id TEXT PRIMARY KEY,
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    response TEXT NOT NULL CHECK(response IN ('yes', 'no', 'unsure', 'unsupported')),
    predicted_label TEXT NOT NULL,
    corrected_label TEXT NOT NULL DEFAULT '',
    image_sha256 TEXT NOT NULL DEFAULT '',
    predicted_taxon_version_id TEXT REFERENCES taxa(taxon_version_id),
    model_version_id TEXT REFERENCES model_versions(model_version_id),
    model_score REAL,
    score_type TEXT NOT NULL DEFAULT 'softmax_score',
    original_typed_name TEXT NOT NULL DEFAULT '',
    resolved_taxon_version_id TEXT REFERENCES taxa(taxon_version_id),
    resolver_method TEXT NOT NULL DEFAULT '',
    duplicate_of_feedback_id TEXT REFERENCES feedback(feedback_id),
    verification_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(verification_status IN ('pending', 'verified', 'rejected', 'needs_review')),
    retention_consent INTEGER NOT NULL DEFAULT 0 CHECK(retention_consent IN (0, 1)),
    retention_consent_at TEXT,
    retention_withdrawn_at TEXT,
    training_use_consent INTEGER NOT NULL DEFAULT 0 CHECK(training_use_consent IN (0, 1)),
    training_use_consent_at TEXT,
    training_use_withdrawn_at TEXT
);
CREATE TABLE feedback_images (
    feedback_image_id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL UNIQUE REFERENCES feedback(feedback_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    upload_sha256 TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL DEFAULT '',
    attribution TEXT NOT NULL DEFAULT '',
    rights_review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(rights_review_status IN ('pending', 'approved', 'rejected')),
    quality_review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(quality_review_status IN ('pending', 'approved', 'rejected')),
    withdrawn_at TEXT
);
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES feedback(feedback_id),
    previous_status TEXT NOT NULL,
    new_status TEXT NOT NULL CHECK(new_status IN ('verified', 'rejected', 'needs_review')),
    reviewer_reference TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE promotion_policies (
    policy_id TEXT PRIMARY KEY,
    approved_by TEXT NOT NULL,
    approval_reference TEXT NOT NULL,
    criteria_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE model_registry_events (
    event_id TEXT PRIMARY KEY,
    action TEXT NOT NULL CHECK(action IN ('candidate_registered', 'promoted', 'rejected', 'rolled_back')),
    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
    previous_model_version_id TEXT REFERENCES model_versions(model_version_id),
    policy_id TEXT REFERENCES promotion_policies(policy_id),
    actor_reference TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect_database(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    version_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if version_table is None:
        connection.executescript(
            "BEGIN EXCLUSIVE;\n"
            "CREATE TABLE schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);\n"
            + _SCHEMA
            + "\nINSERT INTO schema_migrations(version) VALUES (1);\nCOMMIT;"
        )
    version = connection.execute(
        "SELECT MAX(version) FROM schema_migrations"
    ).fetchone()[0]
    if version < 2:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            _ensure_column(connection, "dataset_manifests", "manifest_path", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, "feedback", "duplicate_of_feedback_id", "TEXT")
            _ensure_column(connection, "feedback", "retention_consent_at", "TEXT")
            _ensure_column(connection, "feedback", "training_use_consent_at", "TEXT")
            _ensure_column(connection, "feedback", "retention_withdrawn_at", "TEXT")
            _ensure_column(connection, "feedback", "training_use_withdrawn_at", "TEXT")
            _ensure_column(connection, "dataset_items", "feedback_id", "TEXT REFERENCES feedback(feedback_id)")
            _ensure_column(connection, "feedback_images", "upload_sha256", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, "feedback_images", "source", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, "feedback_images", "license", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(connection, "feedback_images", "attribution", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(
                connection,
                "feedback_images",
                "rights_review_status",
                "TEXT NOT NULL DEFAULT 'pending'",
            )
            _ensure_column(
                connection,
                "feedback_images",
                "quality_review_status",
                "TEXT NOT NULL DEFAULT 'pending'",
            )
            _ensure_column(connection, "feedback_images", "withdrawn_at", "TEXT")
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (2)"
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        version = 2
    if version < 3:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS promotion_policies (
                   policy_id TEXT PRIMARY KEY,
                   approved_by TEXT NOT NULL,
                   approval_reference TEXT NOT NULL,
                   criteria_json TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                   )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS model_registry_events (
                   event_id TEXT PRIMARY KEY,
                   action TEXT NOT NULL CHECK(action IN (
                       'candidate_registered', 'promoted', 'rejected', 'rolled_back'
                   )),
                   model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
                   previous_model_version_id TEXT REFERENCES model_versions(model_version_id),
                   policy_id TEXT REFERENCES promotion_policies(policy_id),
                   actor_reference TEXT NOT NULL,
                   rationale TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                   )"""
            )
            connection.execute("INSERT INTO schema_migrations(version) VALUES (3)")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        version = 3
    if version < 4:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            _ensure_column(
                connection,
                "feedback",
                "corrected_label",
                "TEXT NOT NULL DEFAULT ''",
            )
            _ensure_column(
                connection,
                "feedback",
                "image_sha256",
                "TEXT NOT NULL DEFAULT ''",
            )
            feedback_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(feedback)")
            }
            if {"response", "predicted_label"} <= feedback_columns:
                connection.execute(
                    """UPDATE feedback SET corrected_label = predicted_label
                       WHERE response = 'yes' AND corrected_label = ''"""
                )
            connection.execute("INSERT INTO schema_migrations(version) VALUES (4)")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        version = 4
    if version < 5:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            connection.execute(
                """CREATE TABLE application_model_selection (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
                    validation_report_path TEXT NOT NULL,
                    validation_report_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                """CREATE TABLE automatic_training_batches (
                    batch_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('running', 'accepted', 'rejected', 'failed')),
                    candidate_model_version_id TEXT REFERENCES model_versions(model_version_id),
                    evaluation_json TEXT NOT NULL DEFAULT '{}',
                    message TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                )"""
            )
            connection.execute(
                """CREATE TABLE automatic_training_batch_feedback (
                    batch_id TEXT NOT NULL REFERENCES automatic_training_batches(batch_id),
                    feedback_id TEXT NOT NULL UNIQUE REFERENCES feedback(feedback_id),
                    PRIMARY KEY(batch_id, feedback_id)
                )"""
            )
            connection.execute("INSERT INTO schema_migrations(version) VALUES (5)")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return connection


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {
        row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
