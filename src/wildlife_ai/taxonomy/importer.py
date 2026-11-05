from __future__ import annotations

import csv
import hashlib
import uuid
from pathlib import Path

from wildlife_ai.storage.database import connect_database
from wildlife_ai.taxonomy.normalization import normalize_name

REQUIRED_COLUMNS = {
    "source_taxon_id",
    "accepted_scientific_name",
    "rank",
    "taxonomic_status",
}


def import_taxonomy_csv(
    csv_path: str | Path,
    database_path: str | Path,
    *,
    source_namespace: str,
    release: str,
    license: str,
    attribution: str,
) -> dict[str, object]:
    path = Path(csv_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"Taxonomy CSV is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not source_namespace.strip() or not release.strip():
        raise ValueError("Taxonomy source namespace and release must be non-empty")
    if not license.strip() or not attribution.strip():
        raise ValueError("Taxonomy licence and attribution must be documented")

    records: dict[str, dict[str, str]] = {}
    for row in rows:
        source_id = (row.get("source_taxon_id") or "").strip()
        scientific_name = (row.get("accepted_scientific_name") or "").strip()
        if not source_id or not scientific_name:
            raise ValueError("Every taxonomy row needs a source ID and accepted scientific name")
        if source_id in records:
            raise ValueError(f"Duplicate source taxon ID in import: {source_id}")
        records[source_id] = {key: (value or "").strip() for key, value in row.items()}

    import_id = str(uuid.uuid4())
    version_ids = {
        source_id: hashlib.sha256(
            f"{source_namespace}\0{release}\0{source_id}".encode("utf-8")
        ).hexdigest()
        for source_id in records
    }
    connection = connect_database(database_path)
    try:
        with connection:
            connection.execute(
                """INSERT INTO taxonomy_imports
                   (taxonomy_import_id, source_namespace, release, license, attribution, source_sha256)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (import_id, source_namespace, release, license, attribution, digest),
            )
            for source_id, row in records.items():
                parent_source_id = row.get("parent_source_taxon_id", "")
                connection.execute(
                    """INSERT INTO taxa
                       (taxon_version_id, taxonomy_import_id, source_namespace, source_taxon_id,
                        accepted_scientific_name, accepted_common_name, rank,
                        parent_taxon_version_id, parent_source_taxon_id, taxonomic_status, source_release)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        version_ids[source_id],
                        import_id,
                        source_namespace,
                        source_id,
                        row["accepted_scientific_name"],
                        row.get("accepted_common_name", ""),
                        row["rank"],
                        version_ids.get(parent_source_id),
                        parent_source_id,
                        row["taxonomic_status"],
                        release,
                    ),
                )
                names = [
                    (row["accepted_scientific_name"], "scientific"),
                    (row.get("accepted_common_name", ""), "common"),
                ]
                names.extend((name.strip(), "common") for name in row.get("other_common_names", "").split("|"))
                names.extend((name.strip(), "synonym") for name in row.get("synonyms", "").split("|"))
                for display_name, kind in names:
                    normalized = normalize_name(display_name)
                    if not normalized:
                        continue
                    connection.execute(
                        """INSERT OR IGNORE INTO taxon_names
                           (taxon_version_id, display_name, normalized_name, name_kind, provenance)
                           VALUES (?, ?, ?, ?, ?)""",
                        (version_ids[source_id], display_name, normalized, kind, source_namespace),
                    )
    finally:
        connection.close()

    return {
        "taxonomy_import_id": import_id,
        "source_namespace": source_namespace,
        "release": release,
        "row_count": len(records),
        "sha256": digest,
    }
