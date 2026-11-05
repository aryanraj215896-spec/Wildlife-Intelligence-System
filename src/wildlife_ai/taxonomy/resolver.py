from __future__ import annotations

import sqlite3
from typing import Any

from rapidfuzz import fuzz

from wildlife_ai.taxonomy.normalization import normalize_name


def resolve_name(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("Candidate limit must be positive")
    normalized = normalize_name(query)
    if not normalized:
        return {
            "query": query,
            "normalized_query": normalized,
            "method": "unresolved",
            "requires_confirmation": True,
            "ambiguous": False,
            "candidates": [],
        }
    rows = connection.execute(
        """SELECT DISTINCT t.taxon_version_id, t.source_namespace, t.source_taxon_id,
                  t.source_release, t.accepted_scientific_name, t.accepted_common_name,
                  t.rank, t.taxonomic_status, n.display_name, n.normalized_name, n.name_kind,
                  p.accepted_scientific_name AS parent_scientific_name
           FROM taxon_names n
           JOIN taxa t ON t.taxon_version_id = n.taxon_version_id
           LEFT JOIN taxa p ON p.taxon_version_id = t.parent_taxon_version_id"""
    ).fetchall()
    matches = [row for row in rows if row["normalized_name"] == normalized]
    method = "exact" if matches else "fuzzy"
    if not matches:
        scored: dict[str, tuple[float, sqlite3.Row]] = {}
        for row in rows:
            score = float(fuzz.WRatio(normalized, row["normalized_name"]))
            current = scored.get(row["taxon_version_id"])
            if current is None or score > current[0]:
                scored[row["taxon_version_id"]] = (score, row)
        ranked = sorted(scored.values(), key=lambda item: (-item[0], item[1]["accepted_scientific_name"]))
        selected = [(score, row) for score, row in ranked[:limit] if score > 0]
    else:
        selected = [(100.0, row) for row in matches[:limit]]

    candidates = [
        {
            "taxon_version_id": row["taxon_version_id"],
            "source_namespace": row["source_namespace"],
            "source_taxon_id": row["source_taxon_id"],
            "source_release": row["source_release"],
            "accepted_scientific_name": row["accepted_scientific_name"],
            "accepted_common_name": row["accepted_common_name"],
            "rank": row["rank"],
            "taxonomic_status": row["taxonomic_status"],
            "matched_name": row["display_name"],
            "matched_name_kind": row["name_kind"],
            "parent_scientific_name": row["parent_scientific_name"],
            "score": score,
        }
        for score, row in selected
    ]
    return {
        "query": query,
        "normalized_query": normalized,
        "method": method if candidates else "unresolved",
        "requires_confirmation": True,
        "ambiguous": len(candidates) > 1 and method == "exact",
        "candidates": candidates,
    }
