from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.data.preparation import (
    assign_splits,
    inspect_dataset,
    split_counts,
    write_manifest,
)


def load_metadata_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise ValueError("Metadata CSV must include a relative_path column")
        metadata: dict[str, dict[str, str]] = {}
        for row in reader:
            relative_path = (row.pop("relative_path") or "").replace("\\", "/").strip()
            if not relative_path:
                raise ValueError("Metadata CSV contains a row without relative_path")
            if relative_path in metadata:
                raise ValueError(f"Duplicate metadata row for {relative_path}")
            metadata[relative_path] = {
                key: value or "" for key, value in row.items() if key is not None
            }
    return metadata


def hash_manifest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    settings = ProjectSettings.from_repo_root()
    parser = argparse.ArgumentParser(
        description="Inspect class-folder images and write a reproducible dataset manifest."
    )
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_data_dir)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.manifests_dir / "dataset_manifest.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="Optional CSV with relative_path and provenance/review/group columns.",
    )
    args = parser.parse_args()

    metadata = load_metadata_csv(args.metadata_csv)
    inspection = inspect_dataset(args.raw_dir, metadata=metadata)
    split_records = assign_splits(inspection.records, seed=args.seed)
    manifest_path = write_manifest(split_records, args.manifest)
    manifest_hash = hash_manifest(manifest_path)
    summary = inspection.summary
    summary["seed"] = args.seed
    summary["split_counts"] = split_counts(split_records)
    summary["manifest_path"] = str(manifest_path.resolve())
    summary["manifest_sha256"] = manifest_hash
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
