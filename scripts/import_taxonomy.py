from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.taxonomy.importer import import_taxonomy_csv


def main() -> None:
    settings = ProjectSettings.from_repo_root()
    parser = argparse.ArgumentParser(description="Import a reviewed taxonomy CSV snapshot.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--source-namespace", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--attribution", required=True)
    args = parser.parse_args()
    result = import_taxonomy_csv(
        args.csv,
        args.database,
        source_namespace=args.source_namespace,
        release=args.release,
        license=args.license,
        attribution=args.attribution,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
