from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.evaluation.metrics import evaluate_checkpoint


def main() -> None:
    settings = ProjectSettings.from_repo_root()
    parser = argparse.ArgumentParser(description="Evaluate a trusted local checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=settings.manifests_dir / "dataset_manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_data_dir)
    parser.add_argument("--output-dir", type=Path, default=settings.root_dir / "reports" / "metrics")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.raw_dir,
        args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
