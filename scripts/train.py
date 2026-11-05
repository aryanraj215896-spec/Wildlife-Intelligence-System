from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.training.engine import TrainingConfig, train_model


def main() -> None:
    settings = ProjectSettings.from_repo_root()
    parser = argparse.ArgumentParser(description="Train a local animal classifier.")
    parser.add_argument("--model", choices=("scratch_cnn", "resnet18"), required=True)
    parser.add_argument("--manifest", type=Path, default=settings.manifests_dir / "dataset_manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_data_dir)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or (
        settings.root_dir
        / "models"
        / "runs"
        / f"{args.model}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    config = TrainingConfig(
        model_name=args.model,
        manifest_path=args.manifest,
        raw_root=args.raw_dir,
        output_dir=output_dir,
        database_path=settings.database_path,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.workers,
        seed=args.seed,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
    )
    print(json.dumps(train_model(config), indent=2))


if __name__ == "__main__":
    main()
