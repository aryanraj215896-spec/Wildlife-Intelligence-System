import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.data.loading import ManifestImageDataset, build_transforms, make_data_loader
from wildlife_ai.data.preparation import assign_splits, inspect_dataset, write_manifest
from wildlife_ai.evaluation.metrics import _abstention_curve, _classification_metrics
from wildlife_ai.models.architectures import build_model
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.training.engine import _save_checkpoint


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="PNG")


def test_inspection_reports_valid_corrupt_and_unknown_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "Cheetah" / "valid.png", (1, 2, 3))
    corrupt_path = raw_dir / "jaguar" / "corrupt.png"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not an image")
    (raw_dir / "Cheetah" / "notes.txt").write_text("not an image", encoding="utf-8")
    (raw_dir / "unknown").mkdir(parents=True)

    result = inspect_dataset(raw_dir)

    assert result.unknown_class_directories == ("unknown",)
    assert result.summary["total_discovered"] == 2
    assert result.summary["total_valid"] == 1
    assert result.summary["total_rejected"] == 1
    assert result.summary["class_availability"]["Cheetah"] == "available"
    assert result.summary["class_availability"]["Gray wolf"] == "awaiting_data"
    by_path = {record.relative_path: record for record in result.records}
    assert by_path["Cheetah/valid.png"].label == "Cheetah"
    assert by_path["jaguar/corrupt.png"].image_status == "rejected"
    assert "Cheetah/notes.txt" not in by_path


def test_capitalized_cat_and_dog_folders_use_canonical_labels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "Cat" / "cat.png", (1, 2, 3))
    _write_image(raw_dir / "Dog" / "dog.png", (3, 2, 1))

    records = inspect_dataset(raw_dir).records

    labels = {record.relative_path: record.label for record in records}
    assert labels == {
        "Cat/cat.png": "Domestic Cat",
        "Dog/dog.png": "Domestic Dog",
    }


def test_grouped_split_is_reproducible_and_keeps_exact_duplicates_together(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    for index in range(10):
        _write_image(
            raw_dir / "Cat" / f"cat-{index}.png",
            (index, index + 1, index + 2),
        )
    duplicate = raw_dir / "Cat" / "cat-duplicate.png"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_dir / "cat" / "cat-0.png", duplicate)

    records = inspect_dataset(raw_dir).records
    first = assign_splits(records, seed=7)
    repeated = assign_splits(records, seed=7)

    assert first == repeated
    split_by_path = {record.relative_path: record.split for record in first}
    assert split_by_path["Cat/cat-0.png"] == split_by_path["Cat/cat-duplicate.png"]
    assert {record.split for record in first} == {"train", "validation", "test"}


def test_three_independent_groups_populate_all_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    for index in range(3):
        _write_image(raw_dir / "tiger" / f"tiger-{index}.png", (50, 20, index))

    split_records = assign_splits(inspect_dataset(raw_dir).records)

    assert {record.split for record in split_records} == {"train", "validation", "test"}


def test_observation_group_is_not_split_across_partitions(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "dog" / "dog-1.png", (20, 1, 2))
    _write_image(raw_dir / "dog" / "dog-2.png", (20, 1, 3))
    metadata = {
        "dog/dog-1.png": {"group_id": "sequence-a"},
        "dog/dog-2.png": {"group_id": "sequence-a"},
    }
    records = inspect_dataset(raw_dir, metadata=metadata).records

    split_records = assign_splits(records)

    assert split_records[0].split == split_records[1].split


def test_conflicting_labels_for_same_duplicate_group_are_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    cat_path = raw_dir / "cat" / "same.png"
    _write_image(cat_path, (100, 100, 100))
    dog_path = raw_dir / "dog" / "same.png"
    dog_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cat_path, dog_path)

    records = inspect_dataset(raw_dir).records

    with pytest.raises(ValueError, match="conflicting labels"):
        assign_splits(records)


def test_manifest_includes_rejected_records_and_provenance_fields(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    corrupt_path = raw_dir / "jaguar" / "broken.jpg"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"bad")
    records = assign_splits(inspect_dataset(raw_dir).records)
    destination = tmp_path / "manifests" / "manifest.csv"

    write_manifest(records, destination)

    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["label"] == "Jaguar"
    assert rows[0]["image_status"] == "rejected"
    assert rows[0]["license"] == ""
    assert rows[0]["canonical_taxon_id"] == ""


def test_metadata_is_attached_and_unknown_paths_are_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "jaguar" / "image.png", (90, 20, 1))
    metadata = {
        "jaguar/image.png": {
            "source": "Example collection",
            "license": "CC BY 4.0",
            "review_status": "reviewed",
        }
    }

    record = inspect_dataset(raw_dir, metadata=metadata).records[0]
    assert record.source == "Example collection"
    assert record.license == "CC BY 4.0"
    assert record.review_status == "reviewed"

    with pytest.raises(ValueError, match="not found"):
        inspect_dataset(raw_dir, metadata={"jaguar/missing.png": {}})


def test_prepare_dataset_cli_writes_manifest_and_hash(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "cheetah" / "sample.png", (4, 5, 6))
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_dataset.py"
    manifest_path = tmp_path / "output" / "manifest.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--raw-dir",
            str(raw_dir),
            "--manifest",
            str(manifest_path),
            "--seed",
            "19",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert report["class_counts"]["Cheetah"]["valid"] == 1
    assert report["split_counts"]["Cheetah"]["train"] == 1
    assert report["manifest_sha256"] == expected_hash


def test_manifest_loader_uses_valid_rows_and_canonical_class_indices(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "Cat" / "sample.png", (8, 20, 40))
    records = assign_splits(inspect_dataset(raw_dir).records)
    manifest_path = write_manifest(records, tmp_path / "manifest.csv")
    dataset = ManifestImageDataset(
        manifest_path,
        raw_dir,
        "train",
        transform=build_transforms("scratch", training=False, image_size=32),
    )

    image, target = dataset[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert SUPPORTED_CLASSES[target] == "Domestic Cat"


def test_manifest_loader_rejects_paths_outside_raw_root(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_status,split,label,relative_path\n"
        "valid,train,Domestic Cat,../outside.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes raw data root"):
        ManifestImageDataset(manifest_path, raw_dir, "train")


def test_training_loader_uses_pinned_prefetched_persistent_workers() -> None:
    import torch
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.zeros(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))

    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        device="cuda",
        seed=42,
    )

    assert loader.pin_memory
    assert loader.persistent_workers
    assert loader.prefetch_factor == 2


def test_cpu_loader_disables_cuda_pinning_without_worker_processes() -> None:
    import torch
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.zeros(2, 3, 8, 8), torch.zeros(2, dtype=torch.long))
    loader = make_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        device="cpu",
        seed=42,
    )

    assert not loader.pin_memory
    assert not loader.persistent_workers
    assert loader.prefetch_factor is None


def test_models_return_eight_class_logits() -> None:
    import torch

    inputs = torch.rand(2, 3, 224, 224)
    scratch = build_model("scratch_cnn")
    resnet = build_model("resnet18", pretrained=False)

    assert tuple(scratch(inputs).shape) == (2, 8)
    assert tuple(resnet(inputs).shape) == (2, 8)


def test_checkpoint_round_trip_uses_restricted_torch_loading(tmp_path: Path) -> None:
    import torch

    model = build_model("scratch_cnn")
    metadata = {
        "architecture": "scratch_cnn",
        "class_names": list(SUPPORTED_CLASSES),
        "model_version": "test-version",
        "normalization": "scratch",
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint_path, model, metadata)

    loaded, loaded_metadata = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert loaded.training is False
    assert loaded_metadata["model_version"] == "test-version"
    assert tuple(loaded(torch.zeros(1, 3, 224, 224)).shape) == (1, 8)


def test_classification_metrics_compute_macro_and_per_class_values() -> None:
    aggregate, per_class = _classification_metrics(
        [
            [2, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            *([[0] * 8] * 6),
        ]
    )

    assert aggregate["sample_count"] == 4
    assert aggregate["accuracy"] == 0.75
    assert per_class[0]["support"] == 3
    assert len(per_class) == 8


def test_abstention_curve_reports_coverage_without_selecting_a_threshold() -> None:
    curve = _abstention_curve([(0, 0, 0.9), (0, 1, 0.4)])

    assert curve[0]["coverage"] == 1.0
    assert curve[-1]["coverage"] == 0.0
    assert curve[10]["threshold"] == 0.5
    assert curve[10]["coverage"] == 0.5
