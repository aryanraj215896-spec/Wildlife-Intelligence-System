from __future__ import annotations

import csv
import hashlib
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from wildlife_ai.config.settings import CLASS_DIRECTORIES, SUPPORTED_CLASSES

_CLASS_DIRECTORY_TO_LABEL_CASEFOLD = {
    folder.casefold(): label for folder, label in CLASS_DIRECTORIES
}
_SPLITS = ("train", "validation", "test")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetImage:
    image_id: str
    relative_path: str
    label: str
    sha256: str | None
    duplicate_group: str | None
    group_id: str | None
    image_status: str
    rejection_reason: str | None
    width: int | None
    height: int | None
    split: str | None = None
    source: str = ""
    source_image_id: str = ""
    license: str = ""
    attribution: str = ""
    collection_date: str = ""
    canonical_taxon_id: str = ""
    review_status: str = "unreviewed"
    quality_notes: str = ""
    feedback_id: str = ""


@dataclass(frozen=True)
class DatasetInspection:
    records: tuple[DatasetImage, ...]
    unknown_class_directories: tuple[str, ...]

    @property
    def summary(self) -> dict[str, object]:
        counts = {
            label: {"discovered": 0, "valid": 0, "rejected": 0}
            for label in SUPPORTED_CLASSES
        }
        duplicate_counts: Counter[str] = Counter()
        for record in self.records:
            class_counts = counts[record.label]
            class_counts["discovered"] += 1
            class_counts["valid" if record.image_status == "valid" else "rejected"] += 1
            if record.sha256:
                duplicate_counts[record.sha256] += 1

        return {
            "class_counts": counts,
            "class_availability": {
                label: "available" if class_counts["valid"] else "awaiting_data"
                for label, class_counts in counts.items()
            },
            "total_discovered": len(self.records),
            "total_valid": sum(item["valid"] for item in counts.values()),
            "total_rejected": sum(item["rejected"] for item in counts.values()),
            "exact_duplicate_groups": sum(count > 1 for count in duplicate_counts.values()),
            "rejection_reasons": dict(
                Counter(
                    record.rejection_reason
                    for record in self.records
                    if record.rejection_reason is not None
                )
            ),
            "unknown_class_directories": list(self.unknown_class_directories),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_image(path: Path) -> tuple[int, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("unsupported_image_format")
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("invalid_image_dimensions")
    return width, height


def inspect_dataset(
    raw_root: str | Path,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> DatasetInspection:
    """Inspect class-folder images without modifying or deleting source files."""

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_directories = {
        path.name: _CLASS_DIRECTORY_TO_LABEL_CASEFOLD[path.name.casefold()]
        for path in root.iterdir()
        if path.is_dir() and path.name.casefold() in _CLASS_DIRECTORY_TO_LABEL_CASEFOLD
    }
    unknown_directories = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name not in class_directories
        )
    )
    supplied_metadata = metadata or {}
    records: list[DatasetImage] = []

    for folder, label in sorted(class_directories.items(), key=lambda item: item[0].casefold()):
        class_root = root / folder
        for path in sorted(class_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not path.is_file() or path.suffix.casefold() not in _IMAGE_SUFFIXES:
                continue
            relative_path = path.relative_to(root).as_posix()
            image_metadata = supplied_metadata.get(relative_path, {})
            try:
                digest = _sha256_file(path)
            except OSError:
                digest = None

            stable_id_material = f"{relative_path}\0{digest or ''}".encode("utf-8")
            image_id = hashlib.sha256(stable_id_material).hexdigest()[:24]
            width: int | None = None
            height: int | None = None
            rejection_reason: str | None = None
            try:
                if digest is None:
                    raise OSError("image_file_unreadable")
                width, height = _inspect_image(path)
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as exc:
                rejection_reason = str(exc) or "image_decode_failed"

            records.append(
                DatasetImage(
                    image_id=image_id,
                    relative_path=relative_path,
                    label=label,
                    sha256=digest,
                    duplicate_group=digest,
                    group_id=image_metadata.get("group_id") or None,
                    image_status="valid" if rejection_reason is None else "rejected",
                    rejection_reason=rejection_reason,
                    width=width,
                    height=height,
                    source=image_metadata.get("source", ""),
                    source_image_id=image_metadata.get("source_image_id", ""),
                    license=image_metadata.get("license", ""),
                    attribution=image_metadata.get("attribution", ""),
                    collection_date=image_metadata.get("collection_date", ""),
                    canonical_taxon_id=image_metadata.get("canonical_taxon_id", ""),
                    review_status=image_metadata.get("review_status", "unreviewed"),
                    quality_notes=image_metadata.get("quality_notes", ""),
                )
            )

    discovered_paths = {record.relative_path for record in records}
    unknown_metadata_paths = set(supplied_metadata) - discovered_paths
    if unknown_metadata_paths:
        paths = ", ".join(sorted(unknown_metadata_paths))
        raise ValueError(f"Metadata references files not found in the dataset: {paths}")

    return DatasetInspection(tuple(records), unknown_directories)


def _group_valid_records(records: Sequence[DatasetImage]) -> list[list[DatasetImage]]:
    valid_records = [record for record in records if record.image_status == "valid"]
    parents = {record.image_id: record.image_id for record in valid_records}

    def find(image_id: str) -> str:
        parent = parents[image_id]
        if parent != image_id:
            parents[image_id] = find(parent)
        return parents[image_id]

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    group_members: dict[tuple[str, str], list[str]] = {}
    for record in valid_records:
        if record.duplicate_group:
            group_members.setdefault(("duplicate", record.duplicate_group), []).append(record.image_id)
        if record.group_id:
            group_members.setdefault(("observation", record.group_id), []).append(record.image_id)
    for members in group_members.values():
        for image_id in members[1:]:
            union(members[0], image_id)

    grouped: dict[str, list[DatasetImage]] = {}
    for record in valid_records:
        grouped.setdefault(find(record.image_id), []).append(record)
    for group in grouped.values():
        labels = {record.label for record in group}
        if len(labels) != 1:
            paths = ", ".join(sorted(record.relative_path for record in group))
            raise ValueError(f"Duplicate/observation group has conflicting labels: {paths}")
    return list(grouped.values())


def assign_splits(
    records: Sequence[DatasetImage],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[DatasetImage, ...]:
    """Assign deterministic, class-stratified splits without separating groups."""

    if len(ratios) != len(_SPLITS) or any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")

    grouped_records = _group_valid_records(records)
    groups_by_label: dict[str, list[list[DatasetImage]]] = {
        label: [] for label in SUPPORTED_CLASSES
    }
    for group in grouped_records:
        label = group[0].label
        if label not in groups_by_label:
            raise ValueError(f"Unsupported dataset label: {label}")
        groups_by_label[label].append(group)

    split_by_image_id: dict[str, str] = {}
    for class_index, label in enumerate(SUPPORTED_CLASSES):
        groups = groups_by_label[label]
        total_images = sum(len(group) for group in groups)
        if not total_images:
            continue
        rng = random.Random(seed + class_index)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        targets = {split: ratio * total_images for split, ratio in zip(_SPLITS, ratios)}
        counts = dict.fromkeys(_SPLITS, 0)
        for index, group in enumerate(groups):
            empty_splits = [split for split in _SPLITS if counts[split] == 0]
            remaining_groups = len(groups) - index
            candidates = empty_splits if remaining_groups <= len(empty_splits) else _SPLITS
            selected_split = max(
                candidates,
                key=lambda split: targets[split] - counts[split],
            )
            counts[selected_split] += len(group)
            for record in group:
                split_by_image_id[record.image_id] = selected_split

    return tuple(
        replace(record, split=split_by_image_id.get(record.image_id))
        for record in records
    )


def write_manifest(records: Sequence[DatasetImage], destination: str | Path) -> Path:
    """Write all records, including rejected/unreviewed records, as a CSV manifest."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DatasetImage.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return output_path


def split_counts(records: Sequence[DatasetImage]) -> dict[str, dict[str, int]]:
    counts = {
        label: dict.fromkeys(_SPLITS, 0)
        for label in SUPPORTED_CLASSES
    }
    for record in records:
        if record.image_status == "valid" and record.split is not None:
            counts[record.label][record.split] += 1
    return counts
