
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.data.loading import (
    ManifestImageDataset,
    build_transforms,
    make_data_loader,
)
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.training.engine import resolve_device
from wildlife_ai.evaluation.visualization import (
    write_confusion_svg,
    write_history_svg,
)


def _classification_metrics(
    confusion: list[list[int]],
) -> tuple[dict[str, Any], list[dict[str, float | int | str]]]:
    total = sum(sum(row) for row in confusion)
    correct = sum(
        confusion[index][index]
        for index in range(len(confusion))
    )

    per_class: list[dict[str, float | int | str]] = []

    for index, label in enumerate(SUPPORTED_CLASSES):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)

        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro = {
        metric: sum(float(row[metric]) for row in per_class)
        / len(per_class)
        for metric in ("precision", "recall", "f1")
    }

    return (
        {
            "accuracy": correct / total if total else 0.0,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
            "sample_count": total,
        },
        per_class,
    )


def _abstention_curve(
    prediction_rows: list[tuple[int, int, float]],
) -> list[dict[str, float | int]]:
    total = len(prediction_rows)
    curve = []

    for step in range(21):
        threshold = step / 20
        accepted = [
            row for row in prediction_rows
            if row[2] >= threshold
        ]

        correct = sum(
            expected == predicted
            for expected, predicted, _ in accepted
        )

        coverage = len(accepted) / total if total else 0.0

        curve.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": coverage,
                "selective_accuracy": (
                    correct / len(accepted) if accepted else 0.0
                ),
                "supported_false_rejection_rate": 1.0 - coverage,
            }
        )

    return curve


def _write_misclassification_artifacts(
    misclassified_images: list[tuple[Path, float]],
    output_dir: Path,
    split: str,
) -> tuple[Path, Path]:

    csv_path = (
        output_dir
        / f"cheetah-as-leopard-{split}-misclassifications.csv"
    )

    contact_sheet_path = (
        output_dir
        / f"cheetah-as-leopard-{split}-contact-sheet.jpg"
    )

    # Export the original image paths and model scores.
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.writer(stream)

        writer.writerow(
            [
                "image_path",
                "true_label",
                "predicted_label",
                "top1_score",
            ]
        )

        for image_path, score in misclassified_images:
            writer.writerow(
                [
                    str(image_path),
                    "Cheetah",
                    "Leopard",
                    score,
                ]
            )

    # Build a contact sheet for visual inspection.
    columns = 3
    cell_width = 360
    image_height = 230
    caption_height = 52

    image_count = len(misclassified_images)

    rows = max(
        1,
        (image_count + columns - 1) // columns,
    )

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * (image_height + caption_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    if not misclassified_images:
        draw.text(
            (20, 20),
            "No Cheetah-to-Leopard misclassifications found.",
            fill="black",
        )

    for index, (image_path, score) in enumerate(
        misclassified_images
    ):
        x = (index % columns) * cell_width
        y = (index // columns) * (
            image_height + caption_height
        )

        with Image.open(image_path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail(
                (cell_width - 20, image_height - 20)
            )

            paste_x = x + (
                cell_width - thumbnail.width
            ) // 2

            paste_y = y + (
                image_height - thumbnail.height
            ) // 2

            sheet.paste(
                thumbnail,
                (paste_x, paste_y),
            )

        draw.text(
            (x + 8, y + image_height + 4),
            f"{index + 1}. {image_path.name[:42]}",
            fill="black",
        )

        draw.text(
            (x + 8, y + image_height + 25),
            f"Cheetah -> Leopard | Top-1 score: {score:.3f}",
            fill="black",
        )

    sheet.save(
        contact_sheet_path,
        quality=90,
    )

    return csv_path, contact_sheet_path


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:

    if split not in {"validation", "test"}:
        raise ValueError(
            "Evaluation split must be validation or test"
        )

    if batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "batch_size must be positive and num_workers non-negative"
        )

    device = resolve_device(device_name)

    model, metadata = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    normalization = metadata.get("normalization")

    if normalization not in {"scratch", "imagenet"}:
        raise ValueError(
            f"Invalid checkpoint normalization: {normalization}"
        )

    dataset = ManifestImageDataset(
        manifest_path,
        raw_root,
        split,
        build_transforms(
            normalization,
            training=False,
            image_size=int(metadata["input_size"][0]),
        ),
    )

    if not dataset:
        raise ValueError(
            f"The {split} split contains no valid images"
        )

    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device.type,
        seed=seed,
    )

    class_count = len(SUPPORTED_CLASSES)

    confusion = [
        [0 for _ in range(class_count)]
        for _ in range(class_count)
    ]

    confidence_values: list[float] = []
    prediction_rows: list[tuple[int, int, float]] = []
    forward_seconds: list[float] = []

    # Store paths of images that are actually Cheetahs
    # but are predicted as Leopards.
    cheetah_index = SUPPORTED_CLASSES.index("Cheetah")
    leopard_index = SUPPORTED_CLASSES.index("Leopard")

    misclassified_images: list[tuple[Path, float]] = []

    total_seen = 0

    model.eval()

    with torch.inference_mode():
        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )

            labels = labels.to(
                device,
                non_blocking=device.type == "cuda",
            )

            if total_seen == 0:
                _ = model(images)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            started = time.perf_counter()

            logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            forward_seconds.append(
                time.perf_counter() - started
            )

            probabilities = torch.softmax(logits, dim=1)

            scores, predictions = probabilities.max(dim=1)

            expected_values = labels.cpu().tolist()
            predicted_values = predictions.cpu().tolist()
            score_values = scores.cpu().tolist()

            # DataLoader uses shuffle=False, so these paths
            # correspond to the current batch in exact order.
            batch_samples = dataset.samples[
                total_seen:total_seen + len(expected_values)
            ]

            if len(batch_samples) != len(expected_values):
                raise RuntimeError(
                    "Dataset samples and evaluation batch are out of sync"
                )

            for (
                (image_path, _),
                expected,
                predicted,
                score,
            ) in zip(
                batch_samples,
                expected_values,
                predicted_values,
                score_values,
            ):

                confusion[expected][predicted] += 1

                confidence_values.append(
                    float(score)
                )

                prediction_rows.append(
                    (expected, predicted, float(score))
                )

                # Capture only Cheetah images predicted as Leopard.
                if (
                    expected == cheetah_index
                    and predicted == leopard_index
                ):
                    misclassified_images.append(
                        (image_path, float(score))
                    )

            total_seen += labels.size(0)

    aggregate, per_class = _classification_metrics(
        confusion
    )

    elapsed_forward = sum(forward_seconds)

    checkpoint = Path(checkpoint_path)

    destination = Path(output_dir)
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = (
        destination.parent
        / "figures"
        / str(metadata["model_version"])
    )

    figures = {
        "confusion_matrix_svg": str(
            write_confusion_svg(
                SUPPORTED_CLASSES,
                confusion,
                figures_dir / f"{split}-confusion.svg",
            ).resolve()
        )
    }

    training_config = metadata.get(
        "training_config",
        {},
    )

    if (
        isinstance(training_config, dict)
        and training_config.get("output_dir")
    ):
        history_path = (
            Path(str(training_config["output_dir"]))
            / f"history-{metadata['architecture']}.csv"
        )

        if history_path.is_file():
            figures["training_history_svg"] = str(
                write_history_svg(
                    history_path,
                    figures_dir / "training-history.svg",
                ).resolve()
            )

    # Export the targeted errors.
    misclassification_csv, contact_sheet = (
        _write_misclassification_artifacts(
            misclassified_images,
            destination,
            split,
        )
    )

    report: dict[str, Any] = {
        "model_version": metadata["model_version"],
        "architecture": metadata["architecture"],
        "split": split,
        "dataset_manifest_sha256": metadata[
            "dataset_manifest_sha256"
        ],
        "sample_count": total_seen,
        "metrics": aggregate,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(SUPPORTED_CLASSES),
            "counts": confusion,
        },
        "targeted_misclassifications": {
            "true_label": "Cheetah",
            "predicted_label": "Leopard",
            "count": len(misclassified_images),
            "csv_path": str(
                misclassification_csv.resolve()
            ),
            "contact_sheet_path": str(
                contact_sheet.resolve()
            ),
        },
        "inference": {
            "measurement": (
                "model forward pass only; first batch warmed up; "
                "CUDA synchronized; data loading excluded"
            ),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "batch_size": batch_size,
            "total_forward_seconds": elapsed_forward,
            "images_per_second": (
                total_seen / elapsed_forward
                if elapsed_forward
                else None
            ),
            "mean_batch_forward_seconds": (
                elapsed_forward / len(forward_seconds)
                if forward_seconds
                else None
            ),
            "mean_model_score": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            ),
        },
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "figures": figures,
        "abstention_validation": (
            {
                "method": (
                    "maximum softmax score threshold sweep; "
                    "scores are not calibrated probabilities"
                ),
                "curve": _abstention_curve(prediction_rows),
                "selected_threshold": None,
                "ood_evaluation": (
                    "not_run: no separately versioned OOD "
                    "development/final datasets were supplied"
                ),
            }
            if split == "validation"
            else None
        ),
    }

    report_path = (
        destination
        / f"{metadata['architecture']}-{split}-metrics.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    confusion_path = (
        destination
        / f"{metadata['architecture']}-{split}-confusion.csv"
    )

    with confusion_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.writer(stream)

        writer.writerow(
            ["actual/predicted", *SUPPORTED_CLASSES]
        )

        for label, row in zip(
            SUPPORTED_CLASSES,
            confusion,
        ):
            writer.writerow(
                [label, *row]
            )

    report["report_path"] = str(
        report_path.resolve()
    )

    report["confusion_csv"] = str(
        confusion_path.resolve()
    )

    return report
