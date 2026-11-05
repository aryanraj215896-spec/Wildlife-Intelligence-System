from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torchvision import __version__ as torchvision_version

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.curation.eligibility import validate_training_manifest_consent
from wildlife_ai.data.loading import ManifestImageDataset, build_transforms, make_data_loader
from wildlife_ai.models.architectures import ModelName, build_model
from wildlife_ai.utils.memory import system_memory_snapshot


@dataclass(frozen=True)
class TrainingConfig:
    model_name: ModelName
    manifest_path: Path
    raw_root: Path
    output_dir: Path
    database_path: Path | None = None
    epochs: int = 10
    fine_tune_epochs: int = 0
    batch_size: int | None = None
    learning_rate: float | None = None
    fine_tune_learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    num_workers: int = 2
    seed: int = 42
    device: Literal["auto", "cuda", "cpu"] = "auto"
    mixed_precision: bool = True
    image_size: int = 224


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_epoch(
    model: nn.Module,
    loader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    if any(parameter.device.type != device.type for parameter in model.parameters()):
        raise RuntimeError(f"Model parameters are not on the selected device {device}")
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if images.device.type != device.type or labels.device.type != device.type:
                raise RuntimeError(f"Training batch did not transfer to selected device {device}")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(images)
                loss = loss_function(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.detach().item() * labels.size(0)
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    return total_loss / total, correct / total


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(
            {"state_dict": model.state_dict(), "metadata": metadata},
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _default_batch_size(model_name: ModelName) -> int:
    return 16 if model_name == "resnet18" else 32


def train_model(config: TrainingConfig) -> dict[str, object]:
    if config.epochs <= 0 or config.fine_tune_epochs < 0:
        raise ValueError("epochs must be positive and fine_tune_epochs non-negative")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if config.batch_size is not None and config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not config.manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {config.manifest_path}")
    if not config.raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {config.raw_root}")
    if config.database_path is not None and config.database_path.is_file():
        validate_training_manifest_consent(config.database_path, config.manifest_path)
    else:
        with config.manifest_path.open(newline="", encoding="utf-8") as stream:
            columns = next(csv.reader(stream), [])
        if "feedback_id" in columns:
            raise FileNotFoundError(
                "A consent registry database is required to train from curated feedback"
            )

    device = resolve_device(config.device)
    batch_size = (
        config.batch_size
        if config.batch_size is not None
        else _default_batch_size(config.model_name)
    )
    learning_rate = config.learning_rate or (0.001 if config.model_name == "scratch_cnn" else 0.001)
    mixed_precision = config.mixed_precision and device.type == "cuda"
    _seed_everything(config.seed)

    normalization = "imagenet" if config.model_name == "resnet18" else "scratch"
    train_dataset = ManifestImageDataset(
        config.manifest_path,
        config.raw_root,
        "train",
        build_transforms(normalization, training=True, image_size=config.image_size),
    )
    validation_dataset = ManifestImageDataset(
        config.manifest_path,
        config.raw_root,
        "validation",
        build_transforms(normalization, training=False, image_size=config.image_size),
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("Training and validation splits must both contain valid images")
    train_loader = make_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device.type,
        seed=config.seed,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device.type,
        seed=config.seed,
    )

    pretrained = config.model_name == "resnet18"
    model = build_model(
        config.model_name,
        pretrained=pretrained,
        freeze_backbone=pretrained,
    ).to(device)
    if any(parameter.device.type != device.type for parameter in model.parameters()):
        raise RuntimeError(f"Model failed to move to selected device {device}")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    started_at = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    system_memory_before = system_memory_snapshot()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_accuracy = -1.0
    best_checkpoint = config.output_dir / f"best-{config.model_name}.pt"
    last_checkpoint = config.output_dir / f"last-{config.model_name}.pt"
    history: list[dict[str, object]] = []
    total_epochs = config.epochs + config.fine_tune_epochs

    for epoch in range(total_epochs):
        if epoch == config.epochs and config.fine_tune_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.fine_tune_learning_rate,
                weight_decay=config.weight_decay,
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started_at = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, loss_function, device, optimizer, scaler, mixed_precision
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_elapsed_seconds = time.perf_counter() - train_started_at
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, loss_function, device, None, scaler, mixed_precision
        )
        ram_snapshot = system_memory_snapshot()
        row = {
            "epoch": epoch + 1,
            "stage": "fine_tune" if epoch >= config.epochs else "head" if pretrained else "scratch",
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "train_samples_per_second": len(train_dataset) / train_elapsed_seconds,
            "system_ram_used_bytes": ram_snapshot["used_bytes"],
            "system_ram_available_bytes": ram_snapshot["available_bytes"],
            "gpu_memory_allocated_bytes": (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else None
            ),
            "gpu_memory_reserved_bytes": (
                torch.cuda.memory_reserved(device) if device.type == "cuda" else None
            ),
            "gpu_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
        history.append(row)
        metadata = _checkpoint_metadata(
            config,
            device,
            normalization,
            started_at,
            history,
            epoch + 1,
            _sha256(config.manifest_path),
        )
        _save_checkpoint(last_checkpoint, model, metadata)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            _save_checkpoint(best_checkpoint, model, metadata)

    elapsed_seconds = time.perf_counter() - start_clock
    history_path = config.output_dir / f"history-{config.model_name}.csv"
    _write_history(history_path, history)
    return {
        "model_name": config.model_name,
        "device": str(device),
        "model_device": str(next(model.parameters()).device),
        "batch_device": str(next(model.parameters()).device),
        "cuda_verified": (
            device.type == "cuda"
            and all(parameter.device.type == "cuda" for parameter in model.parameters())
        ),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "vram_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None
        ),
        "mixed_precision": mixed_precision,
        "batch_size": batch_size,
        "epochs_completed": total_epochs,
        "best_validation_accuracy": best_accuracy,
        "elapsed_seconds": elapsed_seconds,
        "system_memory_before": system_memory_before,
        "system_memory_after": system_memory_snapshot(),
        "peak_gpu_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "history_csv": str(history_path.resolve()),
    }


def _checkpoint_metadata(
    config: TrainingConfig,
    device: torch.device,
    normalization: str,
    started_at: datetime,
    history: list[dict[str, object]],
    epoch: int,
    manifest_hash: str,
) -> dict[str, object]:
    from torch import __version__ as torch_version

    return {
        "model_version": f"{config.model_name}-{started_at.strftime('%Y%m%dT%H%M%SZ')}",
        "architecture": config.model_name,
        "class_names": list(SUPPORTED_CLASSES),
        "class_to_index": {name: index for index, name in enumerate(SUPPORTED_CLASSES)},
        "class_taxonomy_ids": {name: None for name in SUPPORTED_CLASSES},
        "input_size": [config.image_size, config.image_size],
        "normalization": normalization,
        "training_config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "batch_size": (
                config.batch_size
                if config.batch_size is not None
                else _default_batch_size(config.model_name)
            ),
        },
        "dataset_manifest_sha256": manifest_hash,
        "dataset_split_seed": config.seed,
        "epoch": epoch,
        "validation_metrics": dict(history[-1]),
        "pretrained_weights": (
            "Torchvision ResNet18_Weights.DEFAULT (ImageNet-1K V1)"
            if config.model_name == "resnet18"
            else None
        ),
        "software": {
            "python": platform.python_version(),
            "torch": str(torch_version),
            "torchvision": torchvision_version,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "git_revision": _git_revision(),
        "started_at": started_at.isoformat(),
        "history": history,
    }
