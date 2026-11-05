from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.data.loading import ManifestImageDataset, build_transforms, make_data_loader
from wildlife_ai.models.architectures import build_model
from wildlife_ai.training.engine import resolve_device
from wildlife_ai.utils.memory import system_memory_snapshot


def _gpu_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _benchmark_loader(
    dataset: ManifestImageDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, object]:
    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        device=device.type,
        seed=42,
    )
    iterator = iter(loader)
    batches = 0
    samples = 0
    started_at = 0.0
    while batches < warmup_batches + measured_batches:
        try:
            images, _ = next(iterator)
        except StopIteration:
            break
        batches += 1
        if batches == warmup_batches:
            samples = 0
            started_at = time.perf_counter()
        elif batches > warmup_batches:
            samples += images.size(0)
    if batches > warmup_batches:
        elapsed = time.perf_counter() - started_at
        measured = batches - warmup_batches
        throughput = samples / elapsed if elapsed else 0.0
    else:
        elapsed = 0.0
        measured = 0
        throughput = 0.0
    del iterator, loader
    return {
        "batch_size": batch_size,
        "workers": workers,
        "measured_batches": measured,
        "samples_per_second": round(throughput, 2),
        "elapsed_seconds": round(elapsed, 3),
    }


def _benchmark_training(
    dataset: ManifestImageDataset,
    *,
    model_name: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    warmup_batches: int,
    measured_batches: int,
    mixed_precision: bool,
) -> dict[str, object]:
    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        device=device.type,
        seed=42,
    )
    model = build_model(model_name).to(device)
    if any(parameter.device.type != device.type for parameter in model.parameters()):
        raise RuntimeError("Benchmark model parameters are not on the selected device")
    model_device = str(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    amp_enabled = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.train()
    iterator = iter(loader)
    batches = 0
    samples = 0
    started_at = 0.0
    batch_device: str | None = None
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while batches < warmup_batches + measured_batches:
            try:
                images, labels = next(iterator)
            except StopIteration:
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if images.device.type != device.type or labels.device.type != device.type:
                raise RuntimeError("Benchmark batch did not transfer to the selected device")
            batch_device = str(images.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            if batches == warmup_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples = 0
                started_at = time.perf_counter()
            elif batches > warmup_batches:
                samples += images.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        if batches > warmup_batches:
            elapsed = time.perf_counter() - started_at
            measured = batches - warmup_batches
            throughput = samples / elapsed if elapsed else 0.0
            status = "ok"
        else:
            elapsed = 0.0
            measured = 0
            throughput = 0.0
            status = "insufficient_batches"
    except torch.cuda.OutOfMemoryError:
        status = "out_of_memory"
        elapsed = 0.0
        measured = batches
        throughput = 0.0
    finally:
        del iterator, loader, model, optimizer, scaler, loss_function
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "workers": workers,
        "status": status,
        "model_device": model_device,
        "batch_device": batch_device,
        "measured_batches": measured,
        "samples_per_second": round(throughput, 2),
        "elapsed_seconds": round(elapsed, 3),
        "mixed_precision": amp_enabled,
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "gpu_memory_after": _gpu_memory_snapshot(device),
    }


def main() -> None:
    settings = ProjectSettings.from_repo_root()
    parser = argparse.ArgumentParser(
        description="Benchmark data loading and real training steps on the selected device."
    )
    parser.add_argument("--model", choices=("scratch_cnn", "resnet18"), default="scratch_cnn")
    parser.add_argument("--manifest", type=Path, default=settings.manifests_dir / "dataset_manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_data_dir)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--worker-counts", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--measured-batches", type=int, default=12)
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()
    if any(size <= 0 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if any(count < 0 for count in args.worker_counts):
        parser.error("worker counts cannot be negative")
    if args.warmup_batches < 1 or args.measured_batches < 1:
        parser.error("warmup and measured batch counts must be positive")

    device = resolve_device(args.device)
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {args.manifest}")
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {args.raw_dir}")
    dataset = ManifestImageDataset(
        args.manifest,
        args.raw_dir,
        "train",
        build_transforms("imagenet" if args.model == "resnet18" else "scratch", training=True),
    )
    if not dataset:
        raise ValueError("The training split contains no valid images")

    memory_before = system_memory_snapshot()
    gpu_before = _gpu_memory_snapshot(device)
    loader_results = [
        _benchmark_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            device=device,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
        )
        for workers in args.worker_counts
        for batch_size in args.batch_sizes
    ]
    training_results = [
        _benchmark_training(
            dataset,
            model_name=args.model,
            batch_size=batch_size,
            workers=2 if 2 in args.worker_counts else args.worker_counts[0],
            device=device,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
            mixed_precision=not args.no_mixed_precision,
        )
        for batch_size in args.batch_sizes
    ]
    print(
        json.dumps(
            {
                "model": args.model,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cuda_verified": (
                    device.type == "cuda"
                    and all(
                        result["model_device"].startswith("cuda")
                        and result["batch_device"] is not None
                        and result["batch_device"].startswith("cuda")
                        for result in training_results
                    )
                ),
                "batch_size_sweep": list(args.batch_sizes),
                "system_memory_before": memory_before,
                "system_memory_after": system_memory_snapshot(),
                "gpu_memory_before": gpu_before,
                "data_loader": loader_results,
                "training_steps": training_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
