from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.data.loading import build_transforms
from wildlife_ai.training.engine import resolve_device
from wildlife_ai.utils.images import decode_sanitized_image


def prepare_model_input(
    metadata: dict[str, Any],
    image_bytes: bytes,
) -> tuple[torch.Tensor, Image.Image]:
    normalization = metadata.get("normalization")
    if normalization not in {"scratch", "imagenet"}:
        raise ValueError(f"Unsupported checkpoint normalization: {normalization}")
    class_names = metadata.get("class_names")
    if class_names != list(SUPPORTED_CLASSES):
        raise ValueError("Checkpoint class order does not match configured supported classes")
    input_size = metadata.get("input_size")
    if not isinstance(input_size, (list, tuple)) or not input_size:
        raise ValueError("Checkpoint is missing a valid input_size")
    image = decode_sanitized_image(image_bytes)
    image_size = int(input_size[0])
    transform = build_transforms(
        normalization,
        training=False,
        image_size=image_size,
    )
    batch = transform(image).unsqueeze(0)
    resize_size = round(image_size * 256 / 224)
    model_view = transforms.Compose(
        [transforms.Resize(resize_size), transforms.CenterCrop(image_size)]
    )(image)
    return batch, model_view


def predict_image(
    model: nn.Module,
    metadata: dict[str, Any],
    image_bytes: bytes,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    target_device = device or resolve_device("auto")
    batch, _ = prepare_model_input(metadata, image_bytes)
    batch = batch.to(
        target_device, non_blocking=target_device.type == "cuda"
    )
    model.eval()
    with torch.inference_mode():
        logits = model(batch)
        scores, indexes = torch.softmax(logits, dim=1).topk(
            min(3, len(SUPPORTED_CLASSES)), dim=1
        )
    top_outputs = [
        {"label": SUPPORTED_CLASSES[index], "score": float(score)}
        for index, score in zip(indexes[0].cpu().tolist(), scores[0].cpu().tolist())
    ]
    return {
        "predicted_label": top_outputs[0]["label"],
        "model_score": top_outputs[0]["score"],
        "score_type": "softmax model score; not a calibrated probability",
        "top_outputs": top_outputs,
        "model_version": metadata.get("model_version", "unversioned"),
        "architecture": metadata.get("architecture", "unknown"),
        "device": str(target_device),
    }
