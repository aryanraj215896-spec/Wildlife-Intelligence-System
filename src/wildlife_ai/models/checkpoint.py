from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.models.architectures import ModelName, build_model


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trusted local checkpoint using PyTorch's restricted unpickler."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a mapping")
    state_dict = checkpoint.get("state_dict")
    metadata = checkpoint.get("metadata")
    if not isinstance(state_dict, dict) or not isinstance(metadata, dict):
        raise ValueError("Checkpoint must contain state_dict and metadata mappings")
    if metadata.get("class_names") != list(SUPPORTED_CLASSES):
        raise ValueError("Checkpoint class order does not match configured supported classes")
    architecture = metadata.get("architecture")
    if architecture not in {"scratch_cnn", "resnet18"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model = build_model(architecture, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, metadata
