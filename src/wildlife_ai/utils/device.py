from __future__ import annotations

from typing import Literal

DeviceName = Literal["cpu", "cuda"]


def detect_device() -> tuple[DeviceName, bool]:
    """Return the active PyTorch device and whether CUDA is available."""

    try:
        import torch
    except ImportError:  # pragma: no cover - dependency setup is validated elsewhere.
        return "cpu", False

    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False
