from __future__ import annotations

import io
from pathlib import Path

import torch
import pytest
from PIL import Image

from wildlife_ai.config.settings import SUPPORTED_CLASSES
from wildlife_ai.explainability.gradcam import generate_gradcam
from wildlife_ai.inference import predict_image
from wildlife_ai.models.architectures import build_model


def test_prediction_returns_supported_top_three_and_safe_score_type() -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(image_buffer, format="PNG")
    model = build_model("scratch_cnn", pretrained=False)
    metadata = {
        "architecture": "scratch_cnn",
        "model_version": "test-version",
        "class_names": list(SUPPORTED_CLASSES),
        "input_size": [224, 224],
        "normalization": "scratch",
    }

    result = predict_image(
        model,
        metadata,
        image_buffer.getvalue(),
        device=torch.device("cpu"),
    )

    assert result["predicted_label"] in SUPPORTED_CLASSES
    assert len(result["top_outputs"]) == 3
    assert all(row["label"] in SUPPORTED_CLASSES for row in result["top_outputs"])
    assert 0 <= result["model_score"] <= 1
    assert "not a calibrated probability" in result["score_type"]
    assert result["device"] == "cpu"


def test_gradcam_generates_heatmap_for_resnet18_and_rejects_other_architectures() -> None:
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False)
    input_tensor = torch.randn(1, 3, 64, 64)
    model_view = Image.new("RGB", (64, 64), (20, 40, 60))
    heatmap = generate_gradcam(
        model,
        input_tensor,
        model_view,
        class_index=0,
    )
    assert heatmap.size == model_view.size

    scratch = build_model("scratch_cnn", pretrained=False)
    with pytest.raises(ValueError, match="only for ResNet18"):
        generate_gradcam(
            scratch,
            input_tensor,
            model_view,
            class_index=0,
        )
