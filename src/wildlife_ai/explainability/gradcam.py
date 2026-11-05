from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_view: Image.Image,
    *,
    class_index: int,
    alpha: float = 0.45,
) -> Image.Image:
    if not hasattr(model, "layer4") or not hasattr(model.layer4[-1], "conv2"):
        raise ValueError("Grad-CAM is currently supported only for ResNet18")
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires a single BCHW model input")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    target_layer = model.layer4[-1].conv2
    activations: list[torch.Tensor] = []
    handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        if class_index < 0 or class_index >= logits.shape[1]:
            raise ValueError(f"Class index is out of range: {class_index}")
        if not activations:
            raise RuntimeError("Grad-CAM target layer did not produce activations")
        gradients = torch.autograd.grad(logits[0, class_index], activations[0])[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap,
            size=(model_view.height, model_view.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        maximum = heatmap.max()
        if not torch.isfinite(maximum.detach()).item() or maximum.detach().item() <= 0:
            raise RuntimeError("Grad-CAM produced no finite positive activation")
        values = (heatmap / maximum).detach().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    color_map = np.stack((values, np.zeros_like(values), 1.0 - values), axis=-1)
    colored = Image.fromarray(np.uint8(np.clip(color_map * 255, 0, 255)), mode="RGB")
    return Image.blend(model_view.convert("RGB"), colored, alpha=alpha)
