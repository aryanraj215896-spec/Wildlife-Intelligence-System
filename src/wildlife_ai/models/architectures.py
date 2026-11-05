from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from wildlife_ai.config.settings import SUPPORTED_CLASSES

ModelName = Literal["scratch_cnn", "resnet18"]


class ScratchCNN(nn.Module):
    """Compact convolutional baseline for 224x224 RGB images."""

    def __init__(self, num_classes: int = len(SUPPORTED_CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_model(
    model_name: ModelName,
    *,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a model with the configured class count."""

    if model_name == "scratch_cnn":
        if pretrained or freeze_backbone:
            raise ValueError("ScratchCNN does not support pretrained or frozen weights")
        return ScratchCNN()
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(SUPPORTED_CLASSES))
        return model
    raise ValueError(f"Unsupported model architecture: {model_name}")
