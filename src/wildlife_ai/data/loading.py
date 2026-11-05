from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from wildlife_ai.config.settings import SUPPORTED_CLASSES

Normalization = Literal["scratch", "imagenet"]


def _initialize_worker(_: int) -> None:
    import torch

    torch.set_num_threads(1)


def build_transforms(
    normalization: Normalization,
    *,
    training: bool,
    image_size: int = 224,
) -> transforms.Compose:
    if normalization == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif normalization == "scratch":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    if training:
        image_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        resize_size = round(image_size * 256 / 224)
        image_transforms = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]
    image_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(image_transforms)


class ManifestImageDataset(Dataset):
    """Load only validated rows from a dataset manifest split."""

    def __init__(
        self,
        manifest_path: str | Path,
        raw_root: str | Path,
        split: str,
        transform=None,
    ) -> None:
        self.raw_root = Path(raw_root).resolve()
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        class_indices = {label: index for index, label in enumerate(SUPPORTED_CLASSES)}
        manifest = Path(manifest_path)
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("image_status") != "valid" or row.get("split") != split:
                    continue
                label = row.get("label", "")
                if label not in class_indices:
                    raise ValueError(f"Unsupported manifest label: {label}")
                relative_path = Path(row["relative_path"])
                if relative_path.parts[:2] == ("data", "feedback"):
                    allowed_root = self.raw_root.parent.parent / "data" / "feedback"
                    image_path = (self.raw_root.parent.parent / relative_path).resolve()
                else:
                    allowed_root = self.raw_root
                    image_path = (self.raw_root / relative_path).resolve()
                if not image_path.is_relative_to(allowed_root.resolve()):
                    raise ValueError(f"Manifest image path escapes raw data root: {row['relative_path']}")
                self.samples.append((image_path, class_indices[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


def make_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: str,
    seed: int,
) -> DataLoader:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        generator=generator,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_initialize_worker if num_workers > 0 else None,
    )
