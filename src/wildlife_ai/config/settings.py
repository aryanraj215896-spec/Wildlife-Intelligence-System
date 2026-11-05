from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CLASS_DIRECTORIES: tuple[tuple[str, str], ...] = (
    ("cat", "Domestic Cat"),
    ("dog", "Domestic Dog"),
    ("lion", "Lion"),
    ("tiger", "Tiger"),
    ("leopard", "Leopard"),
    ("wolf", "Gray wolf"),
    ("cheetah", "Cheetah"),
    ("jaguar", "Jaguar"),
)
SUPPORTED_CLASSES: tuple[str, ...] = tuple(label for _, label in CLASS_DIRECTORIES)


@dataclass(frozen=True)
class ProjectSettings:
    """Project-wide configuration and path registry for the local MVP."""

    root_dir: Path
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    taxonomy_dir: Path = field(init=False)
    feedback_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    manifests_dir: Path = field(init=False)
    database_path: Path = field(init=False)
    supported_classes: tuple[str, ...] = SUPPORTED_CLASSES
    batch_size: int = 32
    input_size: tuple[int, int] = (224, 224)
    device_name: str = "cpu"

    def __post_init__(self) -> None:
        root = Path(self.root_dir).resolve()
        object.__setattr__(self, "root_dir", root)
        object.__setattr__(self, "data_dir", root / "data")
        object.__setattr__(self, "raw_data_dir", root / "data" / "raw")
        object.__setattr__(self, "processed_data_dir", root / "data" / "processed")
        object.__setattr__(self, "taxonomy_dir", root / "data" / "taxonomy")
        object.__setattr__(self, "feedback_dir", root / "data" / "feedback")
        object.__setattr__(self, "checkpoint_dir", root / "models" / "checkpoints")
        object.__setattr__(self, "manifests_dir", root / "data" / "manifests")
        object.__setattr__(self, "database_path", root / "wildlife_ai.db")

    @classmethod
    def from_repo_root(cls, repo_root: str | Path | None = None) -> "ProjectSettings":
        candidate = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[3]
        return cls(root_dir=candidate)

    @property
    def dataset_directories(self) -> tuple[Path, ...]:
        return (
            self.raw_data_dir,
            self.processed_data_dir,
            self.taxonomy_dir,
            self.feedback_dir,
            self.manifests_dir,
        )

    def ensure_directories(self) -> Iterable[Path]:
        directories = [self.data_dir, *self.dataset_directories, self.checkpoint_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return tuple(directories)
