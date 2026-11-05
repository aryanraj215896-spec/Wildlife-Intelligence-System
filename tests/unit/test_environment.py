from pathlib import Path

from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.utils.device import detect_device


def test_project_settings_resolve_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    settings = ProjectSettings.from_repo_root(repo_root)

    assert settings.root_dir == repo_root
    assert settings.data_dir == repo_root / "data"
    assert settings.raw_data_dir == repo_root / "data" / "raw"
    assert settings.checkpoint_dir == repo_root / "models" / "checkpoints"
    assert len(settings.supported_classes) == 8
    assert settings.supported_classes[0] == "Domestic Cat"
    assert settings.supported_classes[-2:] == ("Cheetah", "Jaguar")


def test_detect_device_returns_valid_device() -> None:
    device_name, cuda_available = detect_device()

    assert device_name in {"cpu", "cuda"}
    assert isinstance(cuda_available, bool)
