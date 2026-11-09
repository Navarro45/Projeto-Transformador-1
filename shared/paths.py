import os
from pathlib import Path
from typing import Optional

_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))


def repo_root() -> str:
    """Raiz do repositório TCC (pasta que contém `shared/` e os modelos)."""
    return os.path.abspath(os.path.join(_SHARED_DIR, ".."))


def default_dataset_root() -> str:
    """Pasta única do dataset: `<repo>/dataset`."""
    return os.path.join(repo_root(), "dataset")


def relocate_dataset_path(
    path_value: str | os.PathLike | None,
    data_root: str | os.PathLike | None = None,
) -> Path | None:
    """Resolve a dataset path after the repository has been moved."""
    if path_value is None or not str(path_value).strip():
        return None

    root = Path(data_root or default_dataset_root()).resolve()
    path = Path(str(path_value))
    candidates: list[Path] = []

    lowered_parts = [part.lower() for part in path.parts]
    if "dataset" in lowered_parts:
        dataset_index = lowered_parts.index("dataset")
        candidates.append(root.joinpath(*path.parts[dataset_index + 1 :]))

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve() if candidates else path.resolve()


def dataset_layout(data_root: Optional[str] = None) -> dict:
    root = os.path.abspath(data_root or default_dataset_root())
    return {
        "root": root,
        "images": os.path.join(root, "images"),
        "raw": os.path.join(root, "raw"),
        "train": os.path.join(root, "train"),
        "val": os.path.join(root, "val"),
        "test": os.path.join(root, "test"),
        "metadata": os.path.join(root, "metadata.csv"),
    }
