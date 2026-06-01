import os
from typing import Optional

_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))


def repo_root() -> str:
    """Raiz do repositório TCC (pasta que contém `shared/` e os modelos)."""
    return os.path.abspath(os.path.join(_SHARED_DIR, ".."))


def default_dataset_root() -> str:
    """Pasta única do dataset: `<repo>/dataset`."""
    return os.path.join(repo_root(), "dataset")


def dataset_layout(data_root: Optional[str] = None) -> dict:
    root = os.path.abspath(data_root or default_dataset_root())
    return {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "train": os.path.join(root, "train"),
        "val": os.path.join(root, "val"),
        "test": os.path.join(root, "test"),
        "metadata": os.path.join(root, "metadata.csv"),
    }
