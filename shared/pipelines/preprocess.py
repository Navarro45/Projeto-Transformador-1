import os
import random
import shutil
from typing import Dict, Optional, Tuple

from PIL import Image

from shared.dataset_defaults import DEFAULT_SPLIT, IMAGE_CLASS_FOLDERS
from shared.paths import dataset_layout, default_dataset_root


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_structure(output_dir: str, classes: Tuple[str, ...], split_names: Tuple[str, ...]) -> None:
    for split in split_names:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


def process_class(cls: str, raw_dir: str, output_dir: str, split: Dict[str, float], rng: random.Random) -> None:
    print(f"\nProcessando {cls}")

    class_path = os.path.join(raw_dir, cls)
    if not os.path.isdir(class_path):
        print(f" Aviso: pasta em falta {class_path}")
        return

    files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    valid_samples = []

    for f in files:
        img_path = os.path.join(class_path, f)
        if not is_valid_image(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass
            continue
        valid_samples.append(f)

    print(f"Válidos: {len(valid_samples)}")
    rng.shuffle(valid_samples)

    n = len(valid_samples)
    train_end = int(n * split["train"])
    val_end = train_end + int(n * split["val"])

    splits = {
        "train": valid_samples[:train_end],
        "val": valid_samples[train_end:val_end],
        "test": valid_samples[val_end:],
    }

    for split_name, split_files in splits.items():
        for f in split_files:
            src_img = os.path.join(class_path, f)
            dst_img = os.path.join(output_dir, split_name, cls, f)
            shutil.copy2(src_img, dst_img)

    print(f"{cls} OK")


def run(
    data_root: Optional[str] = None,
    classes: Optional[Tuple[str, ...]] = None,
    split: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> None:
    """
    Valida JPEGs em ``raw/``, divide train/val/test e copia para ``<data_root>/{train,val,test}/``.
    """
    paths = dataset_layout(data_root or default_dataset_root())
    raw_dir = paths["raw"]
    output_dir = paths["root"]
    classes = classes or IMAGE_CLASS_FOLDERS
    split = split or DEFAULT_SPLIT

    rng = random.Random(seed)
    create_structure(output_dir, classes, tuple(split.keys()))

    for cls in classes:
        process_class(cls, raw_dir, output_dir, split, rng)

    print("\nDataset pronto (materializado a partir de raw).")


if __name__ == "__main__":
    run()
