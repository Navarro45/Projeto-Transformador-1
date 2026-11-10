from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from shared.stellar_hierarchy import ROMAN_LUMINOSITY_ORDER

from .augmentation import (
    IMAGE_SIZE,
    build_balanced_training_records,
    extract_centered_object,
    generate_synthetic_image,
    validate_image_size,
)
from .registry import HierarchyNode


@lru_cache(maxsize=2048)
def _cached_synthetic(path: str, seed: int, image_size: int) -> np.ndarray:
    with Image.open(Path(path)) as source:
        image = source.convert("RGB")
        synthetic, _, _ = generate_synthetic_image(
            image,
            seed=int(seed),
            image_size=image_size,
        )
    return np.asarray(synthetic, dtype=np.uint8)


@lru_cache(maxsize=8192)
def _validate_augmentation_source(path: str, image_size: int) -> tuple[bool, str]:
    try:
        with Image.open(Path(path)) as source:
            extract_centered_object(source.convert("RGB"), image_size=image_size)
        return True, ""
    except (OSError, ValueError) as error:
        return False, str(error)


class ManifestImageDataset(Dataset):
    def __init__(self, records: pd.DataFrame, class_names: list[str], img_size: int):
        if img_size != IMAGE_SIZE:
            raise ValueError(f"A hierarquia visual exige imagens {IMAGE_SIZE}x{IMAGE_SIZE}")
        self.records = records.reset_index(drop=True).copy()
        self.class_names = list(class_names)
        self.class_to_idx = {label: index for index, label in enumerate(self.class_names)}
        self.img_size = int(img_size)
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        if str(row.get("sample_kind", "real")) == "synthetic":
            pixels = _cached_synthetic(
                str(row["image_path"]),
                int(row["synthetic_seed"]),
                self.img_size,
            )
            image = Image.fromarray(pixels.copy(), mode="RGB")
        else:
            with Image.open(Path(row["image_path"])) as source:
                image = source.convert("RGB")
                validate_image_size(image, self.img_size)
                image = image.copy()
        return self.transform(image), self.class_to_idx[str(row["target_label"])]


class InferenceImageDataset(Dataset):
    def __init__(self, records: pd.DataFrame, img_size: int):
        if img_size != IMAGE_SIZE:
            raise ValueError(f"A hierarquia visual exige imagens {IMAGE_SIZE}x{IMAGE_SIZE}")
        self.records = records.reset_index(drop=True).copy()
        self.img_size = int(img_size)
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        with Image.open(Path(self.records.iloc[index]["image_path"])) as source:
            image = source.convert("RGB")
            validate_image_size(image, self.img_size)
            return self.transform(image)


@dataclass
class NodeData:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    train_records: pd.DataFrame
    validation_records: pd.DataFrame
    test_records: pd.DataFrame
    class_names: list[str]
    augmentation_summary: dict = field(default_factory=dict)


def load_hierarchy_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifesto hierarquico nao encontrado: {path}. Execute: "
            "python dataset_scripts/prepare_hierarchical_dataset.py"
        )
    manifest = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "object_id",
        "spec_object_id",
        "split",
        "image_path",
        "object_class",
        "stellar_level2_label",
        "stellar_level3_label",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError("Manifesto sem colunas obrigatorias: " + ", ".join(missing))
    if manifest["object_id"].duplicated().any():
        duplicates = manifest.loc[manifest["object_id"].duplicated(), "object_id"].head(5)
        raise ValueError("Objetos repetidos no manifesto: " + ", ".join(duplicates))
    nonempty_spec = manifest.loc[manifest["spec_object_id"].ne(""), "spec_object_id"]
    if nonempty_spec.duplicated().any():
        raise ValueError("spec_object_id repetido no manifesto")
    invalid_splits = sorted(set(manifest["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError("Splits invalidos no manifesto: " + ", ".join(invalid_splits))
    missing_images = [value for value in manifest["image_path"] if not Path(value).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Imagem do manifesto nao encontrada: {missing_images[0]}")
    return manifest


def records_for_node(manifest: pd.DataFrame, node: HierarchyNode, split: str) -> pd.DataFrame:
    records = manifest.loc[manifest["split"].eq(split)].copy()
    if node.level >= 2:
        records = records.loc[records["object_class"].eq("STAR")]
    if node.level == 3:
        eligible_level2 = (
            node.eligible_level2_labels or ROMAN_LUMINOSITY_ORDER
        )
        records = records.loc[
            records["stellar_level2_label"].isin(eligible_level2)
        ]
    records["target_label"] = records[node.target_column].map(node.map_target_label)
    records = records.loc[records["target_label"].isin(node.classes)].copy()
    records["sample_kind"] = "real"
    records["source_object_id"] = records["object_id"]
    return records.reset_index(drop=True)


def build_node_data(
    manifest: pd.DataFrame,
    node: HierarchyNode,
    img_size: int,
    batch_size: int,
    seed: int,
    max_train_images: int | None = None,
    max_validation_images: int | None = None,
    max_test_images: int | None = None,
    augmentation_enabled: bool = True,
    balance_target: int = 500,
    min_synthetic_fraction: float = 0.10,
    max_synthetic_per_source: int = 10,
) -> NodeData:
    if not node.requires_model:
        raise ValueError(f"O no {node.key} nao requer classificador")
    if img_size != IMAGE_SIZE:
        raise ValueError(f"A hierarquia visual exige --img-size {IMAGE_SIZE}")

    records = {
        split: records_for_node(manifest, node, split)
        for split in ("train", "val", "test")
    }
    for split, frame in records.items():
        if frame.empty:
            raise ValueError(f"O no {node.key} nao possui exemplos em {split}")

    if augmentation_enabled and node.level >= 2:
        records["train"] = build_balanced_training_records(
            records["train"],
            node_key=node.key,
            target_per_class=balance_target,
            min_synthetic_fraction=min_synthetic_fraction,
            max_synthetic_per_source=max_synthetic_per_source,
            seed=seed,
            source_validator=lambda path: _validate_augmentation_source(path, img_size),
        )

    train_dataset = ManifestImageDataset(records["train"], list(node.classes), img_size)
    val_dataset = ManifestImageDataset(records["val"], list(node.classes), img_size)
    test_dataset = ManifestImageDataset(records["test"], list(node.classes), img_size)

    train_dataset, train_records = _limit_dataset(
        train_dataset, records["train"], max_train_images, seed
    )
    val_dataset, val_records = _limit_dataset(
        val_dataset, records["val"], max_validation_images, seed + 1
    )
    test_dataset, test_records = _limit_dataset(
        test_dataset, records["test"], max_test_images, seed + 2
    )
    generator = torch.Generator().manual_seed(seed)
    summary = _augmentation_summary(train_records, list(node.classes))
    return NodeData(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        validation_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_records=train_records,
        validation_records=val_records,
        test_records=test_records,
        class_names=list(node.classes),
        augmentation_summary=summary,
    )


def build_inference_loader(records: pd.DataFrame, img_size: int, batch_size: int) -> DataLoader:
    return DataLoader(
        InferenceImageDataset(records, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
    )


def _augmentation_summary(records: pd.DataFrame, class_names: list[str]) -> dict:
    classes = {}
    for label in class_names:
        group = records.loc[records["target_label"].eq(label)]
        real = int(group["sample_kind"].eq("real").sum())
        synthetic = int(group["sample_kind"].eq("synthetic").sum())
        classes[label] = {
            "real_images": real,
            "synthetic_images": synthetic,
            "training_samples": int(len(group)),
            "unique_real_objects": int(group["source_object_id"].nunique()),
            "synthetic_fraction": float(synthetic / len(group)) if len(group) else 0.0,
            "approved_synthetic_sources": int(
                group.loc[
                    group["sample_kind"].eq("real")
                    & group.get("augmentation_source_approved", False).astype(bool),
                    "source_object_id",
                ].nunique()
            ) if "augmentation_source_approved" in group else real,
            "rejected_synthetic_sources": int(
                group.loc[
                    group["sample_kind"].eq("real")
                    & ~group.get("augmentation_source_approved", True).astype(bool)
                ].shape[0]
            ) if "augmentation_source_approved" in group else 0,
        }
    return {"classes": classes, "total_samples": int(len(records))}


def _limit_dataset(dataset, records, limit: int | None, seed: int):
    if limit is None or limit < 0 or limit >= len(dataset):
        return dataset, records.reset_index(drop=True)
    indices = sorted(random.Random(seed).sample(range(len(dataset)), limit))
    return Subset(dataset, indices), records.iloc[indices].reset_index(drop=True)
