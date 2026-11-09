import csv
import os
from pathlib import Path

from PIL import Image

from shared.pipelines.preprocess import run


def _create_images(source_root: Path, count: int = 20) -> None:
    for class_name in ("star", "galaxy", "quasar"):
        class_dir = source_root / class_name
        class_dir.mkdir(parents=True)
        for index in range(count):
            image = Image.new("RGB", (8, 8), color=(index, 20, 40))
            image.save(class_dir / f"{class_name}_{index:02d}.jpg")


def test_prepare_dataset_creates_disjoint_70_15_15_splits(tmp_path):
    data_root = tmp_path / "dataset"
    source_root = data_root / "images"
    _create_images(source_root)

    summary = run(
        data_root=str(data_root),
        materialization="copy",
        seed=42,
    )

    assert summary["counts_by_split"] == {"train": 42, "val": 9, "test": 9}
    assert summary["total_valid_images"] == 60

    split_names = {}
    for split in ("train", "val", "test"):
        names = {
            f"{path.parent.name}/{path.name}"
            for path in (data_root / split).glob("*/*.jpg")
        }
        split_names[split] = names

    assert split_names["train"].isdisjoint(split_names["val"])
    assert split_names["train"].isdisjoint(split_names["test"])
    assert split_names["val"].isdisjoint(split_names["test"])
    assert len(list(source_root.glob("*/*.jpg"))) == 60
    assert (data_root / "manifests" / "image_split_manifest.csv").is_file()
    assert (data_root / "manifests" / "image_split_summary.json").is_file()


def test_prepare_dataset_can_resume_same_assignment(tmp_path):
    data_root = tmp_path / "dataset"
    _create_images(data_root / "images")

    first = run(data_root=str(data_root), materialization="copy", seed=7)
    second = run(data_root=str(data_root), materialization="copy", seed=7)

    assert first["counts_by_split"] == second["counts_by_split"]
    assert second["materialization_counts"] == {"copy_existing": 60}


def test_prepare_dataset_uses_hardlinks_without_changing_source(tmp_path):
    data_root = tmp_path / "dataset"
    source_root = data_root / "images"
    _create_images(source_root)

    summary = run(data_root=str(data_root), seed=11)
    manifest_path = Path(summary["manifest"])
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))
    source_path = Path(first_row["source_path"])
    destination_path = Path(first_row["destination_path"])

    assert source_path.is_file()
    assert destination_path.is_file()
    assert os.path.samefile(source_path, destination_path)
