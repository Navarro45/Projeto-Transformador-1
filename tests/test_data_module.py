from pathlib import Path

from PIL import Image

from legacy.data.data_module import DataModule


def _create_split(root: Path, split: str, count: int) -> None:
    for class_name in ("galaxy", "quasar", "star"):
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8), color=(index, 30, 60)).save(
                class_dir / f"{class_name}_{index}.jpg"
            )


def test_data_module_uses_materialized_validation_split(tmp_path):
    _create_split(tmp_path, "train", 4)
    _create_split(tmp_path, "val", 2)
    _create_split(tmp_path, "test", 3)

    data = DataModule(
        train_dir=tmp_path / "train",
        validation_dir=tmp_path / "val",
        test_dir=tmp_path / "test",
        img_size=8,
        batch_size=2,
        seed=42,
    )

    assert len(data.train_loader.dataset) == 12
    assert len(data.validation_loader.dataset) == 6
    assert len(data.test_loader.dataset) == 9
    assert data.class_names == ["galaxy", "quasar", "star"]
