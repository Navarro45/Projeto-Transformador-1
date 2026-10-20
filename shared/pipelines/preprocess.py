from __future__ import annotations

import csv
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

from shared.dataset_defaults import DEFAULT_SPLIT, IMAGE_CLASS_FOLDERS
from shared.paths import dataset_layout, default_dataset_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def is_valid_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _validate_split(split: Dict[str, float]) -> None:
    if set(split) != set(SPLIT_NAMES):
        raise ValueError("A divisao precisa conter train, val e test.")
    if any(value < 0 for value in split.values()):
        raise ValueError("As proporcoes da divisao nao podem ser negativas.")
    if abs(sum(split.values()) - 1.0) > 1e-9:
        raise ValueError("As proporcoes de train, val e test precisam somar 1.0.")


def _class_assignments(
    files: list[Path],
    split: Dict[str, float],
    seed: int,
) -> dict[str, list[Path]]:
    shuffled = sorted(files, key=lambda path: path.name.lower())
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * split["train"])
    val_end = train_end + int(len(shuffled) * split["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _partition_valid_images(
    candidates: list[Path],
    validation_workers: int,
) -> tuple[list[Path], list[Path]]:
    if validation_workers < 1:
        raise ValueError("validation_workers precisa ser maior ou igual a 1.")

    with ThreadPoolExecutor(max_workers=validation_workers) as executor:
        validity = executor.map(is_valid_image, candidates)

    valid_files = []
    invalid_files = []
    for image_path, is_valid in zip(candidates, validity):
        target = valid_files if is_valid else invalid_files
        target.append(image_path)
    return valid_files, invalid_files


def _remove_existing_splits(output_root: Path) -> None:
    resolved_root = output_root.resolve()
    for split_name in SPLIT_NAMES:
        split_dir = (output_root / split_name).resolve()
        if split_dir.parent != resolved_root:
            raise ValueError(f"Diretorio de split inseguro: {split_dir}")
        if split_dir.exists():
            shutil.rmtree(split_dir)


def _existing_output_images(output_root: Path) -> set[Path]:
    paths: set[Path] = set()
    for split_name in SPLIT_NAMES:
        split_dir = output_root / split_name
        if not split_dir.exists():
            continue
        paths.update(
            path.resolve()
            for path in split_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return paths


def _materialize(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "hardlink_existing"
        except OSError:
            pass
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return "copy_existing"
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(
    data_root: Optional[str] = None,
    source_dir: Optional[str] = None,
    classes: Optional[Tuple[str, ...]] = None,
    split: Optional[Dict[str, float]] = None,
    seed: int = 42,
    materialization: str = "hardlink",
    overwrite: bool = False,
    validation_workers: int = 8,
) -> dict:
    """Materializa train/val/test a partir das imagens locais, sem treinar modelos."""
    paths = dataset_layout(data_root or default_dataset_root())
    output_root = Path(paths["root"])
    source_root = Path(source_dir) if source_dir else Path(paths["images"])
    classes = classes or IMAGE_CLASS_FOLDERS
    split = split or DEFAULT_SPLIT

    _validate_split(split)
    if materialization not in {"hardlink", "copy"}:
        raise ValueError("materialization precisa ser hardlink ou copy.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Diretorio de imagens nao encontrado: {source_root}")

    if overwrite:
        _remove_existing_splits(output_root)

    assignments: list[dict] = []
    invalid_rows: list[dict] = []
    counts_by_class: dict[str, dict[str, int]] = {}

    for class_index, class_name in enumerate(classes):
        class_dir = source_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Classe nao encontrada: {class_dir}")

        candidates = sorted(
            (
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        print(
            f"Validando {class_name}: {len(candidates)} imagens...",
            flush=True,
        )
        valid_files, invalid_files = _partition_valid_images(
            candidates,
            validation_workers,
        )
        invalid_rows.extend(
            {
                "class_name": class_name,
                "source_path": str(image_path.resolve()),
                "reason": "invalid_image",
            }
            for image_path in invalid_files
        )
        print(
            f"{class_name}: {len(valid_files)} validas, "
            f"{len(invalid_files)} invalidas.",
            flush=True,
        )

        class_splits = _class_assignments(
            valid_files,
            split,
            seed + class_index,
        )
        counts_by_class[class_name] = {
            split_name: len(split_files)
            for split_name, split_files in class_splits.items()
        }

        for split_name, split_files in class_splits.items():
            for source_path in split_files:
                destination = output_root / split_name / class_name / source_path.name
                assignments.append(
                    {
                        "class_name": class_name,
                        "filename": source_path.name,
                        "split": split_name,
                        "source_path": str(source_path.resolve()),
                        "destination_path": str(destination.resolve()),
                        "size_bytes": source_path.stat().st_size,
                    }
                )

    expected_destinations = {
        Path(row["destination_path"]).resolve()
        for row in assignments
    }
    unexpected = _existing_output_images(output_root) - expected_destinations
    if unexpected:
        examples = ", ".join(str(path) for path in sorted(unexpected)[:3])
        raise RuntimeError(
            "Os splits existentes nao correspondem a esta configuracao. "
            f"Use --overwrite para recria-los. Exemplos: {examples}"
        )

    method_counts: dict[str, int] = {}
    for row in assignments:
        method = _materialize(
            Path(row["source_path"]),
            Path(row["destination_path"]),
            materialization,
        )
        row["materialization"] = method
        method_counts[method] = method_counts.get(method, 0) + 1

    manifests_dir = output_root / "manifests"
    manifest_path = manifests_dir / "image_split_manifest.csv"
    invalid_path = manifests_dir / "image_split_invalid.csv"
    summary_path = manifests_dir / "image_split_summary.json"

    _write_csv(
        manifest_path,
        assignments,
        [
            "class_name",
            "filename",
            "split",
            "source_path",
            "destination_path",
            "size_bytes",
            "materialization",
        ],
    )
    _write_csv(
        invalid_path,
        invalid_rows,
        ["class_name", "source_path", "reason"],
    )

    total = len(assignments)
    counts_by_split = {
        split_name: sum(
            class_counts[split_name]
            for class_counts in counts_by_class.values()
        )
        for split_name in SPLIT_NAMES
    }
    summary = {
        "source_dir": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "validation_workers": validation_workers,
        "requested_split": split,
        "total_valid_images": total,
        "total_invalid_images": len(invalid_rows),
        "counts_by_split": counts_by_split,
        "actual_split": {
            name: (count / total if total else 0.0)
            for name, count in counts_by_split.items()
        },
        "counts_by_class": counts_by_class,
        "materialization_counts": method_counts,
        "manifest": str(manifest_path.resolve()),
        "invalid_manifest": str(invalid_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Origem preservada: {source_root}")
    print(f"Treino: {counts_by_split['train']}")
    print(f"Validacao: {counts_by_split['val']}")
    print(f"Teste: {counts_by_split['test']}")
    print(f"Imagens invalidas ignoradas: {len(invalid_rows)}")
    print(f"Manifesto: {manifest_path}")
    print("Dataset preparado. Nenhum modelo foi treinado.")
    return summary


if __name__ == "__main__":
    run()
