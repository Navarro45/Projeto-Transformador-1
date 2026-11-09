"""Gera um painel auditavel sem modificar as imagens astronomicas originais."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hierarchy.augmentation import generate_synthetic_image, stable_seed  # noqa: E402
from hierarchy.image_data import load_hierarchy_manifest  # noqa: E402


DEFAULT_CLASSES = ("V", "IV", "III", "WD", "CARBON", "CV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera exemplos do augmentation 256x256.")
    parser.add_argument(
        "--manifest",
        default="dataset/hierarchy/roman_spectral_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/hierarchy/augmentation_examples",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=6)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _ROOT / value


def generate_examples(
    manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    examples: int = 6,
) -> tuple[Path, Path]:
    manifest = load_hierarchy_manifest(manifest_path)
    candidates = manifest.loc[
        manifest["split"].eq("train")
        & manifest["object_class"].eq("STAR")
        & ~manifest["stellar_level2_label"].isin(["", "UNRESOLVED"])
    ].copy()
    labels = [label for label in DEFAULT_CLASSES if label in set(candidates["stellar_level2_label"])]
    if examples > len(labels):
        extra = sorted(set(candidates["stellar_level2_label"]) - set(labels))
        labels.extend(extra)
    labels = labels[:examples]

    generated = []
    metadata_rows = []
    for label in labels:
        group = candidates.loc[candidates["stellar_level2_label"].eq(label)]
        group = group.sample(frac=1.0, random_state=stable_seed(seed, label))
        last_error = ""
        for _, row in group.head(100).iterrows():
            recipe_seed = stable_seed(seed, label, row["object_id"], "example")
            try:
                with Image.open(row["image_path"]) as source:
                    original = source.convert("RGB")
                    synthetic, recipe, extraction = generate_synthetic_image(
                        original,
                        seed=recipe_seed,
                    )
                generated.append((row, original, synthetic, extraction))
                metadata_rows.append(
                    {
                        "object_id": row["object_id"],
                        "spec_object_id": row["spec_object_id"],
                        "level2_label": row["stellar_level2_label"],
                        "level3_label": row["stellar_level3_label"],
                        "source_image_path": row["image_path"],
                        **recipe,
                        "validation_status": "approved",
                    }
                )
                break
            except ValueError as error:
                last_error = str(error)
        else:
            metadata_rows.append(
                {
                    "level2_label": label,
                    "validation_status": "rejected",
                    "rejection_reason": last_error or "sem exemplo extraivel",
                }
            )

    if not generated:
        raise RuntimeError("Nenhum exemplo de augmentation foi aprovado")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "augmentation_examples.png"
    metadata_path = output_dir / "augmentation_examples.csv"
    figure, axes = plt.subplots(len(generated), 4, figsize=(14, 3.4 * len(generated)))
    axes = np.atleast_2d(axes)
    for row_index, (row, original, synthetic, extraction) in enumerate(generated):
        original_array = np.asarray(original)
        overlay = original_array.astype(np.float32).copy()
        overlay[..., 0] = np.where(
            extraction.alpha > 0.05,
            0.65 * overlay[..., 0] + 0.35 * 255,
            overlay[..., 0],
        )
        isolated = np.clip(
            extraction.foreground.astype(np.float32) * extraction.alpha[..., None],
            0,
            255,
        ).astype(np.uint8)
        images = (original_array, overlay.astype(np.uint8), isolated, np.asarray(synthetic))
        titles = ("Original 256x256", "Mascara central", "Objeto isolado", "Sintetica 256x256")
        for column, (pixels, title) in enumerate(zip(images, titles)):
            axes[row_index, column].imshow(pixels)
            axes[row_index, column].set_title(title)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(
            f"N2={row['stellar_level2_label']} | N3={row['stellar_level3_label'] or '-'}\nobj={row['object_id']}",
            fontsize=9,
        )
    figure.suptitle("Data augmentation astronomico: alvo central sobre fundo preto", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    return panel_path, metadata_path


def main() -> int:
    args = parse_args()
    panel, metadata = generate_examples(
        _resolve(args.manifest),
        _resolve(args.output_dir),
        seed=args.seed,
        examples=args.examples,
    )
    print(f"Painel: {panel}")
    print(f"Metadados: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
