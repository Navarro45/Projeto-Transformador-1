from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from hierarchy.augmentation import (
    build_balanced_training_records,
    generate_synthetic_image,
)


def _centered_star() -> Image.Image:
    image = Image.new("RGB", (256, 256), (12, 13, 15))
    draw = ImageDraw.Draw(image)
    draw.ellipse((121, 121, 135, 135), fill=(240, 210, 180))
    draw.ellipse((125, 125, 131, 131), fill=(255, 255, 255))
    return image


def test_synthetic_image_is_256_black_and_reproducible():
    source = _centered_star()
    first, metadata, _ = generate_synthetic_image(source, seed=123)
    second, second_metadata, _ = generate_synthetic_image(source, seed=123)
    pixels = np.asarray(first)
    assert first.size == (256, 256)
    assert np.array_equal(pixels, np.asarray(second))
    assert metadata == second_metadata
    assert np.count_nonzero(pixels) > 0
    assert np.count_nonzero(np.all(pixels == 0, axis=2)) > 60000


def test_balancing_mixes_real_and_synthetic_in_every_class():
    rows = []
    for label, count in (("F", 600), ("B", 10)):
        for index in range(count):
            rows.append(
                {
                    "object_id": f"{label}-{index}",
                    "image_path": f"{label}-{index}.jpg",
                    "target_label": label,
                }
            )
    balanced = build_balanced_training_records(
        pd.DataFrame(rows),
        node_key="star_spectral",
        target_per_class=500,
        min_synthetic_fraction=0.10,
        max_synthetic_per_source=10,
        seed=42,
    )
    abundant = balanced.loc[balanced["target_label"].eq("F")]
    scarce = balanced.loc[balanced["target_label"].eq("B")]
    assert abundant["sample_kind"].value_counts().to_dict() == {"real": 450, "synthetic": 50}
    assert scarce["sample_kind"].value_counts().to_dict() == {"synthetic": 100, "real": 10}
    assert scarce["source_object_id"].nunique() == 10
    assert scarce.groupby("source_object_id").size().max() == 11


def test_balancing_is_deterministic():
    records = pd.DataFrame(
        {
            "object_id": [f"obj-{index}" for index in range(30)],
            "image_path": [f"obj-{index}.jpg" for index in range(30)],
            "target_label": ["G"] * 30,
        }
    )
    first = build_balanced_training_records(records, "star_spectral", seed=7)
    second = build_balanced_training_records(records, "star_spectral", seed=7)
    pd.testing.assert_frame_equal(first, second)
