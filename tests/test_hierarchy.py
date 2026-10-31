from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hierarchy.image_data import records_for_node
from hierarchy.image_pipeline import (
    _conditional_probabilities_from_logits,
    _node_resume_action,
    _validate_node_metadata,
)
from hierarchy.registry import HierarchyRegistry
from scripts.prepare_hierarchical_dataset import build_manifest, build_summary


def _manifest_frame() -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        rows.extend(
            [
                {"object_id": f"galaxy-{split}", "split": split, "object_class": "GALAXY", "stellar_level2_label": "", "stellar_level3_label": ""},
                {"object_id": f"qso-{split}", "split": split, "object_class": "QSO", "stellar_level2_label": "", "stellar_level3_label": ""},
            ]
        )
    for parent, classes in {"V": ("A", "F"), "IV": ("G", "K")}.items():
        for label in classes:
            for index in range(2):
                rows.append(
                    {
                        "object_id": f"{parent}-{label}-{index}",
                        "split": "train",
                        "object_class": "STAR",
                        "stellar_level2_label": parent,
                        "stellar_level3_label": label,
                    }
                )
            for split in ("val", "test"):
                rows.append(
                    {
                        "object_id": f"{parent}-{label}-{split}",
                        "split": split,
                        "object_class": "STAR",
                        "stellar_level2_label": parent,
                        "stellar_level3_label": label,
                    }
                )
    for index in range(2):
        rows.append(
            {
                "object_id": f"wd-{index}",
                "split": "train",
                "object_class": "STAR",
                "stellar_level2_label": "WD",
                "stellar_level3_label": "",
            }
        )
    return pd.DataFrame(rows)


def test_registry_creates_one_global_spectral_node():
    registry = HierarchyRegistry.from_manifest(_manifest_frame(), min_train_per_class=2)

    assert registry.root.requires_model
    assert registry.star_luminosity.requires_model
    assert registry.star_luminosity.classes == ("IV", "V", "WD")
    assert registry.star_spectral.classes == ("A", "F", "G", "K")
    assert registry.star_spectral.requires_model
    assert [node.key for node in registry.trainable_nodes()] == [
        "root",
        "star_luminosity",
        "star_spectral",
    ]
    assert registry.to_dict()["method"] == "hierarchical_global_spectral"


def test_registry_marks_global_spectral_node_unavailable_below_minimum():
    registry = HierarchyRegistry.from_manifest(_manifest_frame(), min_train_per_class=3)
    assert registry.star_spectral.status == "unavailable"
    assert registry.star_spectral.classes == ()


def test_level3_records_combine_all_roman_parents_and_exclude_special_classes():
    manifest = _manifest_frame()
    registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    records = records_for_node(manifest, registry.star_spectral, "train")
    assert set(records["stellar_level2_label"]) == {"IV", "V"}
    assert set(records["target_label"]) == {"A", "F", "G", "K"}
    assert "WD" not in set(records["stellar_level2_label"])


def test_conditional_softmax_remains_numerically_stable():
    logits = np.asarray([[1000.0, -1000.0, -1001.0]])
    masked, has_supported, finite = _conditional_probabilities_from_logits(
        logits,
        np.asarray(["M"]),
        ["F", "M0", "M2"],
        {"F": "F", "M0": "M", "M2": "M"},
    )
    np.testing.assert_allclose(masked[0], [0.0, 0.73105858, 0.26894142])
    assert has_supported.tolist() == [True]
    assert finite.tolist() == [True]


def test_resume_metadata_rejects_incompatible_image_size():
    expected = {
        "model": "resnet",
        "model_family": "resnet",
        "class_names": ["A", "F"],
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "img_size": 256,
        "seed": 44,
        "split_counts": {"train": 10, "val": 2, "test": 2},
        "hierarchy_node": {"key": "star_spectral"},
        "augmentation": {"enabled": True},
    }
    actual = dict(expected)
    actual["img_size"] = 224
    with pytest.raises(ValueError, match="img_size"):
        _validate_node_metadata(actual, expected)


def test_resume_action_distinguishes_complete_and_incomplete_checkpoints(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    node = type("Node", (), {"key": "root"})()
    with patch("hierarchy.image_pipeline._node_is_complete", side_effect=[True, False]):
        assert _node_resume_action(True, str(checkpoint), object(), node) == "skip_complete"
        assert _node_resume_action(True, str(checkpoint), object(), node) == "evaluate_checkpoint"
    assert _node_resume_action(False, str(checkpoint), object(), node) == "train"


def test_manifest_builder_preserves_unique_split_assignments(tmp_path):
    dataset = tmp_path / "dataset"
    metadata_rows = []
    counter = 0
    for split in ("train", "val", "test"):
        for folder, object_class, subclass in (
            ("galaxy", "GALAXY", ""),
            ("quasar", "QSO", ""),
            ("star", "STAR", "F5"),
        ):
            counter += 1
            object_id = str(counter)
            image_dir = dataset / split / folder
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"{object_id}.jpg").write_bytes(b"local")
            metadata_rows.append(
                {"objID": object_id, "specObjID": f"spec-{counter}", "class": object_class, "subclass": subclass}
            )
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_csv, index=False)
    manifest = build_manifest(dataset, metadata_csv, tmp_path / "missing_gaia.csv")
    summary = build_summary(manifest)
    stars = manifest.loc[manifest["object_class"].eq("STAR")]
    assert len(manifest) == 9
    assert manifest["object_id"].nunique() == 9
    assert set(stars["stellar_level2_label"]) == {"UNRESOLVED"}
    assert summary["total_objects"] == 9
