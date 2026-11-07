from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hierarchy.execution_import import (
    _validate_prerequisite_hierarchy,
    build_hierarchy_import_plan,
    find_latest_dated_execution,
    materialize_hierarchy_import_plan,
)


def test_latest_execution_ignores_legacy_model_folders(tmp_path):
    results = tmp_path / "Resultados"
    (results / "resnet").mkdir(parents=True)
    older = results / "2026-10-30_10-00-00_000001"
    latest = results / "2026-10-31_09-00-00_000001"
    older.mkdir()
    latest.mkdir()

    assert find_latest_dated_execution(results) == str(latest.resolve())


def test_latest_exact_incompatibility_does_not_fall_back(tmp_path):
    results = tmp_path / "Resultados"
    older = results / "2026-10-30_10-00-00_000001"
    latest = results / "2026-10-31_09-00-00_000001"
    older.mkdir(parents=True)
    latest.mkdir()
    family = latest / "resnet_hierarchical" / "execucao_01" / "hierarchy" / "metadata"
    family.mkdir(parents=True)
    (family / "hierarchy.json").write_text(
        json.dumps({"method": "incompativel"}),
        encoding="utf-8",
    )
    registry = SimpleNamespace(
        trainable_nodes=lambda: [SimpleNamespace(level=1, key="root")],
        to_dict=lambda: {"method": "esperado"},
    )
    args = SimpleNamespace(
        hierarchy_start_level=2,
        hierarchy_manifest="manifest.csv",
        hierarchy_min_train_per_class=10,
        seed=42,
        img_size=256,
        batch_size=16,
        max_train_images=-1,
        max_validation_images=-1,
        max_test_images=-1,
        hierarchy_disable_augmentation=False,
        hierarchy_balance_target=500,
        hierarchy_min_synthetic_fraction=0.10,
        hierarchy_max_synthetic_per_source=10,
    )

    with (
        patch("hierarchy.execution_import.load_hierarchy_manifest", return_value=object()),
        patch("hierarchy.execution_import.HierarchyRegistry.from_manifest", return_value=registry),
        patch("hierarchy.execution_import.build_node_data", return_value=object()),
        pytest.raises(ValueError, match="hierarquia incompativel"),
    ):
        build_hierarchy_import_plan(args, ["resnet"], 1, results)


def test_import_accepts_downstream_spectral_change_when_root_is_identical():
    root = {"key": "root", "classes": ["GALAXY", "QSO", "STAR"]}
    actual = {
        "method": "hierarchical_global_spectral",
        "nodes": [root, {"key": "star_spectral", "classes": ["O", "B", "A"]}],
    }
    expected = {
        "method": "hierarchical_global_spectral",
        "nodes": [
            root,
            {"key": "star_luminosity", "classes": ["IV", "V"]},
            {"key": "star_spectral", "classes": ["B", "A"]},
        ],
        "excluded_level2_classes": ["III"],
        "excluded_spectral_classes": ["O"],
    }

    _validate_prerequisite_hierarchy(
        actual=actual,
        expected=expected,
        prerequisite_nodes=[SimpleNamespace(key="root")],
        context="resnet/execucao_01",
    )


def test_materialization_copies_complete_level_and_preserves_source(tmp_path):
    source_execution = tmp_path / "Resultados" / "2026-10-31_09-00-00_000001"
    source_node = (
        source_execution
        / "resnet_hierarchical"
        / "execucao_01"
        / "hierarchy"
        / "nivel_1_classes_principais"
    )
    files = {
        "model/model.pth": b"checkpoint",
        "metrics/metrics.json": b'{"accuracy": 1.0}',
        "metadata/info.json": b'{"model": "resnet"}',
        "metrics/classification_report.json": b"{}",
        "plots/confusion_matrix.png": b"png",
        "predictions/test_predictions.csv": b"label\nSTAR\n",
        "reports/details.txt": b"report",
    }
    for relative, payload in files.items():
        path = source_node / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    source_snapshot = {
        relative: (source_node / relative).read_bytes()
        for relative in files
    }

    plan = {
        "start_level": 2,
        "source_execution": str(source_execution),
        "families": {
            "resnet:execucao_01": {
                "model": "resnet",
                "run_index": 1,
                "source_execution": str(source_execution),
                "source_family_dir": str(source_node.parent),
                "hierarchy": {"method": "hierarchical_global_spectral"},
                "nodes": [
                    {
                        "node": "root",
                        "level": 1,
                        "artifact_name": "nivel_1_classes_principais",
                        "classes": ["GALAXY", "QSO", "STAR"],
                        "source_results_dir": str(source_node),
                        "source_checkpoint": str(source_node / "model" / "model.pth"),
                    }
                ],
            }
        },
    }
    destination = tmp_path / "Resultados" / "2026-11-03_10-00-00_000001"
    destination.mkdir(parents=True)

    materialized = materialize_hierarchy_import_plan(plan, str(destination))
    family = materialized["families"]["resnet:execucao_01"]
    copied_node = Path(family["nodes"][0]["destination_results_dir"])
    for relative, payload in source_snapshot.items():
        assert (copied_node / relative).read_bytes() == payload
        assert (source_node / relative).read_bytes() == payload
    assert family["nodes"][0]["resume_action"] == "copied_prerequisite"
    assert family["nodes"][0]["file_count"] == len(files)
    imported = json.loads(
        (
            destination
            / "resnet_hierarchical"
            / "execucao_01"
            / "hierarchy"
            / "metadata"
            / "imported_levels.json"
        ).read_text(encoding="utf-8")
    )
    assert imported["start_level"] == 2
    assert imported["nodes"][0]["checkpoint_sha256"]


def test_materialization_refuses_existing_destination_level(tmp_path):
    source = tmp_path / "source"
    source_node = source / "node"
    (source_node / "model").mkdir(parents=True)
    (source_node / "model" / "model.pth").write_bytes(b"checkpoint")
    destination = tmp_path / "destination"
    existing = (
        destination
        / "resnet_hierarchical"
        / "execucao_01"
        / "hierarchy"
        / "nivel_1_classes_principais"
    )
    existing.mkdir(parents=True)
    plan = {
        "start_level": 2,
        "source_execution": str(source),
        "families": {
            "resnet:execucao_01": {
                "model": "resnet",
                "run_index": 1,
                "hierarchy": {},
                "nodes": [
                    {
                        "node": "root",
                        "level": 1,
                        "artifact_name": "nivel_1_classes_principais",
                        "classes": [],
                        "source_results_dir": str(source_node),
                        "source_checkpoint": str(source_node / "model" / "model.pth"),
                    }
                ],
            }
        },
    }

    with pytest.raises(FileExistsError, match="ja existe"):
        materialize_hierarchy_import_plan(plan, str(destination))
