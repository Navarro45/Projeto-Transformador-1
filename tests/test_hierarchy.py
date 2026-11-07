from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.ensemble import summarize_metric_runs
from hierarchy.image_data import records_for_node
from hierarchy.image_pipeline import (
    END_TO_END_METRICS_VERSION,
    _aggregate_hierarchical_metrics,
    _balanced_class_weights,
    _class_weight_multipliers_for_node,
    _conditional_probabilities_from_logits,
    _evaluate_end_to_end,
    _evaluate_node,
    _full_path_classification_report,
    _full_path_metrics,
    _hierarchical_path_scores,
    _node_is_complete,
    _node_resume_action,
    _true_path,
    _validate_completed_family,
    _validate_node_metadata,
    run_hierarchical_image_pipeline,
)
from hierarchy.registry import HierarchyRegistry
from scripts.prepare_hierarchical_dataset import build_manifest, build_summary
from shared.stellar_hierarchy import parse_level3_class_weights
from utils.result_manager import ResultsManager


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
    assert registry.star_luminosity.classes == ("IV", "V", "Special_type")
    assert registry.star_luminosity.train_counts["Special_type"] == 2
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


def test_level2_records_group_all_non_roman_classes_as_special_type():
    manifest = _manifest_frame()
    extra = []
    for label in ("CARBON", "CV", "PECULIAR"):
        for split in ("train", "val", "test"):
            extra.append(
                {
                    "object_id": f"{label}-{split}",
                    "split": split,
                    "object_class": "STAR",
                    "stellar_level2_label": label,
                    "stellar_level3_label": "",
                }
            )
    manifest = pd.concat([manifest, pd.DataFrame(extra)], ignore_index=True)
    registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    records = records_for_node(manifest, registry.star_luminosity, "train")

    special = records.loc[records["target_label"].eq("Special_type")]
    assert set(special["stellar_level2_label"]) == {"WD", "CARBON", "CV", "PECULIAR"}
    assert registry.star_luminosity.train_counts["Special_type"] == 5
    assert registry.to_dict()["level2_grouping"] == "all_non_roman_to_special_type"


def test_registry_merges_o_and_b_and_can_exclude_ob_from_global_model():
    manifest = _manifest_frame()
    extra = []
    for spectral in ("O", "B"):
        for index in range(2):
            extra.append(
                {
                    "object_id": f"V-{spectral}-{index}",
                    "split": "train",
                    "object_class": "STAR",
                    "stellar_level2_label": "V",
                    "stellar_level3_label": spectral,
                }
            )
        for split in ("val", "test"):
            extra.append(
                {
                    "object_id": f"V-{spectral}-{split}",
                    "split": split,
                    "object_class": "STAR",
                    "stellar_level2_label": "V",
                    "stellar_level3_label": spectral,
                }
            )
    manifest = pd.concat([manifest, pd.DataFrame(extra)], ignore_index=True)

    merged_registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    merged_records = records_for_node(
        manifest,
        merged_registry.star_spectral,
        "train",
    )
    merged_sources = merged_records.loc[merged_records["target_label"].eq("OB")]
    assert set(merged_sources["stellar_level3_label"]) == {"O", "B"}
    assert merged_registry.star_spectral.train_counts["OB"] == 4
    assert merged_registry.star_spectral.classes == ("OB", "A", "F", "G", "K")

    registry = HierarchyRegistry.from_manifest(
        manifest,
        min_train_per_class=2,
        excluded_spectral_classes="OB",
    )
    records = records_for_node(manifest, registry.star_spectral, "train")

    assert registry.star_spectral.train_counts["OB"] == 4
    assert registry.star_spectral.classes == ("A", "F", "G", "K")
    assert "OB" not in set(records["target_label"])
    assert registry.to_dict()["excluded_spectral_classes"] == ["OB"]


def test_registry_excludes_level2_class_and_keeps_ob_in_spectral_model():
    manifest = _manifest_frame()
    extra = []
    for parent, spectral in (("V", "O"), ("III", "M")):
        for index in range(2):
            extra.append(
                {
                    "object_id": f"{parent}-{spectral}-{index}",
                    "split": "train",
                    "object_class": "STAR",
                    "stellar_level2_label": parent,
                    "stellar_level3_label": spectral,
                }
            )
        for split in ("val", "test"):
            extra.append(
                {
                    "object_id": f"{parent}-{spectral}-{split}",
                    "split": split,
                    "object_class": "STAR",
                    "stellar_level2_label": parent,
                    "stellar_level3_label": spectral,
                }
            )
    manifest = pd.concat([manifest, pd.DataFrame(extra)], ignore_index=True)

    registry = HierarchyRegistry.from_manifest(
        manifest,
        min_train_per_class=2,
        excluded_level2_classes="III",
    )
    level2_records = records_for_node(manifest, registry.star_luminosity, "train")
    level3_records = records_for_node(manifest, registry.star_spectral, "train")

    assert registry.star_luminosity.classes == ("IV", "V", "Special_type")
    assert registry.star_spectral.classes == ("OB", "A", "F", "G", "K")
    assert registry.star_spectral.eligible_level2_labels == ("IV", "V")
    assert "III" not in set(level2_records["target_label"])
    assert "III" not in set(level3_records["stellar_level2_label"])
    assert "OB" in set(level3_records["target_label"])
    assert registry.to_dict()["excluded_level2_classes"] == ["III"]


def test_level3_class_weight_parser_accepts_aliases_and_multiple_classes():
    assert parse_level3_class_weights("O=1.5,M=1.2") == {
        "OB": 1.5,
        "M": 1.2,
    }
    assert parse_level3_class_weights({"B": 2}) == {"OB": 2.0}


@pytest.mark.parametrize(
    "value",
    [
        "OB",
        "X=1.5",
        "OB=0",
        "OB=-1",
        "OB=nan",
        "OB=inf",
        "O=1.5,B=2.0",
    ],
)
def test_level3_class_weight_parser_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_level3_class_weights(value)


def test_level3_multiplier_changes_only_requested_spectral_weight():
    args = SimpleNamespace(hierarchy_level3_class_weights={"OB": 1.5})
    spectral_node = SimpleNamespace(key="star_spectral")
    level2_node = SimpleNamespace(key="star_luminosity")
    class_names = ["OB", "A", "F"]
    records = pd.DataFrame({"target_label": ["OB", "A", "F"] * 2})
    base = _balanced_class_weights(records, class_names)

    spectral_multipliers = _class_weight_multipliers_for_node(
        args,
        spectral_node,
        class_names,
    )
    level2_multipliers = _class_weight_multipliers_for_node(
        args,
        level2_node,
        ["IV", "V", "Special_type"],
    )
    effective = [
        weight * multiplier
        for weight, multiplier in zip(base, spectral_multipliers)
    ]

    assert base == pytest.approx([1.0, 1.0, 1.0])
    assert spectral_multipliers == pytest.approx([1.5, 1.0, 1.0])
    assert effective == pytest.approx([1.5, 1.0, 1.0])
    assert level2_multipliers == pytest.approx([1.0, 1.0, 1.0])


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


@pytest.mark.parametrize(
    ("true_path", "predicted_path", "precision", "recall", "f1", "shared"),
    [
        ("STAR > V > G", "STAR > V > G", 1.0, 1.0, 1.0, 3),
        ("STAR > V > G", "STAR > V", 1.0, 2 / 3, 0.8, 2),
        ("STAR > V > G", "STAR > V > K", 2 / 3, 2 / 3, 2 / 3, 2),
        ("STAR > V > G", "STAR > IV > G", 1 / 3, 1 / 3, 1 / 3, 1),
        ("GALAXY", "QSO", 0.0, 0.0, 0.0, 0),
        ("STAR > Special_type", "STAR > Special_type", 1.0, 1.0, 1.0, 2),
    ],
)
def test_hierarchical_path_scores_use_complete_prefixes(
    true_path,
    predicted_path,
    precision,
    recall,
    f1,
    shared,
):
    scores = _hierarchical_path_scores(true_path, predicted_path)

    assert scores["precisao_hierarquica"] == pytest.approx(precision)
    assert scores["revocacao_hierarquica"] == pytest.approx(recall)
    assert scores["f1_hierarquico"] == pytest.approx(f1)
    assert scores["niveis_corretos"] == shared


def test_true_paths_use_grouped_level2_and_level3_labels():
    special = pd.Series(
        {
            "object_class": "STAR",
            "stellar_level2_label": "WD",
            "stellar_level3_label": "O",
            "evaluated_stellar_level2": "Special_type",
            "evaluated_stellar_level3": "OB",
        }
    )
    roman = pd.Series(
        {
            "object_class": "STAR",
            "stellar_level2_label": "V",
            "stellar_level3_label": "O",
            "evaluated_stellar_level2": "V",
            "evaluated_stellar_level3": "OB",
        }
    )

    assert _true_path(special) == "STAR > Special_type"
    assert _true_path(roman) == "STAR > V > OB"


def test_empty_metric_subsets_are_explicitly_null():
    full_path = _full_path_metrics([], [])
    hierarchy = _aggregate_hierarchical_metrics(pd.DataFrame())
    report = _full_path_classification_report([], [])

    assert full_path["amostras"] == 0
    assert full_path["f1_ponderado"] is None
    assert hierarchy["amostras"] == 0
    assert hierarchy["f1"] is None
    assert report == {"amostras": 0, "classes": [], "relatorio": None}


def test_standard_end_to_end_metrics_are_included_in_global_summaries():
    summary = summarize_metric_runs(
        [
            {
                "name": "resnet_hierarchical",
                    "results_dir": "unused",
                "metrics": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.8,
                    "f1": 0.75,
                },
            }
        ]
    )

    metrics = summary["resnet_hierarchical"]["metrics"]
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert metrics["precision"]["mean"] == pytest.approx(0.7)
    assert metrics["f1"]["mean"] == pytest.approx(0.75)


def test_end_to_end_evaluation_creates_readable_metric_files(tmp_path):
    manifest = _manifest_frame()
    manifest["spec_object_id"] = "spec-" + manifest["object_id"].astype(str)
    manifest["image_path"] = "unused.jpg"
    registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    results = ResultsManager(
        "resnet_hierarchical",
        execution_dir=str(tmp_path),
        run_index=1,
        variant="hierarchy",
    )
    args = SimpleNamespace(
        hierarchy_confidence_threshold=0.0,
        img_size=256,
        batch_size=16,
    )

    def fake_predict(records, node, *_args, **_kwargs):
        if node.key == "root":
            labels = records["object_class"].astype(str).to_numpy()
        elif node.key == "star_luminosity":
            labels = records["stellar_level2_label"].astype(str).to_numpy()
        else:
            labels = records["stellar_level3_label"].astype(str).to_numpy()
        return labels, np.ones(len(records), dtype=float)

    with patch("hierarchy.image_pipeline._predict_records", side_effect=fake_predict):
        evaluated = _evaluate_end_to_end(
            args=args,
            manifest=manifest,
            registry=registry,
            checkpoints={node.key: "unused" for node in registry.trainable_nodes()},
            model_name="resnet",
            device=object(),
            results=results,
        )

    metrics = evaluated["metrics"]
    assert metrics["metrics_version"] == END_TO_END_METRICS_VERSION
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["f1_hierarquico"] == pytest.approx(1.0)

    metric_files = {
        "end_to_end_metrics.json",
        "metricas_ponta_a_ponta.json",
        "relatorio_classes_ponta_a_ponta.json",
    }
    assert metric_files.issubset(
        {path.name for path in Path(results.metrics_dir).iterdir()}
    )
    predictions = pd.read_csv(
        Path(results.predictions_dir) / "hierarchical_test_predictions.csv"
    )
    assert {
        "precisao_hierarquica",
        "revocacao_hierarquica",
        "f1_hierarquico",
        "niveis_corretos",
        "profundidade_real",
        "profundidade_predita",
    }.issubset(predictions.columns)


def test_spectral_node_evaluation_writes_validation_reports(tmp_path):
    class FakeWrapper:
        def __init__(self):
            self.model = torch.nn.Identity()

        def get_model(self):
            return self.model

        def forward_with_features(self, images):
            predictions = torch.tensor([0, 1], device=images.device)
            return {
                "predictions": predictions,
                "confidences": torch.tensor([0.9, 0.8], device=images.device),
                "logits": torch.tensor(
                    [[2.0, 0.0], [0.0, 2.0]],
                    device=images.device,
                ),
            }

    records = pd.DataFrame(
        {
            "object_id": ["ob", "a"],
            "image_path": ["ob.jpg", "a.jpg"],
            "target_label": ["OB", "A"],
        }
    )
    loader = [(torch.zeros((2, 3, 2, 2)), torch.tensor([0, 1]))]
    data = SimpleNamespace(
        test_loader=loader,
        validation_loader=loader,
        test_records=records,
        validation_records=records,
        class_names=["OB", "A"],
    )
    node = SimpleNamespace(key="star_spectral")
    results = ResultsManager(
        "resnet_hierarchical",
        execution_dir=str(tmp_path),
        run_index=1,
        variant="hierarchy/nivel_3_classes_espectrais_globais",
    )

    metrics = _evaluate_node(
        FakeWrapper(),
        data,
        node,
        results,
        torch.device("cpu"),
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert (Path(results.metrics_dir) / "validation_metrics.json").is_file()
    assert (
        Path(results.metrics_dir) / "validation_classification_report.json"
    ).is_file()
    assert (
        Path(results.predictions_dir) / "validation_predictions.csv"
    ).is_file()


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


def test_resume_metadata_rejects_different_class_weight_multiplier():
    expected = {
        "model": "resnet",
        "model_family": "resnet",
        "class_names": ["OB", "A"],
        "class_weights_base": {"OB": 1.0, "A": 1.0},
        "class_weight_multipliers": {"OB": 1.5, "A": 1.0},
        "class_weights": {"OB": 1.5, "A": 1.0},
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "img_size": 256,
        "seed": 44,
        "split_counts": {"train": 1000, "val": 20, "test": 20},
        "hierarchy_node": {"key": "star_spectral"},
        "augmentation": {"enabled": True},
    }
    actual = dict(expected)
    actual["class_weight_multipliers"] = {"OB": 1.0, "A": 1.0}
    actual["class_weights"] = {"OB": 1.0, "A": 1.0}

    with pytest.raises(ValueError, match="class_weight"):
        _validate_node_metadata(actual, expected)


def test_completed_family_rejects_different_level3_weight_configuration():
    manifest = _manifest_frame()
    registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    summary = {
        "name": "resnet_hierarchical",
        "model_key": "resnet",
        "strategy": "hierarchical_global_spectral",
        "run_index": 1,
        "seed": 42,
        "hierarchy": registry.to_dict(),
        "level3_class_weight_multipliers": {},
    }
    args = SimpleNamespace(hierarchy_level3_class_weights={"A": 1.5})

    with pytest.raises(ValueError, match="level3_class_weight_multipliers"):
        _validate_completed_family(
            summary,
            args=args,
            registry=registry,
            model_name="resnet",
            run_index=1,
            seed=42,
            device=torch.device("cpu"),
        )


def test_resume_action_distinguishes_complete_and_incomplete_checkpoints(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    node = type("Node", (), {"key": "root"})()
    with patch("hierarchy.image_pipeline._node_is_complete", side_effect=[True, False]):
        assert _node_resume_action(True, str(checkpoint), object(), node) == "skip_complete"
        assert _node_resume_action(True, str(checkpoint), object(), node) == "evaluate_checkpoint"
    assert _node_resume_action(False, str(checkpoint), object(), node) == "train"


def test_spectral_node_is_incomplete_until_validation_reports_exist(tmp_path):
    results = ResultsManager.from_existing(str(tmp_path))
    node = SimpleNamespace(key="star_spectral")
    required = [
        Path(results.get_model_path()),
        Path(results.get_metrics_path()),
        Path(results.metadata_dir) / "info.json",
        Path(results.metrics_dir) / "classification_report.json",
        Path(results.plots_dir) / "confusion_matrix.png",
        Path(results.predictions_dir) / "test_predictions.csv",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert not _node_is_complete(results, node)

    validation = [
        Path(results.metrics_dir) / "validation_metrics.json",
        Path(results.metrics_dir) / "validation_classification_report.json",
        Path(results.predictions_dir) / "validation_predictions.csv",
    ]
    for path in validation:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert _node_is_complete(results, node)


@pytest.mark.parametrize(
    ("start_level", "imported_keys", "trained_keys"),
    [
        (2, ["root"], ["star_luminosity", "star_spectral"]),
        (3, ["root", "star_luminosity"], ["star_spectral"]),
    ],
)
def test_start_level_imports_prerequisites_and_trains_only_requested_levels(
    tmp_path,
    start_level,
    imported_keys,
    trained_keys,
):
    manifest = _manifest_frame()
    registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    execution_dir = tmp_path / "new_execution"
    required = {
        "model/model.pth": b"checkpoint",
        "metrics/metrics.json": b'{"accuracy": 0.9}',
        "metadata/info.json": b"{}",
        "metrics/classification_report.json": b"{}",
        "plots/confusion_matrix.png": b"png",
        "predictions/test_predictions.csv": b"label\nSTAR\n",
    }
    imported_nodes = []
    for key in imported_keys:
        node = registry.nodes[key]
        destination_dir = (
            execution_dir
            / "resnet_hierarchical"
            / "execucao_01"
            / "hierarchy"
            / node.artifact_name
        )
        for relative, payload in required.items():
            path = destination_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        imported_nodes.append(
            {
                "node": node.key,
                "classes": list(node.classes),
                "source_results_dir": str(tmp_path / f"source_{node.key}"),
                "destination_results_dir": str(destination_dir),
            }
        )

    import_spec = {
        "source_execution": str(tmp_path / "source_execution"),
        "nodes": imported_nodes,
    }
    args = SimpleNamespace(
        hierarchy_manifest="manifest.csv",
        hierarchy_min_train_per_class=2,
        hierarchy_resume_execution=None,
        hierarchy_start_level=start_level,
        seed=42,
        img_size=256,
        hierarchy_confidence_threshold=0.0,
    )

    def fake_run_node(**kwargs):
        node = kwargs["node"]
        return {
            "node": node.key,
            "artifact_name": node.artifact_name,
            "classes": list(node.classes),
            "metrics": {"accuracy": 1.0},
            "results_dir": str(tmp_path / node.key),
            "checkpoint": str(tmp_path / f"{node.key}.pth"),
            "resume_action": "train",
        }

    with (
        patch("hierarchy.image_pipeline.load_hierarchy_manifest", return_value=manifest),
        patch("hierarchy.image_pipeline._load_node_checkpoint", return_value=object()),
        patch("hierarchy.image_pipeline._run_node", side_effect=fake_run_node) as run_node,
        patch(
            "hierarchy.image_pipeline._evaluate_end_to_end",
            return_value={"metrics": {"accuracy": 1.0}, "predictions": pd.DataFrame()},
        ),
        patch("hierarchy.image_pipeline.torch.cuda.is_available", return_value=False),
    ):
        summary = run_hierarchical_image_pipeline(
            args=args,
            model_name="resnet",
            run_index=1,
            execution_dir=str(execution_dir),
            import_spec=import_spec,
        )

    assert [call.kwargs["node"].key for call in run_node.call_args_list] == trained_keys
    assert [record["node"] for record in summary["nodes"][: len(imported_keys)]] == (
        imported_keys
    )
    assert all(
        record["resume_action"] == "copied_prerequisite"
        for record in summary["nodes"][: len(imported_keys)]
    )
    assert summary["start_level"] == start_level


def test_resume_regenerates_only_outdated_end_to_end_metrics(tmp_path):
    manifest = _manifest_frame()
    registry = HierarchyRegistry.from_manifest(manifest, min_train_per_class=2)
    family_results = ResultsManager(
        "resnet_hierarchical",
        execution_dir=str(tmp_path),
        run_index=1,
        variant="hierarchy",
    )
    old_summary = {
        "name": "resnet_hierarchical",
        "model_key": "resnet",
        "strategy": "hierarchical_global_spectral",
        "run_index": 1,
        "seed": 42,
        "metrics": {"accuracy": 0.5},
        "nodes": [
            {
                "node": node.key,
                "classes": list(node.classes),
                "checkpoint": str(tmp_path / f"{node.key}.pth"),
                "results_dir": str(tmp_path / node.key),
            }
            for node in registry.trainable_nodes()
        ],
        "hierarchy": registry.to_dict(),
    }
    family_results.save_json("summary.json", old_summary)
    args = SimpleNamespace(
        hierarchy_manifest="manifest.csv",
        hierarchy_min_train_per_class=2,
        hierarchy_resume_execution=str(tmp_path),
        hierarchy_start_level=1,
        seed=42,
        img_size=256,
        batch_size=16,
        hierarchy_confidence_threshold=0.0,
    )
    refreshed_metrics = {
        "metrics_version": END_TO_END_METRICS_VERSION,
        "accuracy": 0.75,
        "precision": 0.7,
        "recall": 0.75,
        "f1": 0.72,
    }

    with (
        patch("hierarchy.image_pipeline.load_hierarchy_manifest", return_value=manifest),
        patch("hierarchy.image_pipeline._validate_completed_family"),
        patch(
            "hierarchy.image_pipeline._completed_family_nodes_are_complete",
            return_value=True,
        ),
        patch("hierarchy.image_pipeline._end_to_end_metrics_are_current", return_value=False),
        patch(
            "hierarchy.image_pipeline._evaluate_end_to_end",
            return_value={"metrics": refreshed_metrics, "predictions": pd.DataFrame()},
        ) as evaluate,
        patch("hierarchy.image_pipeline._run_node") as run_node,
        patch("hierarchy.image_pipeline.torch.cuda.is_available", return_value=False),
    ):
        summary = run_hierarchical_image_pipeline(
            args=args,
            model_name="resnet",
            run_index=1,
            execution_dir=str(tmp_path),
        )

    run_node.assert_not_called()
    evaluate.assert_called_once()
    assert summary["metrics"] == refreshed_metrics
    assert summary["metrics_version"] == END_TO_END_METRICS_VERSION
    resume_plan = json.loads(
        (Path(family_results.metadata_dir) / "resume_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert resume_plan["family_action"] == "reevaluate_end_to_end_metrics"


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
