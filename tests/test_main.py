from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import main
import pytest


def _args(tmp_path):
    split_dirs = {}
    for split_name in ("train", "val", "test"):
        split_dir = tmp_path / split_name / "star"
        split_dir.mkdir(parents=True)
        (split_dir / "sample.jpg").write_bytes(b"image")
        split_dirs[split_name] = split_dir.parent

    return Namespace(
        mode="train",
        model="all_models",
        runs=1,
        seed=42,
        train_dir=str(split_dirs["train"]),
        validation_dir=str(split_dirs["val"]),
        test_dir=str(split_dirs["test"]),
        batch_size=2,
        epochs=1,
        learning_rate=0.001,
        img_size=8,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        scheduler_patience=1,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
        max_train_images=-1,
        max_validation_images=-1,
        max_test_images=-1,
        heatmaps_per_model=0,
        rf_label_level="both",
        rf_split_manifest=None,
    )


def test_all_models_runs_image_and_random_forest_suites(tmp_path):
    args = _args(tmp_path)
    image_record = {
        "name": "resnet",
        "class_names": ["star"],
        "metrics": {},
    }
    star_record = {
        "name": "flat_random_forest_main",
        "metrics": {},
    }

    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main, "get_model_names", return_value=["resnet"]),
        patch.object(main.ResultsManager, "create_execution_dir", return_value=str(tmp_path / "results")),
        patch.object(main, "run_pipeline", return_value=image_record) as run_image,
        patch.object(main, "evaluate_ensemble", return_value=[]),
        patch.object(main, "_run_star_model_levels", return_value=[star_record]) as run_stars,
        patch.object(main, "save_summary") as save_summary,
        patch.object(main.torch.cuda, "is_available", return_value=False),
    ):
        main.main()

    run_image.assert_called_once()
    run_stars.assert_called_once()
    summary_path, records, ensembles = save_summary.call_args.args
    assert Path(summary_path).name == "summary_all_models.json"
    assert records == [image_record, star_record]
    assert ensembles == []


def test_all_models_requires_prepared_image_splits(tmp_path):
    args = _args(tmp_path)
    for image in Path(args.validation_dir).rglob("*.jpg"):
        image.unlink()

    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main.ResultsManager, "create_execution_dir", return_value=str(tmp_path / "results")),
    ):
        try:
            main.main()
        except FileNotFoundError as error:
            assert "python scripts/prepare_dataset.py" in str(error)
        else:
            raise AssertionError("all_models deveria rejeitar splits vazios")


def test_hierarchical_resnet_uses_dedicated_pipeline(tmp_path):
    args = _args(tmp_path)
    args.model = "resnet"
    args.hierarchical = True
    args.img_size = 256
    args.hierarchy_manifest = str(tmp_path / "manifest.csv")
    args.hierarchy_min_train_per_class = 10
    args.hierarchy_confidence_threshold = 0.0
    args.hierarchy_balance_target = 500
    args.hierarchy_min_synthetic_fraction = 0.10
    args.hierarchy_max_synthetic_per_source = 10
    args.hierarchy_disable_augmentation = False
    args.hierarchy_resume_execution = None
    args.hierarchy_start_level = 1
    record = {"name": "resnet_hierarchical", "metrics": {}}

    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main.ResultsManager, "create_execution_dir", return_value=str(tmp_path / "results")),
        patch.object(main, "run_hierarchical_image_pipeline", return_value=record) as run_hierarchy,
        patch.object(main, "save_summary") as save_summary,
        patch.object(main.torch.cuda, "is_available", return_value=False),
    ):
        main.main()

    run_hierarchy.assert_called_once()
    summary_path, records = save_summary.call_args.args
    assert Path(summary_path).name == "summary_resnet_hierarchical.json"
    assert records == [record]


def test_hierarchical_resume_reuses_existing_execution_directory(tmp_path):
    args = _args(tmp_path)
    args.model = "resnet"
    args.hierarchical = True
    args.img_size = 256
    args.hierarchy_manifest = str(tmp_path / "manifest.csv")
    args.hierarchy_min_train_per_class = 10
    args.hierarchy_confidence_threshold = 0.0
    args.hierarchy_balance_target = 500
    args.hierarchy_min_synthetic_fraction = 0.10
    args.hierarchy_max_synthetic_per_source = 10
    args.hierarchy_disable_augmentation = False
    args.hierarchy_resume_execution = str(tmp_path / "existing_results")
    args.hierarchy_start_level = 1
    Path(args.hierarchy_resume_execution).mkdir()
    record = {"name": "resnet_hierarchical", "metrics": {}}

    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main.ResultsManager, "create_execution_dir") as create_execution,
        patch.object(main, "run_hierarchical_image_pipeline", return_value=record) as run_hierarchy,
        patch.object(main, "save_summary"),
        patch.object(main.torch.cuda, "is_available", return_value=False),
    ):
        main.main()

    create_execution.assert_not_called()
    assert run_hierarchy.call_args.kwargs["execution_dir"] == str(
        Path(args.hierarchy_resume_execution)
    )


def test_hierarchical_start_level_resolves_source_before_destination(tmp_path):
    args = _args(tmp_path)
    args.model = "resnet"
    args.hierarchical = True
    args.img_size = 256
    args.hierarchy_manifest = str(tmp_path / "manifest.csv")
    args.hierarchy_min_train_per_class = 10
    args.hierarchy_confidence_threshold = 0.0
    args.hierarchy_balance_target = 500
    args.hierarchy_min_synthetic_fraction = 0.10
    args.hierarchy_max_synthetic_per_source = 10
    args.hierarchy_disable_augmentation = False
    args.hierarchy_resume_execution = None
    args.hierarchy_start_level = 2
    destination = str(tmp_path / "new_results")
    events = []
    plan = {
        "source_execution": str(tmp_path / "old_results"),
        "families": {"resnet:execucao_01": {"nodes": []}},
    }

    def build_plan(**kwargs):
        events.append("preflight")
        return plan

    def create_execution():
        events.append("create_destination")
        return destination

    def materialize(import_plan, execution_dir):
        events.append("copy")
        import_plan["destination_execution"] = execution_dir
        return import_plan

    record = {"name": "resnet_hierarchical", "metrics": {}}
    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main, "build_hierarchy_import_plan", side_effect=build_plan),
        patch.object(main.ResultsManager, "create_execution_dir", side_effect=create_execution),
        patch.object(main, "materialize_hierarchy_import_plan", side_effect=materialize),
        patch.object(main, "run_hierarchical_image_pipeline", return_value=record) as run_hierarchy,
        patch.object(main, "save_summary"),
        patch.object(main.torch.cuda, "is_available", return_value=False),
    ):
        main.main()

    assert events == ["preflight", "create_destination", "copy"]
    assert run_hierarchy.call_args.kwargs["import_spec"] == plan["families"][
        "resnet:execucao_01"
    ]


def test_hierarchical_start_level_rejects_in_place_resume(tmp_path):
    args = _args(tmp_path)
    args.model = "resnet"
    args.hierarchical = True
    args.img_size = 256
    args.hierarchy_resume_execution = str(tmp_path / "existing")
    args.hierarchy_start_level = 3

    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main.ResultsManager, "create_execution_dir") as create_execution,
        pytest.raises(ValueError, match="nao pode ser combinado"),
    ):
        main.main()

    create_execution.assert_not_called()


def test_level3_class_weight_requires_hierarchical_mode(tmp_path):
    args = _args(tmp_path)
    args.model = "resnet"
    args.hierarchical = False
    args.hierarchy_level3_class_weights = "OB=1.5"

    with (
        patch.object(main, "parse_args", return_value=args),
        patch.object(main.ResultsManager, "create_execution_dir") as create_execution,
        pytest.raises(ValueError, match="hierarchy-level3-class-weights exige"),
    ):
        main.main()

    create_execution.assert_not_called()
