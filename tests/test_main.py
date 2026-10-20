from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import main


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
