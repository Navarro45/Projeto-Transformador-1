from datetime import datetime
from pathlib import Path

from utils.result_manager import ResultsManager


def test_results_manager_uses_execution_model_and_run_folders(tmp_path):
    execution_dir = tmp_path / "2026-10-08_12-00-00_000000"

    results = ResultsManager(
        "efficientnet_b0",
        execution_dir=execution_dir,
        run_index=2,
    )

    assert Path(results.base_dir) == execution_dir / "efficientnet_b0" / "execucao_02"
    assert Path(results.model_dir).is_dir()
    assert Path(results.metrics_dir).is_dir()
    assert Path(results.predictions_dir).is_dir()
    assert Path(results.metadata_dir).is_dir()
    assert Path(results.reports_dir).is_dir()
    assert Path(results.plots_dir).is_dir()


def test_create_execution_dir_uses_start_timestamp(tmp_path):
    started_at = datetime(2026, 10, 8, 12, 30, 5, 123456)

    execution_dir = ResultsManager.create_execution_dir(
        results_root=tmp_path,
        started_at=started_at,
    )

    assert Path(execution_dir) == tmp_path / "2026-10-08_12-30-05_123456"
    assert Path(execution_dir).is_dir()


def test_from_existing_reads_new_layout(tmp_path):
    base_dir = tmp_path / "2026-10-08_12-00-00_000000" / "resnet50" / "execucao_03"

    results = ResultsManager.from_existing(base_dir)

    assert results.model_name == "resnet50"
    assert results.run_index == 3
    assert Path(results.execution_dir) == tmp_path / "2026-10-08_12-00-00_000000"
    assert Path(results.reports_dir).is_dir()


def test_results_manager_supports_result_variant(tmp_path):
    execution_dir = tmp_path / "2026-10-08_12-00-00_000000"

    results = ResultsManager(
        "flat_random_forest",
        execution_dir=execution_dir,
        run_index=1,
        variant="subclass",
    )

    expected = execution_dir / "flat_random_forest" / "execucao_01" / "subclass"
    assert Path(results.base_dir) == expected
    assert results.variant == "subclass"
    assert Path(results.metrics_dir).is_dir()


def test_from_existing_reads_variant_layout(tmp_path):
    base_dir = (
        tmp_path
        / "2026-10-08_12-00-00_000000"
        / "hierarchical_random_forest"
        / "execucao_02"
        / "main"
    )

    results = ResultsManager.from_existing(base_dir)

    assert results.model_name == "hierarchical_random_forest"
    assert results.run_index == 2
    assert results.variant == "main"
    assert Path(results.execution_dir) == tmp_path / "2026-10-08_12-00-00_000000"
