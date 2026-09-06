from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from models.model_factory import ModelFactory
from models.random_forest_modelo import RandomForestModelConfig
from shared.spectral_labels import normalize_spectral_label
from spectral.preprocessing import SpectralPreprocessor, SpectralPreprocessorConfig
from utils.result_manager import ResultsManager


ROOT_LABEL = "STAR"
DEFAULT_STAR_OBJECTS_INDEX = "dataset/star_objects/star_objects_index.csv"
DEFAULT_RF_CACHE_DIR = "dataset/star_supervised/cache"
DEFAULT_RF_SPLIT_MANIFEST = "dataset/star_supervised/splits/split_manifest.csv"


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(inner) for inner in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=4, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_hierarchy(records: pd.DataFrame) -> dict:
    leaf_to_parent = {}
    for _, row in records.iterrows():
        leaf = str(row["spectral_leaf_label"])
        parent = str(row["spectral_parent_label"])
        previous = leaf_to_parent.get(leaf)
        if previous is not None and previous != parent:
            raise ValueError(f"Leaf {leaf} possui parents conflitantes: {previous}, {parent}")
        leaf_to_parent[leaf] = parent

    parent_to_leaves = {}
    for leaf, parent in leaf_to_parent.items():
        parent_to_leaves.setdefault(parent, []).append(leaf)

    parent_to_leaves = {
        parent: sorted(leaves)
        for parent, leaves in sorted(parent_to_leaves.items())
    }
    return {
        "root": ROOT_LABEL,
        "leaf_to_parent": dict(sorted(leaf_to_parent.items())),
        "parent_to_leaves": parent_to_leaves,
        "parent_labels": sorted(parent_to_leaves.keys()),
        "leaf_labels": sorted(leaf_to_parent.keys()),
    }


def load_star_supervised_records(
    star_objects_index: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    index_path = Path(star_objects_index)
    if not index_path.exists():
        raise FileNotFoundError(f"Indice de estrelas nao encontrado: {index_path}")

    index = pd.read_csv(index_path, dtype={"objID": str, "specObjID": str})
    if "star_id" not in index.columns:
        raise ValueError("star_objects_index.csv precisa conter a coluna star_id.")

    if max_samples is not None and max_samples >= 0 and len(index) > max_samples:
        index = index.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    rows = []
    base_dir = index_path.parent

    for _, row in index.iterrows():
        star_id = str(row["star_id"])
        object_dir = Path(str(row.get("object_dir", base_dir / star_id)))
        if not object_dir.is_absolute():
            object_dir = (base_dir / object_dir).resolve()

        target = _read_json(object_dir / "Target" / "target.json")
        spectrogram = _read_json(object_dir / "Spectrogram" / "metadata.json")
        raw_label = target.get("spectral_subclass_raw", row.get("spectral_subclass_raw", "UNKNOWN"))
        labels = normalize_spectral_label(raw_label)

        rows.append(
            {
                "star_id": star_id,
                "objID": row.get("objID"),
                "specObjID": row.get("specObjID"),
                "object_dir": str(object_dir),
                "spectrum_path": spectrogram.get("spectrum_path", ""),
                "spectrum_metadata_path": spectrogram.get("spectrum_metadata_path", ""),
                "target_path": str(object_dir / "Target" / "target.json"),
                **labels,
            }
        )

    records = pd.DataFrame(rows)
    if records.empty:
        raise ValueError("Nenhum objeto STAR foi encontrado para o treinamento supervisionado.")
    return records


def extract_spectrum_features(path: str | Path, preprocessor: SpectralPreprocessor) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS local nao encontrado: {path}")

    with fits.open(path, memmap=False) as hdul:
        if "COADD" in hdul:
            data = hdul["COADD"].data
        else:
            data = hdul[1].data
        if data is None or "flux" not in data.names or "loglam" not in data.names:
            raise ValueError("FITS nao contem colunas COADD flux/loglam.")
        flux = np.asarray(data["flux"], dtype=np.float64)
        wavelength = np.power(10.0, np.asarray(data["loglam"], dtype=np.float64))

    return preprocessor.transform(wavelength, flux)


def _cache_key(records: pd.DataFrame, config: SpectralPreprocessorConfig) -> str:
    payload = {
        "star_ids": records["star_id"].astype(str).tolist(),
        "spectral_length": config.spectral_length,
        "wavelength_min": config.wavelength_min,
        "wavelength_max": config.wavelength_max,
        "min_valid_fraction": config.min_valid_fraction,
        "max_nan_fraction": config.max_nan_fraction,
        "normalization": config.normalization,
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_or_build_spectral_features(
    records: pd.DataFrame,
    preprocessor_config: SpectralPreprocessorConfig,
    cache_dir: str | Path,
    refresh_cache: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = _cache_key(records, preprocessor_config)
    features_path = cache_dir / f"spectral_features_{digest}.npz"
    metadata_path = cache_dir / f"spectral_features_{digest}.csv"
    failures_path = cache_dir / f"spectral_features_{digest}_failures.csv"

    if features_path.exists() and metadata_path.exists() and not refresh_cache:
        cached = np.load(features_path, allow_pickle=False)
        metadata = pd.read_csv(metadata_path, dtype={"objID": str, "specObjID": str})
        failures = (
            pd.read_csv(failures_path, dtype={"objID": str, "specObjID": str})
            if failures_path.exists()
            else pd.DataFrame()
        )
        return cached["X"], metadata, failures, {
            "cache_hit": True,
            "features_path": str(features_path),
            "metadata_path": str(metadata_path),
            "failures_path": str(failures_path),
        }

    preprocessor = SpectralPreprocessor(preprocessor_config)
    features = []
    metadata_rows = []
    failures = []

    for _, row in records.iterrows():
        try:
            vector, quality = extract_spectrum_features(row["spectrum_path"], preprocessor)
            output_row = row.to_dict()
            output_row.update(quality)
            features.append(vector)
            metadata_rows.append(output_row)
        except Exception as error:
            failure = row.to_dict()
            failure["failure_reason"] = str(error)
            failures.append(failure)

    if not features:
        failure_df = pd.DataFrame(failures)
        if not failure_df.empty:
            failure_df.to_csv(failures_path, index=False)
        raise ValueError("Nenhum espectro valido foi extraido dos FITS locais.")

    X = np.stack(features, axis=0).astype(np.float32)
    metadata = pd.DataFrame(metadata_rows)
    failure_df = pd.DataFrame(failures)

    np.savez_compressed(
        features_path,
        X=X,
        star_ids=metadata["star_id"].astype(str).to_numpy(),
        feature_names=np.asarray(feature_names(preprocessor.grid), dtype=str),
    )
    metadata.to_csv(metadata_path, index=False)
    if not failure_df.empty:
        failure_df.to_csv(failures_path, index=False)

    return X, metadata, failure_df, {
        "cache_hit": False,
        "features_path": str(features_path),
        "metadata_path": str(metadata_path),
        "failures_path": str(failures_path),
    }


def feature_names(wavelength_grid: np.ndarray) -> list[str]:
    return [f"flux_{int(round(value))}" for value in wavelength_grid]


def make_split_manifest(
    records: pd.DataFrame,
    split_manifest_path: str | Path,
    seed: int,
    refresh_split: bool = False,
) -> pd.DataFrame:
    split_manifest_path = Path(split_manifest_path)
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if split_manifest_path.exists() and not refresh_split:
        existing = pd.read_csv(split_manifest_path, dtype={"objID": str, "specObjID": str})
        if {"star_id", "split"}.issubset(existing.columns):
            wanted = set(records["star_id"].astype(str))
            available = set(existing["star_id"].astype(str))
            if wanted.issubset(available):
                return records.merge(existing[["star_id", "split"]], on="star_id", how="left")

    rng = random.Random(seed)
    split_rows = []

    for _, group in records.groupby("spectral_leaf_label", sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        n_items = len(indices)

        if n_items == 1:
            assignments = ["train"]
        elif n_items == 2:
            assignments = ["train", "test"]
        else:
            n_val = max(1, int(math.floor(n_items * 0.15)))
            n_test = max(1, int(math.floor(n_items * 0.15)))
            n_train = n_items - n_val - n_test
            if n_train < 1:
                n_train = 1
                if n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1
            assignments = (
                ["train"] * n_train
                + ["validation"] * n_val
                + ["test"] * n_test
            )

        for record_index, split in zip(indices, assignments):
            item = records.loc[record_index]
            split_rows.append(
                {
                    "star_id": item["star_id"],
                    "objID": item.get("objID"),
                    "specObjID": item.get("specObjID"),
                    "spectral_parent_label": item["spectral_parent_label"],
                    "spectral_leaf_label": item["spectral_leaf_label"],
                    "split": split,
                }
            )

    manifest = pd.DataFrame(split_rows).sort_values("star_id").reset_index(drop=True)
    manifest.to_csv(split_manifest_path, index=False)
    return records.merge(manifest[["star_id", "split"]], on="star_id", how="left")


def rare_class_report(records: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    counts = records["spectral_leaf_label"].value_counts().rename_axis("spectral_leaf_label").reset_index(name="count")
    counts["percentage"] = counts["count"] / len(records) * 100.0
    parent_map = records.drop_duplicates("spectral_leaf_label").set_index("spectral_leaf_label")["spectral_parent_label"].to_dict()
    counts["spectral_parent_label"] = counts["spectral_leaf_label"].map(parent_map)
    counts["is_rare"] = counts["count"] <= threshold
    return counts[["spectral_parent_label", "spectral_leaf_label", "count", "percentage", "is_rare"]]


def _manual_log_loss(y_true, probabilities: np.ndarray, labels: list[str]) -> float | None:
    if len(y_true) == 0 or len(labels) == 0:
        return None
    positions = {label: index for index, label in enumerate(labels)}
    losses = []
    for row_index, label in enumerate(y_true):
        index = positions.get(str(label))
        if index is None:
            continue
        probability = float(probabilities[row_index, index])
        losses.append(-math.log(max(min(probability, 1.0 - 1e-15), 1e-15)))
    if not losses:
        return None
    return float(np.mean(losses))


def _classification_metrics(y_true, y_pred, labels: list[str], probabilities: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "status": "no_samples",
            "accuracy": None,
            "balanced_accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "f1_weighted": None,
            "log_loss": None,
        }

    return {
        "status": "ok",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "log_loss": _manual_log_loss(y_true, probabilities, labels),
    }


def path_for_leaf(leaf: str, hierarchy: dict) -> list[str]:
    parent = hierarchy["leaf_to_parent"].get(str(leaf), "UNKNOWN")
    if parent == leaf:
        return [ROOT_LABEL, parent]
    return [ROOT_LABEL, parent, str(leaf)]


def hierarchical_distance(true_leaf: str, pred_leaf: str, hierarchy: dict) -> int:
    true_path = path_for_leaf(true_leaf, hierarchy)
    pred_path = path_for_leaf(pred_leaf, hierarchy)
    shared = 0
    for true_node, pred_node in zip(true_path, pred_path):
        if true_node != pred_node:
            break
        shared += 1
    return (len(true_path) - shared) + (len(pred_path) - shared)


def hierarchical_error_summary(y_true_leaf, y_pred_leaf, hierarchy: dict) -> dict:
    if len(y_true_leaf) == 0:
        return {
            "exact_match": 0,
            "same_parent_error": 0,
            "cross_parent_error": 0,
            "exact_match_pct": None,
            "same_parent_error_pct": None,
            "cross_parent_error_pct": None,
            "average_hierarchical_distance": None,
        }

    exact = 0
    same_parent = 0
    cross_parent = 0
    distances = []
    leaf_to_parent = hierarchy["leaf_to_parent"]

    for true_leaf, pred_leaf in zip(y_true_leaf, y_pred_leaf):
        if true_leaf == pred_leaf:
            exact += 1
        elif leaf_to_parent.get(str(true_leaf)) == leaf_to_parent.get(str(pred_leaf)):
            same_parent += 1
        else:
            cross_parent += 1
        distances.append(hierarchical_distance(str(true_leaf), str(pred_leaf), hierarchy))

    total = len(y_true_leaf)
    return {
        "exact_match": int(exact),
        "same_parent_error": int(same_parent),
        "cross_parent_error": int(cross_parent),
        "exact_match_pct": float(exact / total * 100.0),
        "same_parent_error_pct": float(same_parent / total * 100.0),
        "cross_parent_error_pct": float(cross_parent / total * 100.0),
        "average_hierarchical_distance": float(np.mean(distances)),
    }


def aggregate_parent_probabilities(
    leaf_probabilities: np.ndarray,
    leaf_labels: list[str],
    parent_labels: list[str],
    hierarchy: dict,
) -> np.ndarray:
    output = np.zeros((leaf_probabilities.shape[0], len(parent_labels)), dtype=np.float64)
    parent_positions = {label: index for index, label in enumerate(parent_labels)}
    for leaf_index, leaf in enumerate(leaf_labels):
        parent = hierarchy["leaf_to_parent"][leaf]
        output[:, parent_positions[parent]] += leaf_probabilities[:, leaf_index]
    sums = output.sum(axis=1, keepdims=True)
    return np.divide(output, sums, out=np.zeros_like(output), where=sums > 0)


def evaluate_probability_outputs(
    y_parent_true,
    y_leaf_true,
    parent_probabilities: np.ndarray,
    leaf_probabilities: np.ndarray,
    parent_labels: list[str],
    leaf_labels: list[str],
    hierarchy: dict,
    alpha: float,
) -> dict:
    y_leaf_pred = np.asarray([leaf_labels[index] for index in np.argmax(leaf_probabilities, axis=1)], dtype=str)
    y_parent_pred = np.asarray(
        [hierarchy["leaf_to_parent"].get(str(leaf), "UNKNOWN") for leaf in y_leaf_pred],
        dtype=str,
    )

    parent_metrics = _classification_metrics(y_parent_true, y_parent_pred, parent_labels, parent_probabilities)
    leaf_metrics = _classification_metrics(y_leaf_true, y_leaf_pred, leaf_labels, leaf_probabilities)
    hierarchy_metrics = hierarchical_error_summary(y_leaf_true, y_leaf_pred, hierarchy)

    parent_loss = parent_metrics["log_loss"]
    leaf_loss = leaf_metrics["log_loss"]
    hierarchical_loss_value = None
    if parent_loss is not None and leaf_loss is not None:
        hierarchical_loss_value = float(alpha * parent_loss + (1.0 - alpha) * leaf_loss)

    return {
        "parent": parent_metrics,
        "leaf": leaf_metrics,
        "hierarchical": {
            **hierarchy_metrics,
            "alpha": float(alpha),
            "parent_log_loss": parent_loss,
            "leaf_log_loss": leaf_loss,
            "hierarchical_loss": hierarchical_loss_value,
            "parent_leaf_consistency": 1.0,
        },
        "predicted_parent": y_parent_pred,
        "predicted_leaf": y_leaf_pred,
    }


def _save_confusion_outputs(y_true, y_pred, labels: list[str], metrics_dir: Path, plots_dir: Path, name: str) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(metrics_dir / f"confusion_matrix_{name}.csv")

    size = max(7.0, min(36.0, 0.32 * len(labels) + 4.0))
    plt.figure(figsize=(size, size))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion matrix - {name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6 if len(labels) > 40 else 8)
    plt.yticks(np.arange(len(labels)), labels, fontsize=6 if len(labels) > 40 else 8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plots_dir / f"confusion_matrix_{name}.png", dpi=180)
    plt.close()


def _save_classification_report(y_true, y_pred, labels: list[str], metrics_dir: Path, name: str) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    _write_json(metrics_dir / f"classification_report_{name}.json", report)


def _save_feature_importance(importances: np.ndarray, names: list[str], path: Path, top_n: int = 30) -> pd.DataFrame:
    order = np.argsort(importances)[::-1][:top_n]
    table = pd.DataFrame(
        {
            "feature": [names[index] for index in order],
            "importance": [float(importances[index]) for index in order],
        }
    )
    table.to_csv(path, index=False)
    return table


def _plot_feature_importance(table: pd.DataFrame, path: Path, title: str) -> None:
    if table.empty:
        return
    top = table.head(20).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_alpha_grid(alpha_table: pd.DataFrame, path: Path) -> None:
    if alpha_table.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_table["alpha"], alpha_table["hierarchical_loss"], marker="o", label="hierarchical_loss")
    plt.plot(alpha_table["alpha"], alpha_table["leaf_accuracy"], marker="o", label="leaf_accuracy")
    plt.plot(alpha_table["alpha"], alpha_table["parent_accuracy"], marker="o", label="parent_accuracy")
    plt.xlabel("alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_Sem dados._"
    view = df.head(max_rows).copy()
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Mostrando {max_rows} de {len(df)} linhas._")
    return "\n".join(lines)


def _parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return int(text)


def _parse_optional_float_or_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    parsed = float(text)
    return int(parsed) if parsed.is_integer() else parsed


def _parse_max_features(value):
    if value is None:
        return "sqrt"
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    if text in {"sqrt", "log2"}:
        return text
    try:
        parsed = float(text)
    except ValueError:
        return text
    return int(parsed) if parsed.is_integer() and parsed > 1 else parsed


def _parse_class_weight(value):
    if value is None:
        return "balanced_subsample"
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def _parse_alpha_grid(value: str | None) -> list[float]:
    if _is_missing(value):
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _arg(args, name: str, default):
    return getattr(args, name, default)


def config_from_args(args, seed: int) -> RandomForestModelConfig:
    return RandomForestModelConfig(
        n_estimators=int(_arg(args, "rf_n_estimators", 300)),
        criterion=str(_arg(args, "rf_criterion", "log_loss")),
        max_depth=_parse_optional_int(_arg(args, "rf_max_depth", None)),
        min_samples_split=int(_arg(args, "rf_min_samples_split", 2)),
        min_samples_leaf=int(_arg(args, "rf_min_samples_leaf", 1)),
        max_features=_parse_max_features(_arg(args, "rf_max_features", "sqrt")),
        bootstrap=bool(_arg(args, "rf_bootstrap", True)),
        class_weight=_parse_class_weight(_arg(args, "rf_class_weight", "balanced_subsample")),
        max_samples=_parse_optional_float_or_int(_arg(args, "rf_bootstrap_max_samples", None)),
        random_state=seed,
        n_jobs=int(_arg(args, "rf_n_jobs", -1)),
        prediction_mode=str(_arg(args, "rf_prediction_mode", "global_probability")),
    )


def _evaluate_model_on_split(
    model,
    model_key: str,
    X_split,
    split_records: pd.DataFrame,
    hierarchy: dict,
    alpha: float,
) -> dict:
    parent_labels = hierarchy["parent_labels"]
    leaf_labels = hierarchy["leaf_labels"]
    y_parent = split_records["spectral_parent_label"].astype(str).to_numpy()
    y_leaf = split_records["spectral_leaf_label"].astype(str).to_numpy()

    leaf_probabilities = model.predict_leaf_proba(X_split, leaf_labels)
    if model_key == "hierarchical_random_forest":
        parent_probabilities = model.predict_parent_proba(X_split, parent_labels)
    else:
        parent_probabilities = aggregate_parent_probabilities(
            leaf_probabilities,
            leaf_labels,
            parent_labels,
            hierarchy,
        )

    return evaluate_probability_outputs(
        y_parent,
        y_leaf,
        parent_probabilities,
        leaf_probabilities,
        parent_labels,
        leaf_labels,
        hierarchy,
        alpha,
    )


def _write_predictions(
    records: pd.DataFrame,
    evaluation: dict,
    output_path: Path,
) -> None:
    table = records[
        [
            "star_id",
            "objID",
            "specObjID",
            "spectral_subclass_raw",
            "spectral_parent_label",
            "spectral_leaf_label",
            "spectrum_path",
        ]
    ].copy()
    table["predicted_parent_label"] = evaluation["predicted_parent"]
    table["predicted_leaf_label"] = evaluation["predicted_leaf"]
    table["error_type"] = [
        "EXACT_MATCH"
        if true_leaf == pred_leaf
        else (
            "SAME_PARENT_ERROR"
            if true_parent == pred_parent
            else "CROSS_PARENT_ERROR"
        )
        for true_parent, true_leaf, pred_parent, pred_leaf in zip(
            table["spectral_parent_label"],
            table["spectral_leaf_label"],
            table["predicted_parent_label"],
            table["predicted_leaf_label"],
        )
    ]
    table.to_csv(output_path, index=False)


def _model_metric_row(model_name: str, metrics: dict) -> dict:
    return {
        "model": model_name,
        "parent_accuracy": metrics["parent"]["accuracy"],
        "leaf_accuracy": metrics["leaf"]["accuracy"],
        "macro_f1": metrics["leaf"]["f1_macro"],
        "weighted_f1": metrics["leaf"]["f1_weighted"],
        "hierarchical_loss": metrics["hierarchical"]["hierarchical_loss"],
        "hierarchical_distance": metrics["hierarchical"]["average_hierarchical_distance"],
        "cross_parent_errors": metrics["hierarchical"]["cross_parent_error"],
        "same_parent_errors": metrics["hierarchical"]["same_parent_error"],
    }


def _make_report(
    model_key: str,
    results: ResultsManager,
    config: RandomForestModelConfig,
    records: pd.DataFrame,
    split_records: pd.DataFrame,
    hierarchy: dict,
    rare_classes: pd.DataFrame,
    metrics_by_model: dict,
    cache_info: dict,
    timings: dict,
) -> str:
    lines = [
        f"# {model_key}",
        "",
        "## Configuracao",
        "",
        f"- Amostras validas: {len(records)}",
        f"- Features espectrais: {config.to_dict()}",
        f"- Cache de features: {cache_info.get('features_path')}",
        f"- Split: {split_records['split'].value_counts().to_dict()}",
        f"- Parents: {len(hierarchy['parent_labels'])}",
        f"- Leaf labels: {len(hierarchy['leaf_labels'])}",
        "",
        "## Nota metodologica",
        "",
        "O Random Forest e treinado com o criterio interno do scikit-learn. A hierarchical loss nao substitui backpropagation nem gradient descent; ela e usada para avaliacao e comparacao dos caminhos hierarquicos.",
        "",
        "## Distribuicao",
        "",
        _markdown_table(rare_classes),
        "",
        "## Resultados",
        "",
    ]

    for name, payload in metrics_by_model.items():
        test_metrics = payload.get("test", {})
        if not test_metrics:
            continue
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Parent accuracy: {test_metrics['parent']['accuracy']}",
                f"- Leaf accuracy: {test_metrics['leaf']['accuracy']}",
                f"- Leaf macro F1: {test_metrics['leaf']['f1_macro']}",
                f"- Hierarchical loss: {test_metrics['hierarchical']['hierarchical_loss']}",
                f"- Average hierarchical distance: {test_metrics['hierarchical']['average_hierarchical_distance']}",
                f"- Same-parent errors: {test_metrics['hierarchical']['same_parent_error']}",
                f"- Cross-parent errors: {test_metrics['hierarchical']['cross_parent_error']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Tempos",
            "",
            json.dumps(_json_ready(timings), indent=4),
            "",
            "## Saidas",
            "",
            f"- Modelo: {Path(results.model_dir) / 'model.joblib'}",
            f"- Metricas: {Path(results.metrics_dir) / 'metrics.json'}",
            f"- Predicoes: {Path(results.predictions_dir) / 'test_predictions.csv'}",
            f"- Relatorio: {Path(results.base_dir) / 'reports' / 'hierarchical_random_forest_report.md'}",
        ]
    )
    return "\n".join(lines)


def run_star_random_forest(args, model_name: str, run_index: int = 1) -> dict:
    seed = int(_arg(args, "seed", 42)) + run_index - 1
    random.seed(seed)
    np.random.seed(seed)

    model_key = str(model_name).lower()
    config = config_from_args(args, seed)
    if model_key == "flat_random_forest":
        config.prediction_mode = "global_probability"

    results = ResultsManager(model_key)
    base_dir = Path(results.base_dir)
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    max_spectral_samples = int(_arg(args, "max_spectral_samples", -1))
    if max_spectral_samples < 0:
        max_spectral_samples = None

    start = time.perf_counter()
    source_records = load_star_supervised_records(
        _arg(args, "star_objects_index", DEFAULT_STAR_OBJECTS_INDEX),
        max_samples=max_spectral_samples,
        seed=seed,
    )

    preprocessor_config = SpectralPreprocessorConfig(
        spectral_length=int(_arg(args, "spectral_length", 1024)),
        wavelength_min=float(_arg(args, "spectral_wavelength_min", 3800.0)),
        wavelength_max=float(_arg(args, "spectral_wavelength_max", 9200.0)),
    )

    X, feature_records, failures, cache_info = load_or_build_spectral_features(
        source_records,
        preprocessor_config,
        _arg(args, "rf_cache_dir", DEFAULT_RF_CACHE_DIR),
        refresh_cache=bool(_arg(args, "rf_refresh_cache", False)),
    )
    feature_time = time.perf_counter() - start

    split_start = time.perf_counter()
    split_records = make_split_manifest(
        feature_records,
        _arg(args, "rf_split_manifest", DEFAULT_RF_SPLIT_MANIFEST),
        seed=seed,
        refresh_split=bool(_arg(args, "rf_refresh_split", False)),
    )
    split_time = time.perf_counter() - split_start

    hierarchy = build_hierarchy(split_records)
    rare_classes = rare_class_report(split_records)
    rare_classes.to_csv(Path(results.metadata_dir) / "rare_classes.csv", index=False)
    split_records.to_csv(Path(results.metadata_dir) / "split_manifest.csv", index=False)
    _write_json(Path(results.metadata_dir) / "hierarchy.json", hierarchy)
    _write_json(
        Path(results.metadata_dir) / "feature_config.json",
        SpectralPreprocessor(preprocessor_config).to_dict(),
    )
    if not failures.empty:
        failures.to_csv(Path(results.metadata_dir) / "feature_failures.csv", index=False)

    split_positions = {
        split: split_records.index[split_records["split"].eq(split)].to_numpy()
        for split in ["train", "validation", "test"]
    }

    train_indices = split_positions["train"]
    if len(train_indices) == 0:
        raise ValueError("Split supervisionado nao possui amostras de treino.")

    X_train = X[train_indices]
    y_train_leaf = split_records.iloc[train_indices]["spectral_leaf_label"].astype(str).to_numpy()
    y_train_parent = split_records.iloc[train_indices]["spectral_parent_label"].astype(str).to_numpy()

    train_start = time.perf_counter()
    selected_model = ModelFactory.create_sklearn(model_key, config)
    if model_key == "hierarchical_random_forest":
        selected_model.fit(X_train, y_train_leaf, y_train_parent)
    else:
        selected_model.fit(X_train, y_train_leaf, y_train_parent)
    selected_model.save(Path(results.model_dir) / "model.joblib")

    flat_baseline = None
    if model_key == "hierarchical_random_forest":
        flat_baseline = ModelFactory.create_sklearn("flat_random_forest", config)
        flat_baseline.fit(X_train, y_train_leaf, y_train_parent)
        flat_baseline.save(Path(results.model_dir) / "flat_baseline.joblib")
    training_time = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    alpha = float(_arg(args, "hrf_alpha", 0.35))
    metrics_by_model = {model_key: {}}
    models_to_evaluate = {model_key: selected_model}
    if flat_baseline is not None:
        metrics_by_model["flat_random_forest"] = {}
        models_to_evaluate["flat_random_forest"] = flat_baseline

    for current_name, current_model in models_to_evaluate.items():
        for split, indices in split_positions.items():
            if len(indices) == 0:
                continue
            split_payload = _evaluate_model_on_split(
                current_model,
                current_name,
                X[indices],
                split_records.iloc[indices],
                hierarchy,
                alpha,
            )
            metrics_by_model[current_name][split] = {
                "parent": split_payload["parent"],
                "leaf": split_payload["leaf"],
                "hierarchical": split_payload["hierarchical"],
            }

            if split == "test" and current_name == model_key:
                _write_predictions(
                    split_records.iloc[indices],
                    split_payload,
                    Path(results.predictions_dir) / "test_predictions.csv",
                )
                leaf_probs = current_model.predict_leaf_proba(X[indices], hierarchy["leaf_labels"])
                leaf_positions = {label: index for index, label in enumerate(hierarchy["leaf_labels"])}
                y_true_leaf = split_records.iloc[indices]["spectral_leaf_label"].astype(str).to_numpy()
                y_true_parent = split_records.iloc[indices]["spectral_parent_label"].astype(str).to_numpy()
                y_pred_leaf = split_payload["predicted_leaf"]
                np.save(Path(results.predictions_dir) / "leaf_probabilities.npy", leaf_probs)
                np.save(Path(results.predictions_dir) / "probabilities.npy", leaf_probs)
                np.save(
                    Path(results.predictions_dir) / "labels.npy",
                    np.asarray([leaf_positions[label] for label in y_true_leaf], dtype=np.int64),
                )
                np.save(
                    Path(results.predictions_dir) / "predictions.npy",
                    np.asarray([leaf_positions[label] for label in y_pred_leaf], dtype=np.int64),
                )
                np.save(
                    Path(results.predictions_dir) / "confidences.npy",
                    np.max(leaf_probs, axis=1),
                )
                if current_name == "hierarchical_random_forest":
                    np.save(Path(results.predictions_dir) / "parent_probabilities.npy", current_model.predict_parent_proba(X[indices], hierarchy["parent_labels"]))
                else:
                    np.save(
                        Path(results.predictions_dir) / "parent_probabilities.npy",
                        aggregate_parent_probabilities(leaf_probs, hierarchy["leaf_labels"], hierarchy["parent_labels"], hierarchy),
                    )

                _save_confusion_outputs(
                    y_true_parent,
                    split_payload["predicted_parent"],
                    hierarchy["parent_labels"],
                    Path(results.metrics_dir),
                    Path(results.plots_dir),
                    "parent",
                )
                _save_confusion_outputs(
                    y_true_leaf,
                    split_payload["predicted_leaf"],
                    hierarchy["leaf_labels"],
                    Path(results.metrics_dir),
                    Path(results.plots_dir),
                    "leaf",
                )
                _save_classification_report(
                    y_true_parent,
                    split_payload["predicted_parent"],
                    hierarchy["parent_labels"],
                    Path(results.metrics_dir),
                    "parent",
                )
                _save_classification_report(
                    y_true_leaf,
                    split_payload["predicted_leaf"],
                    hierarchy["leaf_labels"],
                    Path(results.metrics_dir),
                    "leaf",
                )

    evaluation_time = time.perf_counter() - eval_start

    comparison_rows = []
    for current_name, payload in metrics_by_model.items():
        if "test" in payload:
            comparison_rows.append(_model_metric_row(current_name, payload["test"]))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(Path(results.metrics_dir) / "flat_vs_hierarchical.csv", index=False)

    alpha_values = _parse_alpha_grid(_arg(args, "hrf_alpha_grid", ""))
    if alpha_values and "test" in split_positions and len(split_positions["test"]) > 0:
        test_indices = split_positions["test"]
        leaf_probs = selected_model.predict_leaf_proba(X[test_indices], hierarchy["leaf_labels"])
        if model_key == "hierarchical_random_forest":
            parent_probs = selected_model.predict_parent_proba(X[test_indices], hierarchy["parent_labels"])
        else:
            parent_probs = aggregate_parent_probabilities(leaf_probs, hierarchy["leaf_labels"], hierarchy["parent_labels"], hierarchy)
        alpha_rows = []
        for alpha_value in alpha_values:
            alpha_metrics = evaluate_probability_outputs(
                split_records.iloc[test_indices]["spectral_parent_label"].astype(str).to_numpy(),
                split_records.iloc[test_indices]["spectral_leaf_label"].astype(str).to_numpy(),
                parent_probs,
                leaf_probs,
                hierarchy["parent_labels"],
                hierarchy["leaf_labels"],
                hierarchy,
                alpha_value,
            )
            alpha_rows.append(
                {
                    "alpha": alpha_value,
                    "parent_loss": alpha_metrics["hierarchical"]["parent_log_loss"],
                    "leaf_loss": alpha_metrics["hierarchical"]["leaf_log_loss"],
                    "hierarchical_loss": alpha_metrics["hierarchical"]["hierarchical_loss"],
                    "parent_accuracy": alpha_metrics["parent"]["accuracy"],
                    "leaf_accuracy": alpha_metrics["leaf"]["accuracy"],
                    "parent_macro_f1": alpha_metrics["parent"]["f1_macro"],
                    "leaf_macro_f1": alpha_metrics["leaf"]["f1_macro"],
                    "average_hierarchical_distance": alpha_metrics["hierarchical"]["average_hierarchical_distance"],
                }
            )
        alpha_table = pd.DataFrame(alpha_rows)
        alpha_table.to_csv(Path(results.metrics_dir) / "alpha_grid.csv", index=False)
        _plot_alpha_grid(alpha_table, Path(results.plots_dir) / "alpha_grid.png")

    names = feature_names(SpectralPreprocessor(preprocessor_config).grid)
    if model_key == "flat_random_forest":
        importance = _save_feature_importance(
            selected_model.model.feature_importances_,
            names,
            Path(results.metrics_dir) / "feature_importance_flat.csv",
        )
        _plot_feature_importance(importance, Path(results.plots_dir) / "feature_importance_flat.png", "Flat RF feature importance")
    else:
        if selected_model.parent_model is not None:
            importance = _save_feature_importance(
                selected_model.parent_model.feature_importances_,
                names,
                Path(results.metrics_dir) / "feature_importance_parent.csv",
            )
            _plot_feature_importance(importance, Path(results.plots_dir) / "feature_importance_parent.png", "Parent RF feature importance")
        for parent, child in sorted(selected_model.child_models.items())[:10]:
            table = _save_feature_importance(
                child.feature_importances_,
                names,
                Path(results.metrics_dir) / f"feature_importance_child_{parent}.csv",
            )
            _plot_feature_importance(table, Path(results.plots_dir) / f"feature_importance_child_{parent}.png", f"Child RF {parent} feature importance")

    timings = {
        "feature_time_seconds": feature_time,
        "split_time_seconds": split_time,
        "training_time_seconds": training_time,
        "evaluation_time_seconds": evaluation_time,
    }

    metrics_payload = {
        "model": model_key,
        "config": config.to_dict(),
        "splits": split_records["split"].value_counts().to_dict(),
        "classes": {
            "parents": hierarchy["parent_labels"],
            "leaf_labels": hierarchy["leaf_labels"],
        },
        "metrics_by_model": metrics_by_model,
        "timings": timings,
        "cache": cache_info,
    }
    _write_json(Path(results.metrics_dir) / "metrics.json", metrics_payload)

    test_metrics = metrics_by_model[model_key].get("test", {})
    summary_metrics = {
        "accuracy": test_metrics.get("leaf", {}).get("accuracy"),
        "precision": test_metrics.get("leaf", {}).get("precision_macro"),
        "recall": test_metrics.get("leaf", {}).get("recall_macro"),
        "f1": test_metrics.get("leaf", {}).get("f1_weighted"),
        "macro_f1": test_metrics.get("leaf", {}).get("f1_macro"),
        "hierarchical_loss": test_metrics.get("hierarchical", {}).get("hierarchical_loss"),
    }
    summary_metrics = {
        key: value
        for key, value in summary_metrics.items()
        if value is not None
    }

    metadata = {
        "model": model_key,
        "run_index": run_index,
        "seed": seed,
        "results_dir": results.base_dir,
        "star_objects_index": _arg(args, "star_objects_index", DEFAULT_STAR_OBJECTS_INDEX),
        "valid_spectra": int(len(feature_records)),
        "feature_failures": int(len(failures)),
        "random_forest": config.to_dict(),
        "no_remote_downloads": True,
    }
    results.save_metadata(_json_ready(metadata))

    report = _make_report(
        model_key,
        results,
        config,
        split_records,
        split_records,
        hierarchy,
        rare_classes,
        metrics_by_model,
        cache_info,
        timings,
    )
    (reports_dir / "hierarchical_random_forest_report.md").write_text(report, encoding="utf-8")

    print("\nRandom Forest supervisionado finalizado")
    print(f"Modelo: {model_key}")
    print(f"Resultados: {results.base_dir}")
    print(f"Amostras validas: {len(feature_records)} | Falhas: {len(failures)}")

    return {
        "name": model_key,
        "model_key": model_key,
        "run_index": run_index,
        "seed": seed,
        "results_dir": results.base_dir,
        "metrics": summary_metrics,
        "class_names": hierarchy["leaf_labels"],
    }
