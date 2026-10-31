from __future__ import annotations

import gc
import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from models.model_factory import ModelFactory
from shared.metrics import plot_confusion_matrix, plot_training_history
from shared.stellar_hierarchy import ROMAN_LUMINOSITY_ORDER, SPECIAL_STELLAR_CLASSES
from training.trainer import Trainer
from utils.result_manager import ResultsManager

from .image_data import (
    build_inference_loader,
    build_node_data,
    load_hierarchy_manifest,
)
from .registry import HierarchyNode, HierarchyRegistry


def run_hierarchical_image_pipeline(
    args,
    model_name: str,
    run_index: int,
    execution_dir: str,
) -> dict:
    """Treina e avalia os tres niveis da hierarquia visual."""
    manifest = load_hierarchy_manifest(args.hierarchy_manifest)
    registry = HierarchyRegistry.from_manifest(
        manifest,
        min_train_per_class=args.hierarchy_min_train_per_class,
    )
    required_nodes = (
        registry.root,
        registry.star_luminosity,
        registry.star_spectral,
    )
    unavailable = [node for node in required_nodes if not node.requires_model]
    if unavailable:
        details = "; ".join(
            f"{node.key}: {node.reason or 'classes insuficientes'}"
            for node in unavailable
        )
        raise ValueError(f"Nao foi possivel montar a hierarquia: {details}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + run_index - 1
    family_name = f"{model_name}_hierarchical"
    family_results = ResultsManager(
        family_name,
        execution_dir=execution_dir,
        run_index=run_index,
        variant="hierarchy",
    )
    resume = bool(getattr(args, "hierarchy_resume_execution", None))
    resume_plan = {
        "resume_enabled": resume,
        "execution_dir": str(execution_dir),
        "model": model_name,
        "run_index": run_index,
        "nodes": [],
    }
    _validate_or_save_hierarchy(family_results, registry)

    summary_path = os.path.join(family_results.base_dir, "summary.json")
    if resume and os.path.isfile(summary_path):
        summary = _load_json(summary_path)
        _validate_completed_family(
            summary,
            args=args,
            registry=registry,
            model_name=model_name,
            run_index=run_index,
            seed=seed,
            device=device,
        )
        resume_plan["family_action"] = "skip_complete_family"
        resume_plan["nodes"] = [
            {"node": node.key, "action": "skip_complete"}
            for node in registry.trainable_nodes()
        ]
        family_results.save_json("metadata/resume_plan.json", resume_plan)
        print(f"RETOMADA {model_name}: familia completa, treinamento ignorado")
        return summary

    node_records: list[dict] = []
    checkpoints: dict[str, str] = {}
    for node_offset, node in enumerate(registry.trainable_nodes()):
        print("\n" + "=" * 70)
        print(f"HIERARQUIA {model_name} | {node.artifact_name}")
        print("=" * 70)
        record = _run_node(
            args=args,
            manifest=manifest,
            node=node,
            model_name=model_name,
            family_name=family_name,
            run_index=run_index,
            execution_dir=execution_dir,
            device=device,
            seed=seed + node_offset,
            resume=resume,
        )
        node_records.append(record)
        checkpoints[node.key] = record["checkpoint"]
        resume_plan["nodes"].append(
            {"node": node.key, "action": record["resume_action"]}
        )
        family_results.save_json("metadata/resume_plan.json", resume_plan)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    route_results = _evaluate_end_to_end(
        args=args,
        manifest=manifest,
        registry=registry,
        checkpoints=checkpoints,
        model_name=model_name,
        device=device,
        results=family_results,
    )
    summary = {
        "name": family_name,
        "model_key": model_name,
        "strategy": "hierarchical_global_spectral",
        "modality": "image",
        "run_index": run_index,
        "seed": seed,
        "results_dir": family_results.base_dir,
        "metrics": route_results["metrics"],
        "nodes": node_records,
        "hierarchy": registry.to_dict(),
    }
    family_results.save_json("summary.json", summary)
    resume_plan["family_action"] = "completed"
    family_results.save_json("metadata/resume_plan.json", resume_plan)
    return summary


def _run_node(
    args,
    manifest: pd.DataFrame,
    node: HierarchyNode,
    model_name: str,
    family_name: str,
    run_index: int,
    execution_dir: str,
    device: torch.device,
    seed: int,
    resume: bool,
) -> dict:
    _set_seed(seed)
    data = build_node_data(
        manifest=manifest,
        node=node,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        max_train_images=_optional_limit(args.max_train_images),
        max_validation_images=_optional_limit(args.max_validation_images),
        max_test_images=_optional_limit(args.max_test_images),
        augmentation_enabled=(
            node.level >= 2
            and not bool(getattr(args, "hierarchy_disable_augmentation", False))
        ),
        balance_target=int(getattr(args, "hierarchy_balance_target", 500)),
        min_synthetic_fraction=float(
            getattr(args, "hierarchy_min_synthetic_fraction", 0.10)
        ),
        max_synthetic_per_source=int(
            getattr(args, "hierarchy_max_synthetic_per_source", 10)
        ),
    )
    results = ResultsManager(
        family_name,
        execution_dir=execution_dir,
        run_index=run_index,
        variant=os.path.join("hierarchy", node.artifact_name),
    )
    class_weights = _balanced_class_weights(data.train_records, data.class_names)
    data.train_records.to_csv(
        os.path.join(results.metadata_dir, "train_manifest.csv"),
        index=False,
    )
    results.save_json("metadata/augmentation_summary.json", data.augmentation_summary)
    checkpoint = results.get_model_path()
    metadata = _node_metadata(
        args=args,
        node=node,
        model_name=model_name,
        data=data,
        class_weights=class_weights,
        device=device,
        seed=seed,
    )

    resume_action = _node_resume_action(resume, checkpoint, results, node)
    if resume_action != "train":
        metadata_path = os.path.join(results.metadata_dir, "info.json")
        if os.path.isfile(metadata_path):
            _validate_node_metadata(_load_json(metadata_path), metadata)
        wrapper = _load_node_checkpoint(
            checkpoint,
            model_name=model_name,
            node=node,
            device=device,
            image_size=args.img_size,
        )
        if resume_action == "skip_complete":
            metrics = _load_json(results.get_metrics_path())
            action = "skip_complete"
            print(f"RETOMADA {model_name}/{node.key}: checkpoint e avaliacao completos")
        else:
            metrics = _evaluate_node(wrapper, data, node, results, device)
            results.save_metadata(metadata)
            action = "evaluate_checkpoint"
            print(f"RETOMADA {model_name}/{node.key}: checkpoint reutilizado e reavaliado")
        del wrapper
    else:
        wrapper = ModelFactory.create_for_node(
            model_name=model_name,
            node_key=node.key,
            num_classes=len(data.class_names),
            device=device,
            image_size=args.img_size,
        )
        trainer = Trainer(
            model=wrapper.get_model(),
            train_loader=data.train_loader,
            validation_loader=data.validation_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            early_stopping_patience=_optional_limit(args.early_stopping_patience),
            early_stopping_min_delta=args.early_stopping_min_delta,
            scheduler_patience=_optional_limit(args.scheduler_patience),
            scheduler_factor=args.scheduler_factor,
            scheduler_min_lr=args.scheduler_min_lr,
            class_weights=class_weights,
        )
        history = trainer.train()
        plot_training_history(history, results.plots_dir)
        torch.save(wrapper.get_model().state_dict(), checkpoint)
        metrics = _evaluate_node(wrapper, data, node, results, device)
        results.save_metadata(metadata)
        action = "train"
        del wrapper

    return {
        "node": node.key,
        "artifact_name": node.artifact_name,
        "classes": data.class_names,
        "metrics": metrics,
        "results_dir": results.base_dir,
        "checkpoint": checkpoint,
        "resume_action": action,
    }


def _node_metadata(args, node, model_name, data, class_weights, device, seed):
    return {
        "model": model_name,
        "model_family": model_name,
        "hierarchy_node": node.to_dict(),
        "class_names": data.class_names,
        "class_weights": {
            label: float(weight)
            for label, weight in zip(data.class_names, class_weights)
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "img_size": args.img_size,
        "augmentation": {
            "enabled": bool(
                node.level >= 2
                and not getattr(args, "hierarchy_disable_augmentation", False)
            ),
            "balance_target": int(getattr(args, "hierarchy_balance_target", 500)),
            "min_synthetic_fraction": float(
                getattr(args, "hierarchy_min_synthetic_fraction", 0.10)
            ),
            "max_synthetic_per_source": int(
                getattr(args, "hierarchy_max_synthetic_per_source", 10)
            ),
            "summary": data.augmentation_summary,
        },
        "device": str(device),
        "seed": seed,
        "split_counts": {
            "train": len(data.train_records),
            "val": len(data.validation_records),
            "test": len(data.test_records),
        },
    }


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_or_save_hierarchy(results, registry) -> None:
    path = os.path.join(results.metadata_dir, "hierarchy.json")
    expected = registry.to_dict()
    if os.path.isfile(path):
        if _load_json(path) != expected:
            raise ValueError(
                f"A hierarquia da retomada e incompativel com o manifesto atual: {path}"
            )
        return
    results.save_json("metadata/hierarchy.json", expected)


def _validate_node_metadata(actual: dict, expected: dict) -> None:
    exact_fields = (
        "model",
        "model_family",
        "class_names",
        "epochs",
        "batch_size",
        "img_size",
        "seed",
        "split_counts",
        "augmentation",
    )
    mismatches = [
        field for field in exact_fields
        if actual.get(field) != expected.get(field)
    ]
    if not np.isclose(
        float(actual.get("learning_rate", np.nan)),
        float(expected["learning_rate"]),
        rtol=0.0,
        atol=1e-12,
    ):
        mismatches.append("learning_rate")
    actual_node = actual.get("hierarchy_node", {})
    if actual_node.get("key") != expected["hierarchy_node"]["key"]:
        mismatches.append("hierarchy_node.key")
    if mismatches:
        raise ValueError(
            "Checkpoint incompativel com os parametros da retomada: "
            + ", ".join(sorted(set(mismatches)))
        )


def _load_node_checkpoint(checkpoint, model_name, node, device, image_size=256):
    wrapper = ModelFactory.create_for_node(
        model_name=model_name,
        node_key=node.key,
        num_classes=len(node.classes),
        device=device,
        image_size=image_size,
    )
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    try:
        wrapper.get_model().load_state_dict(state)
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint incompativel com {model_name}/{node.key}: {checkpoint}"
        ) from error
    return wrapper


def _node_is_complete(results, node) -> bool:
    required = [
        results.get_model_path(),
        results.get_metrics_path(),
        os.path.join(results.metadata_dir, "info.json"),
        os.path.join(results.metrics_dir, "classification_report.json"),
        os.path.join(results.plots_dir, "confusion_matrix.png"),
        os.path.join(results.predictions_dir, "test_predictions.csv"),
    ]
    if not all(os.path.isfile(path) for path in required):
        return False
    return True


def _node_resume_action(resume, checkpoint, results, node) -> str:
    if not resume or not os.path.isfile(checkpoint):
        return "train"
    if _node_is_complete(results, node):
        return "skip_complete"
    return "evaluate_checkpoint"


def _validate_completed_family(
    summary,
    args,
    registry,
    model_name,
    run_index,
    seed,
    device,
) -> None:
    expected_values = {
        "name": f"{model_name}_hierarchical",
        "model_key": model_name,
        "strategy": "hierarchical_global_spectral",
        "run_index": run_index,
        "seed": seed,
    }
    mismatches = [
        field for field, expected in expected_values.items()
        if summary.get(field) != expected
    ]
    if summary.get("hierarchy") != registry.to_dict():
        mismatches.append("hierarchy")
    if mismatches:
        raise ValueError(
            "Resumo existente incompativel com a retomada: "
            + ", ".join(mismatches)
        )

    records = {record["node"]: record for record in summary.get("nodes", [])}
    for node_offset, node in enumerate(registry.trainable_nodes()):
        if node.key not in records:
            raise ValueError(f"Resumo existente sem o no {node.key}")
        record = records[node.key]
        if record.get("classes") != list(node.classes):
            raise ValueError(f"Classes incompativeis no no {node.key}")
        checkpoint = record.get("checkpoint", "")
        results_dir = record.get("results_dir", "")
        if not checkpoint or not os.path.isfile(checkpoint) or not os.path.isdir(results_dir):
            raise FileNotFoundError(f"Artefatos ausentes para o no {node.key}")
        results = ResultsManager.from_existing(results_dir)
        if not _node_is_complete(results, node):
            raise ValueError(f"Resumo marca no incompleto como concluido: {node.key}")
        metadata = _load_json(os.path.join(results.metadata_dir, "info.json"))
        expected_metadata = {
            "model": model_name,
            "model_family": model_name,
            "class_names": list(node.classes),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "img_size": args.img_size,
            "seed": seed + node_offset,
            "split_counts": metadata.get("split_counts"),
            "hierarchy_node": node.to_dict(),
            "augmentation": metadata.get("augmentation"),
        }
        _validate_node_metadata(metadata, expected_metadata)
        wrapper = _load_node_checkpoint(
            checkpoint, model_name, node, device, image_size=args.img_size
        )
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_node(wrapper, data, node, results, device) -> dict:
    model = wrapper.get_model()
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in data.test_loader:
            outputs = wrapper.forward_with_features(images.to(device))
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(outputs["predictions"].cpu().numpy().tolist())
            confidences.extend(outputs["confidences"].cpu().numpy().tolist())
            logits.append(outputs["logits"].cpu().numpy())

    logits_matrix = np.concatenate(logits, axis=0)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples": len(y_true),
    }
    results.save_json("metrics/metrics.json", metrics)
    results.save_json(
        "metrics/classification_report.json",
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(data.class_names))),
            target_names=data.class_names,
            output_dict=True,
            zero_division=0,
        ),
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        data.class_names,
        os.path.join(results.plots_dir, "confusion_matrix.png"),
    )
    predictions = data.test_records[["object_id", "image_path", "target_label"]].copy()
    predictions["predicted_label"] = [data.class_names[index] for index in y_pred]
    predictions["confidence"] = confidences

    predictions.to_csv(
        os.path.join(results.predictions_dir, "test_predictions.csv"),
        index=False,
    )
    return metrics


def _evaluate_end_to_end(
    args,
    manifest: pd.DataFrame,
    registry: HierarchyRegistry,
    checkpoints: dict[str, str],
    model_name: str,
    device: torch.device,
    results: ResultsManager,
) -> dict:
    test = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True).copy()
    output = test[
        [
            "object_id",
            "spec_object_id",
            "image_path",
            "object_class",
            "stellar_level2_label",
            "stellar_level3_label",
        ]
    ].copy()
    output["predicted_object_class"] = ""
    output["root_confidence"] = np.nan
    output["predicted_stellar_level2"] = ""
    output["level2_confidence"] = np.nan
    output["predicted_stellar_level3"] = ""
    output["level3_confidence"] = np.nan
    output["stop_reason"] = ""

    root_labels, root_conf = _predict_records(
        test,
        registry.root,
        checkpoints[registry.root.key],
        model_name,
        args,
        device,
    )
    output["predicted_object_class"] = root_labels
    output["root_confidence"] = root_conf
    threshold = float(args.hierarchy_confidence_threshold)
    output.loc[root_conf < threshold, "stop_reason"] = "low_confidence_root"

    star_indices = output.index[
        output["predicted_object_class"].eq("STAR")
        & output["root_confidence"].ge(threshold)
    ]
    if len(star_indices):
        level2_labels, level2_confidence = _predict_records(
            test.loc[star_indices].reset_index(drop=True),
            registry.star_luminosity,
            checkpoints[registry.star_luminosity.key],
            model_name,
            args,
            device,
        )
        output.loc[star_indices, "predicted_stellar_level2"] = level2_labels
        output.loc[star_indices, "level2_confidence"] = level2_confidence
        low_level2 = star_indices[np.asarray(level2_confidence) < threshold]
        output.loc[low_level2, "stop_reason"] = "low_confidence_level2"

    spectral_node = registry.star_spectral
    spectral_indices = output.index[
        output["predicted_stellar_level2"].isin(ROMAN_LUMINOSITY_ORDER)
        & output["level2_confidence"].ge(threshold)
    ]
    if len(spectral_indices):
        if spectral_node.requires_model:
            level3_labels, level3_confidence = _predict_records(
                test.loc[spectral_indices].reset_index(drop=True),
                spectral_node,
                checkpoints[spectral_node.key],
                model_name,
                args,
                device,
            )
            output.loc[spectral_indices, "predicted_stellar_level3"] = level3_labels
            output.loc[spectral_indices, "level3_confidence"] = level3_confidence
            low_level3 = spectral_indices[np.asarray(level3_confidence) < threshold]
            output.loc[low_level3, "stop_reason"] = "low_confidence_level3"
        elif spectral_node.is_deterministic:
            output.loc[spectral_indices, "predicted_stellar_level3"] = (
                spectral_node.classes[0]
            )
            output.loc[spectral_indices, "level3_confidence"] = 1.0
        else:
            output.loc[spectral_indices, "stop_reason"] = (
                "no_supported_spectral_classifier"
            )

    supported_level2 = set(registry.star_luminosity.classes)
    supported_level3 = set(registry.star_spectral.classes)

    def target_is_supported(row):
        if row["object_class"] != "STAR":
            return True
        level2 = str(row["stellar_level2_label"])
        if level2 not in supported_level2:
            return False
        if level2 in SPECIAL_STELLAR_CLASSES:
            return True
        if level2 not in ROMAN_LUMINOSITY_ORDER:
            return False
        return str(row["stellar_level3_label"]) in supported_level3

    output["target_supported"] = output.apply(target_is_supported, axis=1)

    output["true_path"] = output.apply(_true_path, axis=1)
    output["predicted_path"] = output.apply(_predicted_path, axis=1)
    output["path_depth"] = output["predicted_path"].str.count(" > ") + 1
    output["exact_path"] = output["true_path"].eq(output["predicted_path"])
    output["hierarchical_distance"] = output.apply(
        lambda row: _path_distance(row["true_path"], row["predicted_path"]),
        axis=1,
    )
    output.to_csv(
        os.path.join(results.predictions_dir, "hierarchical_test_predictions.csv"),
        index=False,
    )

    root_accuracy = accuracy_score(output["object_class"], output["predicted_object_class"])
    predicted_stars = output["predicted_object_class"].eq("STAR")
    true_stars = output["object_class"].eq("STAR")
    supported_targets = output["target_supported"]
    routed_star_denominator = int(predicted_stars.sum())
    true_star_denominator = int(true_stars.sum())
    metrics = {
        "accuracy": float(output["exact_path"].mean()),
        "root_accuracy": float(root_accuracy),
        "exact_path_accuracy": float(output["exact_path"].mean()),
        "exact_path_accuracy_supported_targets": float(
            output.loc[supported_targets, "exact_path"].mean()
        ),
        "mean_hierarchical_distance": float(output["hierarchical_distance"].mean()),
        "full_depth_coverage_all_objects": float(output["path_depth"].ge(3).mean()),
        "full_depth_coverage_predicted_stars": (
            float(output.loc[predicted_stars, "path_depth"].ge(3).mean())
            if routed_star_denominator
            else 0.0
        ),
        "full_depth_coverage_true_stars": (
            float(output.loc[true_stars, "path_depth"].ge(3).mean())
            if true_star_denominator
            else 0.0
        ),
        "stopped_low_confidence": int(output["stop_reason"].str.startswith("low_confidence").sum()),
        "unsupported_target_samples": int((~supported_targets).sum()),
        "level2_labeled_test_stars": int(
            output.loc[true_stars, "stellar_level2_label"].ne("UNRESOLVED").sum()
        ),
        "test_samples": int(len(output)),
    }
    results.save_json("metrics/end_to_end_metrics.json", metrics)
    results.save_json(
        "metrics/stop_reasons.json",
        {str(key): int(value) for key, value in output["stop_reason"].value_counts().items()},
    )
    return {"metrics": metrics, "predictions": output}


def _predict_records(records, node, checkpoint, model_name, args, device):
    outputs = _predict_outputs(
        records,
        node,
        checkpoint,
        model_name,
        args,
        device,
    )
    probabilities = outputs["probabilities"]
    predictions = np.argmax(probabilities, axis=1)
    labels = np.asarray([node.classes[index] for index in predictions], dtype=str)
    confidence = probabilities[np.arange(len(probabilities)), predictions]
    return labels, confidence


def _predict_outputs(records, node, checkpoint, model_name, args, device):
    wrapper = _load_node_checkpoint(
        checkpoint, model_name, node, device, image_size=args.img_size
    )
    wrapper.get_model().eval()
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    loader = build_inference_loader(records, args.img_size, args.batch_size)
    with torch.no_grad():
        for images in loader:
            outputs = wrapper.forward_with_features(images.to(device))
            probabilities.append(outputs["probabilities"].cpu().numpy())
            logits.append(outputs["logits"].cpu().numpy())
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "probabilities": np.concatenate(probabilities, axis=0),
        "logits": np.concatenate(logits, axis=0),
    }


def _conditional_probabilities_from_logits(
    logits: np.ndarray,
    parent_labels: np.ndarray,
    class_names: list[str],
    class_to_parent: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    conditional = np.zeros_like(logits)
    has_supported = np.zeros(len(logits), dtype=bool)
    finite = np.ones(len(logits), dtype=bool)
    positions_by_parent: dict[str, list[int]] = {}
    for position, label in enumerate(class_names):
        positions_by_parent.setdefault(class_to_parent[label], []).append(position)

    for row_index, parent in enumerate(parent_labels):
        positions = positions_by_parent.get(str(parent), [])
        if not positions:
            continue
        has_supported[row_index] = True
        allowed_logits = logits[row_index, positions]
        if not bool(np.all(np.isfinite(allowed_logits))):
            finite[row_index] = False
            continue
        shifted = allowed_logits - np.max(allowed_logits)
        exponentials = np.exp(shifted)
        conditional[row_index, positions] = exponentials / exponentials.sum()
    return conditional, has_supported, finite


def _true_path(row) -> str:
    root = str(row["object_class"])
    if root != "STAR":
        return root
    parent = str(row["stellar_level2_label"])
    leaf = str(row["stellar_level3_label"])
    parts = ["STAR"]
    if parent and parent != "UNRESOLVED":
        parts.append(parent)
    if leaf and leaf != parent and parent not in {"WD", "CARBON", "CV"}:
        parts.append(leaf)
    return " > ".join(parts)


def _predicted_path(row) -> str:
    root = str(row["predicted_object_class"])
    if root != "STAR":
        return root
    parent = str(row["predicted_stellar_level2"])
    leaf = str(row["predicted_stellar_level3"])
    parts = ["STAR"]
    if parent:
        parts.append(parent)
    if leaf and leaf != parent:
        parts.append(leaf)
    return " > ".join(parts)


def _path_distance(true_path: str, predicted_path: str) -> int:
    truth = true_path.split(" > ") if true_path else []
    prediction = predicted_path.split(" > ") if predicted_path else []
    common = 0
    for expected, actual in zip(truth, prediction):
        if expected != actual:
            break
        common += 1
    return len(truth) + len(prediction) - 2 * common


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional_limit(value):
    return None if value is None or value < 0 else value


def _balanced_class_weights(records: pd.DataFrame, class_names: list[str]) -> list[float]:
    counts = records["target_label"].value_counts()
    total = float(len(records))
    class_count = float(len(class_names))
    return [
        total / (class_count * float(counts[label]))
        if counts.get(label, 0) > 0
        else 0.0
        for label in class_names
    ]
