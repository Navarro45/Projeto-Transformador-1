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
    precision_score,
    recall_score,
)

from models.model_factory import ModelFactory
from shared.metrics import plot_confusion_matrix, plot_training_history
from shared.stellar_hierarchy import (
    SPECIAL_STELLAR_CLASSES,
    SPECIAL_TYPE_LABEL,
    parse_level3_class_weights,
)
from training.trainer import Trainer
from utils.result_manager import ResultsManager

from .image_data import (
    build_inference_loader,
    build_node_data,
    load_hierarchy_manifest,
)
from .registry import HierarchyNode, HierarchyRegistry


END_TO_END_METRICS_VERSION = 2
END_TO_END_PREDICTION_COLUMNS = (
    "precisao_hierarquica",
    "revocacao_hierarquica",
    "f1_hierarquico",
    "niveis_corretos",
    "profundidade_real",
    "profundidade_predita",
)


def run_hierarchical_image_pipeline(
    args,
    model_name: str,
    run_index: int,
    execution_dir: str,
    import_spec: dict | None = None,
) -> dict:
    """Treina e avalia os tres niveis da hierarquia visual."""
    manifest = load_hierarchy_manifest(args.hierarchy_manifest)
    registry = HierarchyRegistry.from_manifest(
        manifest,
        min_train_per_class=args.hierarchy_min_train_per_class,
        excluded_level2_classes=getattr(
            args,
            "hierarchy_exclude_level2_classes",
            (),
        ),
        excluded_spectral_classes=getattr(
            args,
            "hierarchy_exclude_spectral_classes",
            (),
        ),
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
    level3_class_weight_overrides = parse_level3_class_weights(
        getattr(args, "hierarchy_level3_class_weights", "")
    )
    unavailable_weight_classes = sorted(
        set(level3_class_weight_overrides).difference(registry.star_spectral.classes)
    )
    if unavailable_weight_classes:
        raise ValueError(
            "Pesos solicitados para classes ausentes do nivel 3: "
            + ", ".join(unavailable_weight_classes)
        )
    args.hierarchy_level3_class_weights = level3_class_weight_overrides
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + run_index - 1
    start_level = int(getattr(args, "hierarchy_start_level", 1))
    if start_level > 1 and import_spec is None:
        raise ValueError(
            f"Nivel inicial {start_level} exige um plano de importacao validado"
        )
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
        "import_enabled": import_spec is not None,
        "start_level": start_level,
        "source_execution": (
            import_spec.get("source_execution") if import_spec else None
        ),
        "execution_dir": str(execution_dir),
        "model": model_name,
        "run_index": run_index,
        "level3_class_weight_multipliers": level3_class_weight_overrides,
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
        if _completed_family_nodes_are_complete(summary, registry):
            resume_plan["nodes"] = [
                {"node": node.key, "action": "skip_complete"}
                for node in registry.trainable_nodes()
            ]
            if not _end_to_end_metrics_are_current(family_results, summary):
                checkpoints = {
                    record["node"]: record["checkpoint"]
                    for record in summary["nodes"]
                }
                route_results = _evaluate_end_to_end(
                    args=args,
                    manifest=manifest,
                    registry=registry,
                    checkpoints=checkpoints,
                    model_name=model_name,
                    device=device,
                    results=family_results,
                )
                summary["metrics"] = route_results["metrics"]
                summary["metrics_version"] = END_TO_END_METRICS_VERSION
                summary["level3_class_weight_multipliers"] = (
                    level3_class_weight_overrides
                )
                family_results.save_json("summary.json", summary)
                resume_plan["family_action"] = "reevaluate_end_to_end_metrics"
                family_results.save_json("metadata/resume_plan.json", resume_plan)
                print(
                    f"RETOMADA {model_name}: metricas ponta a ponta regeneradas "
                    "sem treinamento"
                )
                return summary

            resume_plan["family_action"] = "skip_complete_family"
            family_results.save_json("metadata/resume_plan.json", resume_plan)
            print(f"RETOMADA {model_name}: familia completa, treinamento ignorado")
            return summary
        print(
            f"RETOMADA {model_name}: artefatos de avaliacao ausentes; "
            "checkpoints serao apenas reavaliados"
        )

    node_records: list[dict] = []
    checkpoints: dict[str, str] = {}
    imported_by_node = {
        item["node"]: item
        for item in (import_spec or {}).get("nodes", [])
    }
    for node_offset, node in enumerate(registry.trainable_nodes()):
        print("\n" + "=" * 70)
        print(f"HIERARQUIA {model_name} | {node.artifact_name}")
        print("=" * 70)
        if node.level < start_level:
            record = _imported_node_record(
                node=node,
                descriptor=imported_by_node.get(node.key),
                model_name=model_name,
                family_name=family_name,
                run_index=run_index,
                execution_dir=execution_dir,
                device=device,
                image_size=args.img_size,
            )
        else:
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
        "start_level": start_level,
        "source_execution": (
            import_spec.get("source_execution") if import_spec else None
        ),
        "results_dir": family_results.base_dir,
        "metrics_version": END_TO_END_METRICS_VERSION,
        "metrics": route_results["metrics"],
        "level3_class_weight_multipliers": level3_class_weight_overrides,
        "nodes": node_records,
        "hierarchy": registry.to_dict(),
    }
    family_results.save_json("summary.json", summary)
    resume_plan["family_action"] = "completed"
    family_results.save_json("metadata/resume_plan.json", resume_plan)
    return summary


def _imported_node_record(
    node: HierarchyNode,
    descriptor: dict | None,
    model_name: str,
    family_name: str,
    run_index: int,
    execution_dir: str,
    device: torch.device,
    image_size: int,
) -> dict:
    if descriptor is None:
        raise ValueError(f"Plano de importacao sem o nivel anterior {node.key}")
    if descriptor.get("classes") != list(node.classes):
        raise ValueError(f"Classes importadas incompativeis no nivel {node.key}")

    expected_results = ResultsManager(
        family_name,
        execution_dir=execution_dir,
        run_index=run_index,
        variant=os.path.join("hierarchy", node.artifact_name),
    )
    destination_dir = os.path.abspath(descriptor["destination_results_dir"])
    if os.path.normcase(destination_dir) != os.path.normcase(
        os.path.abspath(expected_results.base_dir)
    ):
        raise ValueError(
            f"Destino importado inesperado para {model_name}/{node.key}: "
            f"{destination_dir}"
        )
    if not _node_is_complete(expected_results, node):
        raise FileNotFoundError(
            f"Nivel importado incompleto em {expected_results.base_dir}"
        )
    checkpoint = expected_results.get_model_path()
    wrapper = _load_node_checkpoint(
        checkpoint,
        model_name=model_name,
        node=node,
        device=device,
        image_size=image_size,
    )
    del wrapper
    metrics = _load_json(expected_results.get_metrics_path())
    print(f"IMPORTACAO {model_name}/{node.key}: nivel anterior copiado e validado")
    return {
        "node": node.key,
        "artifact_name": node.artifact_name,
        "classes": list(node.classes),
        "metrics": metrics,
        "results_dir": expected_results.base_dir,
        "checkpoint": checkpoint,
        "resume_action": "copied_prerequisite",
        "source_results_dir": descriptor["source_results_dir"],
    }


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
    base_class_weights = _balanced_class_weights(data.train_records, data.class_names)
    class_weight_multipliers = _class_weight_multipliers_for_node(
        args,
        node,
        data.class_names,
    )
    class_weights = [
        float(base * multiplier)
        for base, multiplier in zip(base_class_weights, class_weight_multipliers)
    ]
    print(
        f"PESOS {model_name}/{node.key}: "
        + ", ".join(
            f"{label}=base:{base:.6g} x mult:{multiplier:.6g} "
            f"-> final:{effective:.6g}"
            for label, base, multiplier, effective in zip(
                data.class_names,
                base_class_weights,
                class_weight_multipliers,
                class_weights,
            )
        )
    )
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
        base_class_weights=base_class_weights,
        class_weight_multipliers=class_weight_multipliers,
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


def _node_metadata(
    args,
    node,
    model_name,
    data,
    base_class_weights,
    class_weight_multipliers,
    class_weights,
    device,
    seed,
):
    return {
        "model": model_name,
        "model_family": model_name,
        "hierarchy_node": node.to_dict(),
        "class_names": data.class_names,
        "class_weights_base": {
            label: float(weight)
            for label, weight in zip(data.class_names, base_class_weights)
        },
        "class_weight_multipliers": {
            label: float(multiplier)
            for label, multiplier in zip(data.class_names, class_weight_multipliers)
        },
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
        "class_weights",
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
    class_names = list(expected.get("class_names", []))
    actual_base = actual.get("class_weights_base", actual.get("class_weights", {}))
    expected_base = expected.get(
        "class_weights_base",
        expected.get("class_weights", {}),
    )
    if actual_base != expected_base:
        mismatches.append("class_weights_base")
    actual_multipliers = actual.get(
        "class_weight_multipliers",
        {label: 1.0 for label in class_names},
    )
    expected_multipliers = expected.get(
        "class_weight_multipliers",
        {label: 1.0 for label in class_names},
    )
    if actual_multipliers != expected_multipliers:
        mismatches.append("class_weight_multipliers")
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
    if node.key == "star_spectral":
        required.extend(
            [
                os.path.join(results.metrics_dir, "validation_metrics.json"),
                os.path.join(
                    results.metrics_dir,
                    "validation_classification_report.json",
                ),
                os.path.join(
                    results.predictions_dir,
                    "validation_predictions.csv",
                ),
            ]
        )
    if not all(os.path.isfile(path) for path in required):
        return False
    return True


def _completed_family_nodes_are_complete(summary: dict, registry) -> bool:
    records = {record["node"]: record for record in summary.get("nodes", [])}
    for node in registry.trainable_nodes():
        record = records.get(node.key, {})
        results_dir = record.get("results_dir", "")
        if not results_dir or not os.path.isdir(results_dir):
            return False
        if not _node_is_complete(ResultsManager.from_existing(results_dir), node):
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
    expected_weight_overrides = parse_level3_class_weights(
        getattr(args, "hierarchy_level3_class_weights", "")
    )
    if summary.get("level3_class_weight_multipliers", {}) != expected_weight_overrides:
        mismatches.append("level3_class_weight_multipliers")
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
        metadata = _load_json(os.path.join(results.metadata_dir, "info.json"))
        class_names = list(node.classes)
        base_weight_map = metadata.get(
            "class_weights_base",
            metadata.get("class_weights", {}),
        )
        multipliers = _class_weight_multipliers_for_node(
            args,
            node,
            class_names,
        )
        multiplier_map = {
            label: float(multiplier)
            for label, multiplier in zip(class_names, multipliers)
        }
        effective_weight_map = {
            label: float(base_weight_map[label] * multiplier_map[label])
            for label in class_names
        }
        expected_metadata = {
            "model": model_name,
            "model_family": model_name,
            "class_names": class_names,
            "class_weights_base": base_weight_map,
            "class_weight_multipliers": multiplier_map,
            "class_weights": effective_weight_map,
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


def _end_to_end_metrics_are_current(results, summary: dict) -> bool:
    if summary.get("metrics_version") != END_TO_END_METRICS_VERSION:
        return False

    metrics_path = os.path.join(results.metrics_dir, "end_to_end_metrics.json")
    readable_path = os.path.join(results.metrics_dir, "metricas_ponta_a_ponta.json")
    report_path = os.path.join(
        results.metrics_dir,
        "relatorio_classes_ponta_a_ponta.json",
    )
    predictions_path = os.path.join(
        results.predictions_dir,
        "hierarchical_test_predictions.csv",
    )
    required = [metrics_path, readable_path, report_path, predictions_path]
    if not all(os.path.isfile(path) for path in required):
        return False

    try:
        metrics = _load_json(metrics_path)
        prediction_columns = set(pd.read_csv(predictions_path, nrows=0).columns)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    return (
        metrics.get("metrics_version") == END_TO_END_METRICS_VERSION
        and set(END_TO_END_PREDICTION_COLUMNS).issubset(prediction_columns)
    )


def _evaluate_node(wrapper, data, node, results, device) -> dict:
    metrics = _evaluate_node_split(
        wrapper=wrapper,
        loader=data.test_loader,
        records=data.test_records,
        class_names=data.class_names,
        results=results,
        device=device,
        split="test",
    )
    if node.key == "star_spectral":
        _evaluate_node_split(
            wrapper=wrapper,
            loader=data.validation_loader,
            records=data.validation_records,
            class_names=data.class_names,
            results=results,
            device=device,
            split="validation",
        )
    return metrics


def _evaluate_node_split(
    wrapper,
    loader,
    records: pd.DataFrame,
    class_names: list[str],
    results,
    device,
    split: str,
) -> dict:
    model = wrapper.get_model()
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = wrapper.forward_with_features(images.to(device))
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(outputs["predictions"].cpu().numpy().tolist())
            confidences.extend(outputs["confidences"].cpu().numpy().tolist())

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples": len(y_true),
    }
    metrics_name = "metrics.json" if split == "test" else f"{split}_metrics.json"
    report_name = (
        "classification_report.json"
        if split == "test"
        else f"{split}_classification_report.json"
    )
    plot_name = (
        "confusion_matrix.png"
        if split == "test"
        else f"{split}_confusion_matrix.png"
    )
    predictions_name = (
        "test_predictions.csv"
        if split == "test"
        else f"{split}_predictions.csv"
    )
    results.save_json(f"metrics/{metrics_name}", metrics)
    results.save_json(
        f"metrics/{report_name}",
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        os.path.join(results.plots_dir, plot_name),
    )
    predictions = records[["object_id", "image_path", "target_label"]].copy()
    predictions["predicted_label"] = [class_names[index] for index in y_pred]
    predictions["confidence"] = confidences

    predictions.to_csv(
        os.path.join(results.predictions_dir, predictions_name),
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
    eligible_level2 = spectral_node.eligible_level2_labels
    spectral_indices = output.index[
        output["predicted_stellar_level2"].isin(eligible_level2)
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
    output["evaluated_stellar_level2"] = output["stellar_level2_label"].map(
        registry.star_luminosity.map_target_label
    )
    output["evaluated_stellar_level3"] = output["stellar_level3_label"].map(
        registry.star_spectral.map_target_label
    )

    def target_is_supported(row):
        if row["object_class"] != "STAR":
            return True
        level2 = str(row["evaluated_stellar_level2"])
        if level2 not in supported_level2:
            return False
        if level2 == SPECIAL_TYPE_LABEL:
            return True
        if level2 not in eligible_level2:
            return False
        return str(row["evaluated_stellar_level3"]) in supported_level3

    output["target_supported"] = output.apply(target_is_supported, axis=1)

    output["true_path"] = output.apply(_true_path, axis=1)
    output["predicted_path"] = output.apply(_predicted_path, axis=1)
    output["path_depth"] = output["predicted_path"].str.count(" > ") + 1
    output["exact_path"] = output["true_path"].eq(output["predicted_path"])
    output["hierarchical_distance"] = output.apply(
        lambda row: _path_distance(row["true_path"], row["predicted_path"]),
        axis=1,
    )
    hierarchical_rows = pd.DataFrame(
        [
            _hierarchical_path_scores(true_path, predicted_path)
            for true_path, predicted_path in zip(
                output["true_path"],
                output["predicted_path"],
            )
        ],
        index=output.index,
    )
    for column in END_TO_END_PREDICTION_COLUMNS:
        output[column] = hierarchical_rows[column]

    root_accuracy = accuracy_score(output["object_class"], output["predicted_object_class"])
    predicted_stars = output["predicted_object_class"].eq("STAR")
    true_stars = output["object_class"].eq("STAR")
    supported_targets = output["target_supported"]
    supported_output = output.loc[supported_targets]
    routed_star_denominator = int(predicted_stars.sum())
    true_star_denominator = int(true_stars.sum())
    full_path = _full_path_metrics(output["true_path"], output["predicted_path"])
    supported_full_path = _full_path_metrics(
        supported_output["true_path"],
        supported_output["predicted_path"],
    )
    hierarchy = _aggregate_hierarchical_metrics(output)
    supported_hierarchy = _aggregate_hierarchical_metrics(supported_output)
    metrics = {
        "metrics_version": END_TO_END_METRICS_VERSION,
        "accuracy": full_path["acuracia"],
        "precision": full_path["precisao_ponderada"],
        "recall": full_path["revocacao_ponderada"],
        "f1": full_path["f1_ponderado"],
        "acuracia_caminho_completo": full_path["acuracia"],
        "precisao_caminho_completo_macro": full_path["precisao_macro"],
        "revocacao_caminho_completo_macro": full_path["revocacao_macro"],
        "f1_caminho_completo_macro": full_path["f1_macro"],
        "precisao_caminho_completo_ponderada": full_path["precisao_ponderada"],
        "revocacao_caminho_completo_ponderada": full_path["revocacao_ponderada"],
        "f1_caminho_completo_ponderado": full_path["f1_ponderado"],
        "acuracia_caminho_completo_alvos_suportados": supported_full_path[
            "acuracia"
        ],
        "precisao_caminho_completo_macro_alvos_suportados": supported_full_path[
            "precisao_macro"
        ],
        "revocacao_caminho_completo_macro_alvos_suportados": supported_full_path[
            "revocacao_macro"
        ],
        "f1_caminho_completo_macro_alvos_suportados": supported_full_path[
            "f1_macro"
        ],
        "precisao_caminho_completo_ponderada_alvos_suportados": supported_full_path[
            "precisao_ponderada"
        ],
        "revocacao_caminho_completo_ponderada_alvos_suportados": supported_full_path[
            "revocacao_ponderada"
        ],
        "f1_caminho_completo_ponderado_alvos_suportados": supported_full_path[
            "f1_ponderado"
        ],
        "precisao_hierarquica": hierarchy["precisao"],
        "revocacao_hierarquica": hierarchy["revocacao"],
        "f1_hierarquico": hierarchy["f1"],
        "precisao_hierarquica_media_por_objeto": hierarchy[
            "precisao_media_por_objeto"
        ],
        "revocacao_hierarquica_media_por_objeto": hierarchy[
            "revocacao_media_por_objeto"
        ],
        "f1_hierarquico_medio_por_objeto": hierarchy["f1_medio_por_objeto"],
        "precisao_hierarquica_alvos_suportados": supported_hierarchy["precisao"],
        "revocacao_hierarquica_alvos_suportados": supported_hierarchy["revocacao"],
        "f1_hierarquico_alvos_suportados": supported_hierarchy["f1"],
        "precisao_hierarquica_media_por_objeto_alvos_suportados": (
            supported_hierarchy["precisao_media_por_objeto"]
        ),
        "revocacao_hierarquica_media_por_objeto_alvos_suportados": (
            supported_hierarchy["revocacao_media_por_objeto"]
        ),
        "f1_hierarquico_medio_por_objeto_alvos_suportados": supported_hierarchy[
            "f1_medio_por_objeto"
        ],
        "root_accuracy": float(root_accuracy),
        "exact_path_accuracy": full_path["acuracia"],
        "exact_path_accuracy_supported_targets": supported_full_path["acuracia"],
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
    readable_metrics = {
        "versao_metricas": END_TO_END_METRICS_VERSION,
        "caminho_completo": {"todos_os_objetos": full_path},
        "hierarquia": {"todos_os_objetos": hierarchy},
        "alvos_suportados": {
            "caminho_completo": supported_full_path,
            "hierarquia": supported_hierarchy,
        },
        "contagens": {
            "total_objetos_teste": int(len(output)),
            "alvos_suportados": int(supported_targets.sum()),
            "alvos_nao_suportados": int((~supported_targets).sum()),
        },
    }
    class_reports = {
        "versao_metricas": END_TO_END_METRICS_VERSION,
        "todos_os_objetos": _full_path_classification_report(
            output["true_path"],
            output["predicted_path"],
        ),
        "alvos_suportados": _full_path_classification_report(
            supported_output["true_path"],
            supported_output["predicted_path"],
        ),
    }
    output.to_csv(
        os.path.join(results.predictions_dir, "hierarchical_test_predictions.csv"),
        index=False,
    )
    results.save_json("metrics/end_to_end_metrics.json", metrics)
    results.save_json("metrics/metricas_ponta_a_ponta.json", readable_metrics)
    results.save_json(
        "metrics/relatorio_classes_ponta_a_ponta.json",
        class_reports,
    )
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
    parent = str(row.get("evaluated_stellar_level2", row["stellar_level2_label"]))
    leaf = str(row.get("evaluated_stellar_level3", row["stellar_level3_label"]))
    parts = ["STAR"]
    if parent and parent != "UNRESOLVED":
        parts.append(parent)
    special_parents = {*SPECIAL_STELLAR_CLASSES, SPECIAL_TYPE_LABEL}
    if leaf and leaf != parent and parent not in special_parents:
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


def _path_prefixes(path: str) -> tuple[str, ...]:
    parts = [part for part in str(path).split(" > ") if part]
    return tuple(" > ".join(parts[:depth]) for depth in range(1, len(parts) + 1))


def _hierarchical_path_scores(true_path: str, predicted_path: str) -> dict:
    truth = set(_path_prefixes(true_path))
    prediction = set(_path_prefixes(predicted_path))
    shared = len(truth & prediction)
    precision = shared / len(prediction) if prediction else 0.0
    recall = shared / len(truth) if truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precisao_hierarquica": float(precision),
        "revocacao_hierarquica": float(recall),
        "f1_hierarquico": float(f1),
        "niveis_corretos": int(shared),
        "profundidade_real": int(len(truth)),
        "profundidade_predita": int(len(prediction)),
    }


def _full_path_metrics(true_paths, predicted_paths) -> dict:
    true_paths = list(true_paths)
    predicted_paths = list(predicted_paths)
    if not true_paths:
        return {
            "amostras": 0,
            "acuracia": None,
            "precisao_macro": None,
            "revocacao_macro": None,
            "f1_macro": None,
            "precisao_ponderada": None,
            "revocacao_ponderada": None,
            "f1_ponderado": None,
        }
    return {
        "amostras": len(true_paths),
        "acuracia": float(accuracy_score(true_paths, predicted_paths)),
        "precisao_macro": float(
            precision_score(true_paths, predicted_paths, average="macro", zero_division=0)
        ),
        "revocacao_macro": float(
            recall_score(true_paths, predicted_paths, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(true_paths, predicted_paths, average="macro", zero_division=0)
        ),
        "precisao_ponderada": float(
            precision_score(
                true_paths,
                predicted_paths,
                average="weighted",
                zero_division=0,
            )
        ),
        "revocacao_ponderada": float(
            recall_score(
                true_paths,
                predicted_paths,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_ponderado": float(
            f1_score(true_paths, predicted_paths, average="weighted", zero_division=0)
        ),
    }


def _aggregate_hierarchical_metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "amostras": 0,
            "precisao": None,
            "revocacao": None,
            "f1": None,
            "precisao_media_por_objeto": None,
            "revocacao_media_por_objeto": None,
            "f1_medio_por_objeto": None,
        }

    shared = int(rows["niveis_corretos"].sum())
    true_depth = int(rows["profundidade_real"].sum())
    predicted_depth = int(rows["profundidade_predita"].sum())
    precision = shared / predicted_depth if predicted_depth else 0.0
    recall = shared / true_depth if true_depth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "amostras": int(len(rows)),
        "precisao": float(precision),
        "revocacao": float(recall),
        "f1": float(f1),
        "precisao_media_por_objeto": float(rows["precisao_hierarquica"].mean()),
        "revocacao_media_por_objeto": float(rows["revocacao_hierarquica"].mean()),
        "f1_medio_por_objeto": float(rows["f1_hierarquico"].mean()),
    }


def _full_path_classification_report(true_paths, predicted_paths) -> dict:
    true_paths = list(true_paths)
    predicted_paths = list(predicted_paths)
    if not true_paths:
        return {"amostras": 0, "classes": [], "relatorio": None}
    labels = sorted(set(true_paths) | set(predicted_paths))
    raw_report = classification_report(
        true_paths,
        predicted_paths,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    translated_report = {
        label: {
            "precisao": float(raw_report[label]["precision"]),
            "revocacao": float(raw_report[label]["recall"]),
            "f1": float(raw_report[label]["f1-score"]),
            "suporte": int(raw_report[label]["support"]),
        }
        for label in labels
    }
    translated_report["acuracia"] = float(raw_report["accuracy"])
    for source, destination in (
        ("macro avg", "media_macro"),
        ("weighted avg", "media_ponderada"),
    ):
        translated_report[destination] = {
            "precisao": float(raw_report[source]["precision"]),
            "revocacao": float(raw_report[source]["recall"]),
            "f1": float(raw_report[source]["f1-score"]),
            "suporte": int(raw_report[source]["support"]),
        }
    return {
        "amostras": len(true_paths),
        "classes": labels,
        "relatorio": translated_report,
    }


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional_limit(value):
    return None if value is None or value < 0 else value


def _class_weight_multipliers_for_node(
    args,
    node: HierarchyNode,
    class_names: list[str],
) -> list[float]:
    overrides = (
        parse_level3_class_weights(
            getattr(args, "hierarchy_level3_class_weights", "")
        )
        if node.key == "star_spectral"
        else {}
    )
    unavailable = sorted(set(overrides).difference(class_names))
    if unavailable:
        raise ValueError(
            "Pesos solicitados para classes ausentes do nivel 3: "
            + ", ".join(unavailable)
        )
    return [float(overrides.get(label, 1.0)) for label in class_names]


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
