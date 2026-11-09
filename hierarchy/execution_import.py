from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.model_factory import ModelFactory

from .image_data import build_node_data, load_hierarchy_manifest
from .registry import HierarchyNode, HierarchyRegistry


_EXECUTION_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6})"
    r"(?:_(?P<suffix>\d{2}))?$"
)

_REQUIRED_NODE_FILES = (
    os.path.join("model", "model.pth"),
    os.path.join("metrics", "metrics.json"),
    os.path.join("metadata", "info.json"),
    os.path.join("metrics", "classification_report.json"),
    os.path.join("plots", "confusion_matrix.png"),
    os.path.join("predictions", "test_predictions.csv"),
)


def find_latest_dated_execution(results_root: str | os.PathLike = "Resultados") -> str:
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Diretorio de resultados nao encontrado: {root}")

    candidates: list[tuple[datetime, int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _EXECUTION_PATTERN.fullmatch(path.name)
        if not match:
            continue
        timestamp = datetime.strptime(
            match.group("timestamp"),
            "%Y-%m-%d_%H-%M-%S_%f",
        )
        suffix = int(match.group("suffix") or 1)
        candidates.append((timestamp, suffix, path.resolve()))

    if not candidates:
        raise FileNotFoundError(
            f"Nenhuma execucao datada encontrada em {root.resolve()}"
        )
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


def build_hierarchy_import_plan(
    args,
    model_names: list[str],
    runs: int,
    results_root: str | os.PathLike = "Resultados",
) -> dict:
    start_level = int(getattr(args, "hierarchy_start_level", 1))
    if start_level not in {2, 3}:
        raise ValueError("A importacao hierarquica exige nivel inicial 2 ou 3")

    source_execution = find_latest_dated_execution(results_root)
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
    nodes = registry.trainable_nodes()
    prerequisites = [node for node in nodes if node.level < start_level]
    if not prerequisites:
        raise ValueError(f"O nivel {start_level} nao possui niveis anteriores para importar")

    hierarchy_payload = registry.to_dict()
    expected_data: dict[tuple[int, str], object] = {}
    families: dict[str, dict] = {}
    device = torch.device("cpu")

    for run_index in range(1, int(runs) + 1):
        base_seed = int(args.seed) + run_index - 1
        for node_offset, node in enumerate(nodes):
            if node.level >= start_level:
                continue
            node_seed = base_seed + node_offset
            expected_data[(run_index, node.key)] = build_node_data(
                manifest=manifest,
                node=node,
                img_size=args.img_size,
                batch_size=args.batch_size,
                seed=node_seed,
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

        for model_name in model_names:
            family_key = _family_key(model_name, run_index)
            family_base = _family_hierarchy_dir(
                source_execution,
                model_name,
                run_index,
            )
            hierarchy_path = os.path.join(family_base, "metadata", "hierarchy.json")
            if not os.path.isfile(hierarchy_path):
                raise FileNotFoundError(
                    f"Hierarquia ausente na ultima execucao: {hierarchy_path}"
                )
            actual_hierarchy = _load_json(hierarchy_path)
            _validate_prerequisite_hierarchy(
                actual=actual_hierarchy,
                expected=hierarchy_payload,
                prerequisite_nodes=prerequisites,
                context=f"{model_name}/execucao_{run_index:02d}",
            )

            imported_nodes = []
            for node_offset, node in enumerate(nodes):
                if node.level >= start_level:
                    continue
                source_node_dir = os.path.join(family_base, node.artifact_name)
                _require_complete_node(source_node_dir, model_name, node)
                data = expected_data[(run_index, node.key)]
                expected_metadata = _expected_metadata(
                    args=args,
                    node=node,
                    model_name=model_name,
                    data=data,
                    seed=base_seed + node_offset,
                )
                metadata_path = os.path.join(source_node_dir, "metadata", "info.json")
                _validate_metadata(
                    actual=_load_json(metadata_path),
                    expected=expected_metadata,
                    context=f"{model_name}/{node.key}/execucao_{run_index:02d}",
                )
                checkpoint = os.path.join(source_node_dir, "model", "model.pth")
                _validate_checkpoint(
                    checkpoint=checkpoint,
                    model_name=model_name,
                    node=node,
                    image_size=args.img_size,
                    device=device,
                )
                imported_nodes.append(
                    {
                        "node": node.key,
                        "level": node.level,
                        "artifact_name": node.artifact_name,
                        "classes": list(node.classes),
                        "source_results_dir": os.path.abspath(source_node_dir),
                        "source_checkpoint": os.path.abspath(checkpoint),
                    }
                )

            families[family_key] = {
                "model": model_name,
                "run_index": run_index,
                "source_execution": os.path.abspath(source_execution),
                "source_family_dir": os.path.abspath(family_base),
                "hierarchy": hierarchy_payload,
                "nodes": imported_nodes,
            }

    return {
        "start_level": start_level,
        "source_execution": os.path.abspath(source_execution),
        "families": families,
    }


def materialize_hierarchy_import_plan(plan: dict, destination_execution: str) -> dict:
    source_execution = os.path.abspath(plan["source_execution"])
    destination_execution = os.path.abspath(destination_execution)
    if os.path.normcase(source_execution) == os.path.normcase(destination_execution):
        raise ValueError("A execucao de origem e destino nao podem ser a mesma pasta")

    for family in plan["families"].values():
        destination_family = _family_hierarchy_dir(
            destination_execution,
            family["model"],
            family["run_index"],
        )
        metadata_dir = os.path.join(destination_family, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        _save_json(
            os.path.join(metadata_dir, "hierarchy.json"),
            family["hierarchy"],
        )

        copied_nodes = []
        for node in family["nodes"]:
            source_dir = node["source_results_dir"]
            destination_dir = os.path.join(destination_family, node["artifact_name"])
            if os.path.exists(destination_dir):
                raise FileExistsError(
                    f"Destino do nivel importado ja existe: {destination_dir}"
                )
            shutil.copytree(source_dir, destination_dir, copy_function=shutil.copy2)
            verification = _verify_copied_tree(source_dir, destination_dir)
            destination_checkpoint = os.path.join(
                destination_dir,
                "model",
                "model.pth",
            )
            copied = dict(node)
            copied.update(
                {
                    "destination_results_dir": os.path.abspath(destination_dir),
                    "destination_checkpoint": os.path.abspath(destination_checkpoint),
                    "file_count": verification["file_count"],
                    "total_bytes": verification["total_bytes"],
                    "checkpoint_sha256": verification["checkpoint_sha256"],
                    "resume_action": "copied_prerequisite",
                }
            )
            copied_nodes.append(copied)

        family["destination_family_dir"] = os.path.abspath(destination_family)
        family["nodes"] = copied_nodes
        _save_json(
            os.path.join(metadata_dir, "imported_levels.json"),
            {
                "source_execution": source_execution,
                "destination_execution": destination_execution,
                "start_level": int(plan["start_level"]),
                "model": family["model"],
                "run_index": int(family["run_index"]),
                "nodes": copied_nodes,
            },
        )

    plan["destination_execution"] = destination_execution
    return plan


def family_import_spec(plan: dict | None, model_name: str, run_index: int) -> dict | None:
    if not plan:
        return None
    return plan["families"].get(_family_key(model_name, run_index))


def _family_key(model_name: str, run_index: int) -> str:
    return f"{model_name}:execucao_{int(run_index):02d}"


def _family_hierarchy_dir(execution_dir: str, model_name: str, run_index: int) -> str:
    return os.path.join(
        str(execution_dir),
        f"{model_name}_hierarchical",
        f"execucao_{int(run_index):02d}",
        "hierarchy",
    )


def _expected_metadata(args, node, model_name, data, seed) -> dict:
    augmentation_enabled = bool(
        node.level >= 2
        and not getattr(args, "hierarchy_disable_augmentation", False)
    )
    return {
        "model": model_name,
        "model_family": model_name,
        "hierarchy_node": node.to_dict(),
        "class_names": data.class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "img_size": args.img_size,
        "augmentation": {
            "enabled": augmentation_enabled,
            "balance_target": int(getattr(args, "hierarchy_balance_target", 500)),
            "min_synthetic_fraction": float(
                getattr(args, "hierarchy_min_synthetic_fraction", 0.10)
            ),
            "max_synthetic_per_source": int(
                getattr(args, "hierarchy_max_synthetic_per_source", 10)
            ),
            "summary": data.augmentation_summary,
        },
        "seed": int(seed),
        "split_counts": {
            "train": len(data.train_records),
            "val": len(data.validation_records),
            "test": len(data.test_records),
        },
    }


def _validate_prerequisite_hierarchy(
    actual: dict,
    expected: dict,
    prerequisite_nodes: list[HierarchyNode],
    context: str,
) -> None:
    if actual.get("method") != expected.get("method"):
        raise ValueError(
            f"A ultima execucao possui hierarquia incompativel para {context}: method"
        )
    actual_nodes = {
        str(node.get("key")): node
        for node in actual.get("nodes", [])
        if isinstance(node, dict)
    }
    expected_nodes = {
        str(node.get("key")): node
        for node in expected.get("nodes", [])
        if isinstance(node, dict)
    }
    mismatches = [
        node.key
        for node in prerequisite_nodes
        if actual_nodes.get(node.key) != expected_nodes.get(node.key)
    ]
    if mismatches:
        raise ValueError(
            f"A ultima execucao possui hierarquia incompativel para {context}: "
            + ", ".join(mismatches)
        )


def _validate_metadata(actual: dict, expected: dict, context: str) -> None:
    exact_fields = (
        "model",
        "model_family",
        "hierarchy_node",
        "class_names",
        "epochs",
        "batch_size",
        "img_size",
        "seed",
        "split_counts",
        "augmentation",
    )
    mismatches = [
        field for field in exact_fields if actual.get(field) != expected.get(field)
    ]
    if not np.isclose(
        float(actual.get("learning_rate", np.nan)),
        float(expected["learning_rate"]),
        rtol=0.0,
        atol=1e-12,
    ):
        mismatches.append("learning_rate")
    if mismatches:
        raise ValueError(
            f"Nivel anterior incompativel em {context}: "
            + ", ".join(sorted(set(mismatches)))
        )


def _validate_checkpoint(
    checkpoint: str,
    model_name: str,
    node: HierarchyNode,
    image_size: int,
    device: torch.device,
) -> None:
    wrapper = ModelFactory.create_for_node(
        model_name=model_name,
        node_key=node.key,
        num_classes=len(node.classes),
        device=device,
        image_size=image_size,
    )
    try:
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        wrapper.get_model().load_state_dict(state)
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint anterior incompativel com {model_name}/{node.key}: {checkpoint}"
        ) from error
    finally:
        del wrapper
        gc.collect()


def _require_complete_node(
    node_dir: str,
    model_name: str,
    node: HierarchyNode,
) -> None:
    missing = [
        relative
        for relative in _REQUIRED_NODE_FILES
        if not os.path.isfile(os.path.join(node_dir, relative))
    ]
    if missing:
        raise FileNotFoundError(
            f"Nivel anterior incompleto em {model_name}/{node.key}: "
            + ", ".join(missing)
        )


def _verify_copied_tree(source_dir: str, destination_dir: str) -> dict:
    source_files = _tree_files(source_dir)
    destination_files = _tree_files(destination_dir)
    if source_files != destination_files:
        raise IOError(
            f"A copia do nivel nao preservou todos os arquivos: {source_dir}"
        )

    source_checkpoint = os.path.join(source_dir, "model", "model.pth")
    destination_checkpoint = os.path.join(destination_dir, "model", "model.pth")
    source_hash = _sha256(source_checkpoint)
    destination_hash = _sha256(destination_checkpoint)
    if source_hash != destination_hash:
        raise IOError(
            f"O checkpoint copiado difere da origem: {destination_checkpoint}"
        )
    return {
        "file_count": len(source_files),
        "total_bytes": int(sum(source_files.values())),
        "checkpoint_sha256": source_hash,
    }


def _tree_files(root: str) -> dict[str, int]:
    base = Path(root)
    return {
        str(path.relative_to(base)): int(path.stat().st_size)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _optional_limit(value):
    return None if value is None or value < 0 else value
