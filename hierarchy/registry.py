from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import pandas as pd

from shared.stellar_hierarchy import (
    ROMAN_LUMINOSITY_ORDER,
    SPECIAL_STELLAR_CLASSES,
    UNRESOLVED_LABEL,
    ordered_level2_labels,
    ordered_spectral_labels,
)


ROOT_CLASSES = ("GALAXY", "QSO", "STAR")


@dataclass(frozen=True)
class HierarchyNode:
    key: str
    level: int
    parent: str | None
    target_column: str
    classes: tuple[str, ...]
    status: str
    train_counts: dict[str, int]
    reason: str = ""
    class_to_parent: dict[str, str] = field(default_factory=dict)

    @property
    def requires_model(self) -> bool:
        return self.status == "trainable" and len(self.classes) > 1

    @property
    def is_deterministic(self) -> bool:
        return self.status == "deterministic" and len(self.classes) == 1

    @property
    def artifact_name(self) -> str:
        if self.level == 1:
            return "nivel_1_classes_principais"
        if self.level == 2:
            return "nivel_2_luminosidade_e_especiais"
        return "nivel_3_classes_espectrais_globais"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["classes"] = list(self.classes)
        payload["requires_model"] = self.requires_model
        payload["is_deterministic"] = self.is_deterministic
        return payload


class HierarchyRegistry:
    """Registro da cascata: objeto, luminosidade/especial e espectro global."""

    def __init__(self, nodes: Iterable[HierarchyNode], min_train_per_class: int):
        self.nodes = {node.key: node for node in nodes}
        self.min_train_per_class = int(min_train_per_class)

    @classmethod
    def from_manifest(
        cls,
        manifest: pd.DataFrame,
        min_train_per_class: int = 10,
    ) -> "HierarchyRegistry":
        required = {
            "split",
            "object_class",
            "stellar_level2_label",
            "stellar_level3_label",
        }
        missing = sorted(required.difference(manifest.columns))
        if missing:
            raise ValueError(
                "Manifesto hierarquico sem colunas obrigatorias: "
                + ", ".join(missing)
            )
        if min_train_per_class < 1:
            raise ValueError("min_train_per_class deve ser maior ou igual a 1")

        train = manifest.loc[manifest["split"].eq("train")].copy()
        nodes: list[HierarchyNode] = []

        root_counts = _counts(train["object_class"])
        root_classes = tuple(label for label in ROOT_CLASSES if root_counts.get(label, 0))
        nodes.append(
            HierarchyNode(
                key="root",
                level=1,
                parent=None,
                target_column="object_class",
                classes=root_classes,
                status="trainable" if len(root_classes) > 1 else "unavailable",
                train_counts={label: root_counts[label] for label in root_classes},
                reason="" if len(root_classes) > 1 else "menos de duas classes principais",
            )
        )

        stars = train.loc[train["object_class"].eq("STAR")].copy()
        level2 = stars.loc[
            ~stars["stellar_level2_label"].isin(["", UNRESOLVED_LABEL])
        ]
        level2_counts = _counts(level2["stellar_level2_label"])
        level2_classes = ordered_level2_labels(level2_counts)
        nodes.append(
            HierarchyNode(
                key="star_luminosity",
                level=2,
                parent="STAR",
                target_column="stellar_level2_label",
                classes=level2_classes,
                status="trainable" if len(level2_classes) > 1 else "unavailable",
                train_counts={label: level2_counts[label] for label in level2_classes},
                reason="" if len(level2_classes) > 1 else "menos de duas classes de nivel 2",
            )
        )

        spectral_records = level2.loc[
            level2["stellar_level2_label"].isin(ROMAN_LUMINOSITY_ORDER)
            & level2["stellar_level3_label"].ne("")
        ]
        spectral_counts = _counts(spectral_records["stellar_level3_label"])
        supported_spectral = ordered_spectral_labels(
            label
            for label, count in spectral_counts.items()
            if count >= min_train_per_class
        )
        if len(supported_spectral) > 1:
            spectral_status = "trainable"
            spectral_reason = ""
        elif len(supported_spectral) == 1:
            spectral_status = "deterministic"
            spectral_reason = "somente uma classe espectral atingiu o minimo"
        else:
            spectral_status = "unavailable"
            spectral_reason = "nenhuma classe espectral atingiu o minimo"
        nodes.append(
            HierarchyNode(
                key="star_spectral",
                level=3,
                parent="STAR",
                target_column="stellar_level3_label",
                classes=supported_spectral,
                status=spectral_status,
                train_counts=spectral_counts,
                reason=spectral_reason,
            )
        )

        return cls(nodes, min_train_per_class=min_train_per_class)

    @property
    def root(self) -> HierarchyNode:
        return self.nodes["root"]

    @property
    def star_luminosity(self) -> HierarchyNode:
        return self.nodes["star_luminosity"]

    @property
    def star_parent(self) -> HierarchyNode:
        return self.star_luminosity

    @property
    def star_spectral(self) -> HierarchyNode:
        return self.nodes["star_spectral"]

    def trainable_nodes(self) -> list[HierarchyNode]:
        return [
            node
            for node in sorted(self.nodes.values(), key=lambda item: (item.level, item.key))
            if node.requires_model
        ]

    def to_dict(self) -> dict:
        return {
            "method": "hierarchical_global_spectral",
            "min_train_per_class": self.min_train_per_class,
            "special_level2_classes": list(SPECIAL_STELLAR_CLASSES),
            "nodes": [
                node.to_dict()
                for node in sorted(self.nodes.values(), key=lambda item: (item.level, item.key))
            ],
        }


def _counts(values: pd.Series) -> dict[str, int]:
    clean = values.fillna("").astype(str).str.strip()
    clean = clean.loc[clean.ne("")]
    return {str(label): int(count) for label, count in clean.value_counts().items()}
