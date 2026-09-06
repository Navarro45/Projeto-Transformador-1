from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _normalize_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    sums = probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities,
        sums,
        out=np.zeros_like(probabilities),
        where=sums > 0,
    )


@dataclass
class RandomForestModelConfig:
    n_estimators: int = 300
    criterion: str = "log_loss"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | int | None = "sqrt"
    bootstrap: bool = True
    class_weight: str | None = "balanced_subsample"
    max_samples: float | int | None = None
    random_state: int = 42
    n_jobs: int = -1
    prediction_mode: str = "global_probability"

    def to_dict(self) -> dict:
        return asdict(self)

    def classifier_kwargs(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "class_weight": self.class_weight,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


class FlatRandomForestModel:
    model_name = "flat_random_forest"

    def __init__(self, config: RandomForestModelConfig | None = None):
        self.config = config or RandomForestModelConfig()
        self.model = RandomForestClassifier(**self.config.classifier_kwargs())
        self.leaf_labels: list[str] = []

    def fit(self, X, y_leaf, y_parent=None):
        self.leaf_labels = sorted(str(label) for label in set(y_leaf))
        self.model.fit(X, np.asarray(y_leaf, dtype=str))
        return self

    def predict(self, X):
        return self.predict_leaf(X)

    def predict_leaf(self, X):
        return self.model.predict(X).astype(str)

    def predict_proba(self, X):
        return self.predict_leaf_proba(X)

    def predict_leaf_proba(self, X, labels: list[str] | None = None):
        labels = labels or self.leaf_labels
        raw = self.model.predict_proba(X)
        output = np.zeros((len(X), len(labels)), dtype=np.float64)
        positions = {label: index for index, label in enumerate(labels)}
        for source_index, label in enumerate(self.model.classes_.astype(str)):
            if label in positions:
                output[:, positions[label]] = raw[:, source_index]
        return _normalize_rows(output)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        return joblib.load(path)

    def get_params(self) -> dict:
        return self.config.to_dict()

    def set_params(self, **params):
        values = self.config.to_dict()
        values.update(params)
        self.config = RandomForestModelConfig(**values)
        self.model = RandomForestClassifier(**self.config.classifier_kwargs())
        return self


class HierarchicalRandomForestModel:
    model_name = "hierarchical_random_forest"

    def __init__(self, config: RandomForestModelConfig | None = None):
        self.config = config or RandomForestModelConfig()
        self.parent_model: RandomForestClassifier | None = None
        self.child_models: dict[str, RandomForestClassifier] = {}
        self.deterministic_children: dict[str, str] = {}
        self.parent_labels: list[str] = []
        self.leaf_labels: list[str] = []
        self.leaf_to_parent: dict[str, str] = {}
        self.parent_to_leaves: dict[str, list[str]] = {}

    def _new_classifier(self, random_state_offset: int = 0) -> RandomForestClassifier:
        kwargs = self.config.classifier_kwargs()
        kwargs["random_state"] = self.config.random_state + random_state_offset
        return RandomForestClassifier(**kwargs)

    def fit(self, X, y_leaf, y_parent):
        X = np.asarray(X)
        y_leaf = np.asarray(y_leaf, dtype=str)
        y_parent = np.asarray(y_parent, dtype=str)

        self.parent_labels = sorted(set(y_parent))
        self.leaf_labels = sorted(set(y_leaf))
        self.leaf_to_parent = {}
        self.parent_to_leaves = {}

        for leaf, parent in zip(y_leaf, y_parent):
            previous = self.leaf_to_parent.get(leaf)
            if previous is not None and previous != parent:
                raise ValueError(f"Leaf {leaf} possui parents conflitantes: {previous}, {parent}")
            self.leaf_to_parent[leaf] = parent
            self.parent_to_leaves.setdefault(parent, set()).add(leaf)

        self.parent_to_leaves = {
            parent: sorted(leaves)
            for parent, leaves in self.parent_to_leaves.items()
        }

        if len(self.parent_labels) > 1:
            self.parent_model = self._new_classifier()
            self.parent_model.fit(X, y_parent)
        else:
            self.parent_model = None

        self.child_models = {}
        self.deterministic_children = {}

        for offset, parent in enumerate(self.parent_labels, start=1):
            mask = y_parent == parent
            leaves = sorted(set(y_leaf[mask]))
            if len(leaves) <= 1:
                self.deterministic_children[parent] = leaves[0]
                continue
            model = self._new_classifier(offset)
            model.fit(X[mask], y_leaf[mask])
            self.child_models[parent] = model

        return self

    def predict(self, X):
        return self.predict_leaf(X)

    def predict_parent(self, X):
        parent_probabilities = self.predict_parent_proba(X)
        return np.asarray(
            [self.parent_labels[index] for index in np.argmax(parent_probabilities, axis=1)],
            dtype=str,
        )

    def predict_parent_proba(self, X, labels: list[str] | None = None):
        labels = labels or self.parent_labels
        output = np.zeros((len(X), len(labels)), dtype=np.float64)
        positions = {label: index for index, label in enumerate(labels)}

        if self.parent_model is None:
            if self.parent_labels and self.parent_labels[0] in positions:
                output[:, positions[self.parent_labels[0]]] = 1.0
            return output

        raw = self.parent_model.predict_proba(X)
        for source_index, label in enumerate(self.parent_model.classes_.astype(str)):
            if label in positions:
                output[:, positions[label]] = raw[:, source_index]
        return _normalize_rows(output)

    def predict_leaf_proba(self, X, labels: list[str] | None = None):
        labels = labels or self.leaf_labels
        parent_probabilities = self.predict_parent_proba(X, self.parent_labels)
        parent_positions = {label: index for index, label in enumerate(self.parent_labels)}
        leaf_positions = {label: index for index, label in enumerate(labels)}
        output = np.zeros((len(X), len(labels)), dtype=np.float64)

        for parent in self.parent_labels:
            parent_probability = parent_probabilities[:, parent_positions[parent]]
            if parent in self.deterministic_children:
                leaf = self.deterministic_children[parent]
                if leaf in leaf_positions:
                    output[:, leaf_positions[leaf]] = parent_probability
                continue

            child = self.child_models[parent]
            child_raw = child.predict_proba(X)
            for source_index, leaf in enumerate(child.classes_.astype(str)):
                if leaf in leaf_positions:
                    output[:, leaf_positions[leaf]] = parent_probability * child_raw[:, source_index]

        return _normalize_rows(output)

    def predict_proba(self, X):
        return self.predict_leaf_proba(X)

    def predict_leaf(self, X, mode: str | None = None):
        mode = mode or self.config.prediction_mode
        if mode == "top_down":
            return self._predict_leaf_top_down(X)
        if mode != "global_probability":
            raise ValueError("prediction_mode deve ser 'global_probability' ou 'top_down'")

        probabilities = self.predict_leaf_proba(X, self.leaf_labels)
        return np.asarray(
            [self.leaf_labels[index] for index in np.argmax(probabilities, axis=1)],
            dtype=str,
        )

    def _predict_leaf_top_down(self, X):
        parents = self.predict_parent(X)
        predictions = []
        for row_index, parent in enumerate(parents):
            if parent in self.deterministic_children:
                predictions.append(self.deterministic_children[parent])
                continue
            child = self.child_models[parent]
            prediction = child.predict(np.asarray(X)[row_index : row_index + 1])[0]
            predictions.append(str(prediction))
        return np.asarray(predictions, dtype=str)

    def predict_paths(self, X, mode: str | None = None):
        leaves = self.predict_leaf(X, mode=mode)
        parents = np.asarray([self.leaf_to_parent.get(leaf, "UNKNOWN") for leaf in leaves], dtype=str)
        return [
            {"parent": parent, "leaf": leaf, "path": self.path_for_leaf(leaf)}
            for parent, leaf in zip(parents, leaves)
        ]

    def path_for_leaf(self, leaf: str) -> list[str]:
        parent = self.leaf_to_parent.get(str(leaf), "UNKNOWN")
        if parent == leaf:
            return ["STAR", parent]
        return ["STAR", parent, str(leaf)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        return joblib.load(path)

    def get_params(self) -> dict:
        return self.config.to_dict()

    def set_params(self, **params):
        values = self.config.to_dict()
        values.update(params)
        self.config = RandomForestModelConfig(**values)
        return self
