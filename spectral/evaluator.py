from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SPECTRAL_TYPES = ["O", "B", "A", "F", "G", "K", "M"]


def _major_type(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    for spectral_type in SPECTRAL_TYPES:
        if text.startswith(spectral_type):
            return spectral_type
    return None


def cluster_purity(labels, truth) -> float:
    df = pd.DataFrame({"cluster": labels, "truth": truth})
    df = df[(df["cluster"] != -1) & df["truth"].notna()]
    if df.empty:
        return float("nan")

    correct = 0
    total = 0
    for _, group in df.groupby("cluster"):
        counts = group["truth"].value_counts()
        correct += int(counts.max())
        total += int(counts.sum())

    return correct / total if total else float("nan")


def evaluate_clusters(metadata: pd.DataFrame, labels: np.ndarray, output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result = {
        "n_samples": int(len(labels)),
        "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "n_outliers": int(np.sum(labels == -1)),
    }

    metadata = metadata.copy()
    metadata["cluster_id"] = labels

    if "subclass" in metadata.columns:
        truth = metadata["subclass"].map(_major_type)
        valid = truth.notna()

        if valid.sum() > 0:
            result["adjusted_rand_index"] = float(
                adjusted_rand_score(truth[valid], labels[valid])
            )
            result["normalized_mutual_info"] = float(
                normalized_mutual_info_score(truth[valid], labels[valid])
            )
            result["cluster_purity"] = float(cluster_purity(labels[valid], truth[valid]))

        rows = []
        for cluster_id, group in metadata.groupby("cluster_id"):
            counts = group["subclass"].map(_major_type).value_counts()
            row = {
                "cluster_id": cluster_id,
                "total": int(len(group)),
                "dominant_subclass": counts.idxmax() if not counts.empty else "",
            }
            for spectral_type in SPECTRAL_TYPES:
                row[spectral_type] = int(counts.get(spectral_type, 0))
            rows.append(row)

        pd.DataFrame(rows).sort_values("cluster_id").to_csv(
            output / "cluster_subclass_distribution.csv",
            index=False,
        )

    with open(output / "cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result
