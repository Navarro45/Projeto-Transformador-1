import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    # Normalização por linha: proporções em cada classe verdadeira
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure()
    plt.imshow(cm, vmin=0.0, vmax=1.0, cmap="Blues")
    plt.title("Matriz de confusão (normalizada por classe verdadeira)")
    plt.colorbar(label="Proporção")

    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            pct = "" if np.isnan(val) else f"{100.0 * val:.1f}%"
            txt_color = "white" if np.isfinite(val) and val > 0.45 else "black"
            plt.text(j, i, pct, ha="center", va="center", color=txt_color)

    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    import json
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)

    return report