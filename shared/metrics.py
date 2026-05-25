import json
import os
from typing import List, Mapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

Number = Union[int, float]


def compute_metrics(y_true: Sequence, y_pred: Sequence) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    classes: Sequence[str],
    save_path: str,
) -> None:
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


def save_classification_report(
    y_true: Sequence,
    y_pred: Sequence,
    classes: Sequence[str],
    save_path: str,
) -> Mapping:
    report = classification_report(
        y_true, y_pred, target_names=list(classes), output_dict=True, zero_division=0
    )
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    return report


def save_metrics_bundle(
    y_true: Sequence,
    y_pred: Sequence,
    class_names: Sequence[str],
    metrics_dir: str,
    plots_dir: str,
    label: str = "main",
) -> dict:
    """
    Grava o pacote padrão de avaliação usado pelos modelos:
    ``metrics.json``, ``classification_report_{label}.json``,
    ``confusion_matrix_{label}.png``.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred)

    with open(os.path.join(metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        os.path.join(plots_dir, f"confusion_matrix_{label}.png"),
    )

    save_classification_report(
        y_true,
        y_pred,
        class_names,
        os.path.join(metrics_dir, f"classification_report_{label}.json"),
    )

    return metrics


def plot_training_history(
    history: Mapping[str, List[Number]],
    save_dir: str,
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    # Salva o histórico bruto
    history_path = os.path.join(
        save_dir,
        "training_history.json"
    )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs = range(1, len(train_loss) + 1)

    # =========================
    # LOSS
    # =========================

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Loss por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "loss_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =========================
    # ACCURACY
    # =========================

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_acc,
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_acc,
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.title("Accuracy por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "accuracy_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()