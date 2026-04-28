import os
from src.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_multiclass
)
import torch
import json
import numpy as np


def test(model, loader, config):
    model.load_state_dict(torch.load(config.MODEL_DIR + "/best_model.pth"))
    model.to(config.DEVICE)
    model.eval()

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_probs)

    # =========================
    # SALVAR MÉTRICAS
    # =========================
    with open(os.path.join(config.METRIC_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # =========================
    # PLOTS
    # =========================
    cm = np.array(metrics["confusion_matrix"])

    plot_confusion_matrix(
        cm,
        config.CLASS_NAMES,
        os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    )

    plot_roc_multiclass(
        y_true,
        y_probs,
        config.NUM_CLASSES,
        os.path.join(config.PLOT_DIR, "roc_curve.png")
    )

    return metrics