import os
import torch
from src.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    save_classification_report
)
from src.heatmap import save_gradcam_samples

CLASSES = ["star", "galaxy", "quasar"]


def evaluate(model, dataloader, device, config):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    # ================= METRICS =================
    metrics = compute_metrics(y_true, y_pred)

    # ================= OUTPUT DIR =================
    os.makedirs(config.PATHS["plots"], exist_ok=True)
    os.makedirs(config.PATHS["metrics"], exist_ok=True)

    # ================= CONFUSION MATRIX =================
    plot_confusion_matrix(
        y_true,
        y_pred,
        CLASSES,
        os.path.join(config.PATHS["plots"], "confusion_matrix_main.png")
    )

    # ================= REPORT =================
    save_classification_report(
        y_true,
        y_pred,
        CLASSES,
        os.path.join(config.PATHS["metrics"], "classification_report_main.json")
    )

    if getattr(config, "SAVE_HEATMAPS", False):
        save_gradcam_samples(model, dataloader, config, CLASSES)

    return metrics
