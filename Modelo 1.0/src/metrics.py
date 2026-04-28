import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)


# =========================
# MÉTRICAS NUMÉRICAS
# =========================
def compute_metrics(y_true, y_pred, y_probs=None):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if y_probs is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_probs, multi_class="ovr")
        except:
            metrics["auc"] = None

    return metrics


# =========================
# MATRIZ DE CONFUSÃO
# =========================
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Matriz de Confusão")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    plt.xlabel("Predito")
    plt.ylabel("Real")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# HISTÓRICO DE TREINO
# =========================
def plot_training_history(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close()


# =========================
# ROC MULTICLASSE
# =========================
def plot_roc_multiclass(y_true, y_probs, n_classes, save_path):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    plt.figure()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Classe {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC Multiclasse")
    plt.legend()

    plt.savefig(save_path)
    plt.close()