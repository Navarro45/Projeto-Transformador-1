import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def data_output(knn, X_test, y_test):
    # criar pasta output
    os.makedirs("Modelo_KNN/output", exist_ok=True)

    # previsões (caso ainda não tenha feito)
    y_pred = knn.predict(X_test)

    # -------------------------
    # MATRIZ DE CONFUSÃO
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    # salvar como CSV
    np.savetxt("Modelo_KNN/output/confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # salvar como imagem
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.savefig("Modelo_KNN/output/confusion_matrix.png")
    plt.close()

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    report = classification_report(y_test, y_pred)

    # salvar como TXT
    with open("Modelo_KNN/output/classification_report.txt", "w") as f:
        f.write(report)

    # -------------------------
    # METRICS SEPARADAS (opcional)
    # -------------------------
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    with open("Modelo_KNN/output/metrics_summary.txt", "w") as f:
        for label, metrics in report_dict.items():
            f.write(f"{label}:\n")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
            else:
                f.write(f"  {metrics}\n")
            f.write("\n")