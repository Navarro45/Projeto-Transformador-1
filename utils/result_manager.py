import json
import os

from datetime import datetime


class ResultsManager:

    def __init__(self, model_name):

        self.model_name = model_name

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        # =========================
        # DIRETÓRIO BASE
        # =========================

        self.base_dir = os.path.join(
            "Resultados",
            model_name,
            timestamp
        )

        # =========================
        # SUBDIRETÓRIOS
        # =========================

        self.model_dir = os.path.join(
            self.base_dir,
            "model"
        )

        self.metrics_dir = os.path.join(
            self.base_dir,
            "metrics"
        )

        self.predictions_dir = os.path.join(
            self.base_dir,
            "predictions"
        )

        self.gradcam_dir = os.path.join(
            self.base_dir,
            "gradcam"
        )

        self.metadata_dir = os.path.join(
            self.base_dir,
            "metadata"
        )

        # NOVO DIRETÓRIO PARA GRÁFICOS
        self.plots_dir = os.path.join(
            self.base_dir,
            "plots"
        )

        # =========================
        # CRIA PASTAS
        # =========================

        self._create_folders()

    def _create_folders(self):

        folders = [

            self.base_dir,

            self.model_dir,

            self.metrics_dir,

            self.predictions_dir,

            self.gradcam_dir,

            self.metadata_dir,

            self.plots_dir
        ]

        for folder in folders:

            os.makedirs(
                folder,
                exist_ok=True
            )

    # =========================
    # METADATA
    # =========================

    def save_metadata(self, data):

        path = os.path.join(
            self.metadata_dir,
            "info.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

    # =========================
    # CAMINHOS AUXILIARES
    # =========================

    def get_model_path(self):

        return os.path.join(
            self.model_dir,
            "model.pth"
        )

    def get_metrics_path(self):

        return os.path.join(
            self.metrics_dir,
            "metrics.json"
        )

    def get_history_path(self):

        return os.path.join(
            self.plots_dir,
            "training_history.json"
        )

    def get_loss_plot_path(self):

        return os.path.join(
            self.plots_dir,
            "loss_curve.png"
        )

    def get_accuracy_plot_path(self):

        return os.path.join(
            self.plots_dir,
            "accuracy_curve.png"
        )

    def get_confusion_matrix_path(
        self,
        label="main"
    ):

        return os.path.join(
            self.plots_dir,
            f"confusion_matrix_{label}.png"
        )

    # =========================
    # RESUMO
    # =========================

    def summary(self):

        print("\n")
        print("=" * 60)
        print("RESULTS MANAGER")
        print("=" * 60)

        print(f"Modelo: {self.model_name}")
        print(f"Base Dir: {self.base_dir}")

        print("\nDiretórios:")

        print(f"Model: {self.model_dir}")
        print(f"Metrics: {self.metrics_dir}")
        print(f"Predictions: {self.predictions_dir}")
        print(f"GradCAM: {self.gradcam_dir}")
        print(f"Metadata: {self.metadata_dir}")
        print(f"Plots: {self.plots_dir}")

        print("=" * 60)