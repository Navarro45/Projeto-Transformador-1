import json
import os

from datetime import datetime


class ResultsManager:

    def __init__(
        self,
        model_name,
        execution_dir=None,
        run_index=1,
        results_root="Resultados",
        variant=None,
    ):

        self.model_name = model_name
        self.run_index = run_index
        self.variant = str(variant) if variant is not None else None
        self.execution_dir = (
            str(execution_dir)
            if execution_dir is not None
            else self.create_execution_dir(results_root)
        )

        # =========================
        # DIRETORIO BASE
        # =========================

        self.base_dir = os.path.join(
            self.execution_dir,
            model_name,
            self._format_run_dir(run_index)
        )

        if self.variant:

            self.base_dir = os.path.join(
                self.base_dir,
                self.variant
            )

        # =========================
        # SUBDIRETORIOS
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

        self.reports_dir = os.path.join(
            self.base_dir,
            "reports"
        )

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

            self.reports_dir,

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

    def save_json(self, relative_path, data):

        path = os.path.join(
            self.base_dir,
            relative_path
        )

        folder = os.path.dirname(path)

        if folder:

            os.makedirs(
                folder,
                exist_ok=True
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
        print(f"Execucao: {self.execution_dir}")
        print(f"Run: {self.run_index}")
        if self.variant:
            print(f"Variante: {self.variant}")
        print(f"Base Dir: {self.base_dir}")

        print("\nDiretorios:")

        print(f"Model: {self.model_dir}")
        print(f"Metrics: {self.metrics_dir}")
        print(f"Predictions: {self.predictions_dir}")
        print(f"GradCAM: {self.gradcam_dir}")
        print(f"Metadata: {self.metadata_dir}")
        print(f"Reports: {self.reports_dir}")
        print(f"Plots: {self.plots_dir}")

        print("=" * 60)

    @classmethod
    def from_existing(cls, base_dir):

        instance = cls.__new__(cls)

        instance.base_dir = str(base_dir)

        run_dir = os.path.basename(instance.base_dir)
        parent_dir = os.path.dirname(instance.base_dir)
        parent_name = os.path.basename(parent_dir)

        if parent_name.startswith("execucao_"):

            instance.variant = run_dir
            instance.run_index = cls._parse_run_index(parent_name)
            model_dir = os.path.dirname(parent_dir)
            instance.model_name = os.path.basename(model_dir)
            instance.execution_dir = os.path.dirname(model_dir)

        elif run_dir.startswith("execucao_"):

            instance.variant = None
            instance.run_index = cls._parse_run_index(run_dir)
            instance.model_name = os.path.basename(parent_dir)
            instance.execution_dir = os.path.dirname(parent_dir)

        else:

            instance.variant = None
            instance.run_index = None
            instance.model_name = os.path.basename(instance.base_dir)
            instance.execution_dir = os.path.dirname(instance.base_dir)

        instance.model_dir = os.path.join(
            instance.base_dir,
            "model"
        )

        instance.metrics_dir = os.path.join(
            instance.base_dir,
            "metrics"
        )

        instance.predictions_dir = os.path.join(
            instance.base_dir,
            "predictions"
        )

        instance.gradcam_dir = os.path.join(
            instance.base_dir,
            "gradcam"
        )

        instance.metadata_dir = os.path.join(
            instance.base_dir,
            "metadata"
        )

        instance.reports_dir = os.path.join(
            instance.base_dir,
            "reports"
        )

        instance.plots_dir = os.path.join(
            instance.base_dir,
            "plots"
        )

        instance._create_folders()

        return instance

    @staticmethod
    def _format_run_dir(run_index):

        try:

            return f"execucao_{int(run_index):02d}"

        except (TypeError, ValueError):

            return f"execucao_{run_index}"

    @staticmethod
    def _parse_run_index(run_dir):

        try:

            return int(str(run_dir).split("_", 1)[1])

        except (IndexError, TypeError, ValueError):

            return None

    @classmethod
    def create_execution_dir(
        cls,
        results_root="Resultados",
        started_at=None,
    ):

        started_at = started_at or datetime.now()

        timestamp = started_at.strftime(
            "%Y-%m-%d_%H-%M-%S_%f"
        )

        base_dir = os.path.join(
            results_root,
            timestamp
        )

        candidate = base_dir
        suffix = 2

        while os.path.exists(candidate):

            candidate = f"{base_dir}_{suffix:02d}"
            suffix += 1

        os.makedirs(
            candidate,
            exist_ok=True
        )

        return candidate
