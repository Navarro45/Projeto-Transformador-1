import os
import json
from datetime import datetime


class ResultsManager:

    def __init__(self, model_name):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self.base_dir = os.path.join(
            "Resultados",
            model_name,
            timestamp
        )

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

        self._create_folders()

    def _create_folders(self):

        folders = [
            self.base_dir,
            self.model_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.gradcam_dir,
            self.metadata_dir
        ]

        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def save_metadata(self, data):

        path = os.path.join(
            self.metadata_dir,
            "info.json"
        )

        with open(path, "w") as f:
            json.dump(data, f, indent=4)