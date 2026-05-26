import os
import json
import torch
from models.model_factory import (ModelFactory)


class ModelLoader:

    @staticmethod
    def load_model(
        results_folder,
        device
    ):

        # =========================
        # METADATA
        # =========================

        metadata_path = os.path.join(
            results_folder,
            "metadata",
            "info.json"
        )

        with open(metadata_path, "r") as f:

            metadata = json.load(f)

        model_name = metadata["model"]

        num_classes = metadata["num_classes"]

        # =========================
        # CREATE MODEL
        # =========================

        model_wrapper = ModelFactory.create(

            model_name=model_name,

            num_classes=num_classes,

            device=device
        )

        # =========================
        # LOAD WEIGHTS
        # =========================

        model_path = os.path.join(
            results_folder,
            "model",
            "model.pth"
        )

        state_dict = torch.load(

            model_path,

            map_location=device
        )

        model_wrapper.load_state_dict(
            state_dict
        )

        model_wrapper.eval()

        print(f"Loaded model: {model_name}")

        return model_wrapper