from __future__ import annotations

import pickle

import numpy as np
import torch


class SpectralInferencePipeline:
    def __init__(self, autoencoder, clustering_path: str, device):
        self.autoencoder = autoencoder.to(device)
        self.device = device
        with open(clustering_path, "rb") as f:
            payload = pickle.load(f)
        self.clusterer = payload["clusterer"]

    def predict_embedding(self, spectrum: np.ndarray) -> np.ndarray:
        self.autoencoder.eval()
        tensor = torch.tensor(spectrum[None, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            embedding = self.autoencoder.encode(tensor)
        return embedding.cpu().numpy()

    def predict_cluster(self, spectrum: np.ndarray):
        try:
            import hdbscan
        except ImportError as error:
            raise ImportError("hdbscan e necessario para inferencia espectral.") from error

        embedding = self.predict_embedding(spectrum)
        labels, probabilities = hdbscan.approximate_predict(
            self.clusterer,
            embedding,
        )
        return int(labels[0]), float(probabilities[0])
