from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class SpectralClusteringConfig:
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_method: str = "eom"
    enable_umap: bool = False
    random_state: int = 42


class SpectralClusterer:
    def __init__(self, config: SpectralClusteringConfig):
        self.config = config
        self.pca = PCA(n_components=2, random_state=config.random_state)
        self.umap_model = None
        self.clusterer = None

    def fit(self, embeddings: np.ndarray) -> dict:
        pca_2d = self.pca.fit_transform(embeddings)
        umap_2d = None

        if self.config.enable_umap:
            try:
                import umap
            except ImportError as error:
                raise ImportError(
                    "umap-learn nao esta instalado. Remova --enable-umap ou instale a dependencia."
                ) from error

            self.umap_model = umap.UMAP(
                n_components=2,
                random_state=self.config.random_state,
            )
            umap_2d = self.umap_model.fit_transform(embeddings)

        try:
            import hdbscan
        except ImportError as error:
            raise ImportError("hdbscan e necessario para clustering espectral.") from error

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_method=self.config.cluster_selection_method,
            prediction_data=True,
        )
        labels = self.clusterer.fit_predict(embeddings)

        return {
            "labels": labels,
            "pca_2d": pca_2d,
            "umap_2d": umap_2d,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": asdict(self.config),
                    "pca": self.pca,
                    "umap_model": self.umap_model,
                    "clusterer": self.clusterer,
                },
                f,
            )

    @staticmethod
    def plot_projection(points, color_values, output_path, title, color_label):
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(
            points[:, 0],
            points[:, 1],
            c=color_values,
            cmap="tab20",
            s=24,
            alpha=0.85,
        )
        plt.title(title)
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.colorbar(scatter, label=color_label)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
