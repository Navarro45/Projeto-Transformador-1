from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spectral.autoencoder import SpectralAutoencoder
from spectral.clustering import SpectralClusterer, SpectralClusteringConfig
from spectral.evaluator import evaluate_clusters
from spectral.metadata import load_spectral_metadata
from spectral.preprocessing import SpectralPreprocessor, SpectralPreprocessorConfig
from spectral.sdss_resolver import SDSSSpectrumResolver
from spectral.spectrum_generator import SpectrumGenerator
from spectral.trainer import SpectralAutoencoderTrainer, SpectralTrainingConfig
from utils.result_manager import ResultsManager


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _plot_history(history: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_representative_spectra(
    spectra: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    max_per_cluster: int = 3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_clusters = sorted(cluster for cluster in set(labels) if cluster != -1)

    for cluster_id in valid_clusters:
        indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        representative_indices = indices[np.argsort(distances)[:max_per_cluster]]

        plt.figure(figsize=(10, 5))
        for index in representative_indices:
            label = f"idx={index}"
            if "obs_id" in metadata.columns and pd.notna(metadata.iloc[index].get("obs_id")):
                label += f" obs={metadata.iloc[index].get('obs_id')}"
            if "subclass" in metadata.columns and pd.notna(metadata.iloc[index].get("subclass")):
                label += f" subclass={metadata.iloc[index].get('subclass')}"
            plt.plot(spectra[index], alpha=0.85, label=label)

        plt.title(f"Cluster {cluster_id} - espectros representativos")
        plt.xlabel("spectral grid index")
        plt.ylabel("normalized flux")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"cluster_{cluster_id}_representative_spectra.png")
        plt.close()


def _plot_subclass_projection(points, metadata: pd.DataFrame, output_path: Path, title: str) -> None:
    if "subclass" not in metadata.columns:
        return

    subclass = metadata["subclass"].fillna("unknown").astype(str)
    codes, uniques = pd.factorize(subclass)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=codes, cmap="tab20", s=24, alpha=0.85)
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    handles, _ = scatter.legend_elements()
    plt.legend(handles, uniques, title="SUBCLASS", fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_spectral_unsupervised(args) -> dict:
    if not args.spectral_metadata_csv:
        raise ValueError(
            "--spectral-metadata-csv e obrigatorio no modo spectral-unsupervised. "
            "Forneca um CSV confiavel com image_path e obs_id, ra/dec ou plate/mjd/fiber."
        )

    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    results = ResultsManager("spectral_unsupervised")
    base = Path(results.base_dir)
    spectral_dir = _mkdir(base / "spectral")
    spectra_dir = _mkdir(spectral_dir / "spectra")
    embeddings_dir = _mkdir(spectral_dir / "embeddings")
    clusters_dir = _mkdir(spectral_dir / "clusters")
    plots_dir = _mkdir(spectral_dir / "plots")
    models_dir = _mkdir(spectral_dir / "models")
    metadata_dir = _mkdir(spectral_dir / "metadata")

    load_result = load_spectral_metadata(
        csv_path=args.spectral_metadata_csv,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        max_samples=args.max_spectral_samples,
        seed=args.seed,
    )

    load_result.records.to_csv(metadata_dir / "accepted_metadata.csv", index=False)
    if not load_result.rejected.empty:
        load_result.rejected.to_csv(metadata_dir / "rejected_metadata.csv", index=False)

    resolver = SDSSSpectrumResolver()
    generator = SpectrumGenerator(
        cache_dir=args.spectral_cache_dir,
        cache_enabled=True,
        search_radius_arcsec=args.spectral_search_radius_arcsec,
    )
    preprocessor = SpectralPreprocessor(
        SpectralPreprocessorConfig(
            spectral_length=args.spectral_length,
            wavelength_min=args.spectral_wavelength_min,
            wavelength_max=args.spectral_wavelength_max,
        )
    )

    spectra = []
    metadata_rows = []
    failures = []
    cache_hits = 0
    downloads = 0

    for index, row in load_result.records.iterrows():
        try:
            resolution = resolver.resolve(row)
            fetched = generator.fetch(resolution)
            processed, quality = preprocessor.transform(
                fetched.wavelength,
                fetched.flux,
            )

            output_row = dict(fetched.metadata)
            output_row.update(quality)
            output_row["image_path"] = resolution.image_path
            output_row["obs_id"] = resolution.obs_id
            output_row["ra"] = resolution.ra
            output_row["dec"] = resolution.dec

            spectra.append(processed)
            metadata_rows.append(output_row)

            if fetched.cache_hit:
                cache_hits += 1
            else:
                downloads += 1

        except Exception as error:
            failure = row.to_dict()
            failure["failure_reason"] = str(error)
            failures.append(failure)
            print(f"[spectral] falha em {index}: {error}")

    failure_df = pd.DataFrame(failures)
    if not failure_df.empty:
        failure_df.to_csv(metadata_dir / "failures.csv", index=False)

    if len(spectra) < 2:
        summary = {
            "total_metadata_rows": int(load_result.summary["total_rows"]),
            "selected_rows": int(len(load_result.records)),
            "valid_spectra": int(len(spectra)),
            "cache_hits": int(cache_hits),
            "downloads": int(downloads),
            "failures": int(len(failures)),
            "status": "blocked",
            "reason": "menos de 2 espectros validos",
        }
        _save_json(spectral_dir / "cache_summary.json", summary)
        raise ValueError(
            "Menos de 2 espectros validos foram obtidos. Verifique metadata, MAST, FITS e filtros de qualidade."
        )

    spectra_array = np.stack(spectra, axis=0)
    spectral_metadata = pd.DataFrame(metadata_rows)

    np.save(spectra_dir / "spectra.npy", spectra_array)
    spectral_metadata.to_csv(metadata_dir / "spectral_metadata.csv", index=False)
    _save_json(models_dir / "preprocessing.json", preprocessor.to_dict())

    autoencoder = SpectralAutoencoder(
        input_dim=args.spectral_length,
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim,
    )
    training_config = SpectralTrainingConfig(
        epochs=args.ae_epochs,
        learning_rate=args.ae_lr,
        batch_size=args.ae_batch_size,
        early_stopping_patience=args.ae_early_stopping_patience,
    )
    trainer = SpectralAutoencoderTrainer(
        model=autoencoder,
        spectra=spectra_array,
        device=device,
        config=training_config,
        seed=args.seed,
    )
    history = trainer.train()

    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "input_dim": args.spectral_length,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.ae_hidden_dim,
        },
        models_dir / "autoencoder.pth",
    )
    _save_json(plots_dir / "training_history.json", history)
    _plot_history(history, plots_dir / "autoencoder_loss.png")

    embeddings = trainer.embeddings(spectra_array)
    np.save(embeddings_dir / "embeddings.npy", embeddings)

    clusterer = SpectralClusterer(
        SpectralClusteringConfig(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_method=args.hdbscan_cluster_selection_method,
            enable_umap=args.enable_umap,
            random_state=args.seed,
        )
    )
    clustering = clusterer.fit(embeddings)
    labels = clustering["labels"]

    np.save(clusters_dir / "cluster_labels.npy", labels)
    np.save(embeddings_dir / "pca_2d.npy", clustering["pca_2d"])
    if clustering["umap_2d"] is not None:
        np.save(embeddings_dir / "umap_2d.npy", clustering["umap_2d"])

    clusterer.save(models_dir / "hdbscan.pkl")

    spectral_metadata["cluster_id"] = labels
    spectral_metadata.to_csv(metadata_dir / "spectral_metadata.csv", index=False)

    SpectralClusterer.plot_projection(
        clustering["pca_2d"],
        labels,
        plots_dir / "pca_clusters.png",
        "PCA 2D - spectral clusters",
        "cluster_id",
    )
    _plot_subclass_projection(
        clustering["pca_2d"],
        spectral_metadata,
        plots_dir / "pca_subclass.png",
        "PCA 2D - SDSS SUBCLASS",
    )

    if clustering["umap_2d"] is not None:
        SpectralClusterer.plot_projection(
            clustering["umap_2d"],
            labels,
            plots_dir / "umap_clusters.png",
            "UMAP 2D - spectral clusters",
            "cluster_id",
        )
        _plot_subclass_projection(
            clustering["umap_2d"],
            spectral_metadata,
            plots_dir / "umap_subclass.png",
            "UMAP 2D - SDSS SUBCLASS",
        )

    _plot_representative_spectra(
        spectra_array,
        embeddings,
        labels,
        spectral_metadata,
        plots_dir / "representative_spectra",
    )

    metrics = evaluate_clusters(spectral_metadata, labels, clusters_dir)

    summary = {
        "total_metadata_rows": int(load_result.summary["total_rows"]),
        "selected_rows": int(len(load_result.records)),
        "valid_spectra": int(len(spectra_array)),
        "cache_hits": int(cache_hits),
        "downloads": int(downloads),
        "failures": int(len(failures)),
        "discarded_spectra": int(len(failures)),
        "n_clusters": metrics["n_clusters"],
        "n_outliers": metrics["n_outliers"],
        "device": str(device),
        "seed": args.seed,
        "metadata_csv": args.spectral_metadata_csv,
        "results_dir": results.base_dir,
        "autoencoder": {
            "input_dim": args.spectral_length,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.ae_hidden_dim,
            "training": asdict(training_config),
        },
        "clustering": {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
            "cluster_selection_method": args.hdbscan_cluster_selection_method,
            "enable_umap": args.enable_umap,
        },
    }

    _save_json(spectral_dir / "cache_summary.json", summary)
    results.save_metadata(summary)

    print("\nSpectral pipeline finished")
    print(f"Resultados: {results.base_dir}")
    print(f"Clusters: {summary['n_clusters']} | Outliers: {summary['n_outliers']}")
    print(f"Cache hits: {cache_hits} | Downloads: {downloads} | Falhas: {len(failures)}")

    return summary
