import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import Config
from src.dataset import AstroDataset
from src.model import ImageModel
from src.spectral_model import SpectralCNN
from src.hierarchical_model import HierarchicalModel
from src.train import train_model
from src.evaluate import evaluate
from src.utils import set_seed, ensure_dir

from pipelines.download import run as download_pipeline
from pipelines.preprocess import run as preprocess_pipeline
from pipelines.split import run as split_pipeline
from pipelines.unsupervised import run as unsupervised_pipeline

from src.train_spectral import train_spectral_per_class


# =========================
# DATA PIPELINE
# =========================
def run_data_pipeline(force_download=False):
    print("\n Verificando dataset...")

    train_path = os.path.join("data", "train")

    needs_data = force_download

    if not os.path.exists(train_path):
        needs_data = True

    if needs_data:
        print(" Executando pipeline completo de dados...")

        download_pipeline()
        preprocess_pipeline()
        split_pipeline()

    else:
        print("✅ Dataset já disponível")


# =========================
# DATALOADERS
# =========================
def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    train_ds = AstroDataset(config.PATHS["train"], transform)
    val_ds = AstroDataset(config.PATHS["val"], transform)
    test_ds = AstroDataset(config.PATHS["test"], transform)

    print(f"\n Dataset sizes:")
    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"Test: {len(test_ds)}")

    if len(train_ds) == 0:
        raise ValueError(" Train vazio")

    return (
        DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=config.BATCH_SIZE),
        DataLoader(test_ds, batch_size=config.BATCH_SIZE)
    )


# =========================
# TREINO IMAGEM
# =========================
def train_image_model(config, train_loader, val_loader):
    print("\n🏋️ Treinando modelo de imagem...")

    model = ImageModel(config.NUM_CLASSES)

    train_model(model, train_loader, val_loader, config)

    ensure_dir(config.PATHS["models"])

    torch.save(
        model.state_dict(),
        os.path.join(config.PATHS["models"], "image_model.pth")
    )

    return model


# =========================
# TREINO ESPECTRAL
# =========================
def train_spectral(config, train_loader, val_loader):
    print("\n Treinando modelo espectral (subclasses)...")

    train_spectral_per_class(train_loader, val_loader, config)

    print(" Modelo espectral treinado")


# =========================
# MODELO HIERÁRQUICO
# =========================
def build_hierarchical_model(image_model, config):
    print("\n Montando modelo hierárquico...")

    spectral_models = {
        0: SpectralCNN(config.SPECTRAL_LENGTH, 7).to(config.DEVICE),
        1: SpectralCNN(config.SPECTRAL_LENGTH, 3).to(config.DEVICE),
        2: SpectralCNN(config.SPECTRAL_LENGTH, 2).to(config.DEVICE),
    }

    model = HierarchicalModel(image_model, spectral_models)

    return model


# =========================
# AVALIAÇÃO
# =========================
def evaluate_all(model, test_loader, config):
    print("\n Avaliando modelo...")

    metrics = evaluate(model, test_loader, config.DEVICE, config)

    ensure_dir(config.PATHS["metrics"])

    import json
    with open(os.path.join(config.PATHS["metrics"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nResultados:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--unsupervised", action="store_true")

    args = parser.parse_args()

    print("\n INICIANDO PIPELINE COMPLETO\n")

    config = Config()
    set_seed(config.SEED)

    # 1️ Download (opcional)
    if args.download:
        print("\n Download + preprocess completo...\n")

        download_pipeline()
        preprocess_pipeline()
        split_pipeline()
        
    # 2️ Pipeline padrão
    run_data_pipeline()

    # 3️ Subclasses (MUITO IMPORTANTE)
    if args.unsupervised:
        unsupervised_pipeline()

    # 4️ Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 5️ Treino imagem
    image_model = train_image_model(config, train_loader, val_loader)

    # 6️ Treino espectral (usa pseudo-label)
    train_spectral(config, train_loader, val_loader)

    # 7️ Modelo hierárquico
    model = build_hierarchical_model(image_model, config)

    # 8️ Avaliação
    evaluate_all(model, test_loader, config)

    print("\n PIPELINE FINALIZADO")


if __name__ == "__main__":
    main()