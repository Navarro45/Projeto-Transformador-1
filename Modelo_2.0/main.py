import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import Config
from src.dataset import AstroDataset
from src.model import ImageModel
from src.train import train_model
from src.evaluate import evaluate
from src.utils import set_seed, ensure_dir

from pipelines.download import run as download_pipeline
from pipelines.preprocess import run as preprocess_pipeline
from pipelines.split import run as split_pipeline


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
    center_crop_size = int(config.IMAGE_SIZE * config.CENTER_CROP_RATIO)

    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(center_crop_size),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(center_crop_size),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor()
    ])

    train_ds = AstroDataset(config.PATHS["train"], train_transform)
    val_ds = AstroDataset(config.PATHS["val"], eval_transform)
    test_ds = AstroDataset(config.PATHS["test"], eval_transform)

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

    model = ImageModel(
        config.NUM_CLASSES,
        center_focus_sigma=config.CENTER_FOCUS_SIGMA,
        center_focus_strength=config.CENTER_FOCUS_STRENGTH
    )

    train_model(model, train_loader, val_loader, config)

    ensure_dir(config.PATHS["models"])

    torch.save(
        model.state_dict(),
        os.path.join(config.PATHS["models"], "image_model.pth")
    )

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
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Pula download e executa apenas preprocess + split."
    )

    args = parser.parse_args()

    print("\n INICIANDO PIPELINE COMPLETO\n")

    config = Config()
    set_seed(config.SEED)

    # 0️ Somente preparação de dados (sem download)
    if args.prepare_only:
        print("\n Executando apenas preprocess + split (sem download)...\n")
        preprocess_pipeline()
        split_pipeline()
        print("\n PREPARAÇÃO DE DADOS FINALIZADA")
        return

    # 1️ Download (opcional)
    if args.download:
        print("\n Download + preprocess completo...\n")

        download_pipeline()
        preprocess_pipeline()
        split_pipeline()

    # 2️ Pipeline padrão
    run_data_pipeline()

    # 3️ Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 4️ Treino imagem
    model = train_image_model(config, train_loader, val_loader)

    # 5️ Avaliação
    evaluate_all(model, test_loader, config)

    print("\n PIPELINE FINALIZADO")


if __name__ == "__main__":
    main()
