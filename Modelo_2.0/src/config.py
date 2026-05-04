import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    EARLY_STOPPING_PATIENCE = 5

    NUM_CLASSES = 3
    SPECTRAL_LENGTH = 1024
    GENERATE_SPECTRUM_IF_MISSING = True
    SPECTRUM_GENERATOR_BACKEND = "mast_sdss"
    SPECTRUM_GENERATOR_CALLABLE = ""
    SPECTRUM_CACHE_ENABLED = True
    SPECTRUM_ALLOW_HEURISTIC_FALLBACK = True
    MAST_SDSS_PROVENANCE = "SEGUE"

    SEED = 42
    SAVE_HEATMAPS = True
    HEATMAP_MAX_IMAGES = 24

    PATHS = {
    "train": os.path.join(BASE_DIR, "data", "train"),
    "val": os.path.join(BASE_DIR, "data", "val"),
    "test": os.path.join(BASE_DIR, "data", "test"),
    "models": os.path.join(BASE_DIR, "outputs", "models"),
    "metrics": os.path.join(BASE_DIR, "outputs", "metrics"),
    "plots": os.path.join(BASE_DIR, "outputs", "plots"),
    "generated_spectra": os.path.join(BASE_DIR, "outputs", "generated_spectra"),
    "heatmaps": os.path.join(BASE_DIR, "outputs", "heatmaps"),
    "metadata": os.path.join(BASE_DIR, "data", "metadata.csv"),
}