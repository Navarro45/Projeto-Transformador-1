import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

NUM_CLASSES = 3
CLASS_NAMES = ["estrelas", "galaxias", "quasares"]

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

EARLY_STOPPING_PATIENCE = 5