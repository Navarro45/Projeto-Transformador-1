import os
import shutil
import random
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASSES = ["star", "galaxy", "quasar"]

# =========================
# VALIDAR IMAGEM
# =========================
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


# =========================
# CRIAR ESTRUTURA
# =========================
def create_structure():
    for split in SPLIT:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)


# =========================
# PROCESSAR CLASSE
# =========================
def process_class(cls):
    print(f"\nProcessando {cls}")

    class_path = os.path.join(RAW_DIR, cls)
    files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]

    valid_samples = []

    for f in files:
        img_path = os.path.join(class_path, f)

        if not is_valid_image(img_path):
            os.remove(img_path)
            continue

        valid_samples.append(f)

    print(f"Válidos: {len(valid_samples)}")

    random.shuffle(valid_samples)

    n = len(valid_samples)
    train_end = int(n * SPLIT["train"])
    val_end = train_end + int(n * SPLIT["val"])

    splits = {
        "train": valid_samples[:train_end],
        "val": valid_samples[train_end:val_end],
        "test": valid_samples[val_end:]
    }

    for split, files in splits.items():
        for f in files:
            src_img = os.path.join(class_path, f)
            dst_img = os.path.join(OUTPUT_DIR, split, cls, f)

            shutil.copy2(src_img, dst_img)

    print(f"{cls} OK")


def run():
    main()

# =========================
# MAIN
# =========================
def main():
    random.seed(42)

    create_structure()

    for cls in CLASSES:
        process_class(cls)

    print("\nDataset pronto ✔")


if __name__ == "__main__":
    main()
