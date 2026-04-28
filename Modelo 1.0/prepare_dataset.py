import os
import shutil
import random
from PIL import Image

RAW_DIR = "data/raw"
OUTPUT_DIR = "data"

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASSES = ["estrelas", "galaxias", "quasares"]


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
# CRIAR ESTRUTURA FINAL
# =========================
def create_structure():
    for split in SPLIT.keys():
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)


# =========================
# PROCESSAMENTO
# =========================
def process_class(cls):
    print(f"\nProcessando: {cls}")

    class_path = os.path.join(RAW_DIR, cls)

    files = os.listdir(class_path)

    # Filtrar imagens válidas
    valid_files = []

    for f in files:
        path = os.path.join(class_path, f)

        if is_valid_image(path):
            valid_files.append(f)
        else:
            print(f"Removendo inválida: {f}")
            os.remove(path)

    print(f"Válidas: {len(valid_files)}")

    random.shuffle(valid_files)

    n = len(valid_files)
    train_end = int(n * SPLIT["train"])
    val_end = train_end + int(n * SPLIT["val"])

    splits = {
        "train": valid_files[:train_end],
        "val": valid_files[train_end:val_end],
        "test": valid_files[val_end:]
    }

    # Copiar arquivos
    for split, files in splits.items():
        for f in files:
            src = os.path.join(class_path, f)
            dst = os.path.join(OUTPUT_DIR, split, cls, f)

            shutil.copy2(src, dst)

    print(f"{cls} -> train:{len(splits['train'])}, val:{len(splits['val'])}, test:{len(splits['test'])}")


# =========================
# MAIN
# =========================
def main():
    random.seed(42)

    create_structure()

    for cls in CLASSES:
        process_class(cls)

    print("\nDataset pronto para treino ✔")


if __name__ == "__main__":
    main()