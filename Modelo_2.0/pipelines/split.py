import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = os.path.join(BASE_DIR, "data", "raw")
DST = os.path.join(BASE_DIR, "data")

CLASSES = ["star", "galaxy", "quasar"]

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}


def run():
    for cls in CLASSES:
        path = os.path.join(SRC, cls)

        if not os.path.exists(path):
            print(f"⚠️ Classe não encontrada: {cls}")
            continue

        files = [f for f in os.listdir(path) if f.endswith(".jpg")]

        random.shuffle(files)

        n = len(files)
        t1 = int(n * SPLIT["train"])
        t2 = t1 + int(n * SPLIT["val"])

        splits = {
            "train": files[:t1],
            "val": files[t1:t2],
            "test": files[t2:]
        }

        for split in splits:
            os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

            for f in splits[split]:
                src_img = os.path.join(path, f)
                dst_img = os.path.join(DST, split, cls, f)

                shutil.copy(src_img, dst_img)

    print("✅ Split finalizado")


if __name__ == "__main__":
    run()
