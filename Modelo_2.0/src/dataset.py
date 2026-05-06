import os
from PIL import Image
from torch.utils.data import Dataset


class AstroDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        for label, cls in enumerate(["star", "galaxy", "quasar"]):
            cls_path = os.path.join(self.root_dir, cls)

            if not os.path.exists(cls_path):
                print(f"⚠️ Pasta não encontrada: {cls_path}")
                continue

            files = os.listdir(cls_path)
            print(f"🔎 {cls}: {len(files)} arquivos encontrados")

            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_path, file)
                    samples.append((img_path, label))

        print(f"\n📦 TOTAL FINAL DE SAMPLES: {len(samples)}\n")

        return samples

    def __len__(self):
        size = len(self.samples)

        if size == 0:
            print(f"⚠️ Dataset vazio em: {self.root_dir}")

        return size

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
